#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Classes"""

from __future__ import annotations

import logging
import pprint
from collections.abc import Mapping
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from atmodeller import INITIAL_LOG_NUMBER_DENSITY, INITIAL_LOG_STABILITY, TAU
from atmodeller.containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Planet,
    SolverParameters,
    SpeciesCollection,
    TracedParameters,
)
from atmodeller.engine import repeat_solver, solve
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.utilities import (
    get_batch_size,
    partial_rref,
    vmap_axes_spec,
)

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere

    This is the main class that the user interacts with to build interior-atmosphere systems,
    solve them, and retrieve the results.

    Args:
        species: Collection of species
        tau: Tau factor for species stability. Defaults to TAU.
    """

    _solver: Callable | None = None
    _output: Output | None = None

    def __init__(self, species: SpeciesCollection, tau: float = TAU):
        self.species: SpeciesCollection = species
        self.tau: float = tau
        logger.info("species = %s", [species.name for species in self.species])
        logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def output(self) -> Output:
        if self._output is None:
            raise AttributeError("Output has not been set.")

        return self._output

    def solve(
        self,
        *,
        planet: Planet | None = None,
        initial_log_number_density: ArrayLike | None = None,
        initial_log_stability: ArrayLike | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> None:
        """Solves the system and initialises an Output instance for processing the result

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density: Initial log number density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.
            solver_parameters: Solver parameters. Defaults to None.
        """
        if planet is None:
            planet_: Planet = Planet()
        else:
            planet_ = planet

        fugacity_constraints_: FugacityConstraints = FugacityConstraints.create(
            fugacity_constraints
        )
        mass_constraints_: MassConstraints = MassConstraints.create(mass_constraints)

        fixed_parameters_: FixedParameters = self.get_fixed_parameters(
            fugacity_constraints_, mass_constraints_
        )

        traced_parameters_: TracedParameters = TracedParameters(
            planet_, fugacity_constraints_, mass_constraints_
        )

        if solver_parameters is None:
            solver_parameters_: SolverParameters = SolverParameters()
        else:
            solver_parameters_ = solver_parameters

        options: dict[str, Any] = {
            "lower": self.species.get_lower_bound(),
            "upper": self.species.get_upper_bound(),
            "jac": solver_parameters_.jac,
        }

        # Pre-bind fixed configuration to avoid retracing and improve JIT efficiency
        solve_with_bindings: Callable = eqx.Partial(
            solve,
            fixed_parameters=fixed_parameters_,
            solver_parameters=solver_parameters_,
            options=options,
        )

        # Always batch over the initial solution, meaning the initial solution must be broadcast
        # appropriately.
        in_axes: Any = vmap_axes_spec(traced_parameters_)
        self._solver = eqx.filter_jit(eqx.filter_vmap(solve_with_bindings, in_axes=(0, in_axes)))

        batch_size: int = max(get_batch_size(traced_parameters_), 1)
        base_initial_solution: Array = broadcast_initial_solution(
            initial_log_number_density,
            initial_log_stability,
            self.species.number,
            self.species.number_of_stability(),
            batch_size,
        )
        # jax.debug.print("base_initial_solution = {out}", out=base_initial_solution)

        # First solution attempt
        solution, solver_status, solver_steps = self._solver(
            base_initial_solution,
            traced_parameters_,
        )

        # Use repeat solver to ensure all cases solve
        key: Array = jax.random.PRNGKey(0)
        final_i, solution, solver_status, solver_steps = repeat_solver(
            self._solver,
            base_initial_solution,
            solution,
            solver_status,
            solver_steps,
            traced_parameters_,
            multistart_perturbation=solver_parameters_.multistart_perturbation,
            max_attempts=solver_parameters_.multistart,
            key=key,
        )

        self._output = Output(
            self.species,
            solution,
            solver_status,
            solver_steps,
            fixed_parameters_,
            traced_parameters_,
        )

        num_total_models: int = solver_status.size
        num_successful_models: int = jnp.count_nonzero(solver_status).item()
        num_failed_models: int = jnp.count_nonzero(~solver_status).item()

        logger.info(f"Attempted to solve {num_total_models} model(s)")

        if num_failed_models > 0:
            logger.warning(
                f"Solve complete: {num_successful_models} successful model(s) and "
                f"{num_failed_models} model(s) failed.\n"
                "Try increasing 'multistart', for example:\n"
                "    solver_parameters = SolverParameters(multistart=5)\n"
                "and then pass solver_parameters to the solve method.\n"
                "You can also adjust 'multistart_perturbation', for example:\n"
                "    solver_parameters = SolverParameters(multistart=5, "
                "multistart_perturbation=40.0)"
            )
        else:
            required_multistarts: int = max(final_i, 1)
            logger.info(
                f"Solve complete: {num_successful_models} successful model(s)\n"
                f"The number of multistarts required was {required_multistarts}, "
                "which can depend on the choice of the random seed"
            )
            logger.info("solution = %s", solution)

    def get_fixed_parameters(
        self, fugacity_constraints: FugacityConstraints, mass_constraints: MassConstraints
    ) -> FixedParameters:
        """Gets fixed parameters.

        Args:
            fugacity_constraints: Fugacity constraints
            mass_constraints: Mass constraints

        Returns:
            Fixed parameters
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        reaction_stability_matrix: npt.NDArray[np.float_] = self.get_reaction_stability_matrix()
        gas_species_indices: Array = self.species.get_gas_species_indices()
        condensed_species_indices: Array = self.species.get_condensed_species_indices()
        stability_species_indices: Array = self.species.get_stability_species_indices()
        molar_masses: Array = self.species.get_molar_masses()
        diatomic_oxygen_index: int = self.species.get_diatomic_oxygen_index()

        # The complete formula matrix is not required for the calculation but it is used for
        # computing output quantities. So calculate and store it.
        formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix()

        # Formula matrix for elements that are constrained by mass constraints
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
        indices: list[int] = []
        for element in mass_constraints.log_abundance.keys():
            index: int = unique_elements.index(element)
            indices.append(index)
        formula_matrix_constraints: npt.NDArray[np.int_] = formula_matrix.copy()
        formula_matrix_constraints = formula_matrix_constraints[indices, :]

        # Fugacity constraint matrix and indices
        number_fugacity_constraints: int = len(fugacity_constraints.constraints)
        fugacity_species_indices_list: list[int] = []
        species_names: tuple[str, ...] = self.species.get_species_names()
        for species_name in fugacity_constraints.constraints.keys():
            index: int = species_names.index(species_name)
            fugacity_species_indices_list.append(index)
        fugacity_species_indices: Array = jnp.array(fugacity_species_indices_list, dtype=jnp.int_)
        fugacity_matrix: Array = jnp.identity(number_fugacity_constraints)

        # For fixed parameters all objects must be hashable because it is a static argument
        # tolist is important to convert numpy dtypes to standard Python, thus ensuring they are
        # not triggered as arrays by eqx.as_array
        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=formula_matrix,
            formula_matrix_constraints=formula_matrix_constraints,
            reaction_matrix=reaction_matrix,
            reaction_stability_matrix=reaction_stability_matrix,
            stability_species_indices=stability_species_indices,
            fugacity_matrix=fugacity_matrix,
            gas_species_indices=gas_species_indices,
            condensed_species_indices=condensed_species_indices,
            fugacity_species_indices=fugacity_species_indices,
            diatomic_oxygen_index=diatomic_oxygen_index,
            molar_masses=molar_masses,
            tau=self.tau,
        )

        return fixed_parameters

    def get_formula_matrix(self) -> npt.NDArray[np.int_]:
        """Gets the formula matrix.

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            Formula matrix
        """
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
        formula_matrix: npt.NDArray[np.int_] = np.zeros(
            (len(unique_elements), self.species.number), dtype=jnp.int_
        )

        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(self.species):
                count: int = 0
                try:
                    count = species_.data.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def get_unique_elements_in_species(self) -> tuple[str, ...]:
        """Gets unique elements.

        Args:
            species: A list of species

        Returns:
            Unique elements in the species ordered alphabetically
        """
        elements: list[str] = []
        for species_ in self.species:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        logger.debug("unique_elements_in_species = %s", sorted_elements)

        return tuple(sorted_elements)

    def get_reaction_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction matrix.

        Returns:
            A matrix of linearly independent reactions or an empty array if no reactions
        """
        if self.species.number == 1:
            logger.debug("Only one species therefore no reactions")
            return np.array([])

        transpose_formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix().T
        reaction_matrix: npt.NDArray[np.float_] = partial_rref(transpose_formula_matrix)

        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def get_reaction_stability_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction stability matrix.

        Returns:
            Reaction stability matrix
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        mask: npt.NDArray[np.bool_] = np.zeros_like(reaction_matrix, dtype=bool)

        if reaction_matrix.size > 0:
            # Find the species to solve for stability
            stability_bool: Array = self.species.get_stability_species_mask()
            mask[:, stability_bool] = True
            reaction_stability_matrix: npt.NDArray[np.float_] = reaction_matrix * mask
        else:
            reaction_stability_matrix = reaction_matrix

        logger.debug("reaction_stability_matrix = %s", reaction_stability_matrix)

        return reaction_stability_matrix

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        reactions: dict[int, str] = {}
        if reaction_matrix.size != 0:
            for reaction_index in range(reaction_matrix.shape[0]):
                reactants: str = ""
                products: str = ""
                for species_index, species_ in enumerate(self.species):
                    coeff: float = reaction_matrix[reaction_index, species_index].item()
                    if coeff != 0:
                        if coeff < 0:
                            reactants += f"{abs(coeff)} {species_.data.name} + "
                        else:
                            products += f"{coeff} {species_.data.name} + "

                reactants = reactants.rstrip(" + ")
                products = products.rstrip(" + ")
                reaction: str = f"{reactants} = {products}"
                reactions[reaction_index] = reaction

        return reactions


def _broadcast_component(
    component: ArrayLike | None, default_value: float, dim: int, batch_size: int, name: str
) -> Array:
    """Broadcasts a scalar, 1D, or 2D input array to shape (batch_size, dim).

    This function standardizes inputs that may be:
        - None (in which case a default value is used),
        - a scalar (promoted to a 1D array of length `dim`),
        - a 1D array of shape (`dim`,) (broadcast across the batch),
        - or a 2D array of shape (`batch_size`, `dim`) (used as-is).

    Args:
        component: The input data (or None), representing either a scalar, 1D array, or 2D array
        default_value: The default scalar value to use if `component` is None
        dim: The number of features or dimensions per batch item
        batch_size: The number of batch items
        name: Name of the component (used for error messages)

    Returns:
        A JAX array of shape (batch_size, dim), with values broadcast as needed

    Raises:
        ValueError: If the input array has an unexpected shape or inconsistent dimensions
    """
    if component is None:
        base: Array = jnp.full((dim,), default_value, dtype=jnp.float_)
    else:
        component = jnp.asarray(component, dtype=jnp.float_)
        if component.ndim == 0:
            base = jnp.full((dim,), component.item(), dtype=jnp.float_)
        elif component.ndim == 1:
            if component.shape[0] != dim:
                raise ValueError(f"{name} should have shape ({dim},), got {component.shape}")
            base = component
        elif component.ndim == 2:
            if component.shape[0] != batch_size or component.shape[1] != dim:
                raise ValueError(
                    f"{name} should have shape ({batch_size}, {dim}), got {component.shape}"
                )
            # NOTE: 2-D already so return
            return component
        else:
            raise ValueError(
                f"{name} must be a scalar, 1D, or 2D array, got shape {component.shape}"
            )

    # Promote 1D base to (batch_size, dim)
    return jnp.broadcast_to(base[None, :], (batch_size, dim))


def broadcast_initial_solution(
    initial_log_number_density: ArrayLike | None,
    initial_log_stability: ArrayLike | None,
    number_of_species: int,
    number_of_stability: int,
    batch_size: int,
) -> Array:
    """Creates and broadcasts the initial solution to shape (batch_size, D)

    D = number_of_species + number_of_stability, i.e. the total number of solution quantities

    Args:
        initial_log_number_density: Initial log number density
        initial_log_stability: Initial log stability
        number_of_species: Number of species
        number_of_stability: Number of species stability
        batch_size: Batch size

    Returns:
        Initial solution with shape (batch_size, D)
    """
    number_density: Array = _broadcast_component(
        initial_log_number_density,
        INITIAL_LOG_NUMBER_DENSITY,
        number_of_species,
        batch_size,
        name="initial_log_number_density",
    )
    stability: Array = _broadcast_component(
        initial_log_stability,
        INITIAL_LOG_STABILITY,
        number_of_stability,
        batch_size,
        name="initial_log_stability",
    )

    return jnp.concatenate((number_density, stability), axis=-1)
