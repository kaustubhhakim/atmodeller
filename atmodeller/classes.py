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

import logging
import pprint
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Integer, PRNGKeyArray

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
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.mytypes import NpFloat, NpInt
from atmodeller.output import Output
from atmodeller.solver import make_vmapped_solver_function, repeat_solver
from atmodeller.utilities import get_batch_size, partial_rref

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
        planet_: Planet = Planet() if planet is None else planet

        batch_size: int = get_batch_size((planet, fugacity_constraints, mass_constraints))

        fugacity_constraints_: FugacityConstraints = FugacityConstraints.create(
            self.species, fugacity_constraints
        )
        mass_constraints_: MassConstraints = MassConstraints.create(
            self.species, mass_constraints, batch_size
        )
        traced_parameters_: TracedParameters = TracedParameters(
            planet_, fugacity_constraints_, mass_constraints_
        )
        fixed_parameters_: FixedParameters = self.get_fixed_parameters()
        solver_parameters_: SolverParameters = (
            SolverParameters() if solver_parameters is None else solver_parameters
        )
        options: dict[str, Any] = {
            "lower": self.species.get_lower_bound(),
            "upper": self.species.get_upper_bound(),
            "jac": solver_parameters_.jac,
        }

        # NOTE: Determine active entries in the residual. This order must correspond to the order
        # of entries in the residual.
        active: Bool[Array, " res_dim"] = jnp.concatenate(
            (
                fixed_parameters_.active_reactions(),
                fugacity_constraints_.active(),
                mass_constraints_.active(),
                fixed_parameters_.active_stability(),
            )
        )
        # jax.debug.print("active = {out}", out=active)
        active_indices: Integer[Array, "..."] = jnp.where(active)[0]
        # jax.debug.print("active_indices = {out}", out=active_indices)

        base_initial_solution: Array = broadcast_initial_solution(
            initial_log_number_density,
            initial_log_stability,
            self.species.number,
            batch_size,
        )
        # jax.debug.print("base_initial_solution = {out}", out=base_initial_solution)

        self._solver = make_vmapped_solver_function(
            traced_parameters_, fixed_parameters_, solver_parameters_, options
        )

        # First solution attempt
        solution, solver_status, solver_steps = self._solver(
            base_initial_solution,
            active_indices,
            jnp.array(TAU, dtype=float),  # Must declare dtype to avoid recompilation
            traced_parameters_,
        )

        if jnp.all(solver_status):
            logger.info("Solution found with first iteration")
            solver_attempts: Integer[Array, " batch_dim"] = jnp.ones_like(solver_status, dtype=int)

        else:
            logger.info("Some failed")
            logger.info("solver_status = %s", solver_status)
            logger.info("solver_steps = %s", solver_steps)

            logger.info("Initialising multistart")
            key: PRNGKeyArray = jax.random.PRNGKey(0)

            # TODO: Remove. Doesn't seem to help much
            # if jnp.any(fixed_parameters_.active_stability()):
            #     logger.info("Multistart with species' stability")
            #     # Initialize carry
            #     initial_carry: tuple = (
            #         base_initial_solution,
            #         active_indices,
            #         solver_parameters_.multistart_perturbation,
            #         solver_parameters_.multistart,
            #         key,
            #         base_initial_solution,  # Ignore solution because some are meaningless
            #         solver_status,
            #         solver_steps,
            #     )
            #     solve_tau_step: Callable = make_solve_tau_step(self._solver, traced_parameters_)
            #     # Calculate how many steps of 10x reduction are needed
            #     num_steps = int(jnp.log10(TAU_MAX) - jnp.log10(TAU)) + 1  # inclusive range
            #     tau_sequence = jnp.logspace(jnp.log10(TAU_MAX), jnp.log10(TAU), num=num_steps)
            #     _, results = jax.lax.scan(solve_tau_step, initial_carry, tau_sequence)
            #     solution, solver_status, solver_steps, solver_attempts = results

            #     # Just grab the last (final) tau solution
            #     solution = solution[-1]
            #     solver_status = solver_status[-1]
            #     solver_steps = solver_steps[-1]
            #     solver_attempts = solver_attempts[-1]

            # else:
            logger.info("Multistart without species' stability")
            solution, solver_status, solver_steps, solver_attempts = repeat_solver(
                self._solver,
                base_initial_solution,
                active_indices,
                jnp.array(TAU, dtype=float),
                traced_parameters_,
                solution,
                solver_status,
                solver_steps,
                multistart_perturbation=solver_parameters_.multistart_perturbation,
                max_attempts=solver_parameters_.multistart,
                key=key,
            )

        self._output = Output(
            self.species,
            solution,
            active_indices,
            solver_status,
            solver_steps,
            solver_attempts,
            fixed_parameters_,
            traced_parameters_,
            solver_parameters_,
        )

        num_total_models: int = solver_status.size
        num_successful_models: int = jnp.count_nonzero(solver_status).item()
        num_failed_models: int = jnp.count_nonzero(~solver_status).item()
        max_multistarts: int = jnp.max(solver_attempts).item()

        logger.info(f"Attempted to solve {num_total_models} model(s)")

        if num_failed_models > 0:
            logger.warning(
                f"Solve complete: {num_successful_models} successful model(s) and "
                f"{num_failed_models} model(s) failed.\n"
                "Try increasing 'multistart', for example:\n"
                "    solver_parameters = SolverParameters(multistart=20)\n"
                "and then pass solver_parameters to the solve method.\n"
                "You can also adjust 'multistart_perturbation', for example:\n"
                "    solver_parameters = SolverParameters(multistart=20, "
                "multistart_perturbation=40.0)"
            )
        else:
            logger.info(
                f"Solve complete: {num_successful_models} successful model(s)\n"
                f"The number of multistarts required was {max_multistarts}, "
                "which can depend on the choice of the random seed"
            )

        logger.info("Solver steps (max multistarts) = %s (%s)", solver_steps, max_multistarts)

        logger.debug("solution = %s", solution)

    def get_fixed_parameters(self) -> FixedParameters:
        """Gets fixed parameters.

        Returns:
            Fixed parameters
        """
        formula_matrix: NpInt = self.get_formula_matrix()
        reaction_matrix: NpFloat = self.get_reaction_matrix()
        gas_species_mask: Array = self.species.get_gas_species_mask()
        molar_masses: Array = self.species.get_molar_masses()
        diatomic_oxygen_index: int = self.species.get_diatomic_oxygen_index()

        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=jnp.asarray(formula_matrix),
            reaction_matrix=jnp.asarray(reaction_matrix),
            gas_species_mask=gas_species_mask,
            diatomic_oxygen_index=diatomic_oxygen_index,
            molar_masses=molar_masses,
        )

        return fixed_parameters

    def get_formula_matrix(self) -> NpInt:
        """Gets the formula matrix.

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            Formula matrix
        """
        unique_elements: tuple[str, ...] = self.species.get_unique_elements_in_species()
        formula_matrix: NpInt = np.zeros(
            (len(unique_elements), self.species.number), dtype=np.int_
        )

        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(self.species):
                count: int = 0
                try:
                    count = species_.data.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        # logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def get_reaction_matrix(self) -> NpFloat:
        """Gets the reaction matrix.

        Returns:
            A matrix of linearly independent reactions or an empty array if no reactions
        """
        if self.species.number == 1:
            logger.debug("Only one species therefore no reactions")
            return np.array([], dtype=np.float64)

        transpose_formula_matrix: NpInt = self.get_formula_matrix().T
        reaction_matrix: NpFloat = partial_rref(transpose_formula_matrix)

        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        reaction_matrix: NpFloat = self.get_reaction_matrix()
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
) -> NpFloat:
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
        A numpy array of shape (batch_size, dim), with values broadcast as needed

    Raises:
        ValueError: If the input array has an unexpected shape or inconsistent dimensions
    """
    if component is None:
        base: NpFloat = np.full((dim,), default_value, dtype=np.float64)
    else:
        component = np.asarray(component, dtype=jnp.float64)
        if component.ndim == 0:
            base = np.full((dim,), component.item(), dtype=np.float64)
        elif component.ndim == 1:
            if component.shape[0] != dim:
                raise ValueError(f"{name} should have shape ({dim},), got {component.shape}")
            base = component
        elif component.ndim == 2:
            if component.shape[0] != batch_size or component.shape[1] != dim:
                raise ValueError(
                    f"{name} should have shape ({batch_size}, {dim}), got {component.shape}"
                )
            # Replace NaNs with default_value
            component = np.where(np.isnan(component), default_value, component)
            return component
        else:
            raise ValueError(
                f"{name} must be a scalar, 1D, or 2D array, got shape {component.shape}"
            )

    # Promote 1D base to (batch_size, dim)
    return np.broadcast_to(base[None, :], (batch_size, dim))


def broadcast_initial_solution(
    initial_log_number_density: ArrayLike | None,
    initial_log_stability: ArrayLike | None,
    number_of_species: int,
    batch_size: int,
) -> Array:
    """Creates and broadcasts the initial solution to shape (batch_size, D)

    D = number_of_species + number_of_stability, i.e. the total number of solution quantities

    Args:
        initial_log_number_density: Initial log number density. Defaults to None.
        initial_log_stability: Initial log stability. Defaults to None.
        number_of_species: Number of species
        batch_size: Batch size

    Returns:
        Initial solution with shape (batch_size, D)
    """
    number_density: NpFloat = _broadcast_component(
        initial_log_number_density,
        INITIAL_LOG_NUMBER_DENSITY,
        number_of_species,
        batch_size,
        name="initial_log_number_density",
    )
    stability: NpFloat = _broadcast_component(
        initial_log_stability,
        INITIAL_LOG_STABILITY,
        number_of_species,
        batch_size,
        name="initial_log_stability",
    )

    return jnp.concatenate((number_density, stability), axis=-1)
