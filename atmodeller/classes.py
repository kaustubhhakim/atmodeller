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
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray

from atmodeller import INITIAL_LOG_NUMBER_DENSITY, INITIAL_LOG_STABILITY, TAU, TAU_MAX, TAU_NUM
from atmodeller.containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Planet,
    SolverParameters,
    SpeciesCollection,
    TracedParameters,
)
from atmodeller.engine import make_solve_tau_step, repeat_solver, solve
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.mytypes import NpFloat, NpInt
from atmodeller.output import Output
from atmodeller.utilities import get_batch_size, partial_rref, vmap_axes_spec

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

        # Always broadcast tau because the repeat_solver is triggered if some cases fail
        broadcasted_tau: Float[Array, " batch_dim"] = jnp.full((batch_size,), TAU)
        # jax.debug.print("broadcasted_tau = {out}", out=broadcasted_tau)

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
                fugacity_constraints_.active(),
                fixed_parameters_.active_reactions(),
                mass_constraints_.active(),
                fixed_parameters_.active_stability(),
            )
        )
        # jax.debug.print("active = {out}", out=active)
        active_indices: Integer[Array, "..."] = jnp.where(active)[0]
        # jax.debug.print("active_indices = {out}", out=active_indices)

        base_solution_array: Array = broadcast_initial_solution(
            initial_log_number_density,
            initial_log_stability,
            self.species.number,
            batch_size,
        )
        # jax.debug.print("base_solution_array = {out}", out=base_solution_array)

        # Pre-bind fixed configurations
        solver_fn: Callable = eqx.Partial(
            solve,
            # solver_parameters=solver_parameters_,
            fixed_parameters=fixed_parameters_,
            options=options,
        )
        in_axes: TracedParameters = vmap_axes_spec(traced_parameters_)

        # Compile the solver, and this is re-used unless recompilation is triggered
        # Initial solution and tau must be broadcast since they are always batched
        self._solver = eqx.filter_jit(
            eqx.filter_vmap(solver_fn, in_axes=(0, None, 0, in_axes, None))
        )

        # First solution attempt. If the initial guess is close enough we might just find solutions
        # for all cases.
        logger.info(f"Attempting to solve {batch_size} model(s)")
        solution, solver_status, solver_steps = self._solver(
            base_solution_array,
            active_indices,
            broadcasted_tau,
            traced_parameters_,
            solver_parameters_,
        )
        # jax.debug.print("solver_status = {out}", out=solver_status)
        # jax.debug.print("solver_steps = {out}", out=solver_steps)
        solver_attempts: Integer[Array, " batch_dim"] = solver_status.astype(int)
        # jax.debug.print("solver_attempts = {out}", out=solver_attempts)

        if jnp.any(~solver_status):
            num_failed: int = jnp.sum(~solver_status).item()
            logger.warning("%d model(s) failed to converge on the first attempt", num_failed)
            logger.warning(
                "But don't panic! This can happen when starting from a poor initial guess."
            )
            logger.warning(
                "Launching multistart (maximum %d attempts)", solver_parameters_.multistart
            )
            logger.warning(
                "Attempting to solve the %d models(s) that initially failed", num_failed
            )

            # Restore the base solution for cases that failed since this will be perturbed
            solution: Float[Array, "batch_dim sol_dim"] = cast(
                Array, jnp.where(solver_status[:, None], solution, base_solution_array)
            )
            # jax.debug.print("solution = {out}", out=solution)

            # Use repeat solver to ensure all cases solve
            key: PRNGKeyArray = jax.random.PRNGKey(0)
            key, subkey = random.split(key)

            # Prototyping switching the solver for
            # solver_parameters_ = eqx.tree_at(
            #     lambda sp: sp.solver,
            #     solver_parameters_,  # your original instance
            #     optx.LevenbergMarquardt,  # or whatever solver you want to use
            # )
            # print(new_solver_params)

            if jnp.any(fixed_parameters_.active_stability()):
                logger.info(
                    "Multistart with species' stability (TAU_MAX= %.1e, TAU= %.1e, TAU_NUM= %d)",
                    TAU_MAX,
                    TAU,
                    TAU_NUM,
                )
                varying_tau_row: Float[Array, " tau_dim"] = jnp.logspace(
                    jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
                )
                constant_tau_row: Float[Array, " tau_dim"] = jnp.full((TAU_NUM,), TAU)
                tau_templates: Float[Array, "tau_dim 2"] = jnp.stack(
                    [varying_tau_row, constant_tau_row], axis=1
                )
                tau_array: Float[Array, "tau_dim batch_dim"] = tau_templates[
                    :, solver_status.astype(int)
                ]
                # jax.debug.print("tau_array = {out}", out=tau_array)

                initial_carry: tuple[Array, Array] = (subkey, solution)
                solve_tau_step: Callable = make_solve_tau_step(
                    self._solver, active_indices, traced_parameters_, solver_parameters_
                )
                _, results = jax.lax.scan(solve_tau_step, initial_carry, tau_array)
                solution, solver_status_, solver_steps_, solver_attempts = results

                # Debugging output. Requires the complete arrays as given above.
                failed_indices: Integer[Array, "..."] = jnp.where(~solver_status)[0]
                for ii in failed_indices.tolist():
                    logger.debug(f"--- Solve summary for failed index {ii} ---")
                    for tau_i in range(TAU_NUM):
                        status_i: bool = bool(solver_status_[tau_i, ii])
                        steps_i: int = int(solver_steps_[tau_i, ii])
                        attempts_i: int = int(solver_attempts[tau_i, ii])
                        logger.debug(
                            "Tau step %1d: status= %-5s  steps= %3d  attempts= %2d",
                            tau_i,
                            str(status_i),
                            steps_i,
                            attempts_i,
                        )

                # Aggregate output
                solution = solution[-1]  # Only need solution for final TAU
                solver_status_ = solver_status_[-1]  # Only need status for final TAU
                solver_steps_ = jnp.sum(solver_steps_, axis=0)  # Sum steps for all tau
                solver_attempts = jnp.max(solver_attempts, axis=0)  # Max for all tau

                # jax.debug.print("solution = {out}", out=solution)
                # jax.debug.print("solver_status_ = {out}", out=solver_status_)
                # jax.debug.print("solver_steps_ = {out}", out=solver_steps_)
                # jax.debug.print("solver_attempts = {out}", out=solver_attempts)

                # Maximum attempts across all tau and all models
                max_attempts: int = jnp.max(solver_attempts).item()

            else:
                solution, solver_status_, solver_steps_, solver_attempts = repeat_solver(
                    self._solver,
                    active_indices,
                    broadcasted_tau,
                    solution,
                    traced_parameters_,
                    solver_parameters_,
                    subkey,
                )
                max_attempts = jnp.max(solver_attempts).item()
                # Since tau is unaltered, the first multistart just repeats the first calculation,
                # which we already know has some failed cases. So we minus one for the reporting.
                max_attempts -= 1

            logger.info("Multistart complete with %s total attempt(s)", max_attempts)

            # Restore statistics of cases that solved first time
            solver_steps: Integer[Array, " batch_dim"] = jnp.where(
                solver_status, solver_steps, solver_steps_
            )
            solver_status: Bool[Array, " batch_dim"] = solver_status_  # Final status

            # Count unique values and their frequencies
            unique_vals, counts = jnp.unique(solver_attempts, return_counts=True)
            for val, count in zip(unique_vals.tolist(), counts.tolist()):
                logger.info(
                    "Multistart, max attempts: %d, model count: %d (%0.2f%%)",
                    val,
                    count,
                    count * 100 / batch_size,
                )

        num_successful_models: int = jnp.count_nonzero(solver_status).item()
        num_failed_models: int = jnp.count_nonzero(~solver_status).item()

        logger.info(
            "Solve complete: %d (%0.2f%%) successful model(s)",
            num_successful_models,
            num_successful_models * 100 / batch_size,
        )

        if num_failed_models > 0:
            logger.warning(
                "%d (%0.2f%%) model(s) still failed",
                num_failed_models,
                num_failed_models * 100 / batch_size,
            )

        logger.debug("Solver steps = %s", solver_steps)

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
