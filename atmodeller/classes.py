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
from collections.abc import Callable, Mapping
from typing import Optional, cast

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray

from atmodeller._mytypes import NpFloat
from atmodeller.constants import (
    INITIAL_LOG_NUMBER_DENSITY,
    INITIAL_LOG_STABILITY,
    TAU,
    TAU_MAX,
    TAU_NUM,
)
from atmodeller.containers import Parameters, Planet, SolverParameters, SpeciesCollection
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output, OutputSolution
from atmodeller.solvers import get_solver_individual, make_solve_tau_step, repeat_solver
from atmodeller.utilities import get_batch_size

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere

    This is the main class that the user interacts with to build interior-atmosphere systems,
    solve them, and retrieve the results.

    Args:
        species: Collection of species
    """

    _solver: Optional[Callable] = None
    _output: Optional[Output] = None

    def __init__(self, species: SpeciesCollection):
        self.species: SpeciesCollection = species
        logger.info("species = %s", str(self.species))
        # FIXME: To reinstate
        # logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def output(self) -> Output:
        if self._output is None:
            raise AttributeError("Output has not been set.")

        return self._output

    def calculate_disequilibrium(
        self,
        *,
        planet: Planet,
        log_number_density: ArrayLike,
    ) -> None:
        """Computes the Gibbs free energy disequilibrium.

        This method calculates the Gibbs free energy difference (ΔG) for each considered reaction
        relative to equilibrium, based on the current state of the system. A value of zero
        indicates a reaction at equilibrium, while positive or negative values indicate departures
        from equilibrium in terms of energetic favourability.

        Args:
            planet: Planet
            log_number_density: Log number density
        """
        parameters: Parameters = Parameters.create(self.species, planet)
        batch_size: int = get_batch_size(planet)
        solution_array: Array = broadcast_initial_solution(
            log_number_density, None, self.species.number_species, batch_size
        )
        # jax.debug.print("solution_array = {out}", out=solution_array)

        self._output = Output(self.species, solution_array, parameters)

    def solve(
        self,
        *,
        planet: Optional[Planet] = None,
        initial_log_number_density: Optional[ArrayLike] = None,
        initial_log_stability: Optional[ArrayLike] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ) -> None:
        """Solves the system and initialises an Output instance for processing the result

        Args:
            planet: Planet. Defaults to `None`.
            initial_log_number_density: Initial log number density. Defaults to `None`.
            initial_log_stability: Initial log stability. Defaults to `None`.
            fugacity_constraints: Fugacity constraints. Defaults to `None`.
            mass_constraints: Mass constraints. Defaults to `None`.
            solver_parameters: Solver parameters. Defaults to `None`.
        """
        parameters: Parameters = Parameters.create(
            self.species, planet, fugacity_constraints, mass_constraints, solver_parameters
        )

        batch_size: int = get_batch_size((planet, fugacity_constraints, mass_constraints))
        base_solution_array: Array = broadcast_initial_solution(
            initial_log_number_density,
            initial_log_stability,
            self.species.number_species,
            batch_size,
        )
        # jax.debug.print("base_solution_array = {out}", out=base_solution_array)

        self._solver = get_solver_individual(parameters)
        # Another option to solve the batch with a single root find
        # self._solver = get_solver_batch(parameters)

        # First solution attempt. If the initial guess is close enough we might just find solutions
        # for all cases.
        logger.info(f"Attempting to solve {batch_size} model(s)")
        solution, solver_status, solver_steps = self._solver(base_solution_array, parameters)
        jax.debug.print("solution = {out}", out=solution)
        jax.debug.print("solver_status = {out}", out=solver_status)
        jax.debug.print("solver_steps = {out}", out=solver_steps)

        solver_attempts: Integer[Array, "..."] = solver_status.astype(int)
        jax.debug.print("solver_attempts = {out}", out=solver_attempts)

        if jnp.any(~solver_status):
            num_failed: int = jnp.sum(~solver_status).item()
            logger.warning("%d model(s) failed to converge on the first attempt", num_failed)
            logger.warning(
                "But don't panic! This can happen when starting from a poor initial guess."
            )
            logger.warning(
                "Launching multistart (maximum %d attempts)",
                parameters.solver_parameters.multistart,
            )
            logger.warning(
                "Attempting to solve the %d models(s) that initially failed", num_failed
            )

            # Restore the base solution for cases that failed since this will be perturbed
            solution: Float[Array, "batch solution"] = cast(
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

            if jnp.any(parameters.species.active_stability):
                logger.info(
                    "Multistart with species' stability (TAU_MAX= %.1e, TAU= %.1e, TAU_NUM= %d)",
                    TAU_MAX,
                    TAU,
                    TAU_NUM,
                )
                varying_tau_row: Float[Array, " tau"] = jnp.logspace(
                    jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
                )
                constant_tau_row: Float[Array, " tau"] = jnp.full((TAU_NUM,), TAU)
                tau_templates: Float[Array, "tau 2"] = jnp.stack(
                    [varying_tau_row, constant_tau_row], axis=1
                )
                tau_array: Float[Array, "tau batch"] = tau_templates[:, solver_status.astype(int)]
                # jax.debug.print("tau_array = {out}", out=tau_array)

                initial_carry: tuple[Array, Array] = (subkey, solution)
                solve_tau_step: Callable = make_solve_tau_step(self._solver, parameters)
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
                    self._solver, solution, parameters, subkey
                )
                max_attempts = jnp.max(solver_attempts).item()
                # Since tau is unaltered, the first multistart just repeats the first calculation,
                # which we already know has some failed cases. So we minus one for the reporting.
                max_attempts -= 1

            logger.info("Multistart complete with %s total attempt(s)", max_attempts)

            # Restore statistics of cases that solved first time
            solver_steps: Integer[Array, " batch"] = jnp.where(
                solver_status, solver_steps, solver_steps_
            )
            solver_status: Bool[Array, " batch"] = solver_status_  # Final status

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

        logger.info("Solver steps (max) = %s", jnp.max(solver_steps).item())

        self._output = OutputSolution(
            self.species, solution, parameters, solver_status, solver_steps, solver_attempts
        )


def _broadcast_component(
    component: Optional[ArrayLike], default_value: float, dim: int, batch_size: int, name: str
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
    initial_log_number_density: Optional[ArrayLike],
    initial_log_stability: Optional[ArrayLike],
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
