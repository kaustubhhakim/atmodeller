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
"""Solvers"""

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import lax, random
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

from atmodeller.containers import Parameters
from atmodeller.engine import objective_function
from atmodeller.utilities import vmap_axes_spec

LOG_NUMBER_DENSITY_VMAP_AXES: int = 0


# Since this is the core driver function for the solver it remains useful for debugging to see how
# many times recompilation is triggered
# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def solver_single(
    solution: Float[Array, "..."],
    parameters: Parameters,
    objective_function: Callable,
    options: dict[str, Any],
) -> tuple[Float[Array, "..."], Bool[Array, ""], Integer[Array, ""]]:
    """Solves a single system of non-linear equations.

    Args:
        solution: Initial guess for the solution array
        parameters: Model parameters required by the objective function and solver
        objective_function: Callable returning residuals for the system
        options: Additional options to pass to the solver

    Returns:
        - solver_value: Array of computed solution values
        - solver_status: Boolean scalar indicating whether the solver converged successfully
        - solver_steps: Integer scalar giving the number of iterations performed
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        parameters.solver_parameters.get_solver_instance(),
        solution,
        args=parameters,
        throw=parameters.solver_parameters.throw,
        max_steps=parameters.solver_parameters.max_steps,
        options=options,
    )

    solver_value: Float[Array, "..."] = sol.value
    solver_status: Bool[Array, ""] = sol.result == optx.RESULTS.successful
    solver_steps: Integer[Array, ""] = sol.stats["num_steps"]

    # jax.debug.print("solver_value = {out}", out=solver_value)
    # jax.debug.print("solver_steps = {out}", out=solver_steps)
    # jax.debug.print("solver_status = {out}", out=solver_status)

    return solver_value, solver_status, solver_steps


def get_solver_individual(parameters: Parameters, options: dict[str, Any]) -> Callable:
    """Gets a vmapped, JIT-compiled solver for independent batch systems.

    Wraps `solver_single` with `equinox.filter_vmap` and `filter_jit` so that it can solve multiple
    independent systems in a batch efficiently. Each batch element is solved separately, producing
    per-element convergence statistics.

    Args:
        parameters: Model parameters used for all systems in the batch.
        options: Additional solver options to pass to each call of `solver_single`.

    Returns:
        Callable that takes:
            - solution: Array of initial guesses for each batch element
            - parameters: Model parameters
    """
    solver_fn: Callable = eqx.Partial(
        solver_single, objective_function=objective_function, options=options
    )

    return eqx.filter_jit(
        eqx.filter_vmap(
            solver_fn, in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec(parameters))
        )
    )


def get_solver_batch(parameters: Parameters, options: dict[str, Any]) -> Callable:
    """Gets a JIT-compiled solver for batched systems treated as a single problem.

    In this mode, the objective function is already vmapped across the batch dimension, so
    `solver_single` sees the batch as one system. The solver returns a single convergence status
    and iteration count, which are broadcast to match the batch shape.

    Args:
        parameters: Model parameters used for the batch
        options: Additional options to pass to `solver_single`

    Returns:
        Callable that takes:
            - solution: Array of initial guesses for all batch elements
            - parameters: Model parameters
    """
    objective_vmap: Callable = eqx.filter_vmap(
        objective_function,
        in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec(parameters)),
    )

    solver_fn: Callable = eqx.Partial(
        solver_single, objective_function=objective_vmap, options=options
    )

    @eqx.filter_jit
    def solver(
        solution: Array, parameters: Parameters
    ) -> tuple[Float[Array, " batch solution"], Bool[Array, " batch"], Integer[Array, " batch"]]:
        sol_value, solver_status, solver_steps = solver_fn(solution, parameters)

        # Broadcast scalars to match batch dimension
        batch_size: int = solution.shape[0]
        solver_status_b: Bool[Array, " batch"] = jnp.broadcast_to(solver_status, (batch_size,))
        solver_steps_b: Integer[Array, " batch"] = jnp.broadcast_to(solver_steps, (batch_size,))

        return sol_value, solver_status_b, solver_steps_b

    return solver


# TODO: Below need refreshing


@eqx.filter_jit
# Useful for optimising how many times JAX compiles the solve function
# @eqx.debug.assert_max_traces(max_traces=1)
def repeat_solver(
    solver_vmap_fn: Callable,
    solution: Float[Array, "batch solution"],
    parameters: Parameters,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "batch solution"],
    Bool[Array, " batch"],
    Integer[Array, " batch"],
    Integer[Array, " batch"],
]:
    """Repeat solver that perturbs the initial solution for cases that fail and tries again

    Args:
        solver_vmap_fn: Vmapped solver function with pre-bound fixed configuration
        solution: Solution
        parameters: Parameters
        key: Random key

    Returns:
        A tuple with the state: (solution, solver_status, solver_steps, solver_attempts)
    """

    def body_fn(state: tuple[Array, ...]) -> tuple[Array, ...]:
        """Perform one iteration of the solver retry loop

        Args:
            state: Tuple containing:
                i: Current attempt index
                key: PRNG key for random number generation
                solution: Current solution array
                status: Boolean array indicating successful solutions
                steps: Step count
                success_attempt: Integer array recording iteration of success for each entry

        Returns:
            Updated state tuple
        """
        i, key, solution, status, steps, success_attempt = state
        # jax.debug.print("Iteration: {out}", out=i)

        failed_mask: Bool[Array, " batch"] = ~status
        key, subkey = random.split(key)

        # Perturb the (initial) solution for cases that failed. Something more sophisticated could
        # be implemented, such as a regressor or neural network to inform failed cases based on
        # successful solves.
        perturb_shape: tuple[int, int] = (solution.shape[0], solution.shape[1])
        raw_perturb: Float[Array, "batch solution"] = random.uniform(
            subkey, shape=perturb_shape, minval=-1.0, maxval=1.0
        )
        perturbations: Float[Array, "batch solution"] = jnp.where(
            failed_mask[:, None],
            parameters.solver_parameters.multistart_perturbation * raw_perturb,
            jnp.zeros_like(solution),
        )
        new_initial_solution: Float[Array, "batch solution"] = solution + perturbations
        # jax.debug.print("new_initial_solution = {out}", out=new_initial_solution)

        new_solution, new_status, new_steps = solver_vmap_fn(new_initial_solution, parameters)

        # Determine which entries to update: previously failed, now succeeded
        update_mask: Bool[Array, " batch"] = failed_mask & new_status
        updated_i: Integer[Array, "..."] = i + 1
        updated_solution: Float[Array, "batch solution"] = cast(
            Array, jnp.where(update_mask[:, None], new_solution, solution)
        )
        updated_status: Bool[Array, " batch"] = status | new_status
        updated_steps: Integer[Array, " batch"] = cast(
            Array, jnp.where(update_mask, new_steps, steps)
        )
        updated_success_attempt: Array = jnp.where(update_mask, updated_i, success_attempt)

        return (
            updated_i,
            key,
            updated_solution,
            updated_status,
            updated_steps,
            updated_success_attempt,
        )

    def cond_fn(state: tuple[Array, ...]) -> Bool[Array, "..."]:
        """Check if the solver should continue retrying

        Args:
            state: Tuple containing:
                i: Current attempt index
                _: Unused (PRNG key)
                _: Unused (solution)
                status: Boolean array indicating success of each solution
                _: Unused (steps)
                _: Unused (success_attempt)

        Returns:
            A boolean array indicating whether retries should continue (True if any solution
            failed and attempts are still available)
        """
        i, _, _, status, _, _ = state

        # For debugging to force the loop to run to the maximum allowable value
        # return jnp.logical_and(i < parameters.solver_parameters.multistart, True)

        return jnp.logical_and(i < parameters.solver_parameters.multistart, jnp.any(~status))

    # Try first solution
    first_solution, first_solver_status, first_solver_steps = solver_vmap_fn(solution, parameters)
    # jax.debug.print("first_solution = {out}", out=first_solution)
    # jax.debug.print("first_solver_status = {out}", out=first_solver_status)
    # jax.debug.print("first_solver_steps = {out}", out=first_solver_steps)

    # Failback solution
    solution = cast(Array, jnp.where(first_solver_status[:, None], first_solution, solution))
    # jax.debug.print("solution = {out}", out=solution)

    initial_state: tuple[Array, ...] = (
        jnp.array(1),  # First attempt of the repeat_solver
        key,
        solution,
        first_solver_status,
        first_solver_steps,
        first_solver_status.astype(int),  # 1 for solved, otherwise 0
    )

    _, _, final_solution, final_status, final_steps, final_success_attempt = lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    return final_solution, final_status, final_steps, final_success_attempt


def make_solve_tau_step(
    solver_vmap_fn: Callable,
    parameters: Parameters,
    solver_parameters: SolverParameters,
) -> Callable:
    """Wraps the repeat solver to call it for different tau values

    Args:
        solver_vmap_fn: Vmapped solver function with pre-bound fixed configuration
        parameters: Parameters
        solver_parameters: Solver parameters

    Returns:
        Wrapped solver for a single tau value
    """

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve_tau_step(carry: tuple, tau: Float[Array, " batch"]) -> tuple[tuple, tuple]:
        (key, solution) = carry
        key, subkey = jax.random.split(key)

        new_solution, new_status, new_steps, success_attempt = repeat_solver(
            solver_vmap_fn,
            tau,
            solution,
            parameters,
            solver_parameters,
            subkey,
        )

        new_carry: tuple[PRNGKeyArray, Float[Array, "batch solution"]] = (key, new_solution)

        # Output current solution etc for this tau step
        out: tuple[Array, ...] = (new_solution, new_status, new_steps, success_attempt)

        return new_carry, out

    return solve_tau_step
