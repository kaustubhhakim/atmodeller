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
"""Solver wrappers"""

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

from atmodeller.containers import FixedParameters, SolverParameters, TracedParameters
from atmodeller.engine import solve
from atmodeller.utilities import vmap_axes_spec


def make_solver_function(
    fixed_parameters: FixedParameters, solver_parameters: SolverParameters, options: dict[str, Any]
) -> Callable:
    """Makes the solver function with bindings.

    This pre-binds fixed configuration to avoid retracing and improve JIT efficiency.

    Args:
        fixed_parameters: Fixed parameters
        solver_parameters: Solver parameters
        options: Options

    Returns:
        Solver function with bindings
    """
    partial_solve: Callable = eqx.Partial(
        solve,
        fixed_parameters=fixed_parameters,
        solver_parameters=solver_parameters,
        options=options,
    )

    return partial_solve


def make_vmapped_solver_function(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    solver_parameters: SolverParameters,
    options: dict[str, Any],
) -> Callable:
    """Makes the vmapped solver function with bindings

    This pre-binds fixed configuration and applies vmapping over relevant quantities.

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        solver_parameters: Solver parameters
        options: Options

    Returns:
        Vmapped and jitted solver function with bindings
    """
    solver_fn: Callable = make_solver_function(fixed_parameters, solver_parameters, options)
    in_axes: TracedParameters = vmap_axes_spec(traced_parameters)

    # Prepare the vmapped solver function
    # Initial solution must be broadcast since it is always batched
    solver_vmap_fn: Callable = eqx.filter_vmap(solver_fn, in_axes=(0, None, None, in_axes))

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def wrapped_solver(
        solution_array: Float[Array, " batch_dim sol_dim"],
        active_indices: Integer[Array, " res_dim"],
        tau: Float[Array, ""],
        traced_parameters: TracedParameters,
    ) -> tuple[
        Float[Array, " batch_dim sol_dim"], Bool[Array, " batch_dim"], Integer[Array, " batch_dim"]
    ]:
        return solver_vmap_fn(solution_array, active_indices, tau, traced_parameters)

    return wrapped_solver


@eqx.filter_jit
# Useful for optimising how many times JAX compiles the solve function
# @eqx.debug.assert_max_traces(max_traces=1)
def repeat_solver(
    solver_fn: Callable,
    base_initial_solution: Float[Array, " batch_dim sol_dim"],
    active_indices: Integer[Array, " res_dim"],
    tau: Float[Array, ""],
    traced_parameters: TracedParameters,
    initial_solution: Float[Array, " batch_dim sol_dim"],
    initial_status: Bool[Array, " batch_dim"],
    initial_steps: Integer[Array, " batch_dim"],
    multistart_perturbation: float,
    max_attempts: int,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, " batch_dim sol_dim"],
    Bool[Array, " batch_dim"],
    Integer[Array, " batch_dim"],
    Integer[Array, " batch_dim"],
]:
    """Repeat solver with perturbed initial solution

    Args:
        solver_fn: Solver function with pre-bound fixed configuration
        base_initial_solution: Base initial solution to perturb if necessary
        active_indices: Indices of the residual array that are active
        tau: Tau parameter for species' stability
        traced_parameters: Traced parameters
        initial_solution: Initial solution after first solve
        initial_status: Initial status after first solve
        initial_steps: Initial steps after first solve
        multistart_perturbation: Multistart perturbation
        max_attempts: Maximum attempts
        key: Random key

    Returns:
        A tuple with the state
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

        failed_mask: Array = ~status
        key, subkey = jax.random.split(key)

        # Implements a simple perturbation of the base initial solution, but something more
        # sophisticated could be implemented, such as training a network or using a regressor
        # to inform the next guess of the models that failed from the ones that succeeded.
        perturb_shape: tuple[int, int] = (solution.shape[0], solution.shape[1])
        raw_perturb: Array = jax.random.uniform(
            subkey, shape=perturb_shape, minval=-1.0, maxval=1.0
        )
        perturbations: Array = jnp.where(
            failed_mask[:, None],
            multistart_perturbation * raw_perturb,
            jnp.zeros_like(solution),
        )
        new_initial_solution: Array = jnp.where(
            failed_mask[:, None], base_initial_solution + perturbations, solution
        )

        new_solution, new_status, new_steps = solver_fn(
            new_initial_solution, active_indices, tau, traced_parameters
        )

        # Determine which entries to update: previously failed, now succeeded
        update_mask: Array = (~status) & new_status

        # Update fields only for those that just succeeded
        updated_solution: Array = cast(
            Array, jnp.where(update_mask[:, None], new_solution, solution)
        )
        updated_status: Array = status | new_status
        updated_steps: Array = cast(Array, jnp.where(update_mask, new_steps, steps))
        updated_success_attempt: Array = jnp.where(update_mask, i, success_attempt)

        return (
            i + 1,
            key,
            updated_solution,
            updated_status,
            updated_steps,
            updated_success_attempt,
        )

    def cond_fn(state: tuple[Array, ...]) -> Array:
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
            A boolean array indicating whether retries should continue (True if
            any solution failed and attempts are still available)
        """
        i, _, _, status, _, _ = state

        return jnp.logical_and(i < max_attempts, jnp.any(~status))

    initial_state: tuple[Array, ...] = (
        jnp.array(1),  # A first solve has already been attempted before repeat_solver is called
        key,
        initial_solution,
        initial_status,
        initial_steps,
        jnp.asarray(initial_status, dtype=int),  # 1 if already solved, otherwise will be updated
    )
    _, _, final_solution, final_status, final_steps, final_success_attempt = lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    return final_solution, final_status, final_steps, final_success_attempt


def make_solve_tau_step(solver_fn: Callable, traced_parameters: TracedParameters) -> Callable:
    """Wraps the repeat solver to call it for different tau values

    Args:
        solver_fn: Solver function with pre-bound fixed configuration
        traced_parameters: Traced parameters

    Returns:
        Wrapped solver for a single tau value
    """

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve_tau_step(carry, tau):
        # Unpack carry state
        (
            base_initial_solution,
            active_indices,
            multistart_perturbation,
            max_attempts,
            key,
            initial_solution,
            initial_status,
            initial_steps,
        ) = carry

        # Call repeat_solver with current tau and previous solution as initial
        new_solution, new_status, new_steps, success_attempt = repeat_solver(
            solver_fn,
            base_initial_solution,
            active_indices,
            tau,
            traced_parameters,
            initial_solution,
            initial_status,
            initial_steps,
            multistart_perturbation,
            max_attempts,
            key,
        )

        # Update PRNG key for next iteration
        key, _ = jax.random.split(key)

        new_carry = (
            base_initial_solution,
            active_indices,
            multistart_perturbation,
            max_attempts,
            key,
            new_solution,
            # The repeat solver should always run for each value of tau
            jnp.zeros_like(initial_status, dtype=bool),
            jnp.zeros_like(initial_steps, dtype=int),
        )

        # Output current solution etc for this tau step
        out = (new_solution, new_status, new_steps, success_attempt)

        return new_carry, out

    return solve_tau_step
