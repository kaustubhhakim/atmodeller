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
"""Non-linear solvers for chemical equilibrium and parameterised systems.

This module provides JAX-compatible solver utilities for efficiently handling both single-system
and batched systems of non-linear equations. The solvers are designed to integrate seamlessly with
JAX transformations support Equinox-based pytrees for flexible parameter handling.
"""

from collections.abc import Callable
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import lax, random
from jax.tree_util import tree_map
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray

from atmodeller.containers import Parameters
from atmodeller.engine import objective_function

LOG_NUMBER_DENSITY_VMAP_AXES: int = 0


def is_jax_array(element: Any) -> bool:
    """Returns ``True`` if ``element`` is a JAX array."""
    return isinstance(element, jax.Array)


def is_arraylike_batched(x: Any) -> Literal[0, None]:
    """Determines whether an object should be treated as batched along axis 0 for :func:`jax.vmap`.

    This function only considers JAX arrays for batching. While :func:`equinox.is_array` regards
    both JAX and NumPy arrays as arrays for tracing, NumPy arrays are treated here as static
    constants and are never batched. This allows fixed matrices (e.g., reaction matrices,
    stoichiometric formulae) to remain inside pytrees without being inadvertently vectorised.

    Batching rules for JAX arrays:
        - 1-D JAX arrays: Always considered batched along axis 0.
        - 2-D JAX arrays: Considered batched if they have more than one row; the leading axis
          (rows) is the batch dimension.
        - 0-D (scalar) JAX arrays: Not batched.

    All other objects (non-arrays or NumPy arrays) are treated as not batched.

    Args:
        x: Candidate object to check for batching.

    Returns:
        ``0`` if ``x`` should be batched along axis 0, otherwise ``None``.
    """
    if is_jax_array(x):
        # Vectorise over any 1-D array
        if x.ndim == 1:
            return 0
        # Any 2-D array (i.e., log_abundance in MassConstraints) should be vectorised over the
        # the first dimension if it is not unity
        elif x.ndim == 2 and x.shape[0] > 1:
            return 0


def vmap_axes_spec(x: Any) -> Any:
    """Recursively generate ``in_axes`` for :func:`jax.vmap` by checking if each leaf is batched.

    Unlike :func:`equinox.is_array`, which treats both JAX and NumPy arrays as arrays for tracing,
    batching here is only applied to JAX array leaves. This distinction is important because:

    - JAX arrays: These are the only leaves considered for batching. If a JAX array is 1-D or a
      2-D array with more than one row, its leading axis (0) is marked for vectorisation.
    - NumPy arrays: These are traced like arrays by Equinox but are treated as static constants
      here. They are not batched, which allows domain-specific matrices (e.g., formula matrices,
      reaction matrices, stoichiometry tables) to remain inside pytrees without being mistakenly
      vectorised.

    This behaviour ensures that only model inputs and outputs represented as JAX arrays
    participate in :func:`jax.vmap`, while fixed structural data stored as NumPy arrays are
    preserved as-is.

    Args:
        x: A pytree of nested containers possibly containing JAX arrays, NumPy arrays, or scalars.

    Returns:
        A pytree with the same structure as ``x``. Each leaf is ``0`` if batched, or ``None``
        if not.
    """
    return tree_map(is_arraylike_batched, x)


# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def solver_single(
    initial_guess: Float[Array, "..."], parameters: Parameters, objective_function: Callable
) -> tuple[Float[Array, "..."], Bool[Array, ""], Integer[Array, ""]]:
    """Solves a single system of non-linear equations.

    Args:
        initial_guess: Initial guess for the solution
        parameters: Model parameters required by the objective function and solver
        objective_function: Callable returning residuals for the system

    Returns:
        tuple:
            - solution: Array of solution values
            - status: Boolean scalar indicating whether the solver converged
            - steps: Integer scalar giving the number of iterations performed
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        parameters.solver_parameters.get_solver_instance(),
        initial_guess,
        args=parameters,
        throw=parameters.solver_parameters.throw,
        max_steps=parameters.solver_parameters.max_steps,
        options=parameters.solver_parameters.get_options(parameters.species.number_species),
    )

    solution: Float[Array, "..."] = sol.value
    status: Bool[Array, ""] = sol.result == optx.RESULTS.successful
    steps: Integer[Array, ""] = sol.stats["num_steps"]

    # jax.debug.print("solution = {out}", out=solution)
    # jax.debug.print("status = {out}", out=status)
    # jax.debug.print("steps = {out}", out=steps)

    return solution, status, steps


def get_solver_individual(parameters: Parameters) -> Callable:
    """Gets a vmapped, JIT-compiled solver for independent batch systems.

    Wraps :func:`solver_single` with :func:`equinox.filter_vmap` and :func:`filter_jit` so that it
    can solve multiple independent systems in a batch efficiently. Each batch element is solved
    separately, producing per-element convergence statistics.

    Args:
        parameters: Model parameters required by the objective function and solver

    Returns:
        Callable
    """
    solver_fn: Callable = eqx.Partial(solver_single, objective_function=objective_function)

    return eqx.filter_jit(
        eqx.filter_vmap(
            solver_fn, in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec(parameters))
        )
    )


def get_solver_batch(parameters: Parameters) -> Callable:
    """Gets a JIT-compiled solver for batched systems treated as a single problem.

    In this mode, the objective function is already vmapped across the batch dimension, so
    :func:`solver_single` sees the batch as one system. The solver returns a single convergence
    status and iteration count, which are broadcast to match the batch shape.

    Args:
        parameters: Model parameters required by the objective function and solver

    Returns:
        Callable
    """
    objective_vmap: Callable = eqx.filter_vmap(
        objective_function,
        in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec(parameters)),
    )
    solver_fn: Callable = eqx.Partial(solver_single, objective_function=objective_vmap)

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


@eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def repeat_solver(
    solver_fn: Callable,
    initial_guess: Float[Array, "batch solution"],
    parameters: Parameters,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "batch solution"],
    Bool[Array, " batch"],
    Integer[Array, " batch"],
    Integer[Array, " batch"],
]:
    """Repeatable multistart solver with perturbations for failed entries.

    This function attempts to solve a system of equations for a batch of initial solutions. If
    some entries fail to converge, it perturbs only the failing solutions and retries, up to a
    maximum number of attempts specified in
    :attr:`~atmodeller.containers.SolverParameters.multistart`.

    The perturbation is applied proportionally to
    :attr:`~atmodeller.containers.SolverParameters.multistart_perturbation`. Successful solutions
    are kept as-is, and the function tracks the iteration when each entry first succeeded.

    Args:
        solver_fn: Solver function
        initial_guess: Initial guess for the solution
        parameters: Model parameters required by the objective function and solver
        key: A JAX random key for generating perturbations

    Returns:
        tuple:
            - final_solution: Array of solution values
            - final_status: Boolean scalar indicating whether the solver converged
            - final_steps: Integer scalar giving the number of iterations performed
            - final_attempts: Success attempts
    """

    def body_fn(state: tuple[Array, ...]) -> tuple[Array, ...]:
        """Performs one iteration of the solver retry loop.

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

        new_solution, new_status, new_steps = solver_fn(new_initial_solution, parameters)

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
        """Checks if the solver should continue retrying.

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
    first_solution, first_solver_status, first_solver_steps = solver_fn(initial_guess, parameters)
    # jax.debug.print("first_solution = {out}", out=first_solution)
    # jax.debug.print("first_solver_status = {out}", out=first_solver_status)
    # jax.debug.print("first_solver_steps = {out}", out=first_solver_steps)

    # Failback solution
    solution = cast(Array, jnp.where(first_solver_status[:, None], first_solution, initial_guess))
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


def make_solve_tau_step(solver_fn: Callable, parameters: Parameters) -> Callable:
    """Factory function that creates a JIT-compiled solver step for a sequence of tau values.

    This wraps a repeatable, vmapped solver function (``solver_fn``) to update the ``tau``
    parameter dynamically at each step. The returned function is suitable for use in
    :func:`jax.lax.scan` to efficiently sweep over multiple tau values in a single compiled loop.

    Args:
        solver_fn: Solver function
        parameters: Template :class:`~atmodeller.containers.Parameters` object containing the
            full solver configuration. The ``tau`` leaf inside
            :class:`~atmodeller.containers.SolverParameters` will be replaced at each step.

    Returns:
        Callable
    """

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve_tau_step(carry: tuple, tau: Float[Array, " batch"]) -> tuple[tuple, tuple]:
        """Performs a single solver step for a given batch of tau values.

        This function is intended to be used inside :func``jax.lax.scan`` to iterate over multiple
        tau values efficiently. It updates the ``tau`` leaf in the parameters, calls the
        :func:`repeat_solver` for the current batch, and returns the updated carry and results.

        Args:
            carry: Tuple of carry values
            tau: Array of tau values for the current step in the scan.

        Returns:
            new carry tuple, output tuple
        """
        (key, solution) = carry
        key, subkey = jax.random.split(key)

        # Get new parameters with tau value
        get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731
        new_parameters: Parameters = eqx.tree_at(get_leaf, parameters, tau)
        # jax.debug.print("tau = {out}", out=new_parameters.solver_parameters.tau)

        new_solution, new_status, new_steps, success_attempt = repeat_solver(
            solver_fn, solution, new_parameters, subkey
        )

        new_carry: tuple[PRNGKeyArray, Float[Array, "batch solution"]] = (key, new_solution)

        # Output current solution for this tau step
        out: tuple[Array, ...] = (new_solution, new_status, new_steps, success_attempt)

        return new_carry, out

    return solve_tau_step
