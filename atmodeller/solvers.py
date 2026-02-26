# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Non-linear solvers for chemical equilibrium and parameterised systems

This module provides JAX-compatible solver utilities for efficiently handling both single-system
and batched systems of non-linear equations. The solvers are designed to integrate seamlessly with
JAX transformations and support Equinox-based pytrees for flexible parameter handling.
"""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from equinox._enum import EnumerationItem
from jax import lax
from jaxmod.solvers import MultiAttemptSolution, make_batch_retry_solver
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray
from optimistix import Solution

from atmodeller.constants import TAU, TAU_MAX, TAU_NUM
from atmodeller.engine import objective_function
from atmodeller.parameters import Parameters

# TODO: Can remove after refactoring but required for previous output routines
LOG_NUMBER_MOLES_VMAP_AXES: int = 0


# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def solve_single(
    initial_guess: Float[Array, "... n_solution"], parameters: Parameters
) -> optx.Solution:
    """Solves a single system.

    Args:
        initial_guess: Initial guess for the solution
        parameters: Parameters

    Returns:
        :class:`~optimistix.Solution` object
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

    return sol


batch_retry_solver: Callable = make_batch_retry_solver(solve_single, objective_function)


# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def tau_sweep_solver(
    initial_guess: Float[Array, "... solution"], parameters: Parameters, key: PRNGKeyArray
) -> MultiAttemptSolution:
    """Solves a batch of solutions for a sequence of tau values using a solver function.

    This function iterates over a set of tau values and applies the solver function to the
    batch of solutions at each tau step. It dynamically updates the ``tau`` value in the solver
    parameters for each iteration. This function is intended to be used inside
    :func:`jax.lax.scan` to efficiently sweep over multiple tau values in a single compiled
    loop.

    Args:
        initial_guess: Batched array of initial guesses for the solver
        parameters: Template :class:`~atmodeller.containers.Parameters` object containing the
            full solver configuration. The ``tau`` leaf inside
            :class:`~atmodeller.containers.SolverParameters` will be replaced at each step.
        key: JAX PRNG key for reproducible random perturbations

    Returns:
        :class:`~jaxmod.solvers.MultiAttemptSolution` object
    """

    def solve_tau_step(carry: tuple, tau: Float[Array, " batch"]) -> tuple[tuple, tuple]:
        """Performs a single solver step for a given batch of tau values.

        This function is intended to be used inside :func``jax.lax.scan`` to iterate over
        multiple tau values efficiently. It updates the ``tau`` leaf in the parameters, calls
        the :func:`repeat_solver` for the current batch, and returns the updated carry and
        results.

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

        new_sol: MultiAttemptSolution = batch_retry_solver(
            solution,
            new_parameters,
            subkey,
            parameters.solver_parameters.multistart_perturbation,
            parameters.solver_parameters.multistart,
            parameters.solver_parameters.atol,
        )

        new_solution: Float[Array, "batch solution"] = new_sol.value
        new_result: optx.RESULTS = new_sol.result
        new_steps: Integer[Array, " batch"] = new_sol.stats["num_steps"]
        success_attempt: Integer[Array, " batch"] = new_sol.attempts

        new_carry: tuple[PRNGKeyArray, Float[Array, "batch solution"]] = (key, new_solution)

        # Output current solution for this tau step
        out: tuple[Array, ...] = (new_solution, new_result._value, new_steps, success_attempt)  # pyright: ignore

        return new_carry, out

    varying_tau_row: Float[Array, " tau"] = jnp.logspace(
        jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
    )
    constant_tau_row: Float[Array, " tau"] = jnp.full((TAU_NUM,), TAU)
    tau_templates: Float[Array, "tau 2"] = jnp.stack([varying_tau_row, constant_tau_row], axis=1)

    # Create solver_status as a 1-D array of zeros with the batch dimension
    solver_status: Integer[Array, " batch"] = jnp.zeros(initial_guess.shape[0], dtype=int)

    tau_array: Float[Array, "tau batch"] = tau_templates[:, solver_status]

    initial_carry: tuple[Array, Array] = (key, initial_guess)

    _, results = jax.lax.scan(solve_tau_step, initial_carry, tau_array)
    solution, result_value, steps, attempts = results

    # Bundle the final outputs into a single optimistix Solution object
    final_result: optx.RESULTS = cast(
        optx.RESULTS,
        EnumerationItem(result_value[-1], optx.RESULTS),  # pyright: ignore
    )

    # NOTE: This solution instance does not return all the information from the solves, but it
    # encapsulates the most important (final) quantities. Aggregate output, solution and result
    # for final TAU
    sol: Solution = Solution(
        solution[-1], final_result, None, {"num_steps": jnp.max(steps, axis=0)}, None
    )
    multi_sol: MultiAttemptSolution = MultiAttemptSolution(sol, jnp.max(attempts, axis=0))

    return multi_sol


@eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=3)
def solve_with_jit(
    base_solution_array: Float[Array, "... solution"], parameters: Parameters, key: PRNGKeyArray
) -> MultiAttemptSolution:
    """Wrapped version of the solve function with JIT compilation for branching logic.

    Args:
        base_solution_array: Base solution array
        parameters: Parameters
        key: Random key

    Returns:
        :class:`~jaxmod.solvers.MultiAttemptSolution` object
    """
    # Define the condition to check if active stability is enabled
    condition: Bool[Array, ""] = jnp.any(parameters.reaction_system.species.active_stability)

    # HACK to debug jitting issues one solver at a time.

    # def solve_with_stability_multistart(key):
    #     """Function for multistart with stability"""
    #     subkey: PRNGKeyArray = jax.random.split(key)[1]  # Split only once and pass subkey
    #     return tau_sweep_solver(base_solution_array, parameters, subkey)

    def solve_with_generic_multistart(key):
        """Function for generic multistart"""
        subkey = jax.random.split(key)[1]  # Split only once and pass subkey
        return batch_retry_solver(
            base_solution_array,
            parameters,
            subkey,
            parameters.solver_parameters.multistart_perturbation,
            parameters.solver_parameters.multistart,
            parameters.solver_parameters.atol,
        )

    multi_sol = lax.cond(
        condition,
        # lambda _: solve_with_stability_multistart(key),  # True: Use stability solver
        # lambda _: solve_with_stability_multistart(key),  # True: Use stability solver
        lambda _: solve_with_generic_multistart(key),  # False: Use generic solver
        lambda _: solve_with_generic_multistart(key),  # False: Use generic solver
        operand=None,  # Operand not used for decision making
    )

    return multi_sol
