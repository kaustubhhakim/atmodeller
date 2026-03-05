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
from jaxmod.utils import vmap_axes_spec
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray
from optimistix import Solution

from atmodeller.constants import TAU, TAU_MAX, TAU_NUM
from atmodeller.engine import objective_function
from atmodeller.parameters import Parameters

LOG_NUMBER_MOLES_VMAP_AXES: int = 0
"""Axis index for the log number of moles in the vmapped solver."""


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


def make_independent_solver(parameters: Parameters) -> Callable:
    """Gets a vmapped, JIT-compiled solver for independent batch systems.

    Wraps :func:`solve_single` with :func:`equinox.filter_vmap` and
    :func:`equinox.filter_jit` so that it can solve multiple independent systems in a batch
    efficiently. Each batch element is solved separately, producing per-element convergence
    statistics.

    Args:
        parameters: Parameters

    Returns:
        Callable that returns a :class:`MultiAttemptSolution` object
    """
    solver_function_vmapped: Callable = eqx.filter_vmap(
        solve_single, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )

    @eqx.filter_jit
    def solver(solution: Array, parameters: Parameters, *args) -> MultiAttemptSolution:
        """Solver

        Args:
            solution: Solution
            parameters: Parameters
            *args: Unused positional arguments for consistency with the solver interface

        Returns:
            :class:`MultiAttemptSolution` object
        """
        del args
        sol: optx.Solution = solver_function_vmapped(solution, parameters)

        return MultiAttemptSolution(sol, _attempts=1)

    return solver


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

    def solve_tau_step(carry: tuple, tau: Float[Array, " ..."]) -> tuple[tuple, tuple]:
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

        new_solution: Float[Array, "... solution"] = new_sol.value
        new_result: optx.RESULTS = new_sol.result
        new_steps: Integer[Array, "..."] = new_sol.stats["num_steps"]
        success_attempt: Integer[Array, "..."] = new_sol.attempts

        new_carry: tuple[PRNGKeyArray, Float[Array, "... solution"]] = (key, new_solution)

        # Output current solution for this tau step
        out: tuple[Array, ...] = (new_solution, new_result._value, new_steps, success_attempt)  # pyright: ignore

        return new_carry, out

    # Initial solve at TAU to check which entries need the sweep
    key, subkey = jax.random.split(key)
    get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731
    initial_parameters: Parameters = eqx.tree_at(get_leaf, parameters, jnp.array(TAU))

    first_sol: MultiAttemptSolution = batch_retry_solver(
        initial_guess,
        initial_parameters,
        subkey,
        parameters.solver_parameters.multistart_perturbation,
        parameters.solver_parameters.multistart,
        parameters.solver_parameters.atol,
    )
    first_solution: Float[Array, "... solution"] = first_sol.value
    # jax.debug.print("first_solution = {out}", out=first_solution)
    first_converged: Bool[Array, "..."] = first_sol.attempts > 0
    # jax.debug.print("first_converged = {out}", out=first_converged)

    # Build per-entry tau schedules of shape (TAU_NUM, *batch_shape)
    # Converged entries: [TAU, TAU, ..., TAU]
    # Failed entries:    [TAU_MAX, ..., TAU]  (log-spaced)
    varying_schedule: Float[Array, " tau"] = jnp.logspace(
        jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
    )
    # jax.debug.print("varying_schedule = {out}", out=varying_schedule)
    constant_schedule: Float[Array, " tau"] = jnp.full((TAU_NUM,), TAU)
    # jax.debug.print("constant_schedule = {out}", out=constant_schedule)

    batch_shape: tuple[int, ...] = initial_guess.shape[:-1]  # () or (N,) or (N, M)
    # jax.debug.print("batch_shape = {out}", out=batch_shape)

    # Reshape schedules to (TAU_NUM, *batch_shape) via broadcasting
    varying = varying_schedule.reshape((TAU_NUM,) + (1,) * len(batch_shape))
    # jax.debug.print("varying = {out}", out=varying)
    constant = constant_schedule.reshape((TAU_NUM,) + (1,) * len(batch_shape))
    # jax.debug.print("constant = {out}", out=constant)

    # first_converged shape: (*batch_shape,) -> (1, *batch_shape) for broadcasting
    tau_schedule: Float[Array, "tau ..."] = jnp.where(
        first_converged[None, ...], constant, varying
    )  # shape: (TAU_NUM, *batch_shape)

    def run_scan(key_and_guess: tuple) -> MultiAttemptSolution:
        """Run the full tau sweep scan (used when some entries failed the initial solve)."""
        key_, _ = key_and_guess
        initial_carry_: tuple[Array, Array] = (key_, first_solution)
        _, results_ = jax.lax.scan(solve_tau_step, initial_carry_, tau_schedule)
        solution_, result_value_, steps_, attempts_ = results_
        final_result_: optx.RESULTS = cast(
            optx.RESULTS,
            EnumerationItem(result_value_[-1], optx.RESULTS),  # pyright: ignore
        )
        sol_: Solution = Solution(
            solution_[-1], final_result_, None, {"num_steps": jnp.max(steps_, axis=0)}, None
        )
        return MultiAttemptSolution(sol_, jnp.max(attempts_, axis=0))

    def run_single_step(_: tuple) -> MultiAttemptSolution:
        """All entries converged at TAU on the first attempt: return immediately."""
        # jax.debug.print("All entries converged at TAU on the first attempt. Skipping tau sweep.")
        return first_sol

    # If all entries converged at TAU on the first attempt, skip the sweep entirely
    all_converged: Bool[Array, ""] = jnp.all(first_converged)
    multi_sol: MultiAttemptSolution = lax.cond(
        all_converged, run_single_step, run_scan, operand=(key, first_solution)
    )

    return multi_sol


def make_solve_with_jit(parameters: Parameters) -> Callable:
    """Gets a JIT-compiled solver function that conditionally applies the tau sweep solver.

    Args:
        parameters: Parameters

    Returns:
        Callable that returns a :class:`MultiAttemptSolution` object
    """

    batch_solver: Callable = make_independent_solver(parameters)
    batch_retry_solver: Callable = make_batch_retry_solver(batch_solver, objective_function)

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=3)
    def solve_with_jit(
        base_solution_array: Float[Array, "... solution"],
        parameters: Parameters,
        key: PRNGKeyArray,
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
        # jax.debug.print("condition (active stability) = {out}", out=condition)

        def solve_with_stability_multistart(key):
            """Function for multistart with stability"""
            subkey: PRNGKeyArray = jax.random.split(key)[1]  # Split only once and pass subkey
            return tau_sweep_solver(base_solution_array, parameters, subkey)

        def solve_with_generic_multistart(key):
            """Function for generic multistart"""
            _, subkey = jax.random.split(key)  # Split only once and pass subkey
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
            lambda _: solve_with_stability_multistart(key),  # True: Use stability solver
            # lambda _: solve_with_stability_multistart(key),  # True: Use stability solver
            # lambda _: solve_with_generic_multistart(key),  # False: Use generic solver
            lambda _: solve_with_generic_multistart(key),  # False: Use generic solver
            operand=None,  # Operand not used for decision making
        )

        return multi_sol

    return solve_with_jit
