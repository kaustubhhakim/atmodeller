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
from jax.scipy.special import logsumexp
from jaxmod.solvers import MultiAttemptSolution, make_batch_retry_solver
from jaxmod.utils import vmap_axes_spec
from jaxtyping import Array, Bool, Float, Integer, PRNGKeyArray
from optimistix import Solution

from atmodeller.constants import (
    INITIAL_LOG_NUMBER_MOLES,
    INITIAL_LOG_STABILITY,
    TAU,
    TAU_MAX,
    TAU_NUM,
)
from atmodeller.engine import objective_function
from atmodeller.parameters import Parameters

LOG_NUMBER_MOLES_VMAP_AXES: int = 0
"""Axis index for the solution array in the vmapped batch solver"""


def _auto_initial_guess(parameters: Parameters) -> Float[Array, " n_solution"]:
    """Generates an initial solution vector from element mass constraints and fugacity constraints.

    **Step 1 — mass-constrained species:** For each constrained element, distributes its mole
    budget equally across all species that contain it (weighted by total stoichiometric demand). A
    species containing multiple constrained elements takes the minimum implied mole count across
    those elements — a limiting-reagent estimate that avoids over-allocating the tightest element
    budget. Species not covered by any constrained element fall back to
    :const:`~atmodeller.constants.INITIAL_LOG_NUMBER_MOLES`.

    **Step 2 — fugacity-constrained gas species:** The total moles of mass-constrained gas species
    (from step 1) are used to estimate the gas volume via the ideal gas law. The pressure is
    estimated from the gas mass of those species via :meth:`~atmodeller.interfaces.ThermodynamicStateProtocol.get_pressure`.
    Fugacity-constrained gas species (e.g. O₂ set by a redox buffer) are then assigned mole
    counts via :math:`n_i = f_i \\cdot n_\\mathrm{gas,known} / P`.

    Log stability is initialised to :const:`~atmodeller.constants.INITIAL_LOG_STABILITY` for all
    species.

    Intended to be called inside a vmapped context (one batch element at a time).

    Args:
        parameters: Parameters for a single batch element

    Returns:
        Concatenated ``[log_number_moles, log_stability]`` of length ``2 * n_species``
    """
    # formula_matrix: (n_elements, n_species)  — integer stoichiometric counts
    A: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )

    # element moles: (n_elements,) — NaN where element is not mass-constrained.
    # log_abundance() squeezes the leading batch dimension when unbatched, giving a 1-D array.
    b: Float[Array, " n_elements"] = jnp.exp(parameters.mass_constraints.log_abundance())

    # Total stoichiometric count per element summed across all species;
    # guard against zero (element absent from all species) to avoid NaN.
    total_stoich: Float[Array, " n_elements"] = jnp.sum(A, axis=1)
    safe_total: Float[Array, " n_elements"] = jnp.where(total_stoich > 0, total_stoich, 1.0)

    # For each constrained element e, each species s that contains it gets:
    #   b_e / total_stoich_e  moles
    # i.e. the element budget divided evenly by the total stoichiometric demand.
    share: Float[Array, " n_elements"] = b / safe_total

    # Broadcast to (n_elements, n_species); use +inf where a species does not contain the element
    # or where the element is not mass-constrained (NaN in b).
    constrained: Bool[Array, "n_elements n_species"] = (A > 0) & ~jnp.isnan(b[:, None])
    implied: Float[Array, "n_elements n_species"] = jnp.where(constrained, share[:, None], jnp.inf)

    # Limiting-reagent estimate: the tightest element budget for a species caps its mole count
    n_estimate: Float[Array, " n_species"] = jnp.min(implied, axis=0)

    # Fall back for species not covered by any constrained element
    fallback: Float[Array, ""] = jnp.exp(jnp.array(INITIAL_LOG_NUMBER_MOLES, dtype=float))
    n_estimate = jnp.where(jnp.isinf(n_estimate), fallback, n_estimate)

    log_number_moles: Float[Array, " n_species"] = jnp.log(n_estimate)

    # --- Incorporate fugacity constraints for gas species ---
    # Identify which gas species have active fugacity constraints
    gas_mask: Bool[Array, " n_species"] = jnp.asarray(parameters.reaction_system.gas_species_mask)
    fug_active: Bool[Array, " n_species"] = parameters.fugacity_constraints.active()
    gas_no_fug: Bool[Array, " n_species"] = gas_mask & ~fug_active

    # Estimate total moles of mass-constrained gas species in log-space (numerically stable)
    log_n_gas_known_total: Float[Array, ""] = logsumexp(
        jnp.where(gas_no_fug, log_number_moles, -jnp.inf)
    )

    # Estimate pressure from the gas mass of mass-constrained species.  This handles both a
    # fixed pressure (ThermodynamicState) and the mechanical-balance mode (ThinAtmospherePlanet).
    molar_masses: Float[Array, " n_species"] = jnp.asarray(parameters.species.molar_masses)
    mass_gas_known: Float[Array, ""] = jnp.sum(
        jnp.where(gas_no_fug, jnp.exp(log_number_moles), 0.0) * molar_masses
    )
    pressure: Float[Array, ""] = parameters.state.get_pressure(mass_gas_known)

    # Log fugacity for all species; NaN for unconstrained species (FixedFugacityConstraint)
    temperature: Float[Array, ""] = parameters.state.temperature
    log_fug: Float[Array, " n_species"] = parameters.fugacity_constraints.log_fugacity(
        temperature, pressure
    )

    # Ideal gas: n_i = f_i * V / (RT), with V estimated from mass-constrained gas:
    #   V = n_gas_known * RT / P  =>  n_i_fug = f_i * n_gas_known / P
    # In log-space:
    log_n_fug: Float[Array, " n_species"] = log_fug + log_n_gas_known_total - jnp.log(pressure)

    # Update only the gas species that are fugacity-constrained
    log_number_moles = jnp.where(gas_mask & fug_active, log_n_fug, log_number_moles)

    log_stability: Float[Array, " n_species"] = jnp.full_like(
        log_number_moles, INITIAL_LOG_STABILITY
    )

    return jnp.concatenate((log_number_moles, log_stability))


def solve_single(
    initial_guess: Float[Array, "... n_solution"], parameters: Parameters
) -> optx.Solution:
    """Solves a single (unbatched) system via :func:`optimistix.root_find`.

    Intended to be wrapped with :func:`equinox.filter_vmap` by :func:`make_batch_solver`
    rather than called directly. All solver configuration is read from
    ``parameters.solver_parameters``.

    Args:
        initial_guess: 1-D initial guess for the solution vector
        parameters: Parameters providing the solver instance, step limit, and options

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


def solve_single_with_auto_guess(
    initial_guess_in: Float[Array, "... n_solution"], parameters: Parameters
) -> optx.Solution:
    """Solves a single (unbatched) system via :func:`optimistix.root_find`, generating an initial
    guess automatically from ``parameters``.

    Intended to be wrapped with :func:`equinox.filter_vmap` by :func:`make_batch_solver`
    rather than called directly. All solver configuration is read from
    ``parameters.solver_parameters``.

    Args:
        initial_guess_in: Initial guess for the solution vector
        parameters: Parameters providing the solver instance, step limit, and options

    Returns:
        :class:`~optimistix.Solution` object
    """

    jax.debug.print("Initial guess in = {out}", out=initial_guess_in)
    initial_guess: Float[Array, " n_solution"] = _auto_initial_guess(parameters)
    jax.debug.print("Auto-generated initial guess = {out}", out=initial_guess)

    sol: optx.Solution = optx.root_find(
        objective_function,
        parameters.solver_parameters.get_solver_instance(),
        initial_guess,
        args=parameters,
        throw=parameters.solver_parameters.throw,
        max_steps=parameters.solver_parameters.max_steps,
        options=parameters.solver_parameters.get_options(parameters.species.number_species),
    )
    jax.debug.print("Solution = {out}", out=sol.value)

    n: int = parameters.species.number_species
    solution_moles: Float[Array, " n_species"] = sol.value[:n]
    rms_auto: Float[Array, ""] = jnp.sqrt(jnp.mean((initial_guess[:n] - solution_moles) ** 2))
    rms_input: Float[Array, ""] = jnp.sqrt(jnp.mean((initial_guess_in[:n] - solution_moles) ** 2))
    jax.debug.print(
        "RMS(auto_guess - solution) = {auto:.4f}  |  RMS(input_guess - solution) = {inp:.4f}",
        auto=rms_auto,
        inp=rms_input,
    )

    return sol


def make_batch_solver(parameters: Parameters) -> Callable:
    """Gets a vmapped batch solver for independent systems.

    Wraps :func:`solve_single` with :func:`equinox.filter_vmap` so that each batch element is
    solved independently, producing per-element convergence statistics. The vmapping axes are
    fixed from ``parameters`` at construction time. JIT compilation is applied by the outer
    :func:`make_solve_with_jit` context.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable that accepts ``(solution, parameters)`` and returns a
        :class:`MultiAttemptSolution` with ``attempts=1``
    """
    solver_function_vmapped: Callable = eqx.filter_vmap(
        solve_single_with_auto_guess,
        in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters)),
    )

    def solver(solution: Array, parameters: Parameters, *args) -> MultiAttemptSolution:
        """Runs the vmapped single-pass batch solve.

        ``_attempts`` is set to ``1`` unconditionally; objective-based per-element convergence
        checking is delegated to the retry wrapper in ``batch_retry_solver``.

        Args:
            solution: Batched initial guess with shape ``(batch, solution)``
            parameters: Parameters passed through to each vmapped :func:`solve_single` call
            *args: Unused; present for interface consistency with :func:`make_batch_retry_solver`

        Returns:
            :class:`MultiAttemptSolution` with ``attempts=1`` for all batch elements
        """
        del args
        sol: optx.Solution = solver_function_vmapped(solution, parameters)

        return MultiAttemptSolution(sol, _attempts=1)

    return solver


def make_batch_retry_solver_from_parameters(parameters: Parameters) -> Callable:
    """Gets a batch retry solver, constructing the vmapped batch solver and objective internally.

    A convenience wrapper around :func:`~jaxmod.solvers.make_batch_retry_solver` that accepts
    ``parameters`` directly, deriving the ``vmap`` axes and building both the batch solver and
    vmapped objective function at construction time. Use this in preference to calling
    :func:`~jaxmod.solvers.make_batch_retry_solver` directly when the vmap axes are not already
    available.

    Args:
        parameters: Parameters used to derive the vmapping axes at construction time

    Returns:
        Callable with the same interface as :func:`~jaxmod.solvers.batch_retry_solver`
    """
    batch_solver: Callable = make_batch_solver(parameters)
    objective_function_vmapped: Callable = eqx.filter_vmap(
        objective_function, in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec(parameters))
    )
    batch_retry_solver: Callable = make_batch_retry_solver(
        batch_solver, objective_function_vmapped
    )

    return batch_retry_solver


def make_tau_sweep_solver(batch_retry_solver: Callable) -> Callable:
    """Gets a solver function that performs a tau sweep for active stability systems.

    Closes over the provided ``batch_retry_solver``. The returned callable first attempts a solve
    at ``TAU``; if all batch elements converge it returns immediately, otherwise it runs a
    full log-spaced sweep from ``TAU_MAX`` down to ``TAU`` for every batch element.

    Args:
        batch_retry_solver: Pre-built batch retry solver, e.g. from
            :func:`make_batch_retry_solver_from_parameters`

    Returns:
        Callable that returns a :class:`MultiAttemptSolution` object
    """
    get_leaf: Callable = lambda t: t.solver_parameters.tau  # noqa: E731
    varying_schedule: Float[Array, " tau"] = jnp.logspace(
        jnp.log10(TAU_MAX), jnp.log10(TAU), num=TAU_NUM
    )

    def tau_sweep_solver(
        initial_guess: Float[Array, "... solution"], parameters: Parameters, key: PRNGKeyArray
    ) -> MultiAttemptSolution:
        """Attempts a solve at ``TAU`` and, if any element fails, runs a full tau sweep.

        First tries to solve all batch elements at ``TAU`` with multistart retry. If every element
        converges, the result is returned immediately. Otherwise a log-spaced schedule from
        ``TAU_MAX`` down to ``TAU`` is swept via :func:`jax.lax.scan`, applying the solver at each
        step for all batch elements. Because ``tau`` must remain a scalar throughout the scan
        (to keep the vmapping axes consistent), converged and failed elements both run the full
        sweep; converged ones simply re-solve quickly from their existing solution.

        Args:
            initial_guess: Batched array of initial guesses with shape ``(batch, solution)``
            parameters: :class:`~atmodeller.parameters.Parameters` whose ``tau`` leaf will be
                replaced at each scan step.
            key: JAX PRNG key for reproducible random perturbations

        Returns:
            :class:`~jaxmod.solvers.MultiAttemptSolution` object
        """

        def solve_tau_step(carry: tuple, tau: Float[Array, " ..."]) -> tuple[tuple, tuple]:
            """Performs a single batched solver step for one scalar tau value.

            Intended for use inside :func:`jax.lax.scan`. Receives a scalar ``tau`` from
            ``varying_schedule``, injects it into a copy of ``parameters``, and runs the batch
            retry solver on all elements. The solution carried forward is the best result from this
            step; the outputs stacked by ``scan`` capture the full history across tau steps.

            Args:
                carry: Tuple of ``(key, solution)`` where ``solution`` has shape
                    ``(batch, solution)``
                tau: Scalar tau value for this scan step

            Returns:
                Updated ``(key, solution)`` carry tuple and an output tuple of
                ``(solution, result_value, num_steps, attempts)``
            """
            (key, solution) = carry
            key, subkey = jax.random.split(key)

            # Get new parameters with tau value
            new_parameters: Parameters = eqx.tree_at(get_leaf, parameters, tau)
            # jax.debug.print("tau = {out}", out=new_parameters.solver_parameters.tau)

            new_sol: MultiAttemptSolution = batch_retry_solver(
                solution,
                new_parameters,
                subkey,
                parameters.solver_parameters.retry_perturbation,
                parameters.solver_parameters.max_starts - 1,
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
        initial_parameters: Parameters = eqx.tree_at(get_leaf, parameters, jnp.array(TAU))

        first_sol: MultiAttemptSolution = batch_retry_solver(
            initial_guess,
            initial_parameters,
            subkey,
            parameters.solver_parameters.retry_perturbation,
            parameters.solver_parameters.max_starts - 1,
            parameters.solver_parameters.atol,
        )
        first_solution: Float[Array, "... solution"] = first_sol.value
        # jax.debug.print("first_solution = {out}", out=first_solution)
        # jax.debug.print("solver success = {out}", out=first_sol.result._value)
        first_converged: Bool[Array, "..."] = first_sol.attempts > 0
        # jax.debug.print("first_converged = {out}", out=first_converged)

        def run_scan(key_and_guess: tuple) -> MultiAttemptSolution:
            """Run the full tau sweep scan across all batch elements.

            Called when at least one element failed the initial solve at ``TAU``. All batch
            elements (including those that already converged) run every step of the
            ``varying_schedule`` scan. The final solution, result, and the maximum step count
            and attempt index across all tau steps are returned.
            """
            initial_carry_: tuple[PRNGKeyArray, Float[Array, "... solution"]] = key_and_guess
            _, results_ = jax.lax.scan(solve_tau_step, initial_carry_, varying_schedule)
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
            """All batch elements converged at ``TAU`` on the first attempt: return immediately."""
            # jax.debug.print("All entries converged at TAU on the first attempt. Skipping tau sweep.")
            return first_sol

        # If all entries converged at TAU on the first attempt, skip the sweep entirely
        all_converged: Bool[Array, ""] = jnp.all(first_converged)
        multi_sol: MultiAttemptSolution = lax.cond(
            all_converged, run_single_step, run_scan, operand=(key, first_solution)
        )

        return multi_sol

    return tau_sweep_solver


def make_solve_with_jit(parameters: Parameters) -> Callable:
    """General assembly function that constructs and returns the JIT-compiled solver.

    Builds a :func:`make_batch_retry_solver_from_parameters` and a tau sweep solver from
    ``parameters`` at construction time, sharing the same ``batch_retry_solver`` instance between
    both paths. The returned callable dispatches at runtime via :func:`jax.lax.cond` based on
    whether any species have active stability — routing to the tau sweep solver when stability
    species are present, or to the generic multistart solver otherwise.

    Note:
        ``active_stability`` is currently not a traced JAX array; its size must be fixed at
        compile time because it determines the shape of the residual vector. The :func:`lax.cond`
        branch therefore compiles *both* paths even though only one will execute at runtime. This
        retains generality for future capabilities (e.g. dynamically switching solver strategy)
        at the expense of additional — currently unnecessary — compile time.

    Args:
        parameters: Parameters used to derive the vmapping axes and build the sub-solvers at
            construction time

    Returns:
        Callable that returns a :class:`MultiAttemptSolution` object
    """
    batch_retry_solver: Callable = make_batch_retry_solver_from_parameters(parameters)
    tau_sweep_solver: Callable = make_tau_sweep_solver(batch_retry_solver)

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve_with_jit(
        base_solution_array: Float[Array, "... solution"],
        parameters: Parameters,
        key: PRNGKeyArray,
    ) -> MultiAttemptSolution:
        """JIT-compiled entry point that dispatches to the appropriate solver branch.

        Checks whether any species have active stability and routes accordingly via
        :func:`jax.lax.cond`: the tau sweep solver is used when stability species are
        present, otherwise the generic multistart retry solver is used. Both branches are
        compiled at trace time.

        Args:
            base_solution_array: Batched initial guess with shape ``(batch, solution)``
            parameters: Parameters; array leaves are traced, non-array leaves are static
            key: JAX PRNG key

        Returns:
            :class:`~jaxmod.solvers.MultiAttemptSolution` object
        """
        # Define the condition to check if active stability is enabled
        condition: Bool[Array, ""] = jnp.any(parameters.reaction_system.species.active_stability)
        # jax.debug.print("condition (active stability) = {out}", out=condition)

        def solve_with_stability_multistart(key):
            """Routes to the tau sweep solver for systems with active stability species."""
            _, subkey = jax.random.split(key)
            return tau_sweep_solver(base_solution_array, parameters, subkey)

        def solve_with_generic_multistart(key):
            """Routes to the generic multistart retry solver for systems without stability."""
            _, subkey = jax.random.split(key)
            return batch_retry_solver(
                base_solution_array,
                parameters,
                subkey,
                parameters.solver_parameters.retry_perturbation,
                parameters.solver_parameters.max_starts - 1,
                parameters.solver_parameters.atol,
            )

        multi_sol = lax.cond(
            condition, solve_with_stability_multistart, solve_with_generic_multistart, operand=key
        )

        return multi_sol

    return solve_with_jit
