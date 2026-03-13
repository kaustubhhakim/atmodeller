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
from jax import lax, random
from jax.scipy.special import logsumexp
from jaxmod.solvers import POSTCHECK_TOLERANCE, MultiAttemptSolution, expand_mask, max_norm
from jaxmod.utils import vmap_axes_spec
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, PyTree
from optimistix import Solution

from atmodeller.constants import (
    INITIAL_LOG_NUMBER_MOLES,
    INITIAL_LOG_STABILITY,
    TAU,
    TAU_MAX,
    TAU_NUM,
)
from atmodeller.engine import get_min_log_elemental_abundance_per_species, objective_function
from atmodeller.output import Output
from atmodeller.parameters import Parameters

LOG_NUMBER_MOLES_VMAP_AXES: int = 0
"""Axis index for the solution array in the vmapped batch solver"""


def _limiting_reagent(
    formula_matrix: Float[Array, "n_elements n_species"],
    element_abundance: Float[Array, " n_elements"],
    mask: Bool[Array, " n_species"],
    *,
    require_positive_budget: bool = False,
) -> Float[Array, " n_species"]:
    """Limiting-reagent mole estimate for species in ``mask``.

    For each species in ``mask``, the mole count is estimated by asking: given the available
    element budget, how many moles of this species could be formed if that element were shared
    equally among all masked species that contain it? The tightest such constraint across all
    elements that appear in the species formula determines the estimate. Returns ``inf`` for
    species not in ``mask`` or not constrained by any available element.

    Args:
        formula_matrix: Formula matrix
        element_abundance: Element abundance. ``NaN`` for unconstrained elements.
        mask: Boolean mask selecting the species to allocate budget to.
        require_positive_budget: When ``True``, elements with ``element_abundance[e] = 0`` are
            excluded from constraining species. Use ``True`` when ``element_abundance`` is a
            *remaining* budget after prior allocation (zeros mean exhausted, not genuinely zero
            mass); use ``False`` (default) when ``element_abundance`` is the original element
            budget (zero is a real constraint).

    Returns:
        Per-species mole estimates; ``inf`` where unconstrained.
    """
    stoich_total: Float[Array, " n_elements"] = jnp.sum(formula_matrix * mask, axis=1)
    safe_stoich: Float[Array, " n_elements"] = jnp.where(stoich_total > 0, stoich_total, 1.0)
    share: Float[Array, " n_elements"] = element_abundance / safe_stoich
    budget_ok: Bool[Array, " n_elements"] = ~jnp.isnan(element_abundance)

    if require_positive_budget:
        budget_ok = budget_ok & (element_abundance > 0)

    is_constrained: Bool[Array, "n_elements n_species"] = (
        (formula_matrix > 0) & mask & budget_ok[:, None]
    )
    implied: Float[Array, "n_elements n_species"] = jnp.where(
        is_constrained, share[:, None], jnp.inf
    )

    return jnp.min(implied, axis=0)


def _auto_initial_guess(parameters: Parameters) -> Float[Array, " n_solution"]:
    """Generates an initial solution vector from element mass constraints and fugacity constraints.

    **Pre-screen — iterative condensate stability prediction:** Starting from a gas-only element
    distribution, the pre-screen iteratively grows the set of predicted-stable condensates using
    :func:`jax.lax.while_loop` until the set stops changing. Each iteration:

    1. Allocates element budget to currently-predicted-stable condensates (limiting-reagent).
    2. Distributes the *remaining* budget to non-condensate species.
    3. Computes ideal-gas activities from those non-condensate mole estimates.
    4. Evaluates ``stability_matrix[r,c] * (log_Kp[r] - log_Q[r]) > 0`` for each reaction and
       condensate species ``c``; a positive signal means the reaction is driven toward condensate
       formation given the current gas activities.
    5. Takes the monotone union of the new predictions with the current set (condensates already
       predicted stable are never retracted).

    This catches cascading condensation that a single-pass screen misses. Because the returned mask
    is the monotone union of the input and new predictions (condensates are never retracted), the
    set can grow by at least one entry per iteration and convergence to a fixed point is guaranteed
    in at most ``n_condensates`` iterations.

    **Step 1 — predicted-stable condensates (first priority):** Only condensates identified as
    supersaturated in the pre-screen are allocated element budget. This keeps more element budget
    available for the gas-phase species.

    **Step 2 — gas species:** Each element's *remaining* budget (after condensate consumption) is
    distributed across non-condensate species by the same limiting-reagent logic. Species whose
    element budget is fully consumed by condensates fall back to
    :const:`~atmodeller.constants.INITIAL_LOG_NUMBER_MOLES`.

    **Step 3 — fugacity-constrained gas species:** The total moles of mass-constrained gas species
    (from step 2) are used to estimate the gas volume via the ideal gas law. The pressure is
    estimated from the gas mass of those species via
    :meth:`~atmodeller.interfaces.ThermodynamicStateProtocol.get_pressure`. Fugacity-constrained
    gas species (e.g. O2 set by a redox buffer) are then assigned mole counts via
    :math:`n_i = f_i \\cdot n_\\mathrm{gas,known} / P`.

    Log stability is initialised to a strongly negative value (``-60``) for predicted-stable
    condensates and to :const:`~atmodeller.constants.INITIAL_LOG_STABILITY` for all other species.

    Intended to be called inside a vmapped context (one batch element at a time).

    Args:
        parameters: Parameters for a single batch element

    Returns:
        Concatenated ``[log_number_moles, log_stability]`` of length ``2 * n_species``
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)

    # element abundance in moles: (n_elements,) — NaN where element is not mass-constrained.
    # log_abundance() squeezes the leading batch dimension when unbatched, giving a 1-D array.
    element_abundance: Float[Array, " n_elements"] = jnp.exp(
        parameters.mass_constraints.log_abundance()
    )
    # jax.debug.print("element_abundance = {out}", out=element_abundance)

    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.condensates_species_mask
    )
    gas_mask: Bool[Array, " n_species"] = jnp.asarray(parameters.reaction_system.gas_species_mask)
    fug_active: Bool[Array, " n_species"] = parameters.fugacity_constraints.active()
    gas_no_fug: Bool[Array, " n_species"] = gas_mask & ~fug_active
    molar_masses: Float[Array, " n_species"] = jnp.asarray(parameters.species.molar_masses)
    temperature: Float[Array, ""] = parameters.state.temperature
    fallback: Float[Array, ""] = jnp.exp(jnp.array(INITIAL_LOG_NUMBER_MOLES, dtype=float))

    # Pre-compute reaction matrices and log Kp (composition-independent).
    # The order of the species is gas, melt, solid, condensates (pure phases)
    reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)

    stability_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_stability_matrix_full
    )
    # jax.debug.print("stability_matrix = {out}", out=stability_matrix)

    log_Kp: Float[Array, " n_reactions"] = parameters.reaction_system.reaction.get_log_Kp(
        temperature
    )
    # jax.debug.print("log_Kp = {out}", out=log_Kp)

    # Species that are not condensates
    other_mask: Bool[Array, " n_species"] = ~condensate_mask
    # jax.debug.print("other_mask = {out}", out=other_mask)

    def _one_stability_pass(
        condensate_stable_known: Bool[Array, " n_species"],
    ) -> Bool[Array, " n_species"]:
        """One stability-prediction pass.

        Given a set of already-known-stable condensates, allocates element budget to them first,
        distributes the remainder to gas species, computes ideal-gas activities, evaluates
        the reaction K vs Q signal, and returns the monotone union of the new predictions with
        the input mask.

        Args:
            condensate_stable_known: Boolean mask of condensates predicted stable so far.

        Returns:
            Updated mask - superset of ``condensate_stable_known``.
        """
        # Allocate element budget to known-stable condensates (limiting-reagent).
        n_known: Float[Array, " n_species"] = _limiting_reagent(
            formula_matrix, element_abundance, condensate_stable_known
        )
        # Zero out unconstrained condensates so they don't consume element budget.
        n_known_applied: Float[Array, " n_species"] = jnp.where(jnp.isinf(n_known), 0.0, n_known)
        # jax.debug.print("n_known_applied = {out}", out=n_known_applied)

        # element_used[e] = sum_s( A[e,s] * n_known_applied[s] ) for known-stable condensates only:
        # the total moles of element e consumed by the allocated condensate budget.
        element_used: Float[Array, " n_elements"] = jnp.einsum(
            "es,s->e", formula_matrix * condensate_stable_known, n_known_applied
        )
        # jax.debug.print("element_used = {out}", out=element_used)
        # Remaining element budget after known-stable condensate allocation.
        remaining_b: Float[Array, " n_elements"] = jnp.maximum(
            element_abundance - element_used, 0.0
        )
        # jax.debug.print("remaining_b = {out}", out=remaining_b)

        # Distribute remaining budget to non-condensate species (limiting-reagent).
        # Exhausted elements (remaining_b = 0) are excluded; species unconstrained by any
        # available element fall back to INITIAL_LOG_NUMBER_MOLES.
        n_other: Float[Array, " n_species"] = _limiting_reagent(
            formula_matrix, remaining_b, other_mask, require_positive_budget=True
        )
        n_gas_est: Float[Array, " n_species"] = jnp.where(jnp.isinf(n_other), fallback, n_other)
        # jax.debug.print("n_gas_est = {out}", out=n_gas_est)

        # Ideal-gas log activity for gas species: log(x_i * P) = log(n_i/n_total) + log(P).
        # Non-gas species (melt, solid, pure-phase condensates) are assigned log_activity = 0,
        # i.e. unit activity. This ignores dilution (mole fraction < 1) and activity coefficients
        # for melt/solid solution species, but is intentional: the pre-screen is a cheap
        # heuristic and calling the EOS/mixing models here would be circular and expensive.
        # Activity coefficients are in any case unavailable without a complete solution.
        mass_gas: Float[Array, ""] = jnp.sum(jnp.where(gas_no_fug, n_gas_est * molar_masses, 0.0))
        pressure: Float[Array, ""] = parameters.state.get_pressure(mass_gas)
        n_gas_total: Float[Array, ""] = jnp.sum(jnp.where(gas_mask, n_gas_est, 0.0))
        safe_n_gas_total: Float[Array, ""] = jnp.where(n_gas_total > 0, n_gas_total, 1.0)
        log_activity: Float[Array, " n_species"] = jnp.where(
            gas_mask,
            jnp.log(jnp.where(gas_mask, n_gas_est, 1.0))
            - jnp.log(safe_n_gas_total)
            + jnp.log(jnp.where(pressure > 0, pressure, 1.0)),
            0.0,
        )

        # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
        # NaN entries in log_activity propagate to log_Q --> stability_signal NaN --> not > 0
        # (safe).
        log_Q: Float[Array, " n_rxn"] = jnp.einsum("rs,s->r", reaction_matrix, log_activity)
        stability_signal: Float[Array, "n_rxn n_species"] = stability_matrix * (
            log_Kp[:, None] - log_Q[:, None]
        )
        new_predictions: Bool[Array, " n_species"] = (
            jnp.any(stability_signal > 0, axis=0) & condensate_mask
        )
        # jax.debug.print("new_predictions = {out}", out=new_predictions)

        # Monotone union: condensates are never retracted once predicted stable. This ensures the
        # set grows by at least one entry per iteration, guaranteeing fixed-point convergence
        # without a cap. The solver makes the definitive stable/absent decision.
        return condensate_stable_known | new_predictions

    # --- Iterate until the predicted-stable set stops growing ---
    # Initialise with no condensates known stable; the first body call is the gas-only pre-screen.
    # The monotone union in _one_stability_pass guarantees termination; _MAX_STABILITY_ITERS is
    # a defensive cap against unforeseen edge cases (e.g. NaN corruption of the boolean mask).
    _MAX_STABILITY_ITERS: int = 10

    init_stable: Bool[Array, " n_species"] = jnp.zeros_like(condensate_mask)
    # jax.debug.print("init_stable = {out}", out=init_stable)
    first_stable: Bool[Array, " n_species"] = _one_stability_pass(init_stable)
    # jax.debug.print("first_stable = {out}", out=first_stable)

    def _cond_fn(carry: tuple) -> Bool[Array, ""]:
        prev, curr, i = carry
        return jnp.any(prev != curr) & (i < _MAX_STABILITY_ITERS)

    def _body_fn(carry: tuple) -> tuple:
        _, curr, i = carry
        return curr, _one_stability_pass(curr), i + 1

    _, condensate_stable_predicted, _ = lax.while_loop(
        _cond_fn, _body_fn, (init_stable, first_stable, jnp.array(0))
    )

    # Step 1: allocate element budget to predicted-stable condensates
    n_condensate: Float[Array, " n_species"] = _limiting_reagent(
        formula_matrix, element_abundance, condensate_stable_predicted
    )

    # Remaining element budget after stable-condensate allocation
    n_condensate_applied: Float[Array, " n_species"] = jnp.where(
        jnp.isinf(n_condensate), 0.0, n_condensate
    )
    element_used: Float[Array, " n_elements"] = jnp.einsum(
        "es,s->e", formula_matrix * condensate_stable_predicted, n_condensate_applied
    )
    remaining_b: Float[Array, " n_elements"] = jnp.maximum(element_abundance - element_used, 0.0)

    # Step 2: allocate remaining budget to non-condensate species
    n_other: Float[Array, " n_species"] = _limiting_reagent(
        formula_matrix, remaining_b, other_mask, require_positive_budget=True
    )

    # Combine: predicted-stable condensates use their budget estimate; all others use the
    # non-condensate remainder. Fallback for species not covered by any constrained element.
    n_estimate: Float[Array, " n_species"] = jnp.where(
        condensate_stable_predicted, n_condensate, n_other
    )
    n_estimate = jnp.where(jnp.isinf(n_estimate), fallback, n_estimate)

    log_number_moles: Float[Array, " n_species"] = jnp.log(n_estimate)

    # Step 3: fugacity-constrained gas species
    log_n_gas_known_total: Float[Array, ""] = logsumexp(
        jnp.where(gas_no_fug, log_number_moles, -jnp.inf)
    )
    mass_gas_known: Float[Array, ""] = jnp.sum(
        jnp.where(gas_no_fug, jnp.exp(log_number_moles), 0.0) * molar_masses
    )
    pressure: Float[Array, ""] = parameters.state.get_pressure(mass_gas_known)
    log_fug: Float[Array, " n_species"] = parameters.fugacity_constraints.log_fugacity(
        temperature, pressure
    )
    log_n_fug: Float[Array, " n_species"] = log_fug + log_n_gas_known_total - jnp.log(pressure)
    log_number_moles = jnp.where(gas_mask & fug_active, log_n_fug, log_number_moles)

    # Log stability for predicted-stable condensates: initialise at the value that makes the
    # stability residual (log_n + log_s - (min_log_abundance + log_tau)) exactly zero given the
    # current mole estimate. This automatically scales with tau so no magic constant is needed.
    # Falls back to INITIAL_LOG_STABILITY where the expression is non-finite (e.g. zero-budget
    # elements, though those species should not be predicted stable anyway).
    log_tau_val: Float[Array, ""] = jnp.log(parameters.solver_parameters.tau)
    min_log_abundance_per_species: Float[Array, " n_species"] = (
        get_min_log_elemental_abundance_per_species(parameters)
    )
    log_stability_stable: Float[Array, " n_species"] = (
        min_log_abundance_per_species + log_tau_val - log_number_moles
    )
    log_stability: Float[Array, " n_species"] = jnp.where(
        condensate_stable_predicted & jnp.isfinite(log_stability_stable),
        log_stability_stable,
        jnp.full_like(log_number_moles, INITIAL_LOG_STABILITY),
    )

    return jnp.concatenate((log_number_moles, log_stability), axis=-1)


def solve_single_with_auto_guess(
    initial_guess: Float[Array, "..."], parameters: Parameters
) -> optx.Solution:
    """Solves a single (unbatched) system via :func:`optimistix.root_find`, generating an initial
    guess automatically from ``parameters``.

    Intended to be wrapped with :func:`equinox.filter_vmap` by :func:`make_batch_solver`
    rather than called directly. All solver configuration is read from
    ``parameters.solver_parameters``.

    Args:
        initial_guess_in: Initial guess for the solution vector. If any element is ``NaN``,
            the initial guess is replaced by the auto-generated guess from
            :func:`_auto_initial_guess`.
        parameters: Parameters providing the solver instance, step limit, and options

    Returns:
        :class:`~optimistix.Solution` object
    """
    initial_guess = lax.cond(
        jnp.any(jnp.isnan(initial_guess)),
        lambda _: _auto_initial_guess(parameters),
        lambda ig: ig,
        operand=initial_guess,
    )
    # jax.debug.print("initial_guess = {out}", out=initial_guess)

    sol: optx.Solution = optx.root_find(
        objective_function,
        parameters.solver_parameters.get_solver_instance(),
        initial_guess,
        args=parameters,
        throw=parameters.solver_parameters.throw,
        max_steps=parameters.solver_parameters.max_steps,
        options=parameters.solver_parameters.get_options(parameters.species.number_species),
    )
    # jax.debug.print("solution = {out}", out=sol.value)

    # n: int = parameters.species.number_species
    # solution_moles: Float[Array, " n_species"] = sol.value[:n]
    # solution_stability: Float[Array, " n_species"] = sol.value[n:]
    # rms_moles: Float[Array, ""] = jnp.sqrt(jnp.mean((initial_guess[:n] - solution_moles) ** 2))
    # rms_stability: Float[Array, ""] = jnp.sqrt(
    #     jnp.mean((initial_guess[n:] - solution_stability) ** 2)
    # )
    # jax.debug.print(
    #     "RMS log number of moles: = {moles:.4f}, log stability = {stability:.4f}",
    #     moles=rms_moles,
    #     stability=rms_stability,
    # )

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


def make_batch_retry_solver(solver_function: Callable, objective_function: Callable) -> Callable:
    """Makes a batch retry solver.

    ``solver_function`` and ``objective_function`` must be pure JAX-callable functions compatible
    with :func:`equinox.filter_jit``. They must not close over non-JAX state or produce Python side
    effects.

    Args:
        solver_function: Callable that performs a single solve. Must accept arguments of an initial
            guess and a pytree of parameters.
        objective_function: Callable for the objective function

    Returns:
        Callable
    """

    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def batch_retry_solver(
        initial_guess: Float[Array, "... solution"],
        parameters: PyTree,
        key: PRNGKeyArray,
        perturb_scale: ArrayLike,
        max_retries: int,
        tolerance: float = POSTCHECK_TOLERANCE,
    ) -> MultiAttemptSolution:
        """Batched solver with retry and perturbation for failed cases

        Runs a batched solver function on a set of initial guesses. If some entries fail to
        converge, the function perturbs only the failed solutions and retries, up to
        ``max_retries``. Successfully converged solutions are kept fixed throughout.

        This approach is useful when solving large batches of nonlinear systems where certain
        initial guesses may fail. Perturbations help the solver escape poor local minima or flat
        regions of the objective function.

        Note:
            ``solver_function`` may return a solver result indicating success even when the
            objective residual remains above tolerance. Convergence is therefore validated
            independently and the result of that validation is tracked in
            :meth:`MultiAttemptSolution.attempts`.
                - ``solution.result``: solver's internal convergence classification
                - ``attempts``: first iteration satisfying objective-based check
                - ``attempts == 0``: never converged within the initial attempt plus
                  ``max_retries`` retries

        Args:
            initial_guess: Batched array of initial guesses for the solver
            parameters: Model parameters passed to the solver
            key: JAX PRNG key for reproducible random perturbations
            perturb_scale: Array or scalar that scales the random perturbation to the log number
                of moles applied to failed solutions
            max_retries: Maximum number of solver retries per batch entry
            tolerance: Tolerance for the objective-based convergence validation performed after
                each solve attempt. Defaults to :obj:`POSTCHECK_TOLERANCE`.

        Returns:
            :class:`MultiAttemptSolution` instance
        """

        def body_fn(state: tuple[Array, Array, Array, Array, Array, Array]) -> tuple:
            """Performs one retry iteration for failed solutions.

            This function executes a single iteration of the solver retry loop. It perturbs only
            the solutions that previously failed, reruns the solver, and updates the batch state
            accordingly. Successfully converged entries remain unchanged.

            Args:
                tuple:
                    i: Current attempt index
                    key: Random key for perturbation generation
                    solution: Current batch of solution estimates
                    result_value: Current result value of the solver for each entry
                    steps: Number of solver steps recorded for each entry
                    attempt: Attempt index when each entry first succeeded or 0 if it did not
                         converge at all.

            Returns:
                Updated state tuple with the same structure as in the input
            """
            i, key, solution, result_value, steps, attempt = state
            # jax.debug.print("Iteration: {out}", out=i)

            failed_mask: Bool[Array, "..."] = attempt == 0  # Not yet converged per objective check
            # jax.debug.print("failed_mask = {out}", out=failed_mask)

            # Split solution into log_number_moles and log_stability
            log_number_moles, log_stability = jnp.split(solution, 2, axis=-1)

            # Perturbation for log number of moles
            key, subkey = random.split(key)
            perturb_shape: tuple[int, ...] = log_number_moles.shape
            raw_perturb = random.uniform(subkey, shape=perturb_shape, minval=-1.0, maxval=1.0)
            # jax.debug.print("raw_perturb = {out}", out=raw_perturb)
            perturbations = jnp.where(
                expand_mask(failed_mask, raw_perturb),
                perturb_scale * raw_perturb,
                jnp.zeros_like(log_number_moles),
            )
            # jax.debug.print("perturbations = {out}", out=perturbations)
            new_log_number_moles: Float[Array, "... n_species"] = log_number_moles + perturbations

            # Perturbation for stability
            key, subkey = random.split(key)
            log_tau: Float[Array, ""] = jnp.log(parameters.solver_parameters.tau)
            perturb_shape = log_stability.shape

            # The logic here is guided but ultimately arbitrary. The goal is to introduce some
            # diversity in the log stability estimates while keeping them bounded and not too
            # large. This heuristic algorithm maintains the original log stability with 25%
            # probability, and otherwise, also with 25% probability, sets it to either 0.25, 0.5,
            # or 0.75 times log_tau to explore the possible range of allowable stability values.
            # The perturbation is only applied to active stability entries.
            rand_vals = random.uniform(subkey, shape=perturb_shape, minval=0.0, maxval=1.0)
            stability_new: Float[Array, "... n_species"] = jnp.select(
                [rand_vals < 0.25, rand_vals < 0.5, rand_vals < 0.75],
                [0.25 * log_tau, 0.5 * log_tau, 0.75 * log_tau],
                default=log_stability,
            )
            # Only update entries where stability is active
            # jax.debug.print(
            #    "active_stability = {out}", out=parameters.reaction_system.species.active_stability
            # )
            stability_new = jnp.where(
                parameters.reaction_system.species.active_stability, stability_new, log_stability
            )
            # jax.debug.print("stability_new = {out}", out=stability_new)

            # Recombine
            new_initial_solution: Float[Array, "... twice_species"] = jnp.concatenate(
                [new_log_number_moles, stability_new], axis=-1
            )
            # jax.debug.print("new_initial_solution = {out}", out=new_initial_solution)

            new_sol: MultiAttemptSolution = solver_function(new_initial_solution, parameters)
            new_solution: Float[Array, "... solution"] = new_sol.value
            # jax.debug.print("new_solution = {out}", out=new_solution)

            new_result_value: Integer[Array, "..."] = new_sol.result._value  # pyright: ignore
            # jax.debug.print("new_result_value = {out}", out=new_result_value)

            # If the solver result is broadcast from a scalar we can't use it to decide which
            # individual models failed. Instead we must perform a per-system check.
            new_successful: Bool[Array, "..."] = (
                max_norm(objective_function, new_solution, parameters) < 1e-60  # tolerance
            )
            # jax.debug.print("new_successful = {out}", out=new_successful)

            new_num_steps: Integer[Array, "..."] = new_sol.stats["num_steps"]
            # jax.debug.print("new_num_steps = {out}", out=new_num_steps)

            # Determine which entries to update: previously failed, now succeeded
            update_mask: Bool[Array, "..."] = jnp.logical_and(failed_mask, new_successful)
            # jax.debug.print("update_mask = {out}", out=update_mask)
            updated_solution: Float[Array, "... solution"] = cast(
                Array, jnp.where(expand_mask(update_mask, new_solution), new_solution, solution)
            )
            updated_result_value: Integer[Array, "..."] = jnp.where(
                update_mask, new_result_value, result_value
            )
            # jax.debug.print("updated_result_value = {out}", out=updated_result_value)
            updated_num_steps: Integer[Array, "..."] = cast(
                Array, jnp.where(update_mask, new_num_steps, steps)
            )
            # jax.debug.print("updated_num_steps = {out}", out=updated_num_steps)
            updated_attempt: Array = jnp.where(update_mask, i, attempt)  # pyright: ignore
            # jax.debug.print("updated_attempt = {out}", out=updated_attempt)

            return (
                i + 1,
                key,
                updated_solution,
                updated_result_value,
                updated_num_steps,
                updated_attempt,
            )

        def cond_fn(
            state: tuple[Array, Array, Array, Array, Array, Array],
        ) -> Bool[Array, "..."]:
            """Determines whether additional solver retries are needed.

            This condition function controls the ``lax.while_loop``. The retry loop continues as
            long as at least one batch entry has not converged and the maximum number of attempts
            has not been reached.

            Args:
                tuple:
                    i: Current attempt index
                    _: Unused (PRNG key)
                    _: Unused (current batch solution)
                    _: Unused (result value)
                    _: Unused (number of steps)
                    attempts: Unused (success attempt index)

            Returns:
                ``True`` if any entry has failed and the number of attempts is less than
                    ``max_retries``; otherwise ``False``.
            """
            i, _, _, _, _, attempt = state

            # For debugging to force the loop to run to the maximum allowable value
            # return jnp.logical_and(i < max_retries, True)

            # Convergence is determined by `check_convergence`, which enforces the objective
            # tolerance on each batch entry individually. We track the first successful attempt
            # index in `attempts`. An entry is considered converged if attempts > 0, ensuring
            # consistency with the convergence mask used elsewhere in the code.
            # i starts at 2 (second overall attempt), so to allow max_retries retries we need
            # the body to run at i in {2, ..., max_retries+1}, hence the condition
            # i < max_retries+2.
            continue_loop: Bool[Array, "..."] = jnp.logical_and(
                jnp.any(attempt == 0), i < max_retries + 2
            )

            return continue_loop

        # Try first solution
        # jax.debug.print("Iteration: 1")
        first_sol: MultiAttemptSolution = solver_function(initial_guess, parameters)
        first_solution: Float[Array, "... solution"] = first_sol.value
        # jax.debug.print("first_solution = {out}", out=first_solution)

        # Check the solver result
        # jax.debug.print("first_sol.result = {out}", out=first_sol.result)

        # Perform a per-system check
        first_converged: Bool[Array, "..."] = (
            max_norm(objective_function, first_solution, parameters) < tolerance
        )
        # jax.debug.print("first_converged = {out}", out=first_converged)

        first_result_value: Integer[Array, "..."] = jnp.broadcast_to(
            first_sol.result._value, first_converged.shape
        )
        # jax.debug.print("first_result_value = {out}", out=first_result_value)

        first_num_steps: Integer[Array, "..."] = jnp.broadcast_to(
            first_sol.stats["num_steps"], first_converged.shape
        )
        # jax.debug.print("first_num_steps = {out}", out=first_num_steps)

        # Failback solution to initial guess for failed models
        first_converged_bc: Bool[Array, "... 1"] = expand_mask(first_converged, first_solution)
        # jax.debug.print("first_converged_bc = {out}", out=first_converged_bc)

        solution: Float[Array, "... solution"] = cast(
            Array, jnp.where(first_converged_bc, first_solution, initial_guess)
        )
        # jax.debug.print("solution = {out}", out=solution)
        # jax.debug.print("Completed iteration: 1")

        initial_state: tuple = (
            jnp.array(2),  # Second overall attempt
            key,
            solution,
            first_result_value,
            first_num_steps,
            first_converged.astype(int),  # 1 for solved, otherwise 0
        )

        _, _, final_solution, final_result_value, final_num_steps, final_attempt = lax.while_loop(
            cond_fn, body_fn, initial_state
        )
        # jax.debug.print("After lax.while_loop")

        # jax.debug.print("final_solution = {out}", out=final_solution)
        # jax.debug.print("final_result_value = {out}", out=final_result_value)
        # jax.debug.print("final_num_steps = {out}", out=final_num_steps)
        # jax.debug.print("final_attempt = {out}", out=final_attempt)

        # Bundle the final outputs into a single optimistix Solution object
        final_result: optx.RESULTS = cast(
            optx.RESULTS,
            EnumerationItem(final_result_value, optx.RESULTS),  # pyright: ignore
        )

        # This solution instance does not return all the information from the solves, but it
        # encapsulates the most important (final) quantities. Zero out steps for failed entries so
        # that reported steps are not misleadingly non-zero for models that never converged.
        final_num_steps_out: Integer[Array, "..."] = cast(
            Array, jnp.where(final_attempt > 0, final_num_steps, jnp.zeros_like(final_num_steps))
        )
        sol: optx.Solution = optx.Solution(
            final_solution, final_result, None, {"num_steps": final_num_steps_out}, None
        )
        multi_sol: MultiAttemptSolution = MultiAttemptSolution(sol, final_attempt)

        return multi_sol

    return batch_retry_solver


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
    ) -> Output:
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
            :class:`~atmodeller.output.Output` object
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
        output: Output = Output(parameters, multi_sol)

        return output

    return solve_with_jit
