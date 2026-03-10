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

    **Pre-screen — iterative condensate stability prediction:** Starting from a gas-only element
    distribution, the pre-screen iteratively grows the set of predicted-stable condensates using
    :func:`jax.lax.while_loop` until the set stops changing. Each iteration:

    1. Allocates element budget to currently-predicted-stable condensates (limiting-reagent).
    2. Distributes the *remaining* budget to non-condensate species (gas/melt/solid).
    3. Computes ideal-gas activities from those non-condensate mole estimates.
    4. Evaluates ``stability_matrix[r,c] * (log_Kp[r] - log_Q[r]) > 0`` for each reaction and
       condensate species ``c``; a positive signal means the reaction is driven toward condensate
       formation given the current gas activities.
    5. Takes the monotone union of the new predictions with the current set (condensates already
       predicted stable are never retracted).

    This catches cascading condensation (e.g. H₂O_l consuming O budget then revealing C_s
    stability) that a single-pass screen misses. Because the returned mask is the monotone union
    of the input and new predictions (condensates are never retracted), the set can grow by at
    least one entry per iteration and convergence to a fixed point is guaranteed in at most
    ``n_condensates`` iterations.

    **Step 1 — predicted-stable condensates (first priority):** Only condensates identified as
    supersaturated in the pre-screen are allocated element budget. This keeps more element budget
    available for the gas-phase species.

    **Step 2 — gas/melt/solid species:** Each element's *remaining* budget (after condensate
    consumption) is distributed across non-condensate species by the same limiting-reagent logic.
    Species whose element budget is fully consumed by condensates fall back to
    :const:`~atmodeller.constants.INITIAL_LOG_NUMBER_MOLES`.

    **Step 3 — fugacity-constrained gas species:** The total moles of mass-constrained gas species
    (from step 2) are used to estimate the gas volume via the ideal gas law. The pressure is
    estimated from the gas mass of those species via
    :meth:`~atmodeller.interfaces.ThermodynamicStateProtocol.get_pressure`. Fugacity-constrained
    gas species (e.g. O₂ set by a redox buffer) are then assigned mole counts via
    :math:`n_i = f_i \\cdot n_\\mathrm{gas,known} / P`.

    Log stability is initialised to a strongly negative value (``-60``) for predicted-stable
    condensates and to :const:`~atmodeller.constants.INITIAL_LOG_STABILITY` for all other species.

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
    jax.debug.print("formula_matrix = {out}", out=A)

    # element moles: (n_elements,) — NaN where element is not mass-constrained.
    # log_abundance() squeezes the leading batch dimension when unbatched, giving a 1-D array.
    b: Float[Array, " n_elements"] = jnp.exp(parameters.mass_constraints.log_abundance())
    jax.debug.print("element_moles = {out}", out=b)

    # Species that are condensates
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
    # So for the test_graphite_water_stable test, the species are ordered as:
    # H2O(g), H2(g), O2(g), CO(g), CO2(g), CH4(g), H2O(l), C(s) - these are the columns
    reaction_matrix: Float[Array, "n_rxn n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)
    # If we are solving for the species stability, which is common for condensates (either pure
    # phase or in the melt or solid phase), this mask indicates where the stability criteria (also
    # part of the solution array) enters the calculation
    stability_matrix_rxn: Float[Array, "n_rxn n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_stability_matrix_full
    )
    jax.debug.print("stability_matrix_rxn = {out}", out=stability_matrix_rxn)

    # log Kp of the reactions. The number of entries is the same as the number of reactions (i.e.,
    # number of rows of the reaction matrix)
    log_Kp: Float[Array, " n_rxn"] = parameters.reaction_system.reaction.get_log_Kp(temperature)
    jax.debug.print("log_Kp = {out}", out=log_Kp)

    # Species that are not condensates
    other_mask: Bool[Array, " n_species"] = ~condensate_mask
    jax.debug.print("other_mask = {out}", out=other_mask)
    # Total element stoichiometry of non-condensate species
    other_stoich_total: Float[Array, " n_elements"] = jnp.sum(A * other_mask, axis=1)
    jax.debug.print("other_stoich_total = {out}", out=other_stoich_total)
    # Guard to avoid divide by zero
    safe_other_stoich: Float[Array, " n_elements"] = jnp.where(
        other_stoich_total > 0, other_stoich_total, 1.0
    )
    jax.debug.print("safe_other_stoich = {out}", out=safe_other_stoich)

    def _one_stability_pass(
        condensate_stable_known: Bool[Array, " n_species"],
    ) -> Bool[Array, " n_species"]:
        """One stability-prediction pass.

        Given a set of already-known-stable condensates, allocates element budget to them first,
        distributes the remainder to gas/melt/solid, computes ideal-gas activities, evaluates
        the reaction K vs Q signal, and returns the monotone union of the new predictions with
        the input mask.

        Args:
            condensate_stable_known: Boolean mask of condensates predicted stable so far.

        Returns:
            Updated mask — superset of ``condensate_stable_known``.
        """
        # Allocate element budget to known-stable condensates (limiting-reagent).
        known_stoich_total: Float[Array, " n_elements"] = jnp.sum(
            A * condensate_stable_known, axis=1
        )
        jax.debug.print("known_stoich_total = {out}", out=known_stoich_total)
        # Guard to avoid divide by zero
        safe_known_stoich: Float[Array, " n_elements"] = jnp.where(
            known_stoich_total > 0, known_stoich_total, 1.0
        )
        jax.debug.print("safe_known_stoich = {out}", out=safe_known_stoich)

        # For each element e, how many moles of each known-stable condensate species s would
        # be implied if element e were the sole limiting reagent: n_s = b[e] / A[e,s].
        known_share: Float[Array, " n_elements"] = b / safe_known_stoich
        jax.debug.print("known_share = {out}", out=known_share)
        # Only consider (e, s) pairs where species s actually contains element e, s is a
        # known-stable condensate, and element e is mass-constrained (not NaN).
        is_known_constrained: Bool[Array, "n_elements n_species"] = (
            (A > 0) & condensate_stable_known & ~jnp.isnan(b[:, None])
        )
        jax.debug.print("is_known_constrained = {out}", out=is_known_constrained)
        # Fill unconstrained (e, s) pairs with inf so they don't win the min below.
        known_implied: Float[Array, "n_elements n_species"] = jnp.where(
            is_known_constrained, known_share[:, None], jnp.inf
        )
        jax.debug.print("known_implied = {out}", out=known_implied)

        n_known: Float[Array, " n_species"] = jnp.min(known_implied, axis=0)
        jax.debug.print("n_known = {out}", out=n_known)
        n_known_applied: Float[Array, " n_species"] = jnp.where(jnp.isinf(n_known), 0.0, n_known)
        jax.debug.print("n_known_applied = {out}", out=n_known_applied)

        # element_used[e] = sum_s( A[e,s] * n_known_applied[s] ) for known-stable condensates only:
        # the total moles of element e consumed by the allocated condensate budget.
        element_used: Float[Array, " n_elements"] = jnp.einsum(
            "es,s->e", A * condensate_stable_known, n_known_applied
        )
        jax.debug.print("element_used = {out}", out=element_used)
        # This is the amount of remaining element budget after known-stable condensate allocation;
        # it will be distributed to all non-condensate species in the next step.
        remaining_b: Float[Array, " n_elements"] = jnp.maximum(b - element_used, 0.0)
        jax.debug.print("remaining_b = {out}", out=remaining_b)

        # Distribute remaining budget to non-condensate species.
        has_remaining: Bool[Array, " n_elements"] = (remaining_b > 0) & ~jnp.isnan(remaining_b)
        jax.debug.print("has_remaining = {out}", out=has_remaining)

        # Same limiting-reagent logic as above, but applied to non-condensate species using
        # the remaining element budget. other_share[e] = remaining_b[e] / safe_other_stoich[e].
        other_share: Float[Array, " n_elements"] = remaining_b / safe_other_stoich
        jax.debug.print("other_share = {out}", out=other_share)
        # Only consider (e, s) pairs where species s is a non-condensate, contains element e,
        # and element e still has remaining budget after condensate allocation.
        is_other_constrained: Bool[Array, "n_elements n_species"] = (
            (A > 0) & other_mask & has_remaining[:, None]
        )
        jax.debug.print("is_other_constrained = {out}", out=is_other_constrained)
        # Fill unconstrained pairs with inf so they don't win the min below.
        other_implied: Float[Array, "n_elements n_species"] = jnp.where(
            is_other_constrained, other_share[:, None], jnp.inf
        )
        jax.debug.print("other_implied = {out}", out=other_implied)
        # Limiting-reagent: the tightest element constraint sets the mole count for each species.
        n_other: Float[Array, " n_species"] = jnp.min(other_implied, axis=0)
        jax.debug.print("n_other = {out}", out=n_other)
        # Species unconstrained by any element (all-inf column) fall back to INITIAL_LOG_NUMBER_MOLES.
        n_gas_est: Float[Array, " n_species"] = jnp.where(jnp.isinf(n_other), fallback, n_other)
        jax.debug.print("n_gas_est = {out}", out=n_gas_est)

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
            0.0,  # NOTE: Approximation. Species in phases will have non-ideal activities that we
            # are ignoring here, but this is just a heuristic pre-screen.
        )

        # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
        # NaN entries in log_activity propagate to log_Q --> stability_signal NaN --> not > 0
        # (safe).
        log_Q: Float[Array, " n_rxn"] = jnp.einsum("rs,s->r", reaction_matrix, log_activity)
        stability_signal: Float[Array, "n_rxn n_species"] = stability_matrix_rxn * (
            log_Kp[:, None] - log_Q[:, None]
        )
        jax.debug.print(
            "  known={k} log_act={la} log_Kp={kp} log_Q={lq} stab_col={sc}",
            k=condensate_stable_known,
            la=log_activity,
            kp=log_Kp,
            lq=log_Q,
            sc=stability_signal[:, -1],
        )
        new_predictions: Bool[Array, " n_species"] = (
            jnp.any(stability_signal > 0, axis=0) & condensate_mask
        )
        jax.debug.print("new_predictions = {out}", out=new_predictions)

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
    jax.debug.print("init_stable = {out}", out=init_stable)
    first_stable: Bool[Array, " n_species"] = _one_stability_pass(init_stable)
    jax.debug.print("first_stable = {out}", out=first_stable)

    def _cond_fn(carry: tuple) -> Bool[Array, ""]:
        prev, curr, i = carry
        return jnp.any(prev != curr) & (i < _MAX_STABILITY_ITERS)

    def _body_fn(carry: tuple) -> tuple:
        _, curr, i = carry
        return curr, _one_stability_pass(curr), i + 1

    _, condensate_stable_predicted, _ = lax.while_loop(
        _cond_fn, _body_fn, (init_stable, first_stable, jnp.array(0))
    )

    # --- Step 1: allocate element budget to predicted-stable condensates ---
    condensate_stoich_total: Float[Array, " n_elements"] = jnp.sum(
        A * condensate_stable_predicted, axis=1
    )
    safe_condensate_stoich: Float[Array, " n_elements"] = jnp.where(
        condensate_stoich_total > 0, condensate_stoich_total, 1.0
    )
    condensate_share: Float[Array, " n_elements"] = b / safe_condensate_stoich
    is_condensate_constrained: Bool[Array, "n_elements n_species"] = (
        (A > 0) & condensate_stable_predicted & ~jnp.isnan(b[:, None])
    )
    condensate_implied: Float[Array, "n_elements n_species"] = jnp.where(
        is_condensate_constrained, condensate_share[:, None], jnp.inf
    )
    n_condensate: Float[Array, " n_species"] = jnp.min(condensate_implied, axis=0)

    # Remaining element budget after stable-condensate allocation.
    n_condensate_applied: Float[Array, " n_species"] = jnp.where(
        jnp.isinf(n_condensate), 0.0, n_condensate
    )
    element_used: Float[Array, " n_elements"] = jnp.einsum(
        "es,s->e", A * condensate_stable_predicted, n_condensate_applied
    )
    remaining_b: Float[Array, " n_elements"] = jnp.maximum(b - element_used, 0.0)

    # --- Step 2: allocate remaining budget to non-condensate species ---
    other_share: Float[Array, " n_elements"] = remaining_b / safe_other_stoich
    has_remaining: Bool[Array, " n_elements"] = (remaining_b > 0) & ~jnp.isnan(remaining_b)
    is_other_constrained: Bool[Array, "n_elements n_species"] = (
        (A > 0) & other_mask & has_remaining[:, None]
    )
    other_implied: Float[Array, "n_elements n_species"] = jnp.where(
        is_other_constrained, other_share[:, None], jnp.inf
    )
    n_other: Float[Array, " n_species"] = jnp.min(other_implied, axis=0)

    # Combine: predicted-stable condensates use their budget estimate; all others use the
    # non-condensate remainder. Fallback for species not covered by any constrained element.
    n_estimate: Float[Array, " n_species"] = jnp.where(
        condensate_stable_predicted, n_condensate, n_other
    )
    n_estimate = jnp.where(jnp.isinf(n_estimate), fallback, n_estimate)

    log_number_moles: Float[Array, " n_species"] = jnp.log(n_estimate)

    # --- Step 3: fugacity-constrained gas species ---
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

    # Log stability: strongly stable (-60) for predicted-stable condensates so the solver starts
    # near the correct stability regime; INITIAL_LOG_STABILITY for everything else.
    log_stability: Float[Array, " n_species"] = jnp.where(
        condensate_stable_predicted,
        jnp.full_like(log_number_moles, -60.0),
        jnp.full_like(log_number_moles, INITIAL_LOG_STABILITY),
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
    solution_stability: Float[Array, " n_species"] = sol.value[n:]
    rms_auto_moles: Float[Array, ""] = jnp.sqrt(
        jnp.mean((initial_guess[:n] - solution_moles) ** 2)
    )
    rms_input_moles: Float[Array, ""] = jnp.sqrt(
        jnp.mean((initial_guess_in[:n] - solution_moles) ** 2)
    )
    rms_auto_stability: Float[Array, ""] = jnp.sqrt(
        jnp.mean((initial_guess[n:] - solution_stability) ** 2)
    )
    rms_input_stability: Float[Array, ""] = jnp.sqrt(
        jnp.mean((initial_guess_in[n:] - solution_stability) ** 2)
    )
    jax.debug.print(
        "RMS log_n:      auto={auto:.4f}  input={inp:.4f}",
        auto=rms_auto_moles,
        inp=rms_input_moles,
    )
    jax.debug.print(
        "RMS log_stab:   auto={auto:.4f}  input={inp:.4f}",
        auto=rms_auto_stability,
        inp=rms_input_stability,
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
