# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Initial solution estimation"""

import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float

from atmodeller.constants import INITIAL_LOG_STABILITY
from atmodeller.engine import get_min_log_elemental_abundance_per_species
from atmodeller.jax_utils import FloatArray
from atmodeller.parameters import Parameters

LOG_NUMBER_MOLES_VMAP_AXES: int = 0
"""Axis index for the solution array in the vmapped batch solver"""


def _limiting_reagent(
    formula_matrix: Float[Array, "n_elements n_species"],
    element_abundance: Float[Array, "... n_elements"],
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
        # TODO: Might not be required, since we would never impose zero mass of an element as a
        # constraints in the first place.
        require_positive_budget: When ``True``, elements with ``element_abundance[e] = 0`` are
            excluded from constraining species. Use ``True`` when ``element_abundance`` is a
            remaining budget after prior allocation (zeros mean exhausted, not genuinely zero
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


def auto_initial_guess(parameters: Parameters) -> Float[Array, "... twice_species"]:
    r"""Generates an initial solution vector from element mass constraints and activity constraints.

    **Pre-screen — iterative condensate stability prediction:** Starting from a gas-only element
    distribution, the pre-screen iteratively grows the set of predicted-stable condensates using
    :func:`jax.lax.while_loop` until the set stops changing. Each iteration:

    1. Allocates element budget to currently-predicted-stable condensates (limiting-reagent).
    2. Distributes the remaining budget to non-condensate species.
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
    element budget is fully consumed by condensates fall back to the geometric mean of all
    finite species estimates in the system.

    **Step 3 — Fugacity-constrained gas species:** The total moles of mass-constrained gas species
    (from step 2) are used to estimate the gas volume via the ideal gas law. The pressure is
    estimated from the gas mass of those species via
    :meth:`~atmodeller.interfaces.ThermodynamicStateProtocol.get_pressure`. Fugacity-constrained
    gas species (e.g. O2 set by a redox buffer) are then assigned mole counts via
    :math:`n_i = f_i \cdot n_\mathrm{gas,known} / P`.

    # TODO: Below might actually depend on tau?
    Log stability is initialized to a strongly negative value (``-60``) for predicted-stable
    condensates and to :const:`~atmodeller.constants.INITIAL_LOG_STABILITY` for all other species.

    # TODO: Not necessarily now, should allow also natural broadcasting of the input parameters.
    Intended to be called inside a vmapped context (one batch element at a time).

    Args:
        parameters: Parameters for a single batch element

    Returns:
        Concatenated ``[log_number_moles, log_stability]`` of length ``2 * n_species``
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )
    reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    stability_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_stability_matrix_full
    )
    temperature: FloatArray = parameters.state.temperature

    log_Kp: Float[Array, "... n_reactions"] = parameters.reaction_system.reaction.get_log_Kp(
        temperature
    )
    # jax.debug.print("log_Kp = {out}", out=log_Kp)

    element_abundance: Float[Array, "... n_elements"] = parameters.mass_constraints.abundance()
    # jax.debug.print("element_abundance = {out}", out=element_abundance)

    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )
    # jax.debug.print("gas_mask = {out}", out=gas_mask)

    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )
    # jax.debug.print("condensate_mask = {out}", out=condensate_mask)

    # Species that are not condensates (i.e. gas, melt, solid solution species)
    not_condensate_mask: Bool[Array, " n_species"] = ~condensate_mask
    # jax.debug.print("not_condensate_mask = {out}", out=not_condensate_mask)

    active_activity_constraints: Bool[Array, "... n_species"] = (
        parameters.activity_constraints.active()
    )
    # jax.debug.print("active_activity_constraints = {out}", out=active_activity_constraints)

    gas_no_imposed_fugacity: Bool[Array, "... n_species"] = gas_mask & ~active_activity_constraints
    # jax.debug.print("gas_no_imposed_fugacity = {out}", out=gas_no_imposed_fugacity)

    def _one_stability_pass(
        condensate_stable_known: Bool[Array, " n_species"],
    ) -> Bool[Array, " n_species"]:
        """One stability-prediction pass.

        Given a set of already-known-stable condensates, allocates element budget to them first,
        distributes the remainder to gas species, computes ideal activities, evaluates the reaction
        K vs Q signal, and returns the monotone union of the new predictions with the input mask.

        Args:
            condensate_stable_known: Boolean mask of condensates predicted stable so far.

        Returns:
            Updated mask - superset of ``condensate_stable_known``.
        """
        # Allocate element budget to known-stable condensates (limiting-reagent).
        n_known: Float[Array, " n_species"] = _limiting_reagent(
            formula_matrix, element_abundance, condensate_stable_known
        )
        # jax.debug.print("n_known = {out}", out=n_known)
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
        # available element fall back to the geometric mean of the finite estimates (mean in
        # log-space), keeping the magnitude of missing species comparable to known ones.
        n_other: Float[Array, " n_species"] = _limiting_reagent(
            formula_matrix, remaining_b, not_condensate_mask, require_positive_budget=True
        )
        _valid: Bool[Array, " n_species"] = jnp.isfinite(n_other) & (n_other > 0)
        _n_valid: Float[Array, ""] = jnp.sum(_valid).astype(float)
        _log_fb: Float[Array, ""] = jnp.sum(
            jnp.where(_valid, jnp.log(n_other), 0.0)
        ) / jnp.maximum(_n_valid, 1)
        n_other = jnp.where(jnp.isinf(n_other), jnp.exp(_log_fb), n_other)
        # jax.debug.print("n_other = {out}", out=n_other)

        pressure: Float[Array, ""] = parameters.state.get_pressure(jnp.log(n_other))
        # jax.debug.print("pressure = {out}", out=pressure)

        # Ideal-gas log activity for gas species: log(x_i * P) = log(n_i/n_total) + log(P).
        # Non-gas species (melt, solid, pure-phase condensates) are assigned log_activity = 0,
        # i.e. unit activity. This ignores dilution (mole fraction < 1) and activity coefficients
        # for melt/solid solution species, but is intentional: the pre-screen is a cheap
        # heuristic and calling the EOS/mixing models here would be circular and expensive.
        # Activity coefficients are in any case unavailable without a complete solution.
        n_gas_total: Float[Array, ""] = jnp.sum(jnp.where(gas_mask, n_other, 0.0))
        safe_n_gas_total: Float[Array, ""] = jnp.where(n_gas_total > 0, n_gas_total, 1.0)
        log_activity: Float[Array, " n_species"] = jnp.where(
            gas_mask,
            jnp.log(jnp.where(gas_mask, n_other, 1.0))
            - jnp.log(safe_n_gas_total)
            + jnp.log(jnp.where(pressure > 0, pressure, 1.0)),
            0.0,
        )

        # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
        # NaN entries in log_activity propagate to log_Q --> stability_signal NaN --> not > 0
        # (safe).
        log_Q: Float[Array, " n_rxn"] = jnp.einsum("rs,s->r", reaction_matrix, log_activity)

        # FIXME: Was previous, but log_Kp is coming in as a column array when batched
        # stability_signal: Float[Array, "n_rxn n_species"] = stability_matrix * (
        #    log_Kp[:, None] - log_Q[:, None]
        # )

        stability_signal: Float[Array, "n_rxn n_species"] = stability_matrix * (log_Kp - log_Q)

        # jax.debug.print("stability_signal = {out}", out=stability_signal)
        new_predictions: Bool[Array, " n_species"] = (
            jnp.any(stability_signal > 0, axis=0) & condensate_mask
        )
        # jax.debug.print("new_predictions = {out}", out=new_predictions)

        # Monotone union: condensates are never retracted once predicted stable. This ensures the
        # set grows by at least one entry per iteration, guaranteeing fixed-point convergence
        # without a cap. The solver makes the definitive stable/absent decision.
        return condensate_stable_known | new_predictions

    # Iterate until the predicted-stable set stops growing
    # Initialize with no condensates known stable; the first body call is the gas-only pre-screen.
    # The monotone union in _one_stability_pass guarantees termination.
    init_stable: Bool[Array, " n_species"] = jnp.zeros_like(condensate_mask)
    # jax.debug.print("init_stable = {out}", out=init_stable)
    first_stable: Bool[Array, " n_species"] = _one_stability_pass(init_stable)
    # jax.debug.print("first_stable = {out}", out=first_stable)

    def _cond_fn(carry: tuple) -> Bool[Array, ""]:
        prev, curr, i = carry
        return jnp.any(prev != curr)

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
        formula_matrix, remaining_b, not_condensate_mask, require_positive_budget=True
    )

    # Combine: predicted-stable condensates use their budget estimate; all others use the
    # non-condensate remainder. Fallback for species not covered by any constrained element:
    # geometric mean of finite estimates keeps missing species at a comparable magnitude.
    n_estimate: Float[Array, " n_species"] = jnp.where(
        condensate_stable_predicted, n_condensate, n_other
    )
    _valid: Bool[Array, " n_species"] = jnp.isfinite(n_estimate) & (n_estimate > 0)
    _n_valid: Float[Array, ""] = jnp.sum(_valid).astype(float)
    _log_fb: Float[Array, ""] = jnp.sum(jnp.where(_valid, jnp.log(n_estimate), 0.0)) / jnp.maximum(
        _n_valid, 1
    )
    n_estimate = jnp.where(jnp.isinf(n_estimate), jnp.exp(_log_fb), n_estimate)

    log_number_moles: Float[Array, " n_species"] = jnp.log(n_estimate)

    # Step 3: fugacity-constrained gas species
    log_n_gas_known_total: Float[Array, ""] = logsumexp(
        jnp.where(gas_no_imposed_fugacity, log_number_moles, -jnp.inf)
    )
    pressure: Float[Array, ""] = parameters.state.get_pressure(log_number_moles)
    log_fug: Float[Array, "... n_species"] = parameters.activity_constraints.log_activity(
        temperature, pressure
    )
    log_n_fug: Float[Array, "... n_species"] = log_fug + log_n_gas_known_total - jnp.log(pressure)
    log_number_moles = jnp.where(
        gas_mask & active_activity_constraints, log_n_fug, log_number_moles
    )

    # Log stability for predicted-stable condensates: initialize at the value that makes the
    # stability residual (log_n + log_s - (min_log_abundance + log_tau)) exactly zero given the
    # current mole estimate. This automatically scales with tau so no magic constant is needed.
    # Falls back to INITIAL_LOG_STABILITY where the expression is non-finite (e.g. zero-budget
    # elements, though those species should not be predicted stable anyway).
    log_tau_val: Float[Array, ""] = jnp.log(parameters.solver_parameters.tau)
    min_log_abundance_per_species: Float[Array, " n_species"] = (
        get_min_log_elemental_abundance_per_species(parameters)
    )
    log_stability_stable: Float[Array, "... n_species"] = (
        min_log_abundance_per_species + log_tau_val - log_number_moles
    )
    log_stability: Float[Array, "... n_species"] = jnp.where(
        condensate_stable_predicted & jnp.isfinite(log_stability_stable),
        log_stability_stable,
        jnp.full_like(log_number_moles, INITIAL_LOG_STABILITY),
    )

    return jnp.concatenate((log_number_moles, log_stability), axis=-1)
