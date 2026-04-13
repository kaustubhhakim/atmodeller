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

LOG_TRACE_VALUE: float = -20.0
"""Small trace value (in log space) to assign to species that have a negligible element budget."""


def max_moles_by_limiting_element(
    parameters: Parameters,
    element_abundance: Float[Array, "... n_elements"],
    mask: Bool[Array, "... n_species"],
) -> Float[Array, "... n_species"]:
    """Maximum moles by limiting element for species in ``mask``

    For each species in ``mask``, the mole count is estimated by asking: given the available
    element budget, how many moles of this species could be formed if that element were shared
    equally among all masked species that contain it? The tightest such constraint across all
    elements that appear in the species formula determines the estimate. Returns ``NaN`` for
    species not in ``mask`` or not constrained by any available element.

    Args:
        parameters: Parameters
        element_abundance: Element abundance. ``NaN`` for unconstrained elements.
        mask: Boolean mask selecting the species to allocate budget to.

    Returns:
        Per-species mole estimates; ``NaN`` where unconstrained.
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )
    constrained_element: Bool[Array, "... n_elements"] = ~jnp.isnan(element_abundance)

    # Broadcast all masks to (..., n_elements, n_species)
    fm_mask: Bool[Array, "1 n_elements n_species"] = (formula_matrix > 0)[None, ...]
    mask_b: Bool[Array, "... 1 n_species"] = mask[..., None, :]
    constrained_element_b: Bool[Array, "... n_elements 1"] = constrained_element[..., :, None]

    stoich_total: Float[Array, "... n_elements"] = jnp.sum(
        formula_matrix[None, ...] * mask_b, axis=-1
    )
    safe_stoich: Float[Array, "... n_elements"] = jnp.where(stoich_total > 0, stoich_total, 1.0)
    share: Float[Array, "... n_elements"] = element_abundance / safe_stoich

    is_constrained: Bool[Array, "... n_elements n_species"] = (
        fm_mask & mask_b & constrained_element_b
    )
    # jax.debug.print("is_constrained = {out}", out=is_constrained)
    implied: Float[Array, "... n_elements n_species"] = jnp.where(
        is_constrained, share[..., :, None], jnp.nan
    )
    # jax.debug.print("implied = {out}", out=implied)

    max_moles_by_limiting_element: Float[Array, "... n_species"] = jnp.nanmin(implied, axis=-2)
    # jax.debug.print("max_moles_by_limiting_element = {out}", out=max_moles_by_limiting_element)

    return max_moles_by_limiting_element


def allocate_element_budget(
    parameters: Parameters,
    element_abundance: Float[Array, "... n_elements"],
    condensate_stable_mask: Bool[Array, "... n_species"],
) -> tuple[Float[Array, "... n_species"], Float[Array, "... n_species"]]:
    """Allocates the element budget between predicted-stable condensates and other species.

    This function first assigns as much of each element as possible to the species predicted to be
    stable condensates, according to the limiting-reagent principle. It then computes the remaining
    element budget and allocates it to the non-condensate species, again using the limiting-reagent
    logic. Finally, fallback logic is applied to assign small nonzero values or the geometric mean.

    Args:
        parameters: Parameters
        element_abundance: Available abundance of each element
        condensate_stable_mask: Boolean mask indicating which species are predicted-stable
            condensates

    Returns:
        Tuple of two arrays:
            - n_condensate: Moles allocated to each condensate species
            - n_other: Moles allocated to each non-condensate species
    """
    formula_matrix: Float[Array, "n_elements n_species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix, dtype=float
    )
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )

    # Allocate element budget to predicted-stable condensates
    n_condensate: Float[Array, "... n_species"] = max_moles_by_limiting_element(
        parameters, element_abundance, condensate_stable_mask
    )
    # jax.debug.print("n_condensate = {out}", out=n_condensate)

    # Assign fallback values to unconstrained (NaN) entries after stable-condensate allocation.
    # Species with NaN allocation are considered unstable or absent for element accounting and are
    # assigned a small trace value (exp(LOG_TRACE_VALUE)). This ensures they do not consume element
    # budget and prevents NaN propagation. Any values assigned to non-condensate species here are
    # ignored in subsequent steps by masking.
    n_condensate_fallback: Float[Array, "... n_species"] = jnp.where(
        jnp.isnan(n_condensate), jnp.exp(LOG_TRACE_VALUE), n_condensate
    )
    # jax.debug.print("n_condensate_fallback = {out}", out=n_condensate_fallback)

    # Total moles of each element consumed by the allocated condensate budget. Note here that the
    # mask ensures that only predicted-stable condensates contribute to the element consumption.
    element_used: Float[Array, "... n_elements"] = jnp.einsum(
        "es,...s->...e", formula_matrix, condensate_stable_mask * n_condensate_fallback
    )
    # jax.debug.print("element_used = {out}", out=element_used)

    # Remaining element budget after stable-condensate allocation, which must be positive or zero.
    remaining_b: Float[Array, "... n_elements"] = jnp.maximum(
        element_abundance - element_used, 0.0
    )
    # jax.debug.print("remaining_b = {out}", out=remaining_b)

    # Allocate remaining budget to non-condensate species
    n_other: Float[Array, "... n_species"] = max_moles_by_limiting_element(
        parameters, remaining_b, ~condensate_mask
    )
    # jax.debug.print("n_other = {out}", out=n_other)

    # Apply fallback logic to non-condensate species.
    # If a species is unconstrained by any available element or constrained to zero moles, fallback
    # values are assigned. For unconstrained species, the fallback value is the geometric mean of
    # the finite estimates, which keeps the magnitude of missing species comparable to known ones.
    # For exhausted elements, the fallback value is a small trace value.
    unconstrained: Bool[Array, "... n_species"] = jnp.isnan(n_other) & ~condensate_mask
    # jax.debug.print("unconstrained = {out}", out=unconstrained)
    exhausted: Bool[Array, "... n_species"] = n_other == 0 & ~condensate_mask
    # jax.debug.print("exhausted = {out}", out=exhausted)
    valid: Bool[Array, "... n_species"] = ~unconstrained & ~exhausted & ~condensate_mask
    # jax.debug.print("valid = {out}", out=valid)

    log_geometric_mean: FloatArray = jnp.sum(
        jnp.where(valid, jnp.log(n_other), 0.0), axis=-1, keepdims=True
    ) / jnp.sum(valid, axis=-1, keepdims=True).astype(float)
    # jax.debug.print("log_geometric_mean = {out}", out=log_geometric_mean)

    n_other_fallback = jnp.where(unconstrained, jnp.exp(log_geometric_mean), n_other)
    n_other_fallback = jnp.where(exhausted, jnp.exp(LOG_TRACE_VALUE), n_other_fallback)

    # jax.debug.print("n_condensate after fallback = {out}", out=n_condensate_fallback)
    # jax.debug.print("n_other after fallback = {out}", out=n_other_fallback)

    return n_condensate_fallback, n_other_fallback


def get_log_activity_estimate(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Estimates log activity for gas species based on ideal-gas assumptions.

    For gas species, the log activity is estimated as log(x_i * P) = log(n_i/n_total) + log(P).
    For non-gas species (melt, solid, pure-phase condensates), the log activity is set to 0,
    i.e. unit activity. This ignores dilution (mole fraction < 1) and activity coefficients
    for melt/solid solution species, but is intentional: the pre-screen is a cheap heuristic
    and calling the EOS/mixing models here would be circular and expensive. Activity
    coefficients are in any case unavailable without a complete solution.

    Args:
        parameters: Parameters containing the reaction system and state information
        species_abundance: Estimated abundance of each species

    Returns:
        Estimated log activity for each species
    """
    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )

    # Must be broadcast to a column array
    pressure: Float[Array, "... 1"] = parameters.state.get_pressure(jnp.log(species_abundance))[
        ..., None
    ]
    n_gas_total: Float[Array, "... 1"] = jnp.nansum(
        jnp.where(gas_mask, species_abundance, 0.0), axis=-1, keepdims=True
    )
    safe_n_gas_total: Float[Array, "... 1"] = jnp.where(n_gas_total > 0, n_gas_total, 1.0)

    # TODO: Here, activity of dissolved species is also computed as unity. To improve.
    log_activity: Float[Array, "... n_species"] = jnp.where(
        gas_mask,
        jnp.log(jnp.where(gas_mask, species_abundance, 1.0))
        - jnp.log(safe_n_gas_total)
        + jnp.log(jnp.where(pressure > 0, pressure, 1.0)),
        0.0,
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    return log_activity


def get_stability_signal(
    parameters: Parameters, species_abundance: Float[Array, "... n_species"]
) -> Float[Array, "... n_species"]:
    """Computes the stability signal for each reaction and condensate species.

    The stability signal is defined as sm[r,c] * (log_Kp[r] - log_Q[r]), where sm is the
    stability matrix, Kp is the equilibrium constant, and Q is the reaction quotient. A positive
    signal means the reaction is driven toward condensate formation given the current assumed
    activities.

    Args:
        parameters: Parameters containing the reaction system information
        species_abundance: Estimated abundance of each species, used to estimate activities

    Returns:
        Stability signal for each reaction and condensate species
    """
    reaction_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_matrix_full
    )
    # jax.debug.print("reaction_matrix = {out}", out=reaction_matrix)
    stability_matrix: Float[Array, "n_reactions n_species"] = jnp.asarray(
        parameters.reaction_system.reaction.reaction_stability_matrix_full
    )
    # jax.debug.print("stability_matrix = {out}", out=stability_matrix)

    temperature: FloatArray = parameters.state.temperature
    log_Kp: Float[Array, "... n_reactions"] = parameters.reaction_system.reaction.get_log_Kp(
        temperature
    )
    # jax.debug.print("log_Kp = {out}", out=log_Kp)

    log_activity: Float[Array, "... n_species"] = get_log_activity_estimate(
        parameters, species_abundance
    )

    # Condensate c is supersaturated when sm[r,c] * (log_Kp[r] - log_Q[r]) > 0.
    log_Q: Float[Array, "... n_reactions"] = jnp.einsum(
        "rs,...s->...r", reaction_matrix, log_activity
    )
    stability_signal: Float[Array, "... n_reactions n_species"] = (
        stability_matrix * jnp.expand_dims(log_Kp - log_Q, axis=-1)
    )
    # jax.debug.print("stability_signal = {out}", out=stability_signal)

    return stability_signal


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

    Note:
        This function must support native broadcasting to be compatible with output routines.
        However, in the batch solver, the engine applies vmap to this function.

    Args:
        parameters: Parameters for a single batch element

    Returns:
        Concatenated ``[log_number_moles, log_stability]`` of length ``2 * n_species``
    """
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )

    temperature: FloatArray = parameters.state.temperature

    element_abundance: Float[Array, "... n_elements"] = parameters.mass_constraints.abundance()
    # jax.debug.print("element_abundance = {out}", out=element_abundance)

    gas_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.gas_species_mask
    )
    condensate_mask: Bool[Array, " n_species"] = jnp.asarray(
        parameters.reaction_system.phase_system.condensates_species_mask
    )
    active_activity_constraints: Bool[Array, "... n_species"] = (
        parameters.activity_constraints.active()
    )
    # jax.debug.print("active_activity_constraints = {out}", out=active_activity_constraints)

    gas_no_imposed_fugacity: Bool[Array, "... n_species"] = gas_mask & ~active_activity_constraints
    # jax.debug.print("gas_no_imposed_fugacity = {out}", out=gas_no_imposed_fugacity)

    def _one_stability_pass(
        condensate_stable_known: Bool[Array, "... n_species"],
    ) -> Bool[Array, "... n_species"]:
        """One stability-prediction pass

        Given a set of already-known-stable condensates, allocates element budget to them first,
        distributes the remainder to gas species, computes ideal activities, evaluates the reaction
        K vs Q signal, and returns the monotone union of the new predictions with the input mask.

        Args:
            condensate_stable_known: Boolean mask of condensates predicted stable so far

        Returns:
            Updated mask - superset of ``condensate_stable_known``.
        """
        _, n_other = allocate_element_budget(
            parameters, element_abundance, condensate_stable_known
        )
        stability_signal: Float[Array, "... n_reactions n_species"] = get_stability_signal(
            parameters, n_other
        )
        new_predictions: Bool[Array, "... n_species"] = (
            jnp.any(stability_signal > 0, axis=-2) & condensate_mask
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
    first_stable: Bool[Array, "... n_species"] = _one_stability_pass(init_stable)
    # jax.debug.print("first_stable = {out}", out=first_stable)

    init_stable_broadcasted: Bool[Array, "... n_species"] = jnp.broadcast_to(
        init_stable, first_stable.shape
    )

    def _cond_fn(carry: tuple) -> Bool[Array, "..."]:
        prev, curr, i = carry
        return jnp.any(prev != curr)

    def _body_fn(carry: tuple) -> tuple:
        _, curr, i = carry
        return curr, _one_stability_pass(curr), i + 1

    _, condensate_stable_predicted, _ = lax.while_loop(
        _cond_fn, _body_fn, (init_stable_broadcasted, first_stable, jnp.array(0))
    )

    n_condensate, n_other = allocate_element_budget(
        parameters, element_abundance, condensate_stable_predicted
    )

    # Combine: predicted-stable condensates use their budget estimate; all others use the
    # non-condensate remainder. Fallback for species not covered by any constrained element:
    # geometric mean of finite estimates keeps missing species at a comparable magnitude.
    n_estimate: Float[Array, "... n_species"] = jnp.where(condensate_mask, n_condensate, n_other)
    # jax.debug.print("n_estimate after merge = {out}", out=n_estimate)

    log_number_moles: Float[Array, "... n_species"] = jnp.log(n_estimate)
    # jax.debug.print("log_number_moles = {out}", out=log_number_moles)

    # Fugacity-constrained gas species
    log_n_gas_known_total: Float[Array, "..."] = logsumexp(
        jnp.where(gas_no_imposed_fugacity, log_number_moles, -jnp.inf),
        axis=-1,
    )
    pressure: Float[Array, "..."] = parameters.state.get_pressure(log_number_moles)
    log_fug: Float[Array, "... n_species"] = parameters.activity_constraints.log_activity(
        temperature, pressure
    )
    log_n_fug: Float[Array, "... n_species"] = log_fug + log_n_gas_known_total - jnp.log(pressure)
    # jax.debug.print("log_n_fug = {out}", out=log_n_fug)

    log_number_moles = jnp.where(
        gas_mask & active_activity_constraints, log_n_fug, log_number_moles
    )
    # jax.debug.print("log_number_moles after fugacity constraints = {out}", out=log_number_moles)

    # Log stability for predicted-stable condensates: initialize at the value that makes the
    # stability residual (log_n + log_s - (min_log_abundance + log_tau)) exactly zero given the
    # current mole estimate. This automatically scales with tau so no magic constant is needed.
    # Falls back to INITIAL_LOG_STABILITY where the expression is non-finite (e.g. zero-budget
    # elements, though those species should not be predicted stable anyway).
    log_tau_val: Float[Array, ""] = jnp.log(parameters.solver_parameters.tau)
    min_log_abundance_per_species: Float[Array, "... n_species"] = (
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

    result = jnp.concatenate((log_number_moles, log_stability), axis=-1)

    # TODO: Maybe clean this up, but must work for both vmapping and native broadcasting
    # Only squeeze axis=0 if its size is 1 (single case), else return as-is (batched)
    if result.shape[0] == 1:
        return jnp.squeeze(result, axis=0)

    return result
