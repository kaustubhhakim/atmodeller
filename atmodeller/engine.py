# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""JAX-based model functions for atmospheric and chemical equilibrium calculations.

This module defines the core set of single-instance model functions (e.g., thermodynamic property
calculations, equation-of-state relations, reaction masks) that operate on a single set of inputs,
without any implicit batching.

These functions form the building blocks for solving the coupled system of equations governing the
model (e.g., mass balance, fugacity constraints, phase stability), and are intended to be:

    1. Pure: No side effects, deterministic outputs for given inputs.
    2. JAX-compatible: Written with ``jax.numpy`` and compatible with transformations such as
       ``jit``, ``grad``, and ``vmap``.
    3. Shape-consistent: Accept and return arrays with predictable shapes, enabling easy
       vectorisation.

In practice, these functions are rarely called directly in production code. Instead, they are
wrapped with :func:`equinox.filter_vmap` to enable efficient batched evaluation over multiple
scenarios or parameter sets.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxmod.utils import safe_exp
from jaxtyping import Array, Bool, Float, Integer

from atmodeller.parameters import Parameters


def get_active_mask(parameters: Parameters) -> Bool[Array, "... dim"]:
    """Gets the mask of active residual quantities.

    Args:
        parameters: Parameters

    Returns:
        Active mask
    """
    fugacity_mask: Bool[Array, "..."] = jnp.asarray(
        parameters.fugacity_constraints.active(), dtype=bool
    )
    mass_mask: Bool[Array, "..."] = jnp.asarray(parameters.mass_constraints.active(), dtype=bool)

    # Infer batch shape from the potentially 2-D masks
    batch_shape: tuple[int, ...] = jnp.broadcast_shapes(
        fugacity_mask.shape[:-1], mass_mask.shape[:-1]
    )

    reactions_mask: Bool[Array, " reactions"] = jnp.ones(
        parameters.reaction_system.number_reactions, dtype=bool
    )
    jax.debug.print("reactions_mask = {out}", out=reactions_mask)
    stability_mask: Bool[Array, " species"] = jnp.asarray(
        parameters.reaction_system.species.active_stability, dtype=bool
    )
    jax.debug.print("stability_mask = {out}", out=stability_mask)

    # Broadcast 1-D masks to batch shape + their own last dim
    reactions_mask = jnp.broadcast_to(reactions_mask, batch_shape + reactions_mask.shape)
    jax.debug.print("reactions_mask (broadcast) = {out}", out=reactions_mask)
    stability_mask = jnp.broadcast_to(stability_mask, batch_shape + stability_mask.shape)
    jax.debug.print("stability_mask (broadcast) = {out}", out=stability_mask)
    fugacity_mask = jnp.broadcast_to(fugacity_mask, batch_shape + fugacity_mask.shape[-1:])
    jax.debug.print("fugacity_mask (broadcast) = {out}", out=fugacity_mask)
    mass_mask = jnp.broadcast_to(mass_mask, batch_shape + mass_mask.shape[-1:])
    jax.debug.print("mass_mask (broadcast) = {out}", out=mass_mask)

    # NOTE: Order must be identical to objective_function() residual concatenation
    return jnp.concatenate([fugacity_mask, reactions_mask, mass_mask, stability_mask], axis=-1)


def get_min_log_elemental_abundance_per_species(
    parameters: Parameters,
) -> Float[Array, "... species"]:
    """For each species, find the elemental mass constraint with the lowest abundance.

    Args:
        parameters: Parameters

    Returns:
        A vector of the minimum log elemental abundance for each species
    """
    formula_matrix: Integer[Array, "elements species"] = jnp.asarray(
        parameters.reaction_system.formula_matrix
    )
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    mask: Bool[Array, "elements species"] = formula_matrix != 0
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    log_abundance: Float[Array, "... elements"] = parameters.mass_constraints.log_abundance()
    # jax.debug.print("log_abundance = {out}", out=log_abundance)

    # Broadcast over batch dimensions and species, then mask out absent elements.
    masked_abundance: Float[Array, "... elements species"] = jnp.where(
        mask, log_abundance[..., :, None], jnp.nan
    )
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Find the minimum log abundance per species
    min_abundance_per_species: Float[Array, "... species"] = jnp.nanmin(masked_abundance, axis=-2)
    # jax.debug.print("min_abundance_per_species = {out}", out=min_abundance_per_species)

    return min_abundance_per_species


def get_total_pressure(
    parameters: Parameters, log_number_moles: Float[Array, "... n_species"]
) -> Float[Array, "..."]:
    """Gets the total pressure.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Total pressure in bar
    """
    log_number_moles_gas: Float[Array, "... n_gas_species"] = log_number_moles[
        ..., parameters.reaction_system.gas_slice
    ]
    gas_mass: Float[Array, "... 1"] = jnp.exp(
        parameters.reaction_system.gas.get_log_phase_mass(log_number_moles_gas)
    )
    gas_mass_squeeze: Float[Array, "..."] = jnp.squeeze(gas_mass, axis=-1)
    pressure: Float[Array, "..."] = parameters.state.get_pressure(gas_mass_squeeze)

    return pressure


@eqx.filter_jit
def objective_function(
    solution: Float[Array, "... solution"], parameters: Parameters
) -> Float[Array, " ... residual"]:
    """Objective function

    The order of the residual does make a difference to the solution process. More investigations
    are necessary, but justification for the current ordering is as follows:

        1. Fugacity constraints - fixed target, well conditioned
        2. Reaction constraints - log-linear, physics-based coupling
        3. Mass balance constraints - stiffer
        4. Stability constraints - stiffer still

    Args:
        solution: Solution array for all species i.e. log number of moles and log stability
        parameters: Parameters

    Returns:
        Residual
    """
    jax.debug.print("Starting new objective_function evaluation")
    temperature: Float[Array, "..."] = parameters.state.temperature

    log_number_moles, log_stability = jnp.split(solution, 2, axis=-1)
    jax.debug.print("log_number_moles = {out}", out=log_number_moles)
    jax.debug.print("log_stability = {out}", out=log_stability)

    total_pressure: Float[Array, "..."] = get_total_pressure(parameters, log_number_moles)
    jax.debug.print("total_pressure = {out}", out=total_pressure)

    log_activity: Float[Array, "... species"] = parameters.reaction_system.get_log_activity(
        log_number_moles,
        temperature,
        total_pressure,
        jnp.log(parameters.state.molar_mass),
        jnp.log(parameters.state.melt_mass),
        jnp.log(parameters.state.solid_mass),
    )
    jax.debug.print("log_activity = {out}", out=log_activity)

    # Fugacity constraints residual (dimensionless)
    fugacity_residual: Float[Array, "... species"] = (
        log_activity - parameters.fugacity_constraints.log_fugacity(temperature, total_pressure)
    )
    jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
    # jax.debug.print(
    #     "fugacity_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(fugacity_residual),
    #     out2=jnp.nanmax(fugacity_residual),
    # )
    # jax.debug.print(
    #     "fugacity_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(fugacity_residual),
    #     out2=jnp.nanstd(fugacity_residual),
    # )

    reaction_residual: Float[Array, "... reactions"] = parameters.reaction_system.get_residual(
        log_activity, log_stability, temperature, total_pressure
    )
    jax.debug.print("reaction_residual = {out}", out=reaction_residual)

    # jax.debug.print(
    #     "reaction_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(reaction_residual),
    #     out2=jnp.nanmax(reaction_residual),
    # )
    # jax.debug.print(
    #     "reaction_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(reaction_residual),
    #     out2=jnp.nanstd(reaction_residual),
    # )

    # Elemental mass balance residual
    log_element_moles_total: Float[Array, "... elements"] = (
        parameters.reaction_system.get_log_element_moles(log_number_moles)
    )
    jax.debug.print("log_element_moles_total = {out}", out=log_element_moles_total)

    log_target_moles: Float[Array, "... elements"] = parameters.mass_constraints.log_abundance()
    jax.debug.print("log_target_moles = {out}", out=log_target_moles)

    # Dimensionless (ratio error - 1)
    # More robust than log residual for poor initial guesses, which are often the case.
    mass_residual: Float[Array, "... elements"] = (
        safe_exp(log_element_moles_total - log_target_moles) - 1
    )
    # Log residual converges fast when near the solution, but can be very large and unstable for
    # poor initial guesses, which are often the case.
    # mass_residual: Float[Array, " elements"] = log_element_moles_total - log_target_moles

    jax.debug.print("mass_residual = {out}", out=mass_residual)
    # jax.debug.print(
    #     "mass_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(mass_residual),
    #     out2=jnp.nanmax(mass_residual),
    # )
    # jax.debug.print(
    #     "mass_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(mass_residual),
    #     out2=jnp.nanstd(mass_residual),
    # )

    # Stability residual
    log_tau: Float[Array, "..."] = jnp.log(parameters.solver_parameters.tau)
    jax.debug.print("log_tau.shape = {out}", out=log_tau.shape)
    log_min_number_moles: Float[Array, "... species"] = (
        get_min_log_elemental_abundance_per_species(parameters) + log_tau
    )
    jax.debug.print("log_min_number_moles = {out}", out=log_min_number_moles)

    # Dimensionless (log-ratio)
    stability_residual: Float[Array, "... species"] = (
        log_number_moles + log_stability - log_min_number_moles
    )
    jax.debug.print("stability_residual = {out}", out=stability_residual)
    # jax.debug.print(
    #     "stability_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(stability_residual),
    #     out2=jnp.nanmax(stability_residual),
    # )
    # jax.debug.print(
    #     "stability_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(stability_residual),
    #     out2=jnp.nanstd(stability_residual),
    # )

    # NOTE: Order must be identical to get_active_mask()
    residual: Float[Array, "... residual"] = jnp.concatenate(
        [fugacity_residual, reaction_residual, mass_residual, stability_residual],
        axis=-1,
    )
    jax.debug.print("residual (with nans) = {out}", out=residual)

    # This final masking operation drops nans (unused constraint options)
    active_mask: Bool[Array, "... dim"] = get_active_mask(parameters)
    jax.debug.print("active_mask = {out}", out=active_mask)
    size: int = parameters.reaction_system.species.number_solution
    # jax.debug.print("size = {out}", out=size)
    dim: int = active_mask.shape[-1]

    # Replace inactive positions with sentinel `dim`, sort so active indices come first, slice
    # This is a way to get fixed-size index arrays without dynamic shapes.
    active_indices: Integer[Array, "... size"] = jnp.sort(
        jnp.where(active_mask, jnp.arange(dim), dim), axis=-1
    )[..., :size]
    # jax.debug.print("active_indices = {out}", out=active_indices)

    # Broadcast active_indices to match residual ndim (e.g. residual is (1,21), indices is (8,))
    active_indices = jnp.broadcast_to(active_indices, residual.shape[:-1] + (size,))
    jax.debug.print("active_indices = {out}", out=active_indices)

    residual = jnp.take_along_axis(residual, active_indices, axis=-1)
    jax.debug.print("residual = {out}", out=residual)

    return residual
