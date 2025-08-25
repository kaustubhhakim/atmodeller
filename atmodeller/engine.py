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

from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Shaped

from atmodeller.constants import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.containers import Parameters, Planet, SpeciesCollection
from atmodeller.type_aliases import NpBool
from atmodeller.utilities import (
    get_log_number_density_from_log_pressure,
    safe_exp,
    to_hashable,
    unit_conversion,
)


def get_active_mask(parameters: Parameters) -> Bool[Array, " dim"]:
    """Gets the mask of active residual quantities.

    Args:
        parameters: Parameters

    Returns:
        Active mask
    """
    # Get all masks separately
    fugacity_mask: Bool[Array, " dim"] = parameters.fugacity_constraints.active()
    reactions_mask: ArrayLike = parameters.species.active_reactions
    mass_mask: Bool[Array, " dim"] = parameters.mass_constraints.active()
    stability_mask: ArrayLike = parameters.species.active_stability

    # jax.debug.print("fugacity_mask = {out}", out=fugacity_mask)
    # jax.debug.print("reactions_mask = {out}", out=reactions_mask)
    # jax.debug.print("mass_mask = {out}", out=mass_mask)
    # jax.debug.print("stability_mask = {out}", out=stability_mask)

    active_mask: Bool[Array, " dim"] = jnp.concatenate(
        (fugacity_mask, reactions_mask, mass_mask, stability_mask)
    )
    # jax.debug.print("active_mask = {out}", out=active_mask)

    return active_mask


def get_atmosphere_log_molar_mass(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, ""]:
    """Gets the log molar mass of the atmosphere.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Log molar mass of the atmosphere
    """
    gas_log_number_density: Float[Array, " species"] = get_gas_species_data(
        parameters, log_number_density
    )
    gas_molar_mass: Float[Array, " species"] = get_gas_species_data(
        parameters, parameters.species.molar_masses
    )
    molar_mass: Float[Array, ""] = logsumexp(gas_log_number_density, b=gas_molar_mass) - logsumexp(
        gas_log_number_density, b=parameters.species.gas_species_mask
    )
    # jax.debug.print("molar_mass = {out}", out=molar_mass)

    return molar_mass


def get_atmosphere_log_volume(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, ""]:
    """Gets the log volume of the atmosphere.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Log volume of the atmosphere
    """
    log_volume: Float[Array, ""] = (
        jnp.log(GAS_CONSTANT)
        + jnp.log(parameters.planet.temperature)
        - get_atmosphere_log_molar_mass(parameters, log_number_density)
        + jnp.log(parameters.planet.surface_area)
        - jnp.log(parameters.planet.surface_gravity)
    )
    # jax.debug.print("log_volume = {out}", out=log_volume)

    return log_volume


def get_element_density(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " elements"]:
    """Number density of elements in the gas or condensed phase

    Input values are sanitised only for output routines, where partitioning between condensed and
    gas species is required. For the solver itself, this distinction is unnecessary.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Number density of elements in the gas or condensed phase
    """
    species_densities: Array = jnp.nan_to_num(safe_exp(log_number_density), nan=0.0)

    formula_matrix: Integer[Array, "elements species"] = jnp.asarray(
        parameters.species.formula_matrix
    )
    element_density: Float[Array, " elements"] = formula_matrix @ species_densities

    return element_density


def get_element_density_in_melt(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the number density of elements dissolved in melt

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Number density of elements dissolved in melt
    """
    species_melt_density: Float[Array, " species"] = get_species_density_in_melt(
        parameters, log_number_density
    )
    formula_matrix: Integer[Array, "elements species"] = jnp.asarray(
        parameters.species.formula_matrix
    )
    element_melt_density: Float[Array, " species"] = formula_matrix.dot(species_melt_density)

    return element_melt_density


def get_gas_species_data(
    parameters: Parameters, some_array: ArrayLike
) -> Shaped[Array, " species"]:
    """Masks the gas species data from an array.

    Args:
        parameters: Parameters
        some_array: Some array to mask the gas species data from

    Returns:
        An array with gas species data from `some_array` and condensate entries zeroed
    """
    gas_data: Shaped[Array, " species"] = (
        jnp.asarray(some_array) * parameters.species.gas_species_mask
    )

    return gas_data


def get_log_activity(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the log activity.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Log activity
    """
    gas_species_mask: Bool[Array, " species"] = jnp.array(parameters.species.gas_species_mask)
    number_density: Float[Array, " species"] = safe_exp(log_number_density)
    gas_species_number_density: Float[Array, " species"] = gas_species_mask * number_density
    atmosphere_log_number_density: Float[Array, ""] = jnp.log(jnp.sum(gas_species_number_density))

    log_activity_pure_species: Float[Array, " species"] = get_log_activity_pure_species(
        parameters, log_number_density
    )
    # jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)
    log_activity_gas_species: Float[Array, " species"] = (
        log_activity_pure_species + log_number_density - atmosphere_log_number_density
    )
    # jax.debug.print("log_activity_gas_species = {out}", out=log_activity_gas_species)
    log_activity: Float[Array, " species"] = jnp.where(
        gas_species_mask, log_activity_gas_species, log_activity_pure_species
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    return log_activity


def get_log_activity_pure_species(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the log activity of pure species.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Log activity of pure species
    """
    planet: Planet = parameters.planet
    temperature: Float[Array, ""] = planet.temperature
    species: SpeciesCollection = parameters.species
    total_pressure: Float[Array, ""] = get_total_pressure(parameters, log_number_density)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)

    activity_funcs: list[Callable] = [
        to_hashable(species_.activity.log_activity) for species_ in species
    ]

    def apply_activity(index: ArrayLike) -> Float[Array, ""]:
        return lax.switch(
            index,
            activity_funcs,
            temperature,
            total_pressure,
        )

    indices: Integer[Array, " species"] = jnp.arange(len(species))
    vmap_activity: Callable = eqx.filter_vmap(apply_activity, in_axes=(0,))
    log_activity_pure_species: Float[Array, " species"] = vmap_activity(indices)
    # jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)

    return log_activity_pure_species


def get_log_Kp(parameters: Parameters) -> Float[Array, " reactions"]:
    """Gets log of the equilibrium constant of each reaction in terms of partial pressures.

    Args:
        parameters: Parameters

    Returns:
        Log of the equilibrium constant of each reaction in terms of partial pressures
    """
    gibbs_funcs: list[Callable] = [
        to_hashable(species_.data.get_gibbs_over_RT) for species_ in parameters.species
    ]

    def apply_gibbs(
        index: Integer[Array, ""], temperature: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        return lax.switch(index, gibbs_funcs, temperature)

    indices: Integer[Array, " species"] = jnp.arange(len(parameters.species))
    vmap_gibbs: Callable = eqx.filter_vmap(apply_gibbs, in_axes=(0, None))
    gibbs_values: Float[Array, "species 1"] = vmap_gibbs(indices, parameters.planet.temperature)
    # jax.debug.print("gibbs_values = {out}", out=gibbs_values)
    reaction_matrix: Float[Array, "reactions species"] = jnp.asarray(
        parameters.species.reaction_matrix
    )
    log_Kp: Float[Array, "reactions 1"] = -1.0 * reaction_matrix @ gibbs_values

    return jnp.ravel(log_Kp)


def get_log_pressure_from_log_number_density(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets log pressure from log number density.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Log pressure
    """
    log_pressure: Float[Array, " species"] = (
        jnp.log(BOLTZMANN_CONSTANT_BAR)
        + jnp.log(parameters.planet.temperature)
        + log_number_density
    )

    return log_pressure


def get_log_reaction_equilibrium_constant(parameters: Parameters) -> Float[Array, " reactions"]:
    """Gets the log equilibrium constant of each reaction.

    Args:
        parameters: Parameters

    Returns:
        Log equilibrium constant of each reaction, hidden unit base of molecules/m^3
    """
    reaction_matrix: Float[Array, "reactions species"] = jnp.asarray(
        parameters.species.reaction_matrix
    )
    log_Kp: Float[Array, " reactions"] = get_log_Kp(parameters)
    # jax.debug.print("lnKp = {out}", out=lnKp)
    delta_n: Float[Array, " reactions"] = jnp.sum(
        reaction_matrix * parameters.species.gas_species_mask, axis=1
    )
    # jax.debug.print("delta_n = {out}", out=delta_n)
    log_Kc: Float[Array, " reactions"] = log_Kp - delta_n * (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(parameters.planet.temperature)
    )
    # jax.debug.print("log10Kc = {out}", out=log_Kc)

    return log_Kc


def get_min_log_elemental_abundance_per_species(
    parameters: Parameters,
) -> Float[Array, " species"]:
    """For each species, find the elemental mass constraint with the lowest abundance.

    Args:
        parameters: Parameters

    Returns:
        A vector of the minimum log elemental abundance for each species
    """
    formula_matrix: Integer[Array, "elements species"] = jnp.asarray(
        parameters.species.formula_matrix
    )
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    mask: Integer[Array, "elements species"] = (formula_matrix != 0).astype(jnp.int_)
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    # log_abundance is a 1-D array, which cannot be transposed, so make a 2-D array
    log_abundance: Float[Array, "elements 1"] = jnp.atleast_2d(
        parameters.mass_constraints.log_abundance
    ).T
    # jax.debug.print("log_abundance = {out}", out=log_abundance)

    # Element-wise multiplication with broadcasting
    masked_abundance: Float[Array, "elements species"] = mask * log_abundance
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)
    masked_abundance = jnp.where(mask != 0, masked_abundance, jnp.nan)
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Find the minimum log abundance per species
    min_abundance_per_species: Float[Array, " species"] = jnp.nanmin(masked_abundance, axis=0)
    # jax.debug.print("min_abundance_per_species = {out}", out=min_abundance_per_species)

    return min_abundance_per_species


def get_pressure_from_log_number_density(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets pressure from log number density.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Pressure
    """
    return safe_exp(get_log_pressure_from_log_number_density(parameters, log_number_density))


def get_reactions_only_mask(parameters: Parameters) -> Bool[Array, " dim"]:
    """Returns a mask with `True` only for active reactions positions, `False` elsewhere.

    Args:
        parameters: Parameters

    Returns:
        Reactions only mask for the residual array
    """
    # Create a full mask of False
    size: int = parameters.species.number_solution
    mask: Bool[Array, " dim"] = jnp.zeros(size, dtype=bool)

    fugacity_mask: Bool[Array, " dim"] = parameters.fugacity_constraints.active()
    reactions_mask: NpBool = parameters.species.active_reactions
    num_active_fugacity: Integer[Array, ""] = jnp.sum(fugacity_mask)

    # Place the reactions_mask at position num_active_fugacity dynamically.
    # Use lax.dynamic_update_slice: (array_to_update, update, start_indices)
    mask: Bool[Array, " dim"] = lax.dynamic_update_slice(
        mask, reactions_mask, (num_active_fugacity,)
    )

    return mask


def get_species_density_in_melt(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the number density of species dissolved in melt due to species solubility.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Number density of species dissolved in melt
    """
    molar_masses: ArrayLike = parameters.species.molar_masses
    melt_mass: Float[Array, ""] = parameters.planet.melt_mass

    ppmw: Float[Array, " species"] = get_species_ppmw_in_melt(parameters, log_number_density)

    log_volume: Float[Array, ""] = get_atmosphere_log_volume(parameters, log_number_density)
    species_melt_density: Float[Array, " species"] = (
        ppmw
        * unit_conversion.ppm_to_fraction
        * AVOGADRO
        * melt_mass
        / (molar_masses * safe_exp(log_volume))
    )
    # jax.debug.print("species_melt_density = {out}", out=species_melt_density)

    return species_melt_density


def get_species_ppmw_in_melt(
    parameters: Parameters,
    log_number_density: Float[Array, " species"],
) -> Float[Array, " species"]:
    """Gets the ppmw of species dissolved in melt.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        ppmw of species dissolved in melt
    """
    species: SpeciesCollection = parameters.species
    diatomic_oxygen_index: Integer[Array, ""] = jnp.array(parameters.species.diatomic_oxygen_index)
    temperature: Float[Array, ""] = parameters.planet.temperature

    log_activity: Float[Array, " species"] = get_log_activity(parameters, log_number_density)
    fugacity: Float[Array, " species"] = safe_exp(log_activity)
    total_pressure: Float[Array, ""] = get_total_pressure(parameters, log_number_density)
    diatomic_oxygen_fugacity: Float[Array, ""] = jnp.take(fugacity, diatomic_oxygen_index)

    # NOTE: All solubility formulations must return a JAX array to allow vmap
    solubility_funcs: list[Callable] = [
        to_hashable(species_.solubility.jax_concentration) for species_ in species
    ]

    def apply_solubility(
        index: Integer[Array, ""], fugacity: Float[Array, ""]
    ) -> Float[Array, ""]:
        return lax.switch(
            index,
            solubility_funcs,
            fugacity,
            temperature,
            total_pressure,
            diatomic_oxygen_fugacity,
        )

    indices: Integer[Array, " species"] = jnp.arange(len(species))
    vmap_solubility: Callable = eqx.filter_vmap(apply_solubility, in_axes=(0, 0))
    species_ppmw: Float[Array, " species"] = vmap_solubility(indices, fugacity)
    # jax.debug.print("ppmw = {out}", out=ppmw)

    return species_ppmw


def get_total_pressure(
    parameters: Parameters, log_number_density: Float[Array, " species"]
) -> Float[Array, ""]:
    """Gets total pressure.

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Total pressure
    """
    pressure: Float[Array, " species"] = get_pressure_from_log_number_density(
        parameters, log_number_density
    )
    gas_pressure: Float[Array, " species"] = pressure * parameters.species.gas_species_mask
    # jax.debug.print("gas_pressure = {out}", out=gas_pressure)

    return jnp.sum(gas_pressure)


def objective_function(
    solution: Float[Array, " solution"], parameters: Parameters
) -> Float[Array, " residual"]:
    """Objective function

    The order of the residual does make a difference to the solution process. More investigations
    are necessary, but justification for the current ordering is as follows:

        1. Fugacity constraints - fixed target, well conditioned
        2. Reaction constraints - log-linear, physics-based coupling
        3. Mass balance constraints - stiffer, depends on solubility
        4. Stability constraints - stiffer still

    Args:
        solution: Solution array for all species i.e. log number density and log stability
        parameters: Parameters

    Returns:
        Residual
    """
    # jax.debug.print("Starting new objective_function evaluation")
    temperature: Float[Array, ""] = parameters.planet.temperature

    log_number_density, log_stability = jnp.split(solution, 2)
    # jax.debug.print("log_number_density = {out}", out=log_number_density)
    # jax.debug.print("log_stability = {out}", out=log_stability)

    log_activity: Float[Array, " species"] = get_log_activity(parameters, log_number_density)
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # Atmosphere
    total_pressure: Float[Array, ""] = get_total_pressure(parameters, log_number_density)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)
    log_volume: Float[Array, ""] = get_atmosphere_log_volume(parameters, log_number_density)
    # jax.debug.print("log_volume = {out}", out=log_volume)

    # Based on the definition of the reaction constant we need to convert gas activities
    # (fugacities) from bar to effective number density, whilst keeping condensate activities
    # unmodified.
    log_activity_number_density: Float[Array, " species"] = (
        get_log_number_density_from_log_pressure(log_activity, temperature)
    )
    log_activity_number_density = jnp.where(
        parameters.species.gas_species_mask, log_activity_number_density, log_activity
    )
    # jax.debug.print("log_activity_number_density = {out}", out=log_activity_number_density)

    # Fugacity constraints residual (dimensionless, log-ratio of number densities)
    fugacity_residual: Float[Array, " reactions"] = (
        log_activity_number_density
        - parameters.fugacity_constraints.log_number_density(temperature, total_pressure)
    )
    # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
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

    # Reaction network residual
    reaction_matrix: Float[Array, "reactions species"] = jnp.asarray(
        parameters.species.reaction_matrix
    )

    log_reaction_equilibrium_constant: Float[Array, " reactions"] = (
        get_log_reaction_equilibrium_constant(parameters)
    )
    # jax.debug.print(
    #     "log_reaction_equilibrium_constant = {out}", out=log_reaction_equilibrium_constant.shape
    # )
    reaction_residual: Float[Array, " reactions"] = (
        reaction_matrix.dot(log_activity_number_density) - log_reaction_equilibrium_constant
    )
    # jax.debug.print("reaction_residual before stability = {out}", out=reaction_residual.shape)
    reaction_stability_mask: Bool[Array, "reactions species"] = jnp.broadcast_to(
        parameters.species.active_stability, reaction_matrix.shape
    )
    reaction_stability_matrix: Float[Array, "reactions species"] = (
        reaction_matrix * reaction_stability_mask
    )
    # jax.debug.print("reaction_stability_matrix = {out}", out=reaction_stability_matrix.shape)

    # Dimensionless (log K residual)
    reaction_residual = reaction_residual - reaction_stability_matrix.dot(safe_exp(log_stability))
    # jax.debug.print("reaction_residual after stability = {out}", out=reaction_residual.shape)
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
    # Number density of elements in the gas or condensed phase
    element_density: Float[Array, " elements"] = get_element_density(
        parameters, log_number_density
    )
    # jax.debug.print("element_density = {out}", out=element_density)
    element_melt_density: Float[Array, " elements"] = get_element_density_in_melt(
        parameters, log_number_density
    )
    # jax.debug.print("element_melt_density = {out}", out=element_melt_density)

    # Relative mass error, computed in log-space for numerical stability
    element_density_total: Float[Array, " elements"] = element_density + element_melt_density
    log_element_density_total: Float[Array, " elements"] = jnp.log(element_density_total)
    # jax.debug.print("log_element_density_total = {out}", out=log_element_density_total)
    log_target_density: Float[Array, " elements"] = parameters.mass_constraints.log_number_density(
        log_volume
    )
    # jax.debug.print("log_target_density = {out}", out=log_target_density)

    # Dimensionless (ratio error - 1)
    mass_residual: Float[Array, " elements"] = (
        safe_exp(log_element_density_total - log_target_density) - 1
    )
    # Log-space residual can perform better when close to the solution
    # mass_residual = log_element_density_total - log_target_density
    # jax.debug.print("mass_residual = {out}", out=mass_residual)
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
    log_min_number_density: Float[Array, " species"] = (
        get_min_log_elemental_abundance_per_species(parameters)
        - log_volume
        + jnp.log(parameters.solver_parameters.tau)
    )
    # jax.debug.print("log_min_number_density = {out}", out=log_min_number_density)
    # Dimensionless (log-ratio)
    stability_residual: Float[Array, " species"] = (
        log_number_density + log_stability - log_min_number_density
    )
    # jax.debug.print("stability_residual = {out}", out=stability_residual)
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
    residual: Float[Array, " residual"] = jnp.concatenate(
        [fugacity_residual, reaction_residual, mass_residual, stability_residual]
    )
    # jax.debug.print("residual (with nans) = {out}", out=residual)

    active_mask: Bool[Array, " dim"] = get_active_mask(parameters)
    # jax.debug.print("active_mask = {out}", out=active_mask)
    size: int = parameters.species.number_solution
    # jax.debug.print("size = {out}", out=size)

    active_indices: Integer[Array, "..."] = jnp.where(active_mask, size=size)[0]
    # jax.debug.print("active_indices = {out}", out=active_indices)

    residual = jnp.take(
        residual, indices=active_indices, unique_indices=True, indices_are_sorted=True
    )
    # jax.debug.print("residual = {out}", out=residual)

    return residual
