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
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from jaxmod.utils import safe_exp, to_hashable
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, Shaped

from atmodeller.containers import Parameters, SpeciesCollection
from atmodeller.type_aliases import NpBool


def get_active_mask(parameters: Parameters) -> Bool[Array, " dim"]:
    """Gets the mask of active residual quantities.

    Args:
        parameters: Parameters

    Returns:
        Active mask
    """
    fugacity_mask: Bool[Array, " dim"] = parameters.fugacity_constraints.active()
    solubility_mask: Bool[Array, " dim"] = jnp.ones(
        len(parameters.species_collection.data.reservoir_species), dtype=bool
    )
    reactions_mask: ArrayLike = parameters.reaction_network.active_reactions
    mass_mask: Bool[Array, " dim"] = parameters.mass_constraints.active()
    stability_mask: ArrayLike = parameters.reaction_network.data.active_stability

    # jax.debug.print("fugacity_mask = {out}", out=fugacity_mask)
    # jax.debug.print("reactions_mask = {out}", out=reactions_mask)
    # jax.debug.print("mass_mask = {out}", out=mass_mask)
    # jax.debug.print("stability_mask = {out}", out=stability_mask)

    active_mask: Bool[Array, " dim"] = jnp.concatenate(
        (fugacity_mask, solubility_mask, reactions_mask, mass_mask, stability_mask)
    )
    jax.debug.print("active_mask = {out}", out=active_mask)

    return active_mask


def get_atmosphere_log_molar_mass(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, ""]:
    """Gets the log molar mass of the atmosphere.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Log molar mass of the atmosphere
    """
    gas_log_number_moles: Float[Array, " species"] = get_gas_species_data(
        parameters, log_number_moles
    )
    gas_molar_mass: Float[Array, " species"] = get_gas_species_data(
        parameters, parameters.species_collection.data.molar_masses
    )
    molar_mass: Float[Array, ""] = logsumexp(gas_log_number_moles, b=gas_molar_mass) - logsumexp(
        gas_log_number_moles, b=parameters.species_collection.data.gas_species_mask
    )
    # jax.debug.print("molar_mass = {out}", out=molar_mass)

    return molar_mass


def get_element_moles(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, " elements"]:
    """Gets the number of moles of elements in the gas or condensed phase.

    Input values are sanitised only for output routines, where partitioning between condensed and
    gas species is required. For the solver itself, this distinction is unnecessary.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Number of moles of elements in the gas or condensed phase
    """
    species_moles: Array = jnp.nan_to_num(safe_exp(log_number_moles), nan=0.0)

    formula_matrix: Integer[Array, "elements species"] = jnp.asarray(
        parameters.species_collection.data.formula_matrix
    )
    element_moles: Float[Array, " elements"] = formula_matrix @ species_moles

    return element_moles


# def get_element_moles_in_melt(
#     parameters: Parameters, log_number_moles: Float[Array, " species"]
# ) -> Float[Array, " species"]:
#     """Gets the number of moles of elements dissolved in melt.

#     Args:
#         parameters: Parameters
#         log_number_moles: Log number of moles

#     Returns:
#         Number of moles of elements dissolved in melt
#     """
#     species_melt_moles: Float[Array, " species"] = get_species_moles_in_melt(
#         parameters, log_number_moles
#     )
#     formula_matrix: Integer[Array, "elements species"] = jnp.asarray(
#         parameters.reaction_network.formula_matrix
#     )
#     element_melt_moles: Float[Array, " species"] = formula_matrix @ species_melt_moles

#     return element_melt_moles


def get_gas_species_data(
    parameters: Parameters, some_array: ArrayLike
) -> Shaped[Array, " species"]:
    """Masks the gas species data from an array.

    Args:
        parameters: Parameters
        some_array: Some array to mask the gas species data from

    Returns:
        An array with gas species data from `some_array` and other entries zeroed
    """
    gas_data: Shaped[Array, " species"] = (
        jnp.asarray(some_array) * parameters.species_collection.data.gas_species_mask
    )

    return gas_data


def get_log_mole_fraction_in_gas(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the log mole fraction of the species in the gas phase

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Log mole fraction in the gas
    """
    gas_species_mask: Bool[Array, " species"] = jnp.array(
        parameters.species_collection.data.gas_species_mask
    )

    # Represent mask in log space: True -> 0, False -> -inf
    log_mask: Float[Array, " species"] = jnp.where(gas_species_mask, 0.0, -jnp.inf)

    # Masked log number of moles
    log_gas_number_moles: Float[Array, " species"] = log_number_moles + log_mask

    # Log total (sum in linear space)
    total_log_number_moles: Float[Array, ""] = logsumexp(log_gas_number_moles)

    # Log mole fraction = log(n_i) − log(total)
    log_mole_fraction: Float[Array, " species"] = log_gas_number_moles - total_log_number_moles
    jax.debug.print("log_mole_fraction_in_gas = {out}", out=log_mole_fraction)

    return log_mole_fraction


def get_log_mass_fraction_in_melt(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the log mass fraction of the species in the melt phase

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Log mass fraction in the melt
    """
    melt_species_mask: Bool[Array, " species"] = jnp.array(
        parameters.species_collection.data.reservoir_species_mask
    )
    jax.debug.print("melt_species_mask = {out}", out=melt_species_mask)

    # Represent mask in log space: True -> 0, False -> -inf
    log_mask: Float[Array, " species"] = jnp.where(melt_species_mask, 0.0, -jnp.inf)
    jax.debug.print("log_mask = {out}", out=log_mask)

    log_melt_mass: Float[Array, " species"] = (
        log_number_moles + jnp.log(parameters.species_collection.data.molar_masses) + log_mask
    )
    jax.debug.print("log_melt_mass = {out}", out=log_melt_mass)

    log_background_melt_mass: Array = jnp.log(parameters.state.melt_mass)

    if parameters.dilute_limit:
        total_log_mass: Float[Array, ""] = log_background_melt_mass
    else:
        # Must account for a background (silicate) melt mass, given by the thermodynamic state
        log_mass_melt_plus: Float[Array, " species_plus_one"] = jnp.append(
            log_melt_mass, log_background_melt_mass
        )
        jax.debug.print("log_mass_melt_plus = {out}", out=log_mass_melt_plus)

        # Log total (sum in linear space)
        total_log_mass = logsumexp(log_mass_melt_plus)
        jax.debug.print("total_log_mass = {out}", out=total_log_mass)

    # Log mass fraction = log(m_i) − log(total)
    log_mass_fraction: Float[Array, " species"] = log_melt_mass - total_log_mass
    jax.debug.print("log_mass_fraction = {out}", out=log_mass_fraction)

    # Finally, convert to ppmw
    log_mass_ppmw: Float[Array, " species"] = log_mass_fraction + jnp.log(1e6)
    jax.debug.print("log_mass_ppmw = {out}", out=log_mass_ppmw)

    return log_mass_ppmw


def get_log_activity(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the log activity.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Log activity
    """
    gas_species_mask: Bool[Array, " species"] = jnp.array(
        parameters.species_collection.data.gas_species_mask
    )

    melt_species_mask: Bool[Array, " species"] = jnp.array(
        parameters.species_collection.data.reservoir_species_mask
    )

    log_mole_fraction_in_gas: Float[Array, " species"] = get_log_mole_fraction_in_gas(
        parameters, log_number_moles
    )

    log_mass_fraction_in_melt: Float[Array, " species"] = get_log_mass_fraction_in_melt(
        parameters, log_number_moles
    )

    log_activity_pure_species: Float[Array, " species"] = get_log_activity_pure_species(
        parameters, log_number_moles
    )
    # jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)
    log_activity_gas_species: Float[Array, " species"] = (
        log_activity_pure_species + log_mole_fraction_in_gas
    )
    # jax.debug.print("log_activity_gas_species = {out}", out=log_activity_gas_species)
    log_activity_melt_species: Float[Array, " species"] = (
        log_activity_pure_species + log_mass_fraction_in_melt
    )
    # jax.debug.print("log_activity_melt_species = {out}", out=log_activity_melt_species)

    log_activity: Float[Array, " species"] = jnp.where(
        gas_species_mask, log_activity_gas_species, log_activity_pure_species
    )

    log_activity = jnp.where(melt_species_mask, log_activity_melt_species, log_activity)

    return log_activity


def get_log_activity_pure_species(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the log activity of pure species.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Log activity of pure species
    """
    temperature: Float[Array, ""] = parameters.state.temperature
    species_collection: SpeciesCollection = parameters.species_collection
    total_pressure: Float[Array, ""] = get_total_pressure(parameters, log_number_moles)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)

    activity_funcs: list[Callable] = [
        to_hashable(species_.activity.log_activity) for species_ in species_collection.data.species
    ]

    def apply_activity(index: ArrayLike) -> Float[Array, ""]:
        return lax.switch(index, activity_funcs, temperature, total_pressure)

    indices: Integer[Array, " species"] = jnp.arange(species_collection.data.number_species)
    vmap_activity: Callable = eqx.filter_vmap(apply_activity, in_axes=(0,))
    log_activity_pure_species: Float[Array, " species"] = vmap_activity(indices)
    jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)

    return log_activity_pure_species


def get_log_Kp(parameters: Parameters) -> Float[Array, " reactions"]:
    """Gets log of the equilibrium constant of each reaction in terms of partial pressures.

    Args:
        parameters: Parameters

    Returns:
        Log of the equilibrium constant of each reaction in terms of partial pressures
    """
    gibbs_funcs: list[Callable] = [
        to_hashable(species_.get_gibbs_over_RT)
        for species_ in parameters.reaction_network.data.species
    ]

    def apply_gibbs(
        index: Integer[Array, ""], temperature: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        return lax.switch(index, gibbs_funcs, temperature)

    indices: Integer[Array, " species"] = jnp.arange(
        parameters.reaction_network.data.number_species
    )
    vmap_gibbs: Callable = eqx.filter_vmap(apply_gibbs, in_axes=(0, None))
    gibbs_values: Float[Array, "species 1"] = vmap_gibbs(indices, parameters.state.temperature)
    # jax.debug.print("gibbs_values = {out}", out=gibbs_values)
    reaction_matrix: Float[Array, "reactions species"] = jnp.asarray(
        parameters.reaction_network.reaction_matrix
    )
    log_Kp: Float[Array, "reactions 1"] = -1.0 * reaction_matrix @ gibbs_values

    return jnp.ravel(log_Kp)


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
        parameters.species_collection.data.formula_matrix
    )
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    mask: Integer[Array, "elements species"] = (formula_matrix != 0).astype(jnp.int_)
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    # log_abundance is a 1-D array, which cannot be transposed, so make a 2-D array
    log_abundance: Float[Array, "elements 1"] = jnp.atleast_2d(
        parameters.mass_constraints.log_abundance()
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


# TODO: reinstate, but only used by output routines
def get_reactions_only_mask(parameters: Parameters) -> Bool[Array, " dim"]:
    """Returns a mask with `True` only for active reactions positions, `False` elsewhere.

    Args:
        parameters: Parameters

    Returns:
        Reactions only mask for the residual array
    """
    # Create a full mask of False
    size: int = parameters.reaction_network.data.number_solution
    mask: Bool[Array, " dim"] = jnp.zeros(size, dtype=bool)

    fugacity_mask: Bool[Array, " dim"] = parameters.fugacity_constraints.active()
    reactions_mask: NpBool = parameters.reaction_network.active_reactions
    num_active_fugacity: Integer[Array, ""] = jnp.sum(fugacity_mask)

    # Place the reactions_mask at position num_active_fugacity dynamically.
    # Use lax.dynamic_update_slice: (array_to_update, update, start_indices)
    mask: Bool[Array, " dim"] = lax.dynamic_update_slice(
        mask, reactions_mask, (num_active_fugacity,)
    )

    return mask


# TODO: Make this a residual instead
# def get_species_moles_in_melt(
#     parameters: Parameters, log_number_moles: Float[Array, " species"]
# ) -> Float[Array, " species"]:
#     """Gets the number of moles of species dissolved in melt due to species solubility.

#     Args:
#         parameters: Parameters
#         log_number_moles: Log number of moles

#     Returns:
#         Number of moles of species dissolved in melt
#     """
#     molar_masses: ArrayLike = parameters.reaction_network.molar_masses
#     melt_mass: Float[Array, ""] = parameters.state.melt_mass

#     ppmw: Float[Array, " species"] = get_species_ppmw_in_melt(parameters, log_number_moles)

#     species_melt_moles: Float[Array, " species"] = (
#         ppmw * unit_conversion.ppm_to_fraction * melt_mass / molar_masses
#     )
#     # jax.debug.print("species_melt_moles = {out}", out=species_melt_moles)

#     return species_melt_moles


def get_species_ppmw_in_melt(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, " species"]:
    """Gets the ppmw of species dissolved in melt.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        ppmw of species dissolved in melt
    """
    species_collection: SpeciesCollection = parameters.species_collection

    # Return empty array if no reservoir species
    if len(species_collection.data.reservoir_species) == 0:
        return jnp.array([], dtype=jnp.float32)

    # Could be an integer (but represented as a float) or np.nan
    diatomic_oxygen_index: Float[Array, ""] = jnp.array(
        species_collection.data.diatomic_oxygen_index
    )
    temperature: Float[Array, ""] = parameters.state.temperature
    total_pressure: Float[Array, ""] = get_total_pressure(parameters, log_number_moles)

    log_activity: Float[Array, " species"] = get_log_activity(parameters, log_number_moles)
    fugacity: Float[Array, " species"] = safe_exp(log_activity)

    # Need just the fugacity of the gas species associated with the (dissolved) reservoir species.
    fugacity_reservoir_species: Float[Array, " reservoir_species"] = jnp.take(
        fugacity,
        indices=species_collection.data.reservoir_species_map,
        unique_indices=True,
        indices_are_sorted=True,
    )

    # For type consistency, convert to integer array with nan as 0
    diatomic_oxygen_index_: Integer[Array, ""] = jnp.nan_to_num(
        diatomic_oxygen_index, nan=0
    ).astype(jnp.int_)

    # Get fO2, or nan if not present
    diatomic_oxygen_fugacity: Float[Array, ""] = jnp.where(
        jnp.isnan(diatomic_oxygen_index), jnp.nan, jnp.take(fugacity, diatomic_oxygen_index_)
    )

    # NOTE: All solubility formulations must return a JAX array to allow vmap
    solubility_funcs: list[Callable] = [
        to_hashable(species_.solubility.jax_concentration)
        for species_ in species_collection.data.reservoir_species
    ]

    def apply_solubility(
        index: Integer[Array, ""],
        fugacity_val: Float[Array, ""],
        temp: Float[Array, ""],
        press: Float[Array, ""],
        o2_fug: Float[Array, ""],
    ) -> Float[Array, ""]:
        return lax.switch(index, solubility_funcs, fugacity_val, temp, press, o2_fug)

    indices: Integer[Array, " reservoir_species"] = jnp.arange(
        len(species_collection.data.reservoir_species)
    )

    vmap_solubility: Callable = eqx.filter_vmap(apply_solubility, in_axes=(0, 0, None, None, None))
    species_ppmw: Float[Array, " reservoir_species"] = vmap_solubility(
        indices, fugacity_reservoir_species, temperature, total_pressure, diatomic_oxygen_fugacity
    )
    # jax.debug.print("ppmw = {out}", out=ppmw)

    return species_ppmw


def get_gas_mass(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, ""]:
    """Gets the gas mass.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Gas mass
    """
    gas_molar_mass: Float[Array, " species"] = get_gas_species_data(
        parameters, parameters.species_collection.data.molar_masses
    )
    log_gas_mass: Float[Array, ""] = logsumexp(log_number_moles + jnp.log(gas_molar_mass))
    gas_mass: Float[Array, ""] = jnp.exp(log_gas_mass)

    return gas_mass


def get_total_pressure(
    parameters: Parameters, log_number_moles: Float[Array, " species"]
) -> Float[Array, ""]:
    """Gets the total pressure.

    Args:
        parameters: Parameters
        log_number_moles: Log number of moles

    Returns:
        Total pressure in bar
    """
    gas_mass: Float[Array, ""] = get_gas_mass(parameters, log_number_moles)
    pressure: Float[Array, ""] = parameters.state.get_pressure(gas_mass)

    return pressure


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
        solution: Solution array for all species i.e. log number of moles and log stability
        parameters: Parameters

    Returns:
        Residual
    """
    # jax.debug.print("Starting new objective_function evaluation")
    temperature: Float[Array, ""] = parameters.state.temperature

    log_number_moles, log_stability = jnp.split(solution, 2)
    jax.debug.print("log_number_moles = {out}", out=log_number_moles)
    jax.debug.print("log_stability = {out}", out=log_stability)

    total_pressure: Float[Array, ""] = get_total_pressure(parameters, log_number_moles)
    jax.debug.print("total_pressure = {out}", out=total_pressure)

    log_activity: Float[Array, " species"] = get_log_activity(parameters, log_number_moles)
    jax.debug.print("log_activity = {out}", out=log_activity)

    # Fugacity constraints residual (dimensionless)
    fugacity_residual: Float[Array, " reactions"] = (
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

    # TODO: Solubility residual (dimensionless)

    species_ppmw = get_species_ppmw_in_melt(parameters, log_number_moles)
    jax.debug.print("species_ppmw = {out}", out=species_ppmw)

    # Need to get just the log_activity of reservoir species
    log_activity_reservoir_indices: Integer[Array, "..."] = jnp.where(
        parameters.species_collection.data.reservoir_species_mask,
        size=len(parameters.species_collection.data.reservoir_species),
    )[0]
    jax.debug.print("log_activity_reservoir_indices = {out}", out=log_activity_reservoir_indices)
    log_activity_reservoir: Float[Array, " reservoir_species"] = jnp.take(
        log_activity,
        indices=log_activity_reservoir_indices,
        unique_indices=True,
        indices_are_sorted=True,
    )
    jax.debug.print("log_activity_reservoir = {out}", out=log_activity_reservoir)

    solubility_residual: Float[Array, " reservoir_species"] = log_activity_reservoir - jnp.log(
        species_ppmw
    )
    jax.debug.print("solubility_residual = {out}", out=solubility_residual)

    # Reaction network residual
    reaction_matrix: Float[Array, "reactions species"] = jnp.asarray(
        parameters.reaction_network.reaction_matrix
    )

    log_reaction_equilibrium_constant: Float[Array, " reactions"] = get_log_Kp(parameters)
    # jax.debug.print(
    #    "log_reaction_equilibrium_constant = {out}", out=log_reaction_equilibrium_constant.shape
    # )

    # Need to get just the log_activity of species involved in reactions
    log_activity_reaction_indices: Integer[Array, "..."] = jnp.where(
        parameters.species_collection.data.reaction_species_mask,
        size=parameters.species_collection.reaction_network.data.number_species,
    )[0]
    jax.debug.print("log_activity_reaction_indices = {out}", out=log_activity_reaction_indices)
    log_activity_reaction: Float[Array, " reaction_species"] = jnp.take(
        log_activity,
        indices=log_activity_reaction_indices,
        unique_indices=True,
        indices_are_sorted=True,
    )
    jax.debug.print("log_activity_reaction = {out}", out=log_activity_reaction)

    reaction_residual: Float[Array, " reactions"] = (
        reaction_matrix.dot(log_activity_reaction) - log_reaction_equilibrium_constant
    )
    jax.debug.print("reaction_residual before stability = {out}", out=reaction_residual)
    reaction_stability_mask: Bool[Array, "reactions species"] = jnp.broadcast_to(
        parameters.reaction_network.data.active_stability, reaction_matrix.shape
    )
    reaction_stability_matrix: Float[Array, "reactions species"] = (
        reaction_matrix * reaction_stability_mask
    )
    jax.debug.print("reaction_stability_matrix = {out}", out=reaction_stability_matrix.shape)

    # Need to get just the log_stability of species involved in reactions
    log_stability_reaction: Float[Array, " reaction_species"] = jnp.take(
        log_stability,
        indices=log_activity_reaction_indices,
        unique_indices=True,
        indices_are_sorted=True,
    )
    jax.debug.print("log_stability_reaction = {out}", out=log_stability_reaction)

    # Dimensionless (log K residual)
    reaction_residual = reaction_residual - reaction_stability_matrix.dot(
        safe_exp(log_stability_reaction)
    )
    jax.debug.print("reaction_residual after stability = {out}", out=reaction_residual)
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

    # TODO: add solubility residual here
    # solubility_residual = log_activity[idx_N2_melt] - jnp.log(S(fugacity_N2, melt_state))

    # Elemental mass balance residual
    # Number of moles of elements in the gas or condensed phase
    # TODO: Needs to include those in dissolved as well
    element_moles: Float[Array, " elements"] = get_element_moles(parameters, log_number_moles)
    jax.debug.print("element_moles = {out}", out=element_moles)

    # TODO: Make a solubility residual instead
    # element_melt_moles: Float[Array, " elements"] = get_element_moles_in_melt(
    #     parameters, log_number_moles
    # )
    # jax.debug.print("element_melt_moles = {out}", out=element_melt_moles)

    # Relative mass error, computed in log-space for numerical stability
    element_moles_total: Float[Array, " elements"] = element_moles  # + element_melt_moles
    log_element_moles_total: Float[Array, " elements"] = jnp.log(element_moles_total)
    # jax.debug.print("log_element_moles_total = {out}", out=log_element_moles_total)

    log_target_moles: Float[Array, " elements"] = parameters.mass_constraints.log_abundance()
    # jax.debug.print("log_target_moles = {out}", out=log_target_moles)

    # Dimensionless (ratio error - 1)
    mass_residual: Float[Array, " elements"] = (
        safe_exp(log_element_moles_total - log_target_moles) - 1
    )
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
    log_min_number_moles: Float[Array, " species"] = get_min_log_elemental_abundance_per_species(
        parameters
    ) + jnp.log(parameters.solver_parameters.tau)
    # jax.debug.print("log_min_number_moles = {out}", out=log_min_number_moles)
    # Dimensionless (log-ratio)
    stability_residual: Float[Array, " species"] = (
        log_number_moles + log_stability - log_min_number_moles
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
        [
            fugacity_residual,
            solubility_residual,
            reaction_residual,
            mass_residual,
            stability_residual,
        ]
    )
    jax.debug.print("residual (with nans) = {out}", out=residual)

    # This final masking operation drops nans (unused constraint options) as well as dropping
    # meaningless entries associated with imposed condensate activity.

    active_mask: Bool[Array, " dim"] = get_active_mask(parameters)
    jax.debug.print("active_mask = {out}", out=active_mask)
    size: int = parameters.species_collection.data.number_solution
    jax.debug.print("size = {out}", out=size)

    active_indices: Integer[Array, "..."] = jnp.where(active_mask, size=size)[0]
    jax.debug.print("active_indices = {out}", out=active_indices)

    residual = jnp.take(
        residual, indices=active_indices, unique_indices=True, indices_are_sorted=True
    )
    jax.debug.print("residual = {out}", out=residual)

    return residual
