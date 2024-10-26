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
"""JAX-related functionality for solving the system of equations"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, jit, lax
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.utilities import logsumexp, unit_conversion

if TYPE_CHECKING:
    from atmodeller.containers import (
        FixedParameters,
        FugacityConstraints,
        MassConstraints,
        Planet,
        Solution,
        SolverParameters,
        Species,
        TracedParameters,
    )


@partial(jit, static_argnames=["fixed_parameters", "solver_parameters"])
def solve(
    solution: Solution,
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    solver_parameters: SolverParameters,
) -> Array:
    """Solves the system of non-linear equations

    Args:
        solution: Solution
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        solver_parameters: Solver parameters

    Returns:
        The solution
    """

    options: dict[str, ArrayLike] = {
        "lower": np.asarray(solver_parameters.lower),
        "upper": np.asarray(solver_parameters.upper),
    }

    sol = optx.root_find(
        objective_function,
        solver_parameters.solver,
        solution.data,
        args={"traced_parameters": traced_parameters, "fixed_parameters": fixed_parameters},
        throw=solver_parameters.throw,
        max_steps=solver_parameters.max_steps,
        options=options,
    )

    jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

    return sol.value


@jit
def objective_function(solution: Array, kwargs: dict) -> Array:
    """Residual of the reaction network and mass balance

    Args:
        kwargs: Dictionary of pytrees required to compute the residual

    Returns:
        Residual
    """
    traced_parameters: TracedParameters = kwargs["traced_parameters"]
    fixed_parameters: FixedParameters = kwargs["fixed_parameters"]
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.surface_temperature
    fugacity_constraints: FugacityConstraints = traced_parameters.fugacity_constraints
    mass_constraints: MassConstraints = traced_parameters.mass_constraints
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)

    formula_matrix: Array = jnp.array(fixed_parameters.formula_matrix)
    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    fugacity_matrix: Array = jnp.array(fixed_parameters.fugacity_matrix)

    # Species
    log_number_density, log_stability = jnp.split(solution, 2)
    log_activity: Array = get_log_activity(traced_parameters, fixed_parameters, log_number_density)

    # Based on the definition of the reaction constant we need to convert gas activities
    # (fugacities) from bar to effective number density.
    mask: Array = jnp.zeros_like(log_activity, dtype=bool)
    mask = mask.at[gas_species_indices].set(True)
    log_activity_number_density: Array = get_log_number_density_from_log_pressure(
        log_activity, temperature
    )
    log_activity_number_density = jnp.where(mask, log_activity_number_density, log_activity)

    # Bulk atmosphere
    log_volume: Array = get_atmosphere_log_volume(fixed_parameters, log_number_density, planet)

    residual: Array = jnp.array([])

    # Reaction network residual
    if reaction_matrix.size > 0:
        log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
            fixed_parameters, temperature
        )
        reaction_residual: Array = (
            reaction_matrix.dot(log_activity_number_density) - log_reaction_equilibrium_constant
        )
        # jax.debug.print("reaction_residual = {out}", out=reaction_residual)
        # Account for species stability.
        reaction_residual = reaction_residual - reaction_matrix.dot(jnp.exp(log_stability))
        # jax.debug.print("reaction_residual with stability = {out}", out=reaction_residual)
        residual = jnp.concatenate([residual, reaction_residual])

    # Fugacity constraints residual
    if fugacity_matrix.size > 0:
        fugacity_species_indices: Array = jnp.array(fixed_parameters.fugacity_species_indices)
        fugacity_log_activity_number_density: Array = jnp.take(
            log_activity_number_density, fugacity_species_indices
        )
        # jax.debug.print("fugacity_log_activity = {out}", out=fugacity_log_activity)
        fugacity_residual: Array = fugacity_matrix.dot(fugacity_log_activity_number_density)
        # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
        total_pressure: Array = get_atmosphere_pressure(
            fixed_parameters, log_number_density, temperature
        )
        fugacity_residual = fugacity_residual - fugacity_constraints.array(
            temperature, total_pressure
        )
        # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
        residual = jnp.concatenate([residual, fugacity_residual])

    # Elemental mass balance residual
    if formula_matrix.size > 0:
        # Number density of elements in the gas or condensed phase
        element_density: Array = get_element_density(fixed_parameters, log_number_density)
        element_melt_density: Array = get_element_density_in_melt(
            traced_parameters, fixed_parameters, log_number_density, log_activity, log_volume
        )
        log_element_density: Array = jnp.log(element_density + element_melt_density)
        mass_residual = log_element_density - mass_constraints.array(log_volume)
        # jax.debug.print("mass_residual = {out}", out=mass_residual)
        # Stability residual
        # Get minimum scaled log number of molecules
        log_min_number_density: Array = jnp.min(mass_constraints.array(log_volume)) - jnp.log(
            fixed_parameters.tau
        )
        stability_residual: Array = log_number_density + log_stability - log_min_number_density
        # jax.debug.print("stability_residual = {out}", out=stability_residual)
        residual = jnp.concatenate([residual, mass_residual, stability_residual])

    # jax.debug.print("residual = {out}", out=residual)

    return residual


@jit
def get_atmosphere_log_molar_mass(
    fixed_parameters: FixedParameters, log_number_density: Array
) -> Array:
    """Gets log molar mass of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log molar mass of the atmosphere
    """
    gas_log_number_density: Array = get_gas_species_data(fixed_parameters, log_number_density)
    gas_molar_mass: Array = get_gas_species_data(
        fixed_parameters, jnp.array(fixed_parameters.molar_masses)
    )
    molar_mass: Array = logsumexp(gas_log_number_density, gas_molar_mass) - logsumexp(
        gas_log_number_density
    )

    return molar_mass


@jit
def get_atmosphere_log_volume(
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    planet: Planet,
) -> Array:
    """Gets log volume of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """

    log_volume: Array = (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.surface_temperature)
        - get_atmosphere_log_molar_mass(fixed_parameters, log_number_density)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )

    return log_volume


@jit
def get_atmosphere_pressure(
    fixed_parameters: FixedParameters, log_number_density: Array, temperature: ArrayLike
) -> Array:
    """Gets pressure of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Pressure of the atmosphere
    """
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    pressure: Array = get_pressure_from_log_number_density(log_number_density, temperature)
    gas_pressure: Array = jnp.take(pressure, gas_species_indices)

    return jnp.sum(gas_pressure)


@jit
def get_element_density(fixed_parameters: FixedParameters, log_number_density: Array) -> Array:
    """Number density of elements in the gas or condensed phase

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Number density of elements in the gas or condensed phase
    """
    formula_matrix: Array = jnp.array(fixed_parameters.formula_matrix)
    element_gas_density: Array = formula_matrix.dot(jnp.exp(log_number_density))

    return element_gas_density


@jit
def get_element_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    log_activity: Array,
    log_volume: Array,
) -> Array:
    """Gets the number density of elements dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity: Log activity
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of elements dissolved in melt
    """
    formula_matrix: Array = jnp.array(fixed_parameters.formula_matrix)
    species_melt_density: Array = get_species_density_in_melt(
        traced_parameters, fixed_parameters, log_number_density, log_activity, log_volume
    )
    element_melt_density: Array = formula_matrix.dot(species_melt_density)

    return element_melt_density


@jit
def get_gas_species_data(fixed_parameters: FixedParameters, some_array: ArrayLike) -> Array:
    """Gets the gas species data from an array

    Args:
        fixed_parameters: Fixed parameters
        some_array: Some array to extract gas species data from

    Returns:
        An array with just the gas species data from `some_array`
    """
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    gas_data: Array = jnp.take(some_array, gas_species_indices)

    return gas_data


@jit
def get_log_activity(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: ArrayLike,
) -> Array:
    """Gets the log activity

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log activity
    """
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.surface_temperature
    species: tuple[Species, ...] = fixed_parameters.species

    pressure: Array = get_pressure_from_log_number_density(log_number_density, temperature)
    activity_funcs: list[Callable] = [species_.activity.log_activity for species_ in species]

    def apply_activity_function(index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike):
        pressure_: Array = jnp.take(pressure, index)

        return lax.switch(
            index,
            activity_funcs,
            temperature,
            pressure_,
        )

    vmap_apply_function: Callable = jax.vmap(apply_activity_function, in_axes=(0, None, None))
    indices: Array = jnp.arange(len(species))
    log_activity: Array = vmap_apply_function(indices, temperature, pressure)

    return log_activity


@jit
def get_log_number_density_from_log_pressure(
    log_pressure: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets log number density from log pressure

    Args:
        log_pressure: Log pressure
        temperature: Temperature

    Returns:
        Log number density
    """
    log_number_density: Array = (
        -jnp.log(BOLTZMANN_CONSTANT_BAR) - jnp.log(temperature) + log_pressure
    )

    return log_number_density


@jit
def get_log_pressure_from_log_number_density(
    log_number_density: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets log pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Log pressure
    """
    log_pressure: Array = (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature) + log_number_density
    )

    return log_pressure


# pylint: disable=invalid-name
@jit
def get_log_Kp(
    species: tuple[Species, ...], reaction_matrix: Array, temperature: ArrayLike
) -> Array:
    """Gets log of the equilibrium constant in terms of partial pressures

    Args:
        species: Species
        reaction_matrix: Reaction matrix
        temperature: Temperature

    Returns:
        Log of the equilibrium constant in terms of partial pressures
    """
    gibbs_list: list[ArrayLike] = []
    for species_ in species:
        gibbs: ArrayLike = species_.data.get_gibbs_over_RT(temperature)
        gibbs_list.append(gibbs)

    gibbs_jnp: Array = jnp.array(gibbs_list)
    log_Kp: Array = -1.0 * reaction_matrix.dot(gibbs_jnp)

    return log_Kp


# pylint: enable=invalid-name


@jit
def get_log_reaction_equilibrium_constant(
    fixed_parameters: FixedParameters,
    temperature: ArrayLike,
) -> Array:
    """Gets the log equilibrium constant of the reactions

    Args:
        fixed_parameters: Fixed parameters
        temperature: Temperature

    Returns:
        Log equilibrium constant of the reactions
    """
    species: tuple[Species, ...] = fixed_parameters.species
    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)

    # pylint: disable=invalid-name
    log_Kp: Array = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)
    delta_n: Array = jnp.sum(jnp.take(reaction_matrix, gas_species_indices, axis=1), axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)
    log_Kc: Array = log_Kp - delta_n * (jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature))
    # jax.debug.print("log10Kc = {out}", out=log_Kc)
    # pylint: enable=invalid-name

    return log_Kc


@jit
def get_pressure_from_log_number_density(
    log_number_density: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Pressure
    """
    return jnp.exp(get_log_pressure_from_log_number_density(log_number_density, temperature))


@jit
def get_species_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    log_activity: Array,
    log_volume: Array,
) -> Array:
    """Gets the number density of species dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity: Log activity
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of species dissolved in melt
    """
    species: tuple[Species, ...] = fixed_parameters.species
    diatomic_oxygen_index: Array = jnp.array(fixed_parameters.diatomic_oxygen_index)
    molar_masses: Array = jnp.array(fixed_parameters.molar_masses)
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.surface_temperature

    fugacity: Array = jnp.exp(log_activity)
    total_pressure: Array = get_atmosphere_pressure(
        fixed_parameters, log_number_density, temperature
    )
    diatomic_oxygen_fugacity: Array = jnp.take(fugacity, diatomic_oxygen_index)

    solubility_funcs: list[Callable] = [
        species_.solubility.jax_concentration for species_ in species
    ]

    def apply_solubility_function(index: ArrayLike, fugacity: ArrayLike):
        return lax.switch(
            index,
            solubility_funcs,
            fugacity,
            temperature,
            total_pressure,
            diatomic_oxygen_fugacity,
        )

    vmap_apply_function: Callable = jax.vmap(apply_solubility_function, in_axes=(0, 0))
    indices: ArrayLike = jnp.arange(len(species))
    ppmw: Array = vmap_apply_function(indices, fugacity)
    # jax.debug.print("ppmw = {out}", out=ppmw)

    species_melt_density: Array = (
        ppmw
        * unit_conversion.ppm_to_fraction
        * AVOGADRO
        * planet.mantle_melt_mass
        / (molar_masses * jnp.exp(log_volume))
    )
    # jax.debug.print("species_melt_density = {out}", out=species_melt_density)

    return species_melt_density
