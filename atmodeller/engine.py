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

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, jit, lax
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
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
from atmodeller.utilities import (
    log_number_density_from_log_pressure,
    log_pressure_from_log_number_density,
    logsumexp,
    scale_number_density,
    unit_conversion,
    unscale_number_density,
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
def get_log_reaction_equilibrium_constant(
    species: list[Species],
    gas_species_indices: Array,
    reaction_matrix: Array,
    temperature: ArrayLike,
    log_scaling: ArrayLike,
) -> Array:
    """Gets the log equilibrium constant of the reactions

    Args:
        species: List of species
        gas_species_indices: Indices of gas species
        reaction_matrix: Reaction matrix
        temperature: Temperature
        log_scaling: Log scaling

    Returns:
        Log equilibrium constant of the reactions
    """
    # pylint: disable=invalid-name
    log_Kp: Array = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)

    delta_n: Array = jnp.sum(jnp.take(reaction_matrix, gas_species_indices, axis=1), axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)

    log_Kc: Array = log_Kp - delta_n * (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + log_scaling + jnp.log(temperature)
    )
    # jax.debug.print("log10Kc = {out}", out=log_Kc)

    # pylint: enable=invalid-name

    return log_Kc


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

    species: tuple[Species, ...] = fixed_parameters.species
    formula_matrix: Array = jnp.array(fixed_parameters.formula_matrix)
    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    fugacity_matrix: Array = jnp.array(fixed_parameters.fugacity_matrix)
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    molar_masses: Array = jnp.array(fixed_parameters.molar_masses)
    log_scaling: float = fixed_parameters.log_scaling

    planet: Planet = traced_parameters.planet
    fugacity_constraints: FugacityConstraints = traced_parameters.fugacity_constraints
    mass_constraints: MassConstraints = traced_parameters.mass_constraints
    temperature: ArrayLike = planet.surface_temperature

    log_number_density, log_stability = jnp.split(solution, 2)
    gas_log_number_density: Array = jnp.take(log_number_density, gas_species_indices)
    gas_molar_masses: Array = jnp.take(molar_masses, gas_species_indices)
    log_volume: Array = get_atmosphere_log_volume(gas_log_number_density, gas_molar_masses, planet)

    # Need pressures of all species in bar for subsequent operations. Note we must unscale.
    log_pressures: Array = unscale_number_density(
        log_pressure_from_log_number_density(log_number_density, temperature), log_scaling
    )
    pressures: Array = jnp.exp(log_pressures)
    total_pressure: Array = atmosphere_pressure(fixed_parameters, pressures)

    residual: Array = jnp.array([])

    # Reaction network residual
    if reaction_matrix.size > 0:
        log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
            species, gas_species_indices, reaction_matrix, temperature, log_scaling
        )
        log_activity: Array = get_log_activity_scaled(
            traced_parameters, fixed_parameters, pressures
        )
        reaction_residual: Array = (
            reaction_matrix.dot(log_activity) - log_reaction_equilibrium_constant
        )
        # jax.debug.print("reaction_residual = {out}", out=reaction_residual)
        # Account for species stability.
        reaction_residual = reaction_residual - reaction_matrix.dot(jnp.exp(log_stability))
        # jax.debug.print("reaction_residual with stability = {out}", out=reaction_residual)
        residual = jnp.concatenate([residual, reaction_residual])

    # Fugacity constraints residual
    if fugacity_matrix.size > 0:
        fugacity_species_indices: Array = jnp.array(fixed_parameters.fugacity_species_indices)
        fugacity_log_activity: Array = jnp.take(log_activity, fugacity_species_indices)
        # jax.debug.print("fugacity_log_activity = {out}", out=fugacity_log_activity)
        fugacity_residual: Array = fugacity_matrix.dot(fugacity_log_activity)
        # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
        fugacity_residual = fugacity_residual - fugacity_constraints.array(
            temperature, total_pressure
        )
        # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
        residual = jnp.concatenate([residual, fugacity_residual])

    # Elemental mass balance residual
    if formula_matrix.size > 0:
        # Number density of elements in the condensed or gas phase
        element_density: Array = formula_matrix.dot(jnp.exp(log_number_density))
        element_melt_density: Array = element_density_in_melt(
            traced_parameters, fixed_parameters, log_number_density, pressures, log_volume
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
def get_log_activity(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    pressures: ArrayLike,
) -> Array:
    """Gets the log activity of all species

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        pressures: Pressures of all species in bar

    Returns:
        Log activity
    """
    planet: Planet = traced_parameters.planet
    species: tuple[Species, ...] = fixed_parameters.species
    temperature: ArrayLike = planet.surface_temperature
    # Currently not used, but may be required in the future.
    # total_pressure: Array = atmosphere_pressure(fixed_parameters, pressures)

    activity_funcs: list[Callable] = [species_.activity.log_activity for species_ in species]

    def apply_activity_function(index: ArrayLike, temperature: ArrayLike, pressures: ArrayLike):
        # Activity, so far, is only a function of the species pressure, not the total pressure
        pressure: Array = jnp.take(pressures, index)

        return lax.switch(
            index,
            activity_funcs,
            temperature,
            pressure,
        )

    vmap_apply_function: Callable = jax.vmap(apply_activity_function, in_axes=(0, None, None))
    indices: Array = jnp.arange(len(species))
    log_activity: Array = vmap_apply_function(indices, temperature, pressures)

    return log_activity


@jit
def get_log_activity_scaled(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    pressures: ArrayLike,
) -> Array:
    """Gets the log activity of all species in scaled units

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        pressures: Pressures of all species

    Returns:
        Log activity in scaled units
    """
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.surface_temperature
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    log_scaling: float = fixed_parameters.log_scaling
    # Currently not used, but may be required in the future.
    # total_pressure: Array = atmosphere_pressure(fixed_parameters, pressures)

    log_activity: Array = get_log_activity(traced_parameters, fixed_parameters, pressures)

    # Need to convert gas species back to scaled units
    mask: Array = jnp.zeros_like(log_activity, dtype=bool)
    mask = mask.at[gas_species_indices].set(True)
    log_activity_scaled: Array = scale_number_density(
        log_number_density_from_log_pressure(log_activity, temperature), log_scaling
    )

    log_activity_scaled = jnp.where(mask, log_activity_scaled, log_activity)
    # jax.debug.print("log_activity_scaled = {out}", out=log_activity_scaled)

    return log_activity_scaled


@jit
def element_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    pressure: Array,
    log_volume: Array,
) -> Array:
    """Number density of elements dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        pressure: Pressure
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of elements dissolved in melt
    """
    species: tuple[Species, ...] = fixed_parameters.species
    formula_matrix: Array = jnp.array(fixed_parameters.formula_matrix)
    diatomic_oxygen_index: Array = jnp.array(fixed_parameters.diatomic_oxygen_index)
    molar_masses: Array = jnp.array(fixed_parameters.molar_masses)
    log_scaling: float = fixed_parameters.log_scaling

    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.surface_temperature

    log_pressures: Array = unscale_number_density(
        log_pressure_from_log_number_density(log_number_density, temperature), log_scaling
    )
    pressures: Array = jnp.exp(log_pressures)
    total_pressure: Array = atmosphere_pressure(fixed_parameters, pressures)
    diatomic_oxygen_fugacity: Array = jnp.take(pressure, diatomic_oxygen_index)

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
    ppmw: Array = vmap_apply_function(indices, pressure)
    # jax.debug.print("ppmw = {out}", out=ppmw)

    species_melt_density: Array = (
        ppmw
        * unit_conversion.ppm_to_fraction
        * AVOGADRO
        * planet.mantle_melt_mass
        / (molar_masses * jnp.exp(log_volume))
    )
    # jax.debug.print("species_melt_density = {out}", out=species_melt_density)
    element_melt_density: Array = formula_matrix.dot(species_melt_density)
    # Scale back to the scaling of the numerical problem. Perform in regular space to enable
    # addition before logging, hence avoiding any problems with NaNs.
    element_melt_density = element_melt_density / jnp.exp(log_scaling)

    return element_melt_density


@jit
def get_atmosphere_log_molar_mass(gas_log_number_density: Array, gas_molar_masses: Array) -> Array:
    """Gets log molar mass of the atmosphere

    Args:
        gas_log_number_density: Log number density of gas species
        gas_molar_masses: Molar masses of gas species

    Returns:
        Log molar mass of the atmosphere
    """
    molar_mass: Array = logsumexp(gas_log_number_density, gas_molar_masses) - logsumexp(
        gas_log_number_density
    )

    return molar_mass


@jit
def get_atmosphere_log_volume(
    gas_log_number_density: Array, gas_molar_masses: Array, planet: Planet
) -> Array:
    """Gets log volume of the atmosphere"

    Args:
        gas_log_number_density: Log number density of gas species
        gas_molar_masses: Molar masses of gas species
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """
    return (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.surface_temperature)
        - get_atmosphere_log_molar_mass(gas_log_number_density, gas_molar_masses)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )


@jit
def atmosphere_pressure(
    fixed_parameters: FixedParameters,
    pressures: Array,
) -> Array:
    """Pressure of the atmosphere in bar

    Args:
        fixed_parameters: Fixed parameters
        pressure: Pressure of all species

    Returns:
        Pressure of the atmosphere in bar
    """
    gas_species_indices: Array = jnp.array(fixed_parameters.gas_species_indices)
    gas_pressures: Array = jnp.take(pressures, gas_species_indices)

    return jnp.sum(gas_pressures)


# pylint: disable=invalid-name
@jit
def get_log_Kp(
    species: tuple[Species, ...], reaction_matrix: Array, temperature: ArrayLike
) -> Array:
    """Gets the natural log of the equilibrium constant in terms of partial pressures.

    Args:
        species: List of species
        reaction_matrix: Reaction matrix
        temperature: Temperature in K

    Returns:
        Natural log of the equilibrium constant in terms of partial pressures
    """
    gibbs_list: list[ArrayLike] = []
    for species_ in species:
        gibbs: ArrayLike = species_.data.thermodata.get_gibbs_over_RT(temperature)
        gibbs_list.append(gibbs)

    gibbs_jnp: Array = jnp.array(gibbs_list)
    log_Kp: Array = -1.0 * reaction_matrix.dot(gibbs_jnp)

    return log_Kp


# pylint: enable=invalid-name
