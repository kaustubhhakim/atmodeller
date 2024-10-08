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
from atmodeller.jax_containers import (
    FugacityConstraints,
    MassConstraints,
    Parameters,
    Planet,
    Solution,
    SolverParameters,
    Species,
)
from atmodeller.jax_utilities import (
    log_pressure_from_log_number_density,
    logsumexp,
    unscale_number_density,
)
from atmodeller.thermodata.jax_thermo import get_gibbs_over_RT
from atmodeller.utilities import unit_conversion


@partial(jit, static_argnames=["solver_parameters"])
def solve(
    solution: Solution, parameters: Parameters, solver_parameters: SolverParameters
) -> Array:
    """Solves the system of non-linear equations

    Args:
        solution: Solution
        parameters: Parameters
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
        args=(parameters),
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
def objective_function(solution: Array, parameters: Parameters) -> Array:
    """Residual of the reaction network and mass balance

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        Residual
    """
    formula_matrix: Array = parameters.fixed.formula_matrix
    reaction_matrix: Array = parameters.fixed.reaction_matrix
    fugacity_matrix: Array = parameters.fixed.fugacity_matrix
    gas_species_indices: Array = parameters.fixed.gas_species_indices
    fugacity_species_indices: Array = parameters.fixed.fugacity_species_indices
    molar_masses: Array = parameters.fixed.molar_masses
    planet: Planet = parameters.planet
    fugacity_constraints: FugacityConstraints = parameters.fugacity_constraints
    mass_constraints: MassConstraints = parameters.mass_constraints
    species: list[Species] = parameters.fixed.species
    temperature: ArrayLike = planet.surface_temperature
    log_scaling: float = parameters.fixed.log_scaling

    log_number_density, log_stability = jnp.split(solution, 2)

    gas_log_number_density: Array = jnp.take(log_number_density, gas_species_indices)
    gas_molar_masses: Array = jnp.take(molar_masses, gas_species_indices)
    log_volume: Array = atmosphere_log_volume(gas_log_number_density, gas_molar_masses, planet)

    # Need pressures in bar for subsequent operations so unscale. This includes condensate
    # pressures, which are non-physical but are ignored in future calculations.
    log_pressure: Array = unscale_number_density(
        log_pressure_from_log_number_density(log_number_density, temperature), log_scaling
    )
    pressure: Array = jnp.exp(log_pressure)
    total_pressure: Array = atmosphere_pressure(parameters, log_number_density)

    # Reaction network residual
    log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
        species, gas_species_indices, reaction_matrix, temperature, log_scaling
    )
    log_activity: Array = get_log_activity(parameters, log_number_density)
    reaction_residual: Array = (
        reaction_matrix.dot(log_activity) - log_reaction_equilibrium_constant
    )
    # jax.debug.print("reaction_residual = {out}", out=reaction_residual)
    # Account for species stability.
    reaction_residual = reaction_residual - reaction_matrix.dot(jnp.exp(log_stability))
    # jax.debug.print("reaction_residual with stability = {out}", out=reaction_residual)

    # Fugacity constraints
    fugacity_log_activity: Array = jnp.take(log_activity, fugacity_species_indices)
    fugacity_residual: Array = fugacity_matrix.dot(fugacity_log_activity)
    fugacity_residual = fugacity_residual - fugacity_constraints.array(temperature)
    # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)

    # Mass balance residual for elements

    # Number density of elements in the condensed or gas phase
    element_density: Array = formula_matrix.dot(jnp.exp(log_number_density))

    element_melt_density: Array = element_density_in_melt(
        parameters, log_number_density, pressure, log_volume
    )
    log_element_density: Array = jnp.log(element_density + element_melt_density)

    mass_residual = log_element_density - mass_constraints.array(log_volume)
    # jax.debug.print("mass_residual = {out}", out=mass_residual)

    # Stability residual
    # Get minimum scaled log number of molecules
    log_min_number_density: Array = jnp.min(mass_constraints.array(log_volume)) - jnp.log(
        parameters.fixed.tau
    )
    stability_residual: Array = log_number_density + log_stability - log_min_number_density
    # jax.debug.print("stability_residual = {out}", out=stability_residual)

    residual: Array = jnp.concatenate(
        (reaction_residual, fugacity_residual, mass_residual, stability_residual)
    )
    # jax.debug.print("residual = {out}", out=residual)

    return residual


@jit
def get_log_activity(parameters: Parameters, log_number_density: Array) -> Array:
    """Log activity

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Log activity
    """
    planet: Planet = parameters.planet
    species: list[Species] = parameters.fixed.species
    temperature: ArrayLike = planet.surface_temperature
    total_pressure: Array = atmosphere_pressure(parameters, log_number_density)

    activity_funcs: list[Callable] = [species_.activity.log_activity for species_ in species]

    def apply_activity_function(
        index: ArrayLike, log_number_density: Array, temperature: ArrayLike, total_pressure: Array
    ):
        return lax.switch(
            index,
            activity_funcs,
            log_number_density,
            index,
            temperature,
            total_pressure,
        )

    vmap_apply_function: Callable = jax.vmap(
        apply_activity_function, in_axes=(0, None, None, None)
    )
    indices: ArrayLike = jnp.arange(len(species))
    log_activity: Array = vmap_apply_function(
        indices, log_number_density, temperature, total_pressure
    )

    return log_activity


@jit
def get_log_extended_activity(
    parameters: Parameters,
    log_number_density: Array,
    log_stability: Array,
) -> Array:
    """Log extended activity

    Args:
        parameters: Parameters
        log_number_density: Log number density
        log_stability: Log stability

    Returns:
        Log extended activity
    """
    log_extended_activity: Array = get_log_activity(parameters, log_number_density) - jnp.exp(
        log_stability
    )
    # jax.debug.print("log_extended_activity = {out}", out=log_extended_activity)

    return log_extended_activity


@jit
def element_density_in_melt(
    parameters: Parameters, log_number_density: Array, pressure: Array, log_volume: Array
) -> Array:
    """Number density of elements dissolved in melt due to species solubility

    Args:
        parameters: Parameters
        log_number_density: Log number density
        pressure: Pressure
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of elements dissolved in melt
    """
    formula_matrix: Array = parameters.fixed.formula_matrix
    diatomic_oxygen_index: Array = parameters.fixed.diatomic_oxygen_index
    molar_masses: Array = parameters.fixed.molar_masses
    planet: Planet = parameters.planet
    species: list[Species] = parameters.fixed.species
    temperature: ArrayLike = planet.surface_temperature
    log_scaling: float = parameters.fixed.log_scaling

    total_pressure: Array = atmosphere_pressure(parameters, log_number_density)
    diatomic_oxygen_fugacity: Array = jnp.take(pressure, diatomic_oxygen_index)

    solubility_funcs: list[Callable] = [species_.solubility.concentration for species_ in species]

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
def atmosphere_log_molar_mass(gas_log_number_density: Array, gas_molar_masses: Array) -> Array:
    """Log molar mass of the atmosphere

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
def atmosphere_log_volume(
    gas_log_number_density: Array, gas_molar_masses: Array, planet: Planet
) -> Array:
    """Log volume of the atmosphere"

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
        - atmosphere_log_molar_mass(gas_log_number_density, gas_molar_masses)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )


@jit
def atmosphere_pressure(parameters: Parameters, log_number_density: Array) -> Array:
    """Pressure of the atmosphere in bar

    Args:
        parameters: Parameters
        log_number_density: Log number density

    Returns:
        Pressure of the atmosphere in bar
    """
    gas_species_indices: Array = parameters.fixed.gas_species_indices
    planet: Planet = parameters.planet
    temperature: ArrayLike = planet.surface_temperature
    log_scaling: float = parameters.fixed.log_scaling

    gas_log_number_density: Array = jnp.take(log_number_density, gas_species_indices)
    gas_log_pressure: Array = unscale_number_density(
        log_pressure_from_log_number_density(gas_log_number_density, temperature), log_scaling
    )
    gas_pressure: Array = jnp.exp(gas_log_pressure)

    return jnp.sum(gas_pressure)


# pylint: disable=invalid-name
@jit
def get_log_Kp(species: list[Species], reaction_matrix: Array, temperature: ArrayLike) -> Array:
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
        gibbs: ArrayLike = get_gibbs_over_RT(species_.data.thermodata, temperature)
        gibbs_list.append(gibbs)

    gibbs_jnp: Array = jnp.array(gibbs_list)
    log_Kp: Array = -1.0 * reaction_matrix.dot(gibbs_jnp)

    return log_Kp


# pylint: enable=invalid-name
