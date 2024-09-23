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
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller import BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.jax_containers import (
    Constraints,
    Parameters,
    Planet,
    Solution,
    SolverParameters,
    SpeciesData,
    gas_species_mask,
)
from atmodeller.jax_utilities import logsumexp


@partial(jit, static_argnames=["solver_parameters"])
def solve(
    solution: Solution, parameters: Parameters, solver_parameters: SolverParameters
) -> Array:
    """Solves the system

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


def solver_wrapper(solver_parameters) -> Callable:

    @jit
    def wrapped_solve(solution, parameters):
        return solve(solution, parameters, solver_parameters)

    return wrapped_solve


@jit
def get_log_reaction_equilibrium_constant(
    species: list[SpeciesData],
    reaction_matrix: Array,
    temperature: ArrayLike,
    log_scaling: ArrayLike,
) -> Array:
    """Gets the log equilibrium constant of the reactions

    Args:
        species: List of species
        reaction_matrix: Reaction matrix
        temperature: Temperature
        log_scaling: Log scaling

    Returns:
        Log equilibrium constant of the reactions
    """
    # pylint: disable=invalid-name
    log_Kp: Array = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)

    gas_mask: Array = gas_species_mask(species)
    delta_n: Array = jnp.sum(reaction_matrix * gas_mask, axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)

    log_Kc: Array = log_Kp - delta_n * (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + log_scaling + jnp.log(temperature)
    )
    # jax.debug.print("log10Kc = {out}", out=log_Kc)

    # pylint: enable=invalid-name

    return log_Kc


@jit
def get_log_activity(log_number_density: Array, parameters: Parameters) -> Array:
    """Log activity

    Args:
        log_number_density: Log number density
        parameters: Parameters

    Returns:
        Log activity
    """
    species: list[SpeciesData] = parameters.species
    gas_mask: Array = gas_species_mask(species)

    activity_for_gas: Array = log_number_density
    activity_for_condensed: Array = jnp.zeros(len(species))

    log_activity: Array = jnp.where(gas_mask == 1, activity_for_gas, activity_for_condensed)

    # jax.debug.print("log_activity = {out}", out=log_activity)

    return log_activity


@jit
def get_log_extended_activity(
    log_number_density: Array, log_stability: Array, parameters: Parameters
) -> Array:
    """Log extended activity

    Args:
        log_number_density: Log number density
        log_stability: Log stability
        parameters: Parameters

    Returns:
        Log extended activity
    """
    log_extended_activity: Array = get_log_activity(log_number_density, parameters) - jnp.exp(
        log_stability
    )

    # jax.debug.print("log_extended_activity = {out}", out=log_extended_activity)

    return log_extended_activity


@jit
def objective_function(solution: Array, parameters: Parameters) -> Array:
    """Residual of the reaction network and mass balance

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        Residual
    """
    formula_matrix: Array = parameters.formula_matrix
    reaction_matrix: Array = parameters.reaction_matrix
    planet: Planet = parameters.planet
    constraints: Constraints = parameters.constraints
    species: list[SpeciesData] = parameters.species
    temperature: ArrayLike = planet.surface_temperature
    log_scaling: float = parameters.log_scaling

    # jax.debug.print("solution in = {out}", out=solution)

    log_number_density, log_stability = jnp.split(solution, 2)

    # Reaction network residual
    log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
        species, reaction_matrix, temperature, log_scaling
    )
    log_activity: Array = get_log_activity(log_number_density, parameters)
    reaction_residual: Array = (
        reaction_matrix.dot(log_activity) - log_reaction_equilibrium_constant
    )
    # jax.debug.print("reaction_residual = {out}", out=reaction_residual)
    # Account for species stability.
    reaction_residual = reaction_residual - reaction_matrix.dot(jnp.exp(log_stability))
    # jax.debug.print("reaction_residual with stability = {out}", out=reaction_residual)

    # Mass balance residual
    log_volume: Array = atmosphere_log_volume(log_number_density, species, planet)
    log_density_matrix_product: Array = jnp.log(formula_matrix.dot(jnp.exp(log_number_density)))
    mass_residual = log_density_matrix_product - (constraints.array() - log_volume)
    # jax.debug.print("mass_residual = {out}", out=mass_residual)

    # Stability residual
    # Get minimum scaled log number of molecules
    log_min_number_density: Array = (
        jnp.min(constraints.array()) - log_volume - jnp.log(parameters.tau)
    )
    stability_residual: Array = log_number_density + log_stability - log_min_number_density
    # jax.debug.print("stability_residual = {out}", out=stability_residual)

    residual: Array = jnp.concatenate((reaction_residual, mass_residual, stability_residual))
    # jax.debug.print("residual = {out}", out=residual)

    return residual


@jit
def atmosphere_log_molar_mass(log_number_density: Array, species: list[SpeciesData]) -> Array:
    """Log molar mass of the atmosphere

    Args:
        log_number_density: Log number density
        species: List of species

    Returns:
        Log molar mass of the atmosphere
    """
    molar_masses: Array = jnp.array([value.molar_mass for value in species])
    gas_mask: Array = gas_species_mask(species)
    gas_molar_masses: Array = molar_masses * gas_mask

    # jax.debug.print("molar_masses = {out}", out=molar_masses)
    # jax.debug.print("gas_molar_masses = {out}", out=gas_molar_masses)

    molar_mass: Array = logsumexp(log_number_density, gas_molar_masses) - logsumexp(
        log_number_density, gas_mask
    )

    return molar_mass


@jit
def atmosphere_log_volume(
    log_number_density: Array, species: list[SpeciesData], planet: Planet
) -> Array:
    """Log volume of the atmosphere"

    Args:
        log_number_density: Log number density
        species: List of species
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """
    return (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.surface_temperature)
        - atmosphere_log_molar_mass(log_number_density, species)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )


@jit
def gibbs_energy_of_formation(species_data: SpeciesData, temperature: ArrayLike) -> Array:
    r"""Gibbs energy of formation

    Args:
        species_data: Species data
        temperature: Temperature in K

    Returns:
        The standard Gibbs free energy of formation in :math:`\mathrm{J}\mathrm{mol}^{-1}`
    """
    gibbs: Array = (
        # Leading with this term prevents the linter from complaining.
        species_data.gibbs_coefficients[1] * jnp.log(temperature)
        + species_data.gibbs_coefficients[0] / temperature
        + species_data.gibbs_coefficients[2]
        + species_data.gibbs_coefficients[3] * jnp.power(temperature, 1)
        + species_data.gibbs_coefficients[4] * jnp.power(temperature, 2)
    )

    return gibbs * 1000.0  # kilo


# pylint: disable=invalid-name
@jit
def get_log_Kp(
    species: list[SpeciesData], reaction_matrix: Array, temperature: ArrayLike
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
        gibbs: ArrayLike = gibbs_energy_of_formation(species_, temperature)
        gibbs_list.append(gibbs)

    gibbs_array: Array = jnp.array(gibbs_list)

    log_Kp: Array = -1.0 * reaction_matrix.dot(gibbs_array) / (GAS_CONSTANT * temperature)

    return log_Kp


# pylint: enable=invalid-name
