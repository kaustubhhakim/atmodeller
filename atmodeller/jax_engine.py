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

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller import BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.jax_containers import (
    Constraints,
    Parameters,
    Planet,
    Solution,
    SpeciesData,
)
from atmodeller.jax_utilities import logsumexp


@jit
def solve(solution: Solution, parameters: Parameters) -> Array:
    """Solves the system

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        The solution
    """

    tol: float = 1.0e-8
    solver = optx.Dogleg(atol=tol, rtol=tol)
    # solver = optx.Newton(atol=tol, rtol=tol)
    # solver = optx.LevenbergMarquardt(atol=tol, rtol=tol)

    sol = optx.root_find(
        objective_function, solver, solution.data, args=(parameters), throw=True, max_steps=256
    )

    jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

    return sol.value


@jit
def get_log_reaction_equilibrium_constant(
    species: list[SpeciesData],
    reaction_matrix: Array,
    temperature: ArrayLike,
    scaling: ArrayLike,
) -> Array:
    """Gets the log equilibrium constant of the reactions

    Args:
        species: List of species
        reaction_matrix: Reaction matrix
        temperature: Temperature
        scaling: Scaling

    Returns:
        Log equilibrium constant of the reactions
    """
    # pylint: disable=invalid-name
    log_Kp: Array = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)

    phase_codes: Array = jnp.array([s.phase_code for s in species])
    gas_phases: Array = (phase_codes == 0).astype(int)
    delta_n: Array = jnp.sum(reaction_matrix * gas_phases, axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)

    log_Kc: Array = log_Kp - delta_n * (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(scaling) + jnp.log(temperature)
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
    phase_codes: Array = jnp.array([s.phase_code for s in species])

    activity_for_gas: Array = log_number_density
    activity_for_condensed: Array = jnp.zeros(len(species))

    log_activity: Array = jnp.where(phase_codes == 0, activity_for_gas, activity_for_condensed)

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
    scaling: ArrayLike = parameters.scaling

    number_density, stability = jnp.split(solution, 2)

    log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
        species, reaction_matrix, temperature, scaling
    )
    # jax.debug.print("rhs = {out}", out=rhs)

    # Reaction network residual
    log_activity: Array = get_log_activity(number_density, parameters)
    # log10_extended_activity: Array = get_log10_extended_activity(
    #    number_density, stability, parameters
    # )
    reaction_residual: Array = (
        reaction_matrix.dot(log_activity) - log_reaction_equilibrium_constant
    )
    # jax.debug.print("reaction_residual = {out}", out=reaction_residual)

    # Modifications for stability. This is the same as dotting reaction matrix with extended
    # activity
    reaction_residual = reaction_residual - reaction_matrix.dot(jnp.exp(stability))

    # Mass balance residual
    log_volume: Array = atmosphere_log_volume(number_density, species, planet)
    mass_residual: Array = jnp.log(formula_matrix.dot(jnp.exp(number_density)))
    mass_residual = mass_residual - (constraints.array(scaling) - log_volume)
    # jax.debug.print("mass_residual = {out}", out=mass_residual)

    # Get minimum scaled log10 number of molecules
    log_tau_min: Array = jnp.min(constraints.array(scaling))
    jax.debug.print("log_tau_min = {out}", out=log_tau_min)
    # Get minimum number density
    log_tau_min = log_tau_min - log_volume
    jax.debug.print("log10_tau_min = {out}", out=log_tau_min)
    # Fraction
    tau_min = jnp.exp(log_tau_min)
    tau_min = tau_min * 1.0e-15

    # Stability residual
    N: Array = jnp.diag(jnp.exp(number_density))
    jax.debug.print("N = {out}", out=N)
    Z: Array = jnp.diag(jnp.exp(stability))
    jax.debug.print("Z = {out}", out=Z)
    e: Array = jnp.ones_like(stability)
    jax.debug.print("e = {out}", out=e)
    stability_residual: Array = jnp.log(N.dot(Z).dot(e)) - jnp.log(tau_min * e)
    jax.debug.print("out = {out}", out=stability_residual)

    residual: Array = jnp.concatenate((reaction_residual, mass_residual, stability_residual))
    jax.debug.print("residual = {out}", out=residual)

    return residual


@jit
def atmosphere_log_molar_mass(solution: Array, species: list[SpeciesData]) -> Array:
    """Log of the molar mass of the atmosphere

    Args:
        solution: Number density solution array(s)
        species: Species

    Returns:
        Log molar mass of the atmosphere
    """
    molar_masses: Array = jnp.array([value.molar_mass for value in species])
    molar_mass: Array = logsumexp(solution, molar_masses) - logsumexp(solution)

    return molar_mass


@jit
def atmosphere_log_volume(solution: Array, species: list[SpeciesData], planet: Planet) -> Array:
    """Log of the volume of the atmosphere"

    Args:
        solution: Number density solution array(s)
        species: Species
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """
    return (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.surface_temperature)
        - atmosphere_log_molar_mass(solution, species)
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
