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

# import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.jax_containers import (
    Constraints,
    Parameters,
    Planet,
    Solution,
    SpeciesData,
)
from atmodeller.jax_utilities import logsumexp_base10

log_AVOGADRO: ArrayLike = np.log(AVOGADRO)


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
        objective_function,
        solver,
        solution.number_density,
        args=(parameters),
        throw=True,
    )

    return sol.value


@jit
def get_rhs(
    species: list[SpeciesData],
    reaction_matrix: Array,
    temperature: ArrayLike,
    scaling: ArrayLike,
) -> Array:
    """Gets the right-hand side of the reaction network equations

    Args:
        species: List of species
        reaction_matrix: Reaction matrix
        temperature: Temperature
        scaling: Scaling

    Returns:
        Right-hand side of the reaction network equations
    """
    # pylint: disable=invalid-name
    lnKp: Array = get_lnKp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)

    delta_n: Array = jnp.sum(reaction_matrix, axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)

    log10Kc: Array = lnKp - delta_n * (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(scaling) + jnp.log(temperature)
    )
    log10Kc = log10Kc / jnp.log(10)
    # jax.debug.print("log10Kc = {out}", out=log10Kc)

    # pylint: enable=invalid-name

    return log10Kc


@jit
def get_log10_activity(solution: Array, parameters: Parameters) -> Array:
    """Log10 activity

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        Activity
    """
    species: list[SpeciesData] = parameters.species
    phase_codes: Array = jnp.array([s.phase_code for s in species])

    value_for_gas: Array = solution
    value_for_condensed: Array = jnp.zeros(len(species))

    activity_array = jnp.where(phase_codes == 0, value_for_gas, value_for_condensed)

    return activity_array


@jit
def get_log10_extended_activity(solution: Array, parameters: Parameters) -> Array:
    """Log10 extended activity

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        Extended activity
    """
    activity: Array = get_log10_activity(solution, parameters)


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

    rhs: Array = get_rhs(species, reaction_matrix, temperature, scaling)
    # jax.debug.print("rhs = {out}", out=rhs)

    # Reaction network residual
    log10_activity: Array = get_log10_activity(solution, parameters)
    reaction_residual: Array = reaction_matrix.dot(log10_activity) - rhs
    # jax.debug.print("reaction_residual = {out}", out=reaction_residual)

    # Mass balance residual
    log10_volume: Array = atmosphere_log10_volume(solution, species, planet)
    mass_residual: Array = jnp.log10(formula_matrix.dot(jnp.power(10, solution)))
    mass_residual = mass_residual - (constraints.array(scaling) - log10_volume)
    # jax.debug.print("mass_residual = {out}", out=mass_residual)

    residual: Array = jnp.concatenate(
        (
            reaction_residual,
            mass_residual,
        )
    )
    # jax.debug.print("residual = {out}", out=residual)

    return residual


@jit
def atmosphere_log10_molar_mass(solution: Array, species: list[SpeciesData]) -> Array:
    """Log10 of the molar mass of the atmosphere

    Args:
        solution: Number density solution array(s)
        species: Species

    Returns:
        Log10 molar mass of the atmosphere
    """
    molar_masses: Array = jnp.array([value.molar_mass for value in species])
    molar_mass: Array = logsumexp_base10(solution, molar_masses) - logsumexp_base10(solution)

    return molar_mass


@jit
def atmosphere_log10_volume(solution: Array, species: list[SpeciesData], planet: Planet) -> Array:
    """Log10 of the volume of the atmosphere"

    Args:
        solution: Number density solution array(s)
        species: Species
        planet: Planet

    Returns:
        Log10 volume of the atmosphere
    """
    return (
        jnp.log10(GAS_CONSTANT)
        + jnp.log10(planet.surface_temperature)
        - atmosphere_log10_molar_mass(solution, species)
        + jnp.log10(planet.surface_area)
        - jnp.log10(planet.surface_gravity)
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
def get_lnKp(species: list[SpeciesData], reaction_matrix: Array, temperature: ArrayLike) -> Array:
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

    lnKp: Array = -1.0 * reaction_matrix.dot(gibbs_array) / (GAS_CONSTANT * temperature)

    return lnKp


# pylint: enable=invalid-name
