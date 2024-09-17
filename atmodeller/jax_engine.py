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
"""JAX-related functionality for solving the system of equations. Functions are jitted."""

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

log_AVOGADRO = np.log(AVOGADRO)


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
def get_rhs(planet: Planet, scaling: float) -> Array:

    temperature: ArrayLike = planet.surface_temperature

    # Reaction set is linearly independent (determined by Gaussian elimination in a previous step)
    # log10 equilibrium constants
    # 2 H2O = 2 H2 + 1 O2
    reaction0_lnKp: float = -118.38862269303881
    reaction0_delta_n: float = 1.0
    # 4 H2 + 1 CO2 = 2 H2O + 1 CH4
    reaction1_lnKp: float = 22.8840873584512
    reaction1_delta_n: float = -2.0
    # 1 H2 + 1 O2 = 1 H2O + 1 CO
    reaction2_lnKp: float = -6.001590516742649
    reaction2_delta_n: float = 0.0

    def log10Kc_from_lnKp(lnKp: float, delta_n: float, temperature: ArrayLike) -> Array:

        log10Kc: Array = lnKp - delta_n * (
            jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(scaling) + jnp.log(temperature)
        )
        log10Kc = log10Kc / jnp.log(10)

        return log10Kc

    reaction0_log10Kc: Array = log10Kc_from_lnKp(reaction0_lnKp, reaction0_delta_n, temperature)
    reaction1_log10Kc: Array = log10Kc_from_lnKp(reaction1_lnKp, reaction1_delta_n, temperature)
    reaction2_log10Kc: Array = log10Kc_from_lnKp(reaction2_lnKp, reaction2_delta_n, temperature)

    # rhs constraints are the equilibrium constants of the reaction
    rhs: Array = jnp.array([reaction0_log10Kc, reaction1_log10Kc, reaction2_log10Kc])

    return rhs


@jit
def objective_function(solution: Array, parameters: Parameters) -> Array:
    """Residual of the reaction network and mass balance

    Args:
        solution: Solution
        parameters: Parameters

    Returns:
        Residual of the objective function
    """
    formula_matrix: Array = parameters.formula_matrix
    reaction_matrix: Array = parameters.reaction_matrix
    planet: Planet = parameters.planet
    constraints: Constraints = parameters.constraints
    species: list[SpeciesData] = parameters.species
    # jax.debug.print("{out}", out=reaction_matrix)
    # jax.debug.print("{out}", out=planet)

    # TODO: Move constraints into the driver script
    scaling: ArrayLike = parameters.scaling

    # RHS could depend on total pressure, which is part of the initial guess solution.
    rhs = get_rhs(planet, scaling)
    # jax.debug.print("{out}", out=rhs)

    # Reaction network residual
    reaction_residual: Array = reaction_matrix.dot(solution) - rhs

    # Mass balance residual
    log10_volume: Array = atmosphere_log10_volume(solution, species, planet)
    mass_residual: Array = jnp.log10(formula_matrix.dot(jnp.power(10, solution)))
    mass_residual = mass_residual - (constraints.array() - log10_volume)

    residual: Array = jnp.concatenate(
        (
            reaction_residual,
            mass_residual,
        )
    )

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
