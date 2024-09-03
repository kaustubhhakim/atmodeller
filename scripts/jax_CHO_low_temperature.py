#!/usr/bin/env python
"""Minimum working example (MWE) for JAX vmap with atmodeller-like containers for parameters.

Reproduces test_CHO_low_temperature in test_benchmark.py using some hard-coded parameters.
"""
from timeit import timeit
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, jit
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from scipy.constants import Avogadro, Boltzmann, gas_constant

from atmodeller.core import Planet
from atmodeller.myjax import (
    CH4_g,
    CO2_g,
    CO_g,
    H2_g,
    H2O_g,
    O2_g,
    ReactionNetworkJAX,
    SpeciesData,
)
from atmodeller.utilities_jax import logsumexp_base10

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_dtypes", True)

MACHEPS: float = float(jnp.finfo(jnp.float_).eps)
"""Machine epsilon"""

# For scaling
GAS_CONSTANT_BAR: float = gas_constant * 1.0e-5
BOLTZMANN_CONSTANT_BAR: float = Boltzmann * 1.0e-5

# This allows for two scalings, which in practice can be thought of as two that can cancel
AVOGADRO: float = Avogadro
log10_AVOGADRO: float = np.log10(AVOGADRO)  # to convert to moles
log_AVOGADRO: float = np.log(AVOGADRO)
scaling: float = 1  # mole faction scaling
log10_scaling: float = np.log10(scaling)
log_scaling: float = np.log(scaling)

# import traceback
# import warnings

# # Create a custom warning handler
# def warning_handler(message, category, filename, lineno, file=None, line=None):
#     # Print the complete warning message
#     print(f"Warning: {message}")
#     print(f"Category: {category.__name__}")
#     print(f"File: {filename}, Line: {lineno}")
#     # Print stack trace for additional context
#     print("Stack trace:")
#     traceback.print_stack()

# # Redirect warnings to the custom handler
# warnings.showwarning = warning_handler

# Species order is: H2, H2O, CO2, O2, CH4, CO
species_list: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]

# For testing solvers, this is the known solution of the system
known_solution: dict[str, float] = {
    "H2": 26.950804260065272,
    "H2O": 26.109794057030303,
    "CO2": 11.303173861822636,
    "O2": -27.890841758236377,
    "CH4": 26.411827244097612,
    "CO": 9.537726420793389,
}

known_solution_array: Array = jnp.array([val for val in known_solution.values()])

# Species molar masses in kg/mol
molar_masses_dict: dict[str, float] = {
    "H2": 0.002015882,
    "H2O": 0.018015287,
    "CO2": 0.044009549999999995,
    "O2": 0.031998809999999996,
    "CH4": 0.016042504000000003,
    "CO": 0.028010145,
}


@jit
def dimensional_to_scaled_base10(dimensional_number_density: Array):
    return dimensional_number_density - log10_AVOGADRO + log10_scaling


@jit
def scaled_to_dimensional_base10(scaled_number_density: Array):
    return scaled_number_density - dimensional_to_scaled_base10(0)


# Element log10 number of total molecules constraints:
log10_oxygen_constraint: float = dimensional_to_scaled_base10(45.58848007858896)
log10_hydrogen_constraint: float = dimensional_to_scaled_base10(46.96664792007732)
log10_carbon_constraint: float = dimensional_to_scaled_base10(45.89051326565627)
# Initial solution guess number density (molecules/m^3)
initial_solution_default: Array = jnp.array([26, 26, 12, -26, 26, 25], dtype=jnp.float_)
# initial_solution_default: Array = jnp.array([26, 26, 26, 26, 26, 26], dtype=jnp.float_)

# If we start somewhere close to the solution then Optimistix is OK
# Parameters for the perturbation
mean = 10.0  # Mean of the perturbation (usually 0)
std_dev = 5.0  # Standard deviation of the perturbation
# Generate random perturbations
perturbation = np.random.normal(mean, std_dev, size=known_solution_array.shape)

# initial_solution: Array = jnp.array(known_solution_array) + perturbation
initial_solution = initial_solution_default
initial_solution = dimensional_to_scaled_base10(initial_solution)


@jit
def get_rhs(planet: Planet) -> Array:

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
        # return (lnKp - delta_n * (np.log(GAS_CONSTANT_BAR) + np.log(temperature))) / np.log(10)
        log10Kc: Array = lnKp - delta_n * (
            jnp.log(BOLTZMANN_CONSTANT_BAR) + log_AVOGADRO - log_scaling + jnp.log(temperature)
        )
        log10Kc = log10Kc / jnp.log(10)

        return log10Kc

    reaction0_log10Kc: Array = log10Kc_from_lnKp(reaction0_lnKp, reaction0_delta_n, temperature)
    reaction1_log10Kc: Array = log10Kc_from_lnKp(reaction1_lnKp, reaction1_delta_n, temperature)
    reaction2_log10Kc: Array = log10Kc_from_lnKp(reaction2_lnKp, reaction2_delta_n, temperature)

    # rhs constraints are the equilibrium constants of the reaction
    rhs: Array = jnp.array([reaction0_log10Kc, reaction1_log10Kc, reaction2_log10Kc])

    return rhs


class SystemParams(NamedTuple):

    initial_solution: Array


class AdditionalParams(NamedTuple):

    coefficient_matrix: Array
    planet: Planet


@jit
def solve_with_optimistix(system_params, additional_params) -> Array:
    """Solve the system with Optimistix"""

    tol: float = 1.0e-8
    solver = optx.Dogleg(atol=tol, rtol=tol)
    # solver = optx.Newton(atol=tol, rtol=tol)
    # solver = optx.LevenbergMarquardt(atol=tol, rtol=tol)

    system_params = SystemParams(initial_solution)

    sol = optx.root_find(
        objective_function,
        solver,
        system_params.initial_solution,
        args=(additional_params),
        throw=True,
    )

    solution: Array = scaled_to_dimensional_base10(sol.value)

    return solution


@jit
def atmosphere_log10_molar_mass(solution: Array) -> Array:
    """Log10 of the molar mass of the atmosphere"""
    molar_masses: Array = jnp.array([value for value in molar_masses_dict.values()])
    molar_mass: Array = logsumexp_base10(solution, molar_masses) - logsumexp_base10(solution)

    return molar_mass


@jit
def atmosphere_log10_volume(solution: Array, planet: Planet) -> Array:
    """Log10 of the volume of the atmosphere"""
    return (
        jnp.log10(gas_constant)
        + jnp.log10(planet.surface_temperature)
        # Units of solution don't matter because it just weights (unless numerical problems)
        - atmosphere_log10_molar_mass(solution)
        + jnp.log10(planet.surface_area)
        - jnp.log10(planet.surface_gravity)
    )


# These argument specifications are fixed for Optimistix, so can conform the parameter passing
# to adhere to this. Should return a pytree of arrays, not necessarily the same shape as the
# solution.
@jit
def objective_function(solution: Array, additional_params: AdditionalParams) -> Array:
    """Residual of the reaction network and mass balance"""

    # Extract parameters from the pytree
    coefficient_matrix = additional_params.coefficient_matrix
    planet = additional_params.planet
    # jax.debug.print("{out}", out=coefficient_matrix)
    # jax.debug.print("{out}", out=planet)

    # RHS could depend on total pressure, which is part of the initial guess solution.
    rhs = get_rhs(planet)
    # jax.debug.print("{out}", out=rhs)

    # Reaction network
    reaction_residual: Array = coefficient_matrix.dot(solution) - rhs

    log10_volume: Array = atmosphere_log10_volume(solution, planet)

    # Mass balance residuals (stoichiometry coefficients are hard-coded for this MWE)
    oxygen_residual: Array = jnp.array(
        [
            solution[1],
            jnp.log10(2) + solution[2],
            jnp.log10(2) + solution[3],
            solution[5],
        ]
    )
    oxygen_residual = logsumexp_base10(oxygen_residual) - (log10_oxygen_constraint - log10_volume)

    hydrogen_residual: Array = jnp.array(
        [jnp.log10(2) + solution[0], jnp.log10(2) + solution[1], jnp.log10(4) + solution[4]]
    )
    hydrogen_residual = logsumexp_base10(hydrogen_residual) - (
        log10_hydrogen_constraint - log10_volume
    )

    carbon_residual: Array = jnp.array([solution[2], solution[4], solution[5]])
    carbon_residual = logsumexp_base10(carbon_residual) - (log10_carbon_constraint - log10_volume)

    residual: Array = jnp.concatenate(
        (
            reaction_residual,
            jnp.array([oxygen_residual]),
            jnp.array([hydrogen_residual]),
            jnp.array([carbon_residual]),
        )
    )

    return residual


def pytrees_stack(pytrees, axis=0):
    """Stacks an iterable of pytrees along a specified axis."""
    results = tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results


def pytrees_vmap(fn):
    """Vectorizes a function over a batch of pytrees."""

    def g(pytrees):
        stacked = pytrees_stack(pytrees)
        results = jax.vmap(fn)(stacked)
        return results

    return g


def solve_single(species: list[SpeciesData]) -> Array:
    """Solves a single system

    This does the non-JAX parts, such as getting the reaction matrix which is assumed constant
    for a calculation.
    """

    reaction_network: ReactionNetworkJAX = ReactionNetworkJAX()
    coefficient_matrix: Array = reaction_network.reaction_matrix(species)

    planet: Planet = Planet(surface_temperature=450)
    system_params = SystemParams(initial_solution)
    additional_params = AdditionalParams(coefficient_matrix, planet)

    out = solve_with_optimistix(system_params, additional_params)

    return out


def solve_batch(species: list[SpeciesData]) -> Array:
    """Solves a batch system

    This does the non-JAX parts, such as getting the reaction matrix which is assumed constant
    for a batch of calculations.
    """

    reaction_network: ReactionNetworkJAX = ReactionNetworkJAX()
    coefficient_matrix: Array = reaction_network.reaction_matrix(species)

    out = solve_batch_jax(coefficient_matrix)

    return out


@jit
def solve_batch_jax(coefficient_matrix: Array):

    planets: list[Planet] = []
    for surface_temperature in range(450, 2001, 1):
        planets.append(Planet(surface_temperature=surface_temperature))

    # Stacks the entities into one named tuple
    planets_for_vmap = pytrees_stack(planets)
    # jax.debug.print("{out}", out=planets_for_vmap)

    # Replicate the coefficient matrix to match the batch size of planets
    coefficient_matrices_for_vmap = jnp.stack(
        [coefficient_matrix] * len(planets_for_vmap.surface_temperature)
    )

    # Combine into AdditionalParams
    additional_params = AdditionalParams(coefficient_matrices_for_vmap, planets_for_vmap)
    # jax.debug.print("{out}", out=additional_params)

    # For different initial solutions
    # system_params = SystemParams(
    #    jnp.stack([initial_solution] * len(planets_for_vmap.surface_temperature))
    # )
    # For the same initial solution
    system_params = SystemParams(initial_solution)

    # jax.debug.print("{out}", out=system_params)

    # JIT compile the solve function
    jit_solve = jax.jit(solve_with_optimistix)

    vmap_solve = jax.vmap(jit_solve, in_axes=(None, 0))

    solutions = vmap_solve(system_params, additional_params)

    return solutions


def main():

    # out = solve_single(species_list)
    # out = solve_batch(species_list)

    # solve_batch_jit = jax.jit(solve_batch)

    # Pre-compile the function before timing...
    # solve_batch_jit(species_list).block_until_ready()

    # out = solve_batch(species_list).block_until_ready()

    out = solve_batch(species_list)

    print(out)

    # out = timeit(solve_batch(species_list).block_until_ready)

    # print(out)
    # print(solutions)


if __name__ == "__main__":
    main()
