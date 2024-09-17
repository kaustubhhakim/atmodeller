#!/usr/bin/env python
"""Minimum working example (MWE) for JAX vmap with atmodeller-like containers for parameters.

Reproduces test_CHO_low_temperature in test_benchmark.py using some hard-coded parameters.
"""
from timeit import timeit
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, debug_logger
from atmodeller.classes import ReactionNetwork
from atmodeller.jax_containers import (
    CH4_g,
    CO2_g,
    CO_g,
    Constraints,
    H2_g,
    H2O_g,
    H2O_l,
    O2_g,
    Parameters,
    Planet,
    Solution,
    SpeciesData,
)
from atmodeller.jax_engine import solve
from atmodeller.jax_utilities import (
    pytrees_stack,
    scale_number_density,
    unscale_number_density,
)
from atmodeller.utilities import earth_oceans_to_hydrogen_mass, partial_rref

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_dtypes", True)

logger = debug_logger()

# TODO: Move scaling factor definition to when the problem is setup by the user
scaling: float = AVOGADRO
log10_scaling: Array = np.log10(scaling)

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

known_solution_array: ArrayLike = np.array([val for val in known_solution.values()])

# Initial solution guess number density (molecules/m^3)
initial_solution_default: ArrayLike = np.array([26, 26, 12, -26, 26, 25], dtype=np.float_)
# initial_solution_default: Array = jnp.array([26, 26, 26, 26, 26, 26], dtype=jnp.float_)

# If we start somewhere close to the solution then Optimistix is OK
# Parameters for the perturbation
mean = 10.0  # Mean of the perturbation (usually 0)
std_dev = 5.0  # Standard deviation of the perturbation
# Generate random perturbations
perturbation = np.random.normal(mean, std_dev, size=known_solution_array.shape)

# initial_solution: Array = jnp.array(known_solution_array) + perturbation
initial_solution: ArrayLike = initial_solution_default
initial_solution = scale_number_density(initial_solution, log10_scaling)

h_kg: float = earth_oceans_to_hydrogen_mass(1)
c_kg: float = 1 * h_kg
o_kg: float = 1.02999e20

# Unscaled total molecules constraints in alphabetical order
mass_constraints = {
    "C": c_kg,  # 10**45.89051326565627,
    "H": h_kg,  # 10**46.96664792007732,
    "O": o_kg,  # 10**45.58848007858896,
}
constraints: Constraints = Constraints.create(species_list, mass_constraints)

reaction_network: ReactionNetwork = ReactionNetwork()
formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species_list))
reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species_list))


def solve_single(species: list[SpeciesData]) -> Array:
    """Solves a single system

    This does the non-JAX parts, such as getting the reaction matrix which is assumed constant
    for a calculation.
    """
    planet: Planet = Planet(surface_temperature=450)
    # Placeholder for stability
    solution: Solution = Solution(initial_solution, initial_solution)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species, planet, constraints, scaling
    )

    solve(solution, parameters).block_until_ready()

    # Define a wrapper function to be timed
    def timed_func():
        solve(solution, parameters).block_until_ready()

    execution_time = timeit(timed_func, number=1000)  # Run 1000 iterations
    print(f"Execution time for 1000 iterations: {execution_time} seconds")

    out: Array = solve(solution, parameters).block_until_ready()

    return unscale_number_density(out, log10_scaling)


def solve_batch(species: list[SpeciesData]) -> Array:
    """Solves a batch system

    This does the non-JAX parts, such as getting the reaction matrix which is assumed constant
    for a batch of calculations.
    """
    planets: list[Planet] = []
    for surface_temperature in range(450, 2001, 1000):
        planets.append(Planet(surface_temperature=surface_temperature))

    # Stacks the entities into one named tuple
    planets_for_vmap = pytrees_stack(planets)

    mass_constraints_batch = {
        "C": jnp.array((c_kg, c_kg * 1)),
        "H": jnp.array((h_kg, h_kg * 2)),
        "O": jnp.array((o_kg, o_kg * 1)),
    }
    constraints_batch = Constraints.create(species_list, mass_constraints_batch)

    constraints_for_vmap = Constraints(species=None, log10_molecules=0)

    parameters = Parameters(
        formula_matrix, reaction_matrix, species, planets_for_vmap, constraints_batch, scaling
    )
    # jax.debug.print("{out}", out=parameters)

    # For the same initial solution
    solution = Solution(initial_solution, initial_solution)  # type: ignore
    # jax.debug.print("{out}", out=solution)

    additional_for_vmap: Parameters = Parameters(
        formula_matrix=None,  # type: ignore
        reaction_matrix=None,  # type: ignore
        species=None,  # type: ignore
        planet=0,  # type: ignore
        constraints=constraints_for_vmap,  # type: ignore
        scaling=None,  # type: ignore
    )

    vmap_solve: Callable = jit(
        jax.vmap(
            solve,
            in_axes=(
                None,
                additional_for_vmap,
            ),
        )
    )

    vmap_solve(solution, parameters).block_until_ready()

    # Define a wrapper function to be timed
    def timed_func():
        vmap_solve(solution, parameters).block_until_ready()

    # Time the function using timeit
    execution_time = timeit(timed_func, number=1000)  # Run 1000 iterations
    print(f"Execution time for 1000 iterations: {execution_time} seconds")

    out: Array = vmap_solve(solution, parameters).block_until_ready()

    return unscale_number_density(out, log10_scaling)


def simple_system():

    # Species order is: H2, H2O, CO2, O2, CH4, CO
    species_list: list[SpeciesData] = [H2_g, CO_g, H2O_g, H2O_l, CO2_g, O2_g, CH4_g]

    reaction_network: ReactionNetwork = ReactionNetwork()

    # With charge balance, Leal et al (2016) state that some of the mass-balance equations can be
    # linearly dependent on the others. But this can't happen with just elements. Therefore we
    # can directly translate the formula matrix of the system W = A and express the mass balance in
    # matrix form as An=b, so for root finding we can solve An-b=0
    # Hence below gives matrix A (Equation 11)
    formula_matrix: npt.NDArray = reaction_network.formula_matrix(species_list)
    logger.info("formula_matrix = %s", formula_matrix)

    # For a general list of species we want to compute the primary and secondary species. We don't
    # care which are which, just that the system removes linear components to determine a set of
    # minimum, albeit non-unique, reactions that couple the production of secondary species to
    # primary species. We can achieve this by performing row reduction on the transpose of the
    # formula matrix. This is given in Appendix A.  Perform an LU decomposition of A^T, whose
    # dimensions are N X C (number of species x number of linearly independent elements). Below
    # is matrix nu where nu*A^T = 0.
    reaction_matrix: npt.NDArray = partial_rref(formula_matrix.T)


def main():

    # out = solve_single(species_list)
    out = solve_batch(species_list)

    # solve_batch_jit = jax.jit(solve_batch)

    # Pre-compile the function before timing...
    # solve_batch_jit(species_list).block_until_ready()

    # out = solve_batch(species_list).block_until_ready()

    # out = solve_batch(species_list)

    # simple_system()

    print(out)

    # out = timeit(solve_batch(species_list).block_until_ready)

    # print(out)
    # print(solutions)


if __name__ == "__main__":
    main()
