#!/usr/bin/env python
"""Minimum working example (MWE) for JAX vmap with atmodeller-like containers for parameters.

Reproduces test_CHO_low_temperature in test_benchmark.py using some hard-coded parameters.
"""
from typing import Callable

import jax
import numpy as np
import numpy.typing as npt
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, debug_logger
from atmodeller.jax_containers import (
    CH4_g,
    CO2_g,
    CO_g,
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
    ReactionNetworkJAX,
    pytrees_stack,
    scale_number_density,
    unscale_number_density,
)
from atmodeller.utilities import partial_rref

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


def solve_single(species: list[SpeciesData]) -> Array:
    """Solves a single system

    This does the non-JAX parts, such as getting the reaction matrix which is assumed constant
    for a calculation.
    """

    reaction_network: ReactionNetworkJAX = ReactionNetworkJAX()
    reaction_matrix: Array = reaction_network.reaction_matrix(species)

    planet: Planet = Planet(surface_temperature=450)
    # Placeholder for stability
    system_params: Solution = Solution(initial_solution, initial_solution)  # type: ignore
    parameters: Parameters = Parameters(reaction_matrix, species, planet, scaling)

    out: Array = solve(system_params, parameters)

    return unscale_number_density(out, log10_scaling)


def solve_batch(species: list[SpeciesData]) -> Array:
    """Solves a batch system

    This does the non-JAX parts, such as getting the reaction matrix which is assumed constant
    for a batch of calculations.
    """

    reaction_network: ReactionNetworkJAX = ReactionNetworkJAX()
    reaction_matrix: ArrayLike = reaction_network.reaction_matrix(species)

    out = solve_batch_jax(reaction_matrix, species)

    return unscale_number_density(out, log10_scaling)


@jit
def solve_batch_jax(reaction_matrix: Array, species: list[SpeciesData]):

    planets: list[Planet] = []
    for surface_temperature in range(450, 2001, 1):
        planets.append(Planet(surface_temperature=surface_temperature))

    # Stacks the entities into one named tuple
    planets_for_vmap = pytrees_stack(planets)

    parameters = Parameters(reaction_matrix, species, planets_for_vmap, scaling)
    # jax.debug.print("{out}", out=parameters)

    # For the same initial solution
    system_params = Solution(initial_solution, initial_solution)  # type: ignore
    # jax.debug.print("{out}", out=system_params)

    # JIT compile the solve function
    # jit_solve = jax.jit(solve)

    additional_for_vmap: Parameters = Parameters(
        reaction_matrix=None, species=None, planet=0, scaling=None  # type: ignore
    )

    vmap_solve: Callable = jax.vmap(
        solve,
        in_axes=(
            None,
            additional_for_vmap,
        ),
    )

    solutions: Array = vmap_solve(system_params, parameters)

    return solutions


def simple_system():

    # Species order is: H2, H2O, CO2, O2, CH4, CO
    species_list: list[SpeciesData] = [H2_g, CO_g, H2O_g, H2O_l, CO2_g, O2_g, CH4_g]

    reaction_network: ReactionNetworkJAX = ReactionNetworkJAX()

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
