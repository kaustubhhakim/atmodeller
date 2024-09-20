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
"""Tests for new JAX"""

# Convenient to use naming convention so pylint: disable=C0103

import logging

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, debug_logger  # pylint: disable=unused-import
from atmodeller.classes import ReactionNetwork
from atmodeller.jax_containers import (
    C_cr,
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
    SolverParameters,
    SpeciesData,
)
from atmodeller.jax_engine import get_log_extended_activity, solve
from atmodeller.jax_utilities import (
    pressure_from_log_number_density,
    scale_number_density,
    unscale_number_density,
)
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 1.0e-5
"""Tolerance of log output to satisfy comparison with FactSage"""

# Scale the numerical problem from molecules/m^3 to moles/m^3 if scaling is AVODAGRO
scaling: float = 1.0  # AVOGADRO
log_scaling: ArrayLike = np.log(scaling)


def test_CHO_low_temperature() -> None:
    """C-H-O system at 450 K"""

    species: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]
    planet: Planet = Planet(surface_temperature=450)
    tau: float = 1.0e60

    # Mass constraints in alphabetical order
    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg
    o_kg: float = 1.02999e20
    mass_constraints: dict[str, float] = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([60, 60, 30, -60, 60, 30], dtype=np.float_)
    logger.debug("initial_number_density = %s", initial_number_density)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    logger.debug("initial_stability = %s", initial_stability)

    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species))
    constraints: Constraints = Constraints.create(species, mass_constraints, log_scaling)

    initial_solution: Solution = Solution.create(
        initial_number_density, initial_stability, log_scaling
    )
    logger.debug("initial_solution = %s", initial_solution)

    solver_parameters: SolverParameters = SolverParameters(
        solver_class=optx.Newton, atol=ATOL, rtol=RTOL
    )
    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species, planet, constraints, tau, scaling
    )

    # Pre-compile
    # solve(initial_solution, parameters, solver_parameters).block_until_ready()

    scaled_solution: Array = solve(initial_solution, parameters, solver_parameters)
    logger.debug("scaled_solution = %s", scaled_solution)

    unscaled_solution: Array = unscale_number_density(scaled_solution, log_scaling)
    logger.debug("unscaled_solution = %s", unscaled_solution)
    number_density, stability = jnp.split(unscaled_solution, 2)

    pressure: Array = pressure_from_log_number_density(number_density, planet.surface_temperature)
    logger.debug("pressure = %s", pressure)

    log_extended_activity: Array = get_log_extended_activity(number_density, stability, parameters)
    logger.debug("log_extended_activity = %s", log_extended_activity)

    target: npt.NDArray[np.float_] = np.array(
        [
            62.05652013342668,
            60.120022576862524,
            26.01213296322353,
            -64.35958121411358,
            60.81547969099319,
            22.19204281838283,
        ]
    )

    factsage_result: npt.NDArray[np.float_] = np.array(
        [55.475, 8.0, 1.24e-14, 7.85e-54, 16.037, 2.12e-16]
    )

    isclose_target: np.bool_ = np.isclose(target, number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        factsage_result, pressure, rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_condensed() -> None:
    """Graphite stable with around 50% condensed C mass fraction"""

    species_list: list[SpeciesData] = [O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr]
    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species_list))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species_list))

    planet: Planet = Planet(surface_temperature=873)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 5 * h_kg
    # Below is oxygen when C_cr is not present
    # o_kg: float = 6.11072e20
    # This is when C_cr is present
    o_kg: float = 2.73159e19

    # Unscaled total molecules constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }
    constraints: Constraints = Constraints.create(species_list, mass_constraints)

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([1, 60, 60, 60, 60, 60, 60], dtype=np.float_)
    initial_number_density = scale_number_density(initial_number_density, log_scaling)
    logger.debug("initial_number_density = %s", initial_number_density)

    # Stability is a non-dimensional quantity
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density, dtype=np.float_)
    logger.debug("initial_stability = %s", initial_stability)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, scaling
    )

    # Pre-compile
    # solve(solution, parameters).block_until_ready()

    out = solve(solution, parameters)

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, log_scaling)

    log_extended_activity = get_log_extended_activity(number_density, stability, parameters)

    out = jnp.concatenate((number_density, stability))
    logger.debug("solution = %s", out)

    logger.debug("log_extended_activity = %s", log_extended_activity)

    out = jnp.exp(out)
    logger.debug("exp solution = %s", out)

    out = jnp.log10(out)
    logger.debug("log10 solution = %s", out)

    target: npt.NDArray = np.array(
        [
            -7.255672014277081e-02,
            6.006944857339484e01,
            5.471220406509149e01,
            5.881303434479700e01,
            5.443393222498312e01,
            6.186698032315073e01,
            6.171111646176269e01,
        ]
    )

    # factsage_result: dict[str, float] = {
    #     "O2_g": 1.27e-25,
    #     "H2_g": 14.564,
    #     "CO_g": 0.07276,
    #     "H2O_g": 4.527,
    #     "CO2_g": 0.061195,
    #     "CH4_g": 96.74,
    #     "activity_C_cr": 1.0,
    #     "mass_C_cr": 3.54162e20,
    # }

    isclose: np.bool_ = np.isclose(target, number_density, rtol=RTOL, atol=ATOL).all()

    assert isclose


def test_graphite_unstable() -> None:
    """C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    # This test requires tau_min scaling factor around 1.0e-17 and a solver other than Newton

    species_list: list[SpeciesData] = [H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g, C_cr]
    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species_list))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species_list))

    planet: Planet = Planet(surface_temperature=1400)

    h_kg: float = earth_oceans_to_hydrogen_mass(3)
    c_kg: float = 1 * h_kg
    o_kg: float = 2.57180041062295e21

    # Unscaled total molecules constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }
    constraints: Constraints = Constraints.create(species_list, mass_constraints)

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([60, 60, 60, 60, 60, 30, 5], dtype=np.float_)
    initial_number_density = scale_number_density(initial_number_density, log_scaling)
    logger.debug("initial_number_density = %s", initial_number_density)

    # Stability is a non-dimensional quantity
    initial_stability: ArrayLike = -25.0 * np.ones_like(initial_number_density, dtype=np.float_)
    logger.debug("initial_stability = %s", initial_stability)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, scaling
    )

    # Pre-compile
    # solve(solution, parameters).block_until_ready()

    out = solve(solution, parameters)

    logger.debug("solution out = %s", out)

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, log_scaling)

    log_extended_activity = get_log_extended_activity(number_density, stability, parameters)

    out = jnp.concatenate((number_density, stability))
    logger.debug("solution = %s", out)

    logger.debug("log_extended_activity = %s", log_extended_activity)

    out = jnp.exp(out)
    logger.debug("exp solution = %s", out)

    out = jnp.log10(out)
    logger.debug("log10 solution = %s", out)

    target: npt.NDArray = np.array(
        [
            62.3771834542796,
            62.70774275216296,
            60.72474193369101,
            60.31973944921249,
            60.28597770208155,
            28.37080234429034,
            10.168390100198557,
        ]
    )

    # factsage_result: dict[str, float] = {
    #     "O2_g": 4.11e-13,
    #     "H2_g": 236.98,
    #     "CO_g": 46.42,
    #     "H2O_g": 337.16,
    #     "CO2_g": 30.88,
    #     "CH4_g": 28.66,
    #     "activity_C_cr": 0.12202,
    #     # FactSage also predicts no C, so these values are set close to the atmodeller output so
    #     # the test knows to pass.
    #     "mass_C_cr": 941506.7454759097,
    # }

    isclose: np.bool_ = np.isclose(target, number_density, rtol=RTOL, atol=ATOL).all()

    assert isclose


def test_water_condensed_O_abundance() -> None:
    """Condensed water at 10 bar

    This is the same test as above, but this time constraining the total pressure and oxygen
    abundance.
    """

    species_list: list[SpeciesData] = [H2_g, H2O_g, O2_g, H2O_l]
    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species_list))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species_list))

    planet: Planet = Planet(surface_temperature=411.75)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    o_kg: float = 1.14375e21

    # Unscaled total molecules constraints in alphabetical order
    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }
    constraints: Constraints = Constraints.create(species_list, mass_constraints)

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([1, 60, 60, 60], dtype=np.float_)
    initial_number_density = scale_number_density(initial_number_density, log_scaling)
    logger.debug("initial_number_density = %s", initial_number_density)

    # Stability is a non-dimensional quantity
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density, dtype=np.float_)
    logger.debug("initial_stability = %s", initial_stability)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, scaling
    )

    out = solve(solution, parameters)

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, log_scaling)

    log_extended_activity = get_log_extended_activity(number_density, stability, parameters)

    out = jnp.concatenate((number_density, stability))
    logger.debug("solution = %s", out)

    logger.debug("log_extended_activity = %s", log_extended_activity)

    out = jnp.exp(out)
    logger.debug("exp solution = %s", out)

    out = jnp.log10(out)
    logger.debug("log10 solution = %s", out)

    target: npt.NDArray = np.array(
        [
            -7.255672014277081e-02,
            6.006944857339484e01,
            5.471220406509149e01,
            5.881303434479700e01,
            5.443393222498312e01,
            6.186698032315073e01,
            6.171111646176269e01,
        ]
    )

    # factsage_result: dict[str, float] = {
    #     "H2O_g": 3.3596,
    #     "H2_g": 6.5604,
    #     "O2_g": 5.6433e-58,
    #     "activity_H2O_l": 1.0,
    #     "mass_H2O_l": 1.247201e21,
    # }

    isclose: np.bool_ = np.isclose(target, number_density, rtol=RTOL, atol=ATOL).all()

    assert isclose


def test_graphite_water_condensed() -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    species_list: list[SpeciesData] = [H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr]
    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species_list))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species_list))

    planet: Planet = Planet(surface_temperature=430)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    # Specify O, because otherwise a total pressure can give rise to different solutions (with
    # different total O), making it more difficult to compare with a known comparison case.
    o_kg: float = 2.48298883581636e21

    # Unscaled total molecules constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }
    constraints: Constraints = Constraints.create(species_list, mass_constraints)

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array(
        [60, 60, -30, 60, 60, 60, 60, 60], dtype=np.float_
    )
    initial_number_density = scale_number_density(initial_number_density, log_scaling)
    logger.debug("initial_number_density = %s", initial_number_density)

    # Stability is a non-dimensional quantity
    initial_stability: ArrayLike = -40.0 * np.ones_like(initial_number_density, dtype=np.float_)
    logger.debug("initial_stability = %s", initial_stability)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, scaling
    )

    # Pre-compile
    # solve(solution, parameters).block_until_ready()

    out = solve(solution, parameters)

    logger.debug("solution out = %s", out)

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, log_scaling)

    log_extended_activity = get_log_extended_activity(number_density, stability, parameters)

    out = jnp.concatenate((number_density, stability))
    logger.debug("solution = %s", out)

    logger.debug("log_extended_activity = %s", log_extended_activity)

    out = jnp.exp(out)
    logger.debug("exp solution = %s", out)

    out = jnp.log10(out)
    logger.debug("log10 solution = %s", out)
