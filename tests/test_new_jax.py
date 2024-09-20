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
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage"""

SCALING: float = 1.0  # AVOGADRO
"""Scale the numerical problem from molecules/m^3 to moles/m^3 if SCALING is AVODAGRO"""
LOG_SCALING: ArrayLike = np.log(SCALING)
"""Log scaling"""

TAU: float = 1.0e60
"""Tau scaling factor for condensate stability"""


def test_CHO_low_temperature() -> None:
    """C-H-O system at 450 K"""

    species: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]
    planet: Planet = Planet(surface_temperature=450)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg
    o_kg: float = 1.02999e20
    # Mass constraints in alphabetical order
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
    constraints: Constraints = Constraints.create(species, mass_constraints, LOG_SCALING)

    initial_solution: Solution = Solution.create(
        initial_number_density, initial_stability, LOG_SCALING
    )
    logger.debug("initial_solution = %s", initial_solution)

    solver_parameters: SolverParameters = SolverParameters(
        solver_class=optx.Newton, atol=ATOL, rtol=RTOL
    )
    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species, planet, constraints, TAU, SCALING
    )

    # Pre-compile
    # solve(initial_solution, parameters, solver_parameters).block_until_ready()

    scaled_solution: Array = solve(initial_solution, parameters, solver_parameters)
    logger.debug("scaled_solution = %s", scaled_solution)

    unscaled_solution: Array = unscale_number_density(scaled_solution, LOG_SCALING)
    logger.debug("unscaled_solution = %s", unscaled_solution)
    log_number_density, log_stability = jnp.split(unscaled_solution, 2)

    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

    log_extended_activity: Array = get_log_extended_activity(
        log_number_density, log_stability, parameters
    )
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

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), np.log(pressure), rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_condensed() -> None:
    """Graphite stable with around 50% condensed C mass fraction"""

    species: list[SpeciesData] = [O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr]
    planet: Planet = Planet(surface_temperature=873)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 5 * h_kg
    o_kg: float = 2.73159e19
    # Mass constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([1, 60, 60, 60, 60, 60, 60], dtype=np.float_)
    logger.debug("initial_number_density = %s", initial_number_density)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    logger.debug("initial_stability = %s", initial_stability)

    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species))
    constraints: Constraints = Constraints.create(species, mass_constraints, LOG_SCALING)

    initial_solution: Solution = Solution.create(
        initial_number_density, initial_stability, LOG_SCALING
    )
    logger.debug("initial_solution = %s", initial_solution)

    solver_parameters: SolverParameters = SolverParameters(
        solver_class=optx.Newton, atol=ATOL, rtol=RTOL
    )
    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species, planet, constraints, TAU, SCALING
    )

    # Pre-compile
    # solve(initial_solution, parameters, solver_parameters).block_until_ready()

    scaled_solution: Array = solve(initial_solution, parameters, solver_parameters)
    logger.debug("scaled_solution = %s", scaled_solution)

    unscaled_solution: Array = unscale_number_density(scaled_solution, LOG_SCALING)
    logger.debug("unscaled_solution = %s", unscaled_solution)
    log_number_density, log_stability = jnp.split(unscaled_solution, 2)

    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

    log_extended_activity: Array = get_log_extended_activity(
        log_number_density, log_stability, parameters
    )
    logger.debug("log_extended_activity = %s", log_extended_activity)

    target: npt.NDArray[np.float_] = np.array(
        [
            -3.325643260498513e-03,
            6.010618973418672e01,
            5.474681960353263e01,
            5.888439104403002e01,
            5.450316330186538e01,
            6.194046264473452e01,
            6.177865797551661e01,
        ]
    )

    factsage_result: npt.NDArray[np.float_] = np.array(
        [
            1.27e-25,
            14.564,
            0.07276,
            4.527,
            0.061195,
            96.74,
            # Below is the "pressure" of the condensed species if it were a gas
            81.513,
            # "activity_C_cr": 1.0,
            # "mass_C_cr": 3.54162e20,
        ]
    )

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), np.log(pressure), rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_unstable() -> None:
    """C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    species: list[SpeciesData] = [O2_g, H2_g, H2O_g, CO_g, CO2_g, CH4_g, C_cr]
    planet: Planet = Planet(surface_temperature=1400)
    tau: float = 1.0e25

    h_kg: float = earth_oceans_to_hydrogen_mass(3)
    c_kg: float = 1 * h_kg
    o_kg: float = 2.57180041062295e21
    # Mass constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 60, 60, 60, 60, 60, 30], dtype=np.float_)
    logger.debug("initial_number_density = %s", initial_number_density)
    initial_stability: ArrayLike = -50.0 * np.ones_like(initial_number_density)
    logger.debug("initial_stability = %s", initial_stability)

    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species))
    constraints: Constraints = Constraints.create(species, mass_constraints, LOG_SCALING)

    initial_solution: Solution = Solution.create(
        initial_number_density, initial_stability, LOG_SCALING
    )
    logger.debug("initial_solution = %s", initial_solution)

    solver_options: dict[str, ArrayLike] = {
        "upper": jnp.array([70, 70, 70, 70, 70, 70, 70, 10, 10, 10, 10, 10, 10, 10]),
        "lower": jnp.array([1, 1, 1, 1, 1, 1, 1, -100, -100, -100, -100, -100, -100, -100]),
    }

    solver_parameters: SolverParameters = SolverParameters(
        solver_class=optx.Newton, atol=ATOL, rtol=RTOL, options=solver_options, max_steps=1000
    )
    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species, planet, constraints, tau, SCALING
    )

    # Pre-compile
    # solve(initial_solution, parameters, solver_parameters).block_until_ready()

    scaled_solution: Array = solve(initial_solution, parameters, solver_parameters)
    logger.debug("scaled_solution = %s", scaled_solution)

    unscaled_solution: Array = unscale_number_density(scaled_solution, LOG_SCALING)
    logger.debug("unscaled_solution = %s", unscaled_solution)
    log_number_density, log_stability = jnp.split(unscaled_solution, 2)

    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

    log_extended_activity: Array = get_log_extended_activity(
        log_number_density, log_stability, parameters
    )
    logger.debug("log_extended_activity = %s", log_extended_activity)

    target: npt.NDArray[np.float_] = np.array(
        [
            28.370802318309433,
            62.3771834542796,
            62.707742752162964,
            60.72474193369101,
            60.319739449212506,
            60.28597770208155,
            3.260634821216421,
        ]
    )

    factsage_result: npt.NDArray[np.float_] = np.array(
        [
            4.11e-13,
            236.98,
            337.16,
            46.42,
            30.88,
            28.66,
            5.03833e-24,
            # "activity_C_cr": 0.12202,
            # FactSage also predicts no C, so these values are set close to the atmodeller output so
            # the test knows to pass.
            # "mass_C_cr": 941506.7454759097,
        ]
    )

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), np.log(pressure), rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


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
    initial_number_density = scale_number_density(initial_number_density, LOG_SCALING)
    logger.debug("initial_number_density = %s", initial_number_density)

    # Stability is a non-dimensional quantity
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density, dtype=np.float_)
    logger.debug("initial_stability = %s", initial_stability)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, SCALING
    )

    out = solve(solution, parameters)

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, LOG_SCALING)

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
    initial_number_density = scale_number_density(initial_number_density, LOG_SCALING)
    logger.debug("initial_number_density = %s", initial_number_density)

    # Stability is a non-dimensional quantity
    initial_stability: ArrayLike = -40.0 * np.ones_like(initial_number_density, dtype=np.float_)
    logger.debug("initial_stability = %s", initial_stability)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, SCALING
    )

    # Pre-compile
    # solve(solution, parameters).block_until_ready()

    out = solve(solution, parameters)

    logger.debug("solution out = %s", out)

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, LOG_SCALING)

    log_extended_activity = get_log_extended_activity(number_density, stability, parameters)

    out = jnp.concatenate((number_density, stability))
    logger.debug("solution = %s", out)

    logger.debug("log_extended_activity = %s", log_extended_activity)

    out = jnp.exp(out)
    logger.debug("exp solution = %s", out)

    out = jnp.log10(out)
    logger.debug("log10 solution = %s", out)
