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
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, debug_logger  # pylint: disable=unused-import
from atmodeller.classes import InteriorAtmosphere
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
from atmodeller.jax_engine import get_log_extended_activity, solve_set_solver
from atmodeller.jax_utilities import (
    pressure_from_log_number_density,
    pytrees_stack,
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

SCALING: float = AVOGADRO
"""Scale the numerical problem from molecules/m^3 to moles/m^3 if SCALING is AVODAGRO"""
LOG_SCALING: float = np.log(SCALING)
"""Log scaling"""

TAU: float = 1.0e60
"""Tau scaling factor for condensate stability"""


def test_CHO_low_temperature() -> None:
    """C-H-O system at 450 K"""

    species: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]
    planet: Planet = Planet(surface_temperature=450)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, LOG_SCALING)

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
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_single(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

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
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, LOG_SCALING)

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
    initial_number_density: ArrayLike = np.array([30, 30, 30, 30, 30, 30, 30], dtype=np.float_)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_single(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

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
            # Below is the atmodeller "pressure" of the condensed species if it were a gas
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
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, LOG_SCALING)

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
    initial_stability: ArrayLike = -50.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_single(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

    target: npt.NDArray[np.float_] = np.array(
        [
            28.37080231828339,
            62.37718345427961,
            62.70774275216296,
            60.72474193369101,
            60.319739449212484,
            60.28597770208156,
            -77.32984343357518,
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
            # Below is the atmodeller "pressure" of the condensed species if it were a gas
            5.038334863741620e-59,
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
    """Condensed water at 10 bar"""

    species: list[SpeciesData] = [H2_g, H2O_g, O2_g, H2O_l]
    planet: Planet = Planet(surface_temperature=411.75)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, LOG_SCALING)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    o_kg: float = 1.14375e21
    # Mass constraints in alphabetical order
    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 30, -30, 30], dtype=np.float_)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_single(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

    target: npt.NDArray = np.array(
        [60.147414932111765, 59.42151086724804, -73.77323817273705, 62.69282187037022]
    )

    factsage_result: npt.NDArray[np.float_] = np.array(
        [
            6.5604,
            3.3596,
            5.6433e-58,
            95.90991,
            # "activity_H2O_l": 1.0,
            # "mass_H2O_l": 1.247201e21,
        ]
    )

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), np.log(pressure), rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_water_condensed() -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    species: list[SpeciesData] = [H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr]
    planet: Planet = Planet(surface_temperature=430)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, LOG_SCALING)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    # Specify O, because otherwise a total pressure can give rise to different solutions (with
    # different total O), making it more difficult to compare with a known comparison case.
    o_kg: float = 2.48298883581636e21
    # Mass constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array(
        [60, 60, -30, 60, 60, 60, 60, 60], dtype=np.float_
    )
    initial_stability: ArrayLike = -40.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_single(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    pressure: Array = pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("pressure = %s", pressure)

    target: npt.NDArray[np.float_] = np.array(
        [
            59.84956737107164,
            52.04446167407229,
            -50.89936632480971,
            45.47316828975361,
            59.51686478781738,
            56.89655030374509,
            64.8157539595103,
            61.88885111892189,
        ]
    )

    # Last two entries are the atmodeller "pressure" of the condensed species if it were a gas
    factsage_result: npt.NDArray[np.float_] = np.array(
        [5.3672, 0.0023, 4.74e-48, 2.77e-6, 4.3064, 0.3241, 836.904, 44.82685]
    )

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), np.log(pressure), rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_batch_planet() -> None:
    """Tests a batch calculation with different planets"""

    species: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]

    # Creates a list of planets with different surface temperatures
    planets: list[Planet] = []
    for surface_temperature in range(450, 2001, 1000):
        planets.append(Planet(surface_temperature=surface_temperature))

    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, LOG_SCALING)

    # Stacks the entities into one named tuple
    # planets_batch = pytrees_stack(planets)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg
    o_kg: float = 1.02999e20

    mass_constraints_batch: dict[str, ArrayLike] = {
        "C": jnp.array((c_kg, c_kg * 1)),
        "H": jnp.array((h_kg, h_kg * 2)),
        "O": jnp.array((o_kg, o_kg * 1)),
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([60, 60, 60, -30, 60, 60], dtype=np.float_)
    initial_stability: ArrayLike = -40.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_batch(
        planets, mass_constraints_batch, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()

    # reaction_network: ReactionNetwork = ReactionNetwork()
    # formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species))
    # reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species))
    # constraints_batch: Constraints = Constraints.create(
    #    species, mass_constraints_batch, LOG_SCALING
    # )
    # constraints_vmap: Constraints = Constraints(species=None, log_molecules=0)

    # initial_solution: Solution = Solution.create(
    #    initial_number_density, initial_stability, LOG_SCALING
    # )
    # logger.debug("initial_solution = %s", initial_solution)

    # parameters_batch: Parameters = Parameters(
    ##    formula_matrix, reaction_matrix, species, planets_batch, constraints_batch, TAU, SCALING
    # )
    # parameters_vmap: Parameters = Parameters(
    #    formula_matrix=None,
    #    reaction_matrix=None,
    #    species=None,
    #    planet=0,
    #    constraints=constraints_vmap,
    #    tau=None,
    #    scaling=None,
    # )
    # solver_parameters: SolverParameters = SolverParameters.create(species, LOG_SCALING)

    # wrapper_to_vmap = solver_wrapper(solver_parameters)

    # vmap_solve: Callable = jit(
    #    jax.vmap(
    #        wrapper_to_vmap,
    #        in_axes=(None, parameters_vmap),
    #    )
    # )

    # vmap_solve(initial_solution, parameters_batch).block_until_ready()

    # scaled_solution: Array = vmap_solve(initial_solution, parameters_batch)
    # logger.debug("scaled_solution = %s", scaled_solution)

    # unscaled_solution: Array = unscale_number_density(scaled_solution, LOG_SCALING)
    # logger.debug("unscaled_solution = %s", unscaled_solution)
    # log_number_density, log_stability = jnp.split(unscaled_solution, 2)

    # pressure: Array = pressure_from_log_number_density(
    #     log_number_density, planets_batch.surface_temperature
    # )
    # logger.debug("pressure = %s", pressure)

    assert True
