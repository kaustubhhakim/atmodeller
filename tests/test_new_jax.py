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

import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, debug_logger  # pylint: disable=unused-import
from atmodeller.classes import (
    InteriorAtmosphere,
    InteriorAtmosphereABC,
    InteriorAtmosphereBatch,
)
from atmodeller.jax_containers import (
    C_cr_data,
    CH4_g_data,
    CO2_g_data,
    CO_g_data,
    CondensedSpecies,
    FugacityConstraints,
    GasSpecies,
    H2_g_data,
    H2O_g_data,
    H2O_l_data,
    O2_g_data,
    Planet,
    Species,
)
from atmodeller.jax_utilities import log_pressure_from_log_number_density
from atmodeller.solubility.jax_hydrogen_species import H2O_peridotite_sossi
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
"""Tau scaling factor for species stability"""


def test_CHO_low_temperature() -> None:
    """C-H-O system at 450 K"""

    H2_g: Species = GasSpecies(H2_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    CO2_g: Species = GasSpecies(CO2_g_data)
    O2_g: Species = GasSpecies(O2_g_data)
    CH4_g: Species = GasSpecies(CH4_g_data)
    CO_g: Species = GasSpecies(CO_g_data)

    species: list[Species] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]
    planet: Planet = Planet(surface_temperature=450.0)
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

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
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: npt.NDArray[np.float_] = np.array(
        [
            62.05652013342669,
            60.120022576862524,
            26.068387536122835,
            -64.2330163272133,
            60.81547969099319,
            22.015508278236624,
        ]
    )

    factsage_result: npt.NDArray[np.float_] = np.array(
        [55.475, 8.0, 1.24e-14, 7.85e-54, 16.037, 2.12e-16]
    )

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), log_pressure, rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_condensed() -> None:
    """Graphite stable with around 50% condensed C mass fraction"""

    O2_g: Species = GasSpecies(O2_g_data)
    H2_g: Species = GasSpecies(H2_g_data)
    CO_g: Species = GasSpecies(CO_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    CO2_g: Species = GasSpecies(CO2_g_data)
    CH4_g: Species = GasSpecies(CH4_g_data)
    C_cr: Species = CondensedSpecies(C_cr_data)

    species: list[Species] = [O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr]
    planet: Planet = Planet(surface_temperature=873.0)
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 5 * h_kg
    o_kg: float = 2.73159e19
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 30, 30, 30, 30, 30, 30], dtype=np.float_)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: npt.NDArray[np.float_] = np.array(
        [
            4.901997908084610e-02,
            6.006235399383993e01,
            5.475577477744803e01,
            5.888892217538084e01,
            5.456276038698600e01,
            6.195032674577313e01,
            6.177996484193057e01,
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
        np.log(factsage_result), log_pressure, rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_unstable() -> None:
    """C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    O2_g: Species = GasSpecies(O2_g_data)
    H2_g: Species = GasSpecies(H2_g_data)
    CO_g: Species = GasSpecies(CO_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    CO2_g: Species = GasSpecies(CO2_g_data)
    CH4_g: Species = GasSpecies(CH4_g_data)
    C_cr: Species = CondensedSpecies(C_cr_data)

    species: list[Species] = [O2_g, H2_g, H2O_g, CO_g, CO2_g, CH4_g, C_cr]
    planet: Planet = Planet(surface_temperature=1400.0)
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

    h_kg: float = earth_oceans_to_hydrogen_mass(3)
    c_kg: float = 1 * h_kg
    o_kg: float = 2.57180041062295e21
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 60, 60, 60, 60, 60, 30], dtype=np.float_)
    initial_stability: ArrayLike = -50.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: npt.NDArray[np.float_] = np.array(
        [
            28.363706985121755,
            62.371226090440025,
            62.71169832106752,
            60.73335397022391,
            60.29832575363264,
            60.297836444468736,
            -77.33522428358012,
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
        np.log(factsage_result), log_pressure, rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_water_condensed_O_abundance() -> None:
    """Condensed water at 10 bar"""

    H2_g: Species = GasSpecies(H2_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    O2_g: Species = GasSpecies(O2_g_data)
    H2O_l: Species = CondensedSpecies(H2O_l_data)

    species: list[Species] = [H2_g, H2O_g, O2_g, H2O_l]
    planet: Planet = Planet(surface_temperature=411.75)
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    o_kg: float = 1.14375e21
    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 30, -30, 30], dtype=np.float_)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: npt.NDArray = np.array(
        [60.12043903355293, 59.339240557109925, -73.72900084543642, 62.66788570903491]
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
        np.log(factsage_result), log_pressure, rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_graphite_water_condensed() -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    O2_g: Species = GasSpecies(O2_g_data)
    H2_g: Species = GasSpecies(H2_g_data)
    CO_g: Species = GasSpecies(CO_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    CO2_g: Species = GasSpecies(CO2_g_data)
    CH4_g: Species = GasSpecies(CH4_g_data)
    C_cr: Species = CondensedSpecies(C_cr_data)
    H2O_l: Species = CondensedSpecies(H2O_l_data)

    species: list[Species] = [H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr]
    planet: Planet = Planet(surface_temperature=430.0)
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    # Specify O, because otherwise a total pressure can give rise to different solutions (with
    # different total O), making it more difficult to compare with a known comparison case.
    o_kg: float = 2.48298883581636e21
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
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: npt.NDArray[np.float_] = np.array(
        [
            59.777593890329186,
            52.03357698384461,
            -50.87409900184007,
            45.301391021053625,
            59.53599162929261,
            56.92008067039274,
            64.83514439811483,
            61.907576626175874,
        ]
    )

    # Last two entries are the atmodeller "pressure" of the condensed species if it were a gas
    factsage_result: npt.NDArray[np.float_] = np.array(
        [5.3672, 0.0023, 4.74e-48, 2.77e-6, 4.3064, 0.3241, 836.904, 44.82685]
    )

    isclose_target: np.bool_ = np.isclose(target, log_number_density, rtol=RTOL, atol=ATOL).all()
    isclose_factsage: np.bool_ = np.isclose(
        np.log(factsage_result), log_pressure, rtol=TOLERANCE, atol=TOLERANCE
    ).all()

    assert isclose_target
    assert isclose_factsage


def test_batch_planet() -> None:
    """Tests a batch calculation with different planets"""

    O2_g: Species = GasSpecies(O2_g_data)
    H2_g: Species = GasSpecies(H2_g_data)
    CO_g: Species = GasSpecies(CO_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    CO2_g: Species = GasSpecies(CO2_g_data)
    CH4_g: Species = GasSpecies(CH4_g_data)

    species: list[Species] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]

    # Creates a list of planets with different surface temperatures
    planet_list: list[Planet] = []
    for surface_temperature in range(450, 2001, 1000):
        planet_list.append(Planet(surface_temperature=float(surface_temperature)))

    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphereBatch(species, LOG_SCALING)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg
    o_kg: float = 1.02999e20

    # Creates a list of mass constraints
    mass_constraints1: dict[str, float] = {"C": c_kg, "H": h_kg, "O": o_kg}
    mass_constraints2: dict[str, float] = {"C": c_kg, "H": h_kg * 2, "O": o_kg}
    mass_constraints_list: list[dict[str, float]] = [mass_constraints1, mass_constraints2]

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([60, 60, 60, -30, 60, 60], dtype=np.float_)
    initial_stability: ArrayLike = -40.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet_list, mass_constraints_list, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()

    # pressure: Array = log_pressure_from_log_number_density(
    #     log_number_density, planets_batch.surface_temperature
    # )
    # logger.debug("pressure = %s", pressure)

    assert True


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer."""

    O2_g: Species = GasSpecies(O2_g_data)
    H2_g: Species = GasSpecies(H2_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data, solubility=H2O_peridotite_sossi)

    species: list[Species] = [H2O_g, H2_g, O2_g]
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 1.22814e21
    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 30, 30], dtype=np.float_)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: dict[str, float] = {
        "H2O_g": 0.2570770067190733,
        "H2_g": 0.24964688044710354,
        "O2_g": 8.838043080858959e-08,
    }

    # FIXME: Setup test comparison
    assert True


# TODO: Implement buffer
def test_H_fO2_buffer() -> None:
    """Tests H2-H2O at the IW buffer."""

    O2_g: Species = GasSpecies(O2_g_data)
    H2_g: Species = GasSpecies(H2_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data, solubility=H2O_peridotite_sossi)

    species: list[Species] = [H2O_g, H2_g, O2_g]
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphereABC = InteriorAtmosphere(species, LOG_SCALING)

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 1.22814e21

    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }
    fugacity_constraints: FugacityConstraints = FugacityConstraints.create(
        {"O2_g": 8.838043080858959e-08}
    )

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = np.array([30, 30, 30], dtype=np.float_)
    initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
    interior_atmosphere.initialise_solve(
        planet, mass_constraints, initial_number_density, initial_stability
    )
    log_number_density, _ = interior_atmosphere.solve()
    log_pressure: Array = log_pressure_from_log_number_density(
        log_number_density, planet.surface_temperature
    )
    logger.debug("log_pressure = %s", log_pressure)

    target: dict[str, float] = {
        "H2O_g": 0.2570770067190733,
        "H2_g": 0.24964688044710354,
        "O2_g": 8.838043080858959e-08,
    }

    # FIXME: Setup test comparison
    assert True
