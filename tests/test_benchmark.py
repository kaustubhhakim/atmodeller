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
"""Comparisons with FactSage 8.2 and FastChem 3.1.1"""

import logging

import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from atmodeller import INITIAL_LOG_NUMBER_DENSITY, INITIAL_LOG_STABILITY, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import Planet, SpeciesCollection
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""


def test_H_O(helper) -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint."""

    species: SpeciesCollection = SpeciesCollection.create(("H2_g", "H2O_g", "O2_g"))
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: ArrayLike = 6.25774e20
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
        "O": o_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    fastchem_result: dict[str, float] = {
        "H2O_g": 76.45861543,
        "H2_g": 73.84378192,
        "O2_g": 8.91399329e-08,
    }

    assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_reduced(helper) -> None:
    """Tests C-H-O system at IW-2

    Similar to :cite:p:`BHS22{Table E, row 1}`.
    """

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g")
    )
    planet: Planet = Planet(surface_temperature=1400.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(-2)}

    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
        "C": c_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "H2_g": 175.5,
        "H2O_g": 13.8,
        "CO_g": 6.21,
        "CO2_g": 0.228,
        "CH4_g": 38.07,
        "O2_g": 1.25e-15,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_IW(helper) -> None:
    """Tests C-H-O system at IW+0.5

    Similar to :cite:p:`BHS22{Table E, row 2}`.
    """

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g")
    )
    planet: Planet = Planet(surface_temperature=1400.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.5)}

    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
        "C": c_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "CH4_g": 28.66,
        "CO2_g": 30.88,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "H2_g": 236.98,
        "O2_g": 4.11e-13,
    }

    fastchem_result: dict[str, float] = {
        "CH4_g": 29.61919788,
        "CO2_g": 29.82548282,
        "CO_g": 45.94958264,
        "H2O_g": 332.03616807,
        "H2_g": 236.73845646,
        "O2_g": 3.96475584e-13,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_oxidised(helper) -> None:
    """Tests C-H-O system at IW+2

    Similar to :cite:p:`BHS22{Table E, row 3}`.
    """

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g")
    )
    planet: Planet = Planet(surface_temperature=1400.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(2)}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 0.1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
        "C": c_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "CH4_g": 0.00129,
        "CO2_g": 3.25,
        "CO_g": 0.873,
        "H2O_g": 218.48,
        "H2_g": 27.40,
        "O2_g": 1.29e-11,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_highly_oxidised(helper) -> None:
    """Tests C-H-O system at IW+4

    Similar to :cite:p:`BHS22{Table E, row 4}`.
    """

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g")
    )
    planet: Planet = Planet(surface_temperature=1400.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(4)}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 5 * h_kg
    # Mass of O that gives the same solution as applying the buffer at IW+4
    # o_kg: ArrayLike = 3.25196e21
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "CH4_g": 7.13e-05,
        "CO2_g": 357.23,
        "CO_g": 10.21,
        "H2O_g": 432.08,
        "H2_g": 5.78,
        "O2_g": 1.14e-09,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_middle_temperature(helper) -> None:
    """Tests C-H-O system at 873 K"""

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g")
    )
    planet: Planet = Planet(surface_temperature=873.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {
        "C": c_kg,
        "H": h_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "H2_g": 59.066,
        "H2O_g": 18.320,
        "CO_g": 8.91e-4,
        "CO2_g": 7.48e-4,
        "CH4_g": 19.548,
        "O2_g": 1.27e-25,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_low_temperature(helper) -> None:
    """Tests C-H-O system at 450 K"""

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2_g", "H2O_g", "CO_g", "CO2_g", "CH4_g", "O2_g")
    )
    planet: Planet = Planet(surface_temperature=450.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    o_kg: ArrayLike = 1.02999e20
    mass_constraints: dict[str, ArrayLike] = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_log_number_density: ArrayLike = jnp.ones(len(species)) * INITIAL_LOG_NUMBER_DENSITY
    # Correcting for the low CO and O2 helps the solver to latch onto the solution
    initial_log_number_density[2] = 30
    initial_log_number_density[5] = -INITIAL_LOG_NUMBER_DENSITY

    interior_atmosphere.solve(
        planet=planet,
        initial_log_number_density=initial_log_number_density,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "H2_g": 55.475,
        "H2O_g": 8.0,
        "CO2_g": 1.24e-14,
        "O2_g": 7.85e-54,
        "CH4_g": 16.037,
        "CO_g": 2.12e-16,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_condensed(helper) -> None:
    """Tests graphite stable with around 50% condensed C mass fraction"""

    species: SpeciesCollection = SpeciesCollection.create(
        ("O2_g", "H2_g", "CO_g", "H2O_g", "CO2_g", "CH4_g", "C_cr")
    )
    planet: Planet = Planet(surface_temperature=873.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 5 * h_kg
    o_kg: ArrayLike = 2.73159e19
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "O2_g": 1.27e-25,
        "H2_g": 14.564,
        "CO_g": 0.07276,
        "H2O_g": 4.527,
        "CO2_g": 0.061195,
        "CH4_g": 96.74,
        "C_cr_activity": 1.0,
        "mass_C_cr": 3.54162e20,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_unstable(helper) -> None:
    """Tests C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    species: SpeciesCollection = SpeciesCollection.create(
        ("O2_g", "H2_g", "CO_g", "H2O_g", "CO2_g", "CH4_g", "C_cr")
    )
    planet: Planet = Planet(surface_temperature=1400.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.5)}

    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "O2_g": 4.11e-13,
        "H2_g": 236.98,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "CO2_g": 30.88,
        "CH4_g": 28.66,
        "C_cr_activity": 0.12202,
        "mass_C_cr": 0.0,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_water_condensed(helper) -> None:
    """Condensed water at 10 bar"""

    species: SpeciesCollection = SpeciesCollection.create(("H2_g", "H2O_g", "O2_g", "H2O_l"))
    planet: Planet = Planet(surface_temperature=411.75)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 1.14375e21
    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_log_number_density: ArrayLike = INITIAL_LOG_NUMBER_DENSITY * jnp.ones(
        len(species), dtype=np.float_
    )
    # For this case, reducing the fO2 is required for the solver to latch onto the solution
    initial_log_number_density[2] = -INITIAL_LOG_NUMBER_DENSITY
    initial_log_stability: ArrayLike = INITIAL_LOG_STABILITY * jnp.ones(1)

    interior_atmosphere.solve(
        planet=planet,
        initial_log_number_density=initial_log_number_density,
        initial_log_stability=initial_log_stability,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "H2O_l_activity": 1.0,
        "mass_H2O_l": 1.247201e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_water_condensed(helper) -> None:
    """Tests C and water in equilibrium at 430 K and 10 bar"""

    species: SpeciesCollection = SpeciesCollection.create(
        ("H2O_g", "H2_g", "O2_g", "CO_g", "CO2_g", "CH4_g", "H2O_l", "C_cr")
    )
    planet: Planet = Planet(surface_temperature=430.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    o_kg: float = 2.48298883581636e21
    mass_constraints = {
        "C": c_kg,
        "H": h_kg,
        "O": o_kg,
    }

    interior_atmosphere.solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.output
    solution: dict[str, ArrayLike] = output.quick_look()

    factsage_result: dict[str, float] = {
        "CH4_g": 0.3241,
        "CO2_g": 4.3064,
        "CO_g": 2.77e-6,
        "C_cr_activity": 1.0,
        "H2O_g": 5.3672,
        "H2O_l_activity": 1.0,
        "H2_g": 0.0023,
        "O2_g": 4.74e-48,
        "mass_C_cr": 8.75101e19,
        "mass_H2O_l": 2.74821e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
