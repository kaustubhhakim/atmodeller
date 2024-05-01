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
"""Tests for condensed species and condensates"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ActivityConstraint,
    FugacityConstraint,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)


def get_water_system(temperature: float = 300) -> InteriorAtmosphereSystem:
    """Gets an interior-atmosphere system with water as a condensed species.

    Args:
        temperature: Temperature in kelvin
    """

    # Below to use 1 bar data for water
    # phase_data = db.getphasedata(filename="H-065")
    # Below to use 10 bar data for water
    # phase_data = db.getphasedata(filename="H-066")
    # Below to use 100 bar for water
    # phase_data = db.getphasedata(filename="H-067")

    species_with_water: Species = Species(
        [
            GasSpecies(formula="H2O"),
            GasSpecies(formula="H2"),
            GasSpecies(formula="O2"),
            LiquidSpecies(formula="H2O", thermodata_name="Water, 10 Bar"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = temperature
    interior_atmosphere_system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species_with_water, planet=planet
    )

    return interior_atmosphere_system


def test_water_condensed_100bar() -> None:
    """Tests including condensed water."""

    species_with_water: Species = Species(
        [
            GasSpecies(formula="H2O"),
            GasSpecies(formula="H2"),
            GasSpecies(formula="O2"),
            LiquidSpecies(formula="H2O", thermodata_name="Water, 100 Bar"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 550
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species_with_water, planet=planet
    )

    h_kg: float = earth_oceans_to_kg(2)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=53.71953115689841),
            MassConstraint(species="H", value=h_kg),
            ActivityConstraint(species="H2O", value=1),
        ]
    )

    # TODO: Update below when recomputed
    # Calculated by Paolo 19/02/2024
    # factsage_comparison: dict[str, float] = {
    #     "H2": 5.766,
    #     "H2O": 1.790,
    #     "O2": 1.268e-25,
    #     "degree_of_condensation_H": 0.513,
    # }

    target: dict[str, float] = {
        "H2O_g": 46.280468843101616,
        "H2_g": 53.71953115689844,
        "O2_g": 5.5324593476958213e-42,
        "H2O_l": 1.0,
        "degree_of_condensation_H": 0.6414542027845724,
    }

    system.solve(constraints)

    # TODO: Update
    # msg: str = "Compatible with FactSage result"
    # system.isclose_tolerance(factsage_comparison, msg)

    assert system.isclose(target, rtol=RTOL, atol=ATOL)
