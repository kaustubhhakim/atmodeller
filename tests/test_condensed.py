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
    ElementMassConstraint,
    FugacityConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)


def test_water_condensed_100bar() -> None:
    """Tests including condensed water."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 100 Bar")

    species: Species = Species([H2O_g, H2_g, O2_g, H2O_l])

    planet: Planet = Planet()
    planet.surface_temperature = 550
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(2)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=53.71953115689841),
            ElementMassConstraint("H", h_kg),
            ActivityConstraint(H2O_l, 1),
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
