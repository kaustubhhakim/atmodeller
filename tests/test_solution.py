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
"""Tests for ideal C-H-O interior-atmosphere systems"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ActivityConstraint,
    BufferedFugacityConstraint,
    ElementMassConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_kg

TOLERANCE: float = 5.0e-2

logger: logging.Logger = debug_logger()


def test_C_half_condensed(helper) -> None:
    """Graphite with 50% condensed C mass fraction"""

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr])

    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    # TODO: These were updated compared to previous FactSage comparison
    h_kg: float = earth_oceans_to_kg(1)  # earth_oceans_to_kg(0.201991413)
    c_kg: float = 5 * h_kg  # * (1 - 0.4569851350481659)  # 4.950705503505735 * h_kg
    # h_kg = c_kg * 0.201991413  # 3.13087e19
    # print(h_kg / earth_oceans_to_kg(1))

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            # TODO: Should no longer be required to specify
            # ActivityConstraint(C_cr, 1),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 96.74,
        "CO2_g": 0.061195,
        "CO_g": 0.07276,
        "C_cr": 1.0,
        "H2O_g": 4.527,
        "H2_g": 14.564,
        "O2_g": 1.27e-25,
        "degree_of_condensation_C": 0.456983,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
