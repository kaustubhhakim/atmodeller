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
"""Tests for ideal C-H-O interior-atmosphere systems

In particular, these test the stable/unstable condensates algorithm. Note that condensed species
must be at the end of the species list.
"""

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
    """Graphite stable with around 50% condensed C mass fraction"""

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

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 5 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            ActivityConstraint(C_cr, 1),
        ]
    )

    factsage_result: dict[str, float] = {
        "O2_g": 1.27e-25,
        "H2_g": 14.564,
        "CO_g": 0.07276,
        "H2O_g": 4.527,
        "CO2_g": 0.061195,
        "CH4_g": 96.74,
        "C_cr": 1.0,
        "degree_of_condensation_C": 0.456983,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_IW(helper) -> None:
    """C-H-O system at IW+0.5

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g, C_cr])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            ActivityConstraint(C_cr, 1),
        ]
    )

    factsage_result: dict[str, float] = {
        "O2_g": 4.11e-13,
        "H2_g": 236.98,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "CO2_g": 30.88,
        "CH4_g": 28.66,
        "C_cr": 0.12202,
        # FactSage also predicts no C, so these values are set close to the atmodeller output so
        # the test knows to pass.
        "degree_of_condensation_C": 1.1e-15,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
