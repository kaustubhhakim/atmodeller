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
"""Tests for condensates"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintBallhaus,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.core import (
    GasSpecies,
    SolidSpecies,
    ThermodynamicDatasetABC,
    ThermodynamicDatasetJANAF,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)


def test_graphite() -> None:
    """Tests including graphite."""

    thermodynamic_data: ThermodynamicDatasetABC = ThermodynamicDatasetJANAF()

    species: Species = Species(
        [
            GasSpecies(formula="H2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="H2O", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CO", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CO2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CH4", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="O2", thermodynamic_dataset=thermodynamic_data),
            SolidSpecies(
                formula="C",
                thermodynamic_dataset=thermodynamic_data,
                name_in_dataset="graphite",
            ),
        ]
    )

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=44.49334998176607),
        ]
    )

    target_pressures: dict[str, float] = {
        "C": 1.0,
        "CH4": 900.3912797397132,
        "CO": 0.07741709702165529,
        "CO2": 0.0685518295157825,
        "H2": 44.493349981766045,
        "H2O": 14.708340036418534,
        "O2": 1.4458158511932372e-25,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_graphite_mass() -> None:
    """Tests including graphite and mass balance."""

    thermodynamic_data: ThermodynamicDatasetABC = ThermodynamicDatasetJANAF()

    species: Species = Species(
        [
            GasSpecies(formula="H2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="H2O", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CO", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CO2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CH4", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="O2", thermodynamic_dataset=thermodynamic_data),
            SolidSpecies(
                formula="C",
                thermodynamic_dataset=thermodynamic_data,
                name_in_dataset="graphite",
            ),
        ]
    )

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=16),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    target_pressures: dict[str, float] = {
        "C": 1.0,
        "CH4": 116.43432612508063,
        "CO": 0.07288629883964833,
        "CO2": 0.0607627023198603,
        "H2": 16.000000000000007,
        "H2O": 4.979635491972861,
        "O2": 1.2815365949520625e-25,
    }

    system.solve(constraints)

    print(system.solution)

    # assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)
