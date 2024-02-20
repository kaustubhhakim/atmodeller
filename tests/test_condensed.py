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
    FugacityConstraint,
    IronWustiteBufferConstraintBallhaus,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)


def get_graphite_system(temperature: float = 873) -> InteriorAtmosphereSystem:
    """Gets an interior-atmosphere system with graphite as a condensed species.

    Args:
        temperature: Temperature in kelvin
    """

    species_with_graphite: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
            SolidSpecies(formula="C", name_in_dataset="graphite"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = temperature
    interior_atmosphere_system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species_with_graphite, planet=planet
    )

    return interior_atmosphere_system


def test_graphite() -> None:
    """Tests including graphite as a condensed phase with only reaction network constraints."""

    system: InteriorAtmosphereSystem = get_graphite_system(temperature=873)

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=44.49334998176607),
        ]
    )

    target: dict[str, float] = {
        "H2": 44.493349981766045,
        "H2O": 14.708340036418534,
        "CO": 0.07741709702165529,
        "CO2": 0.0685518295157825,
        "CH4": 900.3912797397132,
        "O2": 1.4458158511932372e-25,
        "C": 1.0,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_graphite_half_condensed() -> None:
    """Tests including graphite with around 50% condensed C mass fraction."""

    system: InteriorAtmosphereSystem = get_graphite_system(temperature=873)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=5.8),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    # Calculated by Paolo 19/02/2024
    factsage_comparison: dict[str, float] = {
        "H2": 5.766,
        "H2O": 1.790,
        "CO": 0.072,
        "CO2": 0.060,
        "CH4": 15.33,
        "O2": 1.268e-25,
        "C": 1.0,
        "degree_of_condensation_C": 0.513,
    }

    target: dict[str, float] = {
        "H2": 5.799999999999998,
        "H2O": 1.7900177926108025,
        "CO": 0.07227659435955698,
        "CO2": 0.05975037656249113,
        "CH4": 15.300198167373871,
        "O2": 1.2601857916706088e-25,
        "C": 1.0,
        "degree_of_condensation_C": 0.5136921235780504,
    }

    system.solve(constraints)

    msg: str = "Compatible with FactSage result"
    system.isclose_tolerance(factsage_comparison, msg)

    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_graphite_no_condensed() -> None:
    """Tests including graphite with near 0% condensed C mass fraction."""

    system: InteriorAtmosphereSystem = get_graphite_system(temperature=873)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=8.55),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    # Calculated by Paolo 19/02/2024
    factsage_comparison: dict[str, float] = {
        "H2": 8.497,
        "H2O": 2.644,
        "CO": 0.0721,
        "CO2": 0.0607,
        "CH4": 33.29,
        "O2": 1.274e-25,
        "C": 1.0,
        "degree_of_condensation_C": 0.0057,
    }

    target: dict[str, float] = {
        "H2": 8.549999999999999,
        "H2O": 2.64290770472214,
        "CO": 0.07239093876960544,
        "CO2": 0.05993958099197688,
        "CH4": 33.24859502171367,
        "O2": 1.2641762725257323e-25,
        "C": 1.0,
        "degree_of_condensation_C": 0.006696462037687005,
    }

    system.solve(constraints)

    msg: str = "Compatible with FactSage result"
    system.isclose_tolerance(factsage_comparison, msg)

    assert system.isclose(target, rtol=RTOL, atol=ATOL)
