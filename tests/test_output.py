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
"""Tests for the output"""

# Want to use chemistry symbols so pylint: disable=invalid-name

from __future__ import annotations

import logging

import pytest

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ElementMassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies, Species
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
from atmodeller.jax_containers import Planet
from atmodeller.solution import ELEMENT_PREFIX

logger: logging.Logger = debug_logger()

TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage"""


def test_graphite_water_condensed_output(graphite_water_condensed) -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    system = graphite_water_condensed

    output = system.output(to_dict=True, to_excel=True)

    assert 9.89452e18 == pytest.approx(output[f"{ELEMENT_PREFIX}C"][0]["atmosphere_mass"])
    assert 9.81055e19 == pytest.approx(output[f"{ELEMENT_PREFIX}C"][0]["condensed_mass"])
    assert 3.9873e19 == pytest.approx(output[f"{ELEMENT_PREFIX}O"][0]["atmosphere_mass"])
    assert 2.4431158e21 == pytest.approx(output[f"{ELEMENT_PREFIX}O"][0]["condensed_mass"])
    assert 2.17398e18 == pytest.approx(output[f"{ELEMENT_PREFIX}H"][0]["atmosphere_mass"])
    assert 3.07826e20 == pytest.approx(output[f"{ELEMENT_PREFIX}H"][0]["condensed_mass"])


@pytest.mark.skip("Need to fix C-H-O condensed phase partitioning")
def test_trappist_output() -> None:

    surface_temperature = 400  # K

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O")
    C_cr: SolidSpecies = SolidSpecies("C")

    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, H2O_l, CH4_g, C_cr])

    mantle_mass = 2.912e24
    planet_mass = mantle_mass / (1 - 0.295334691460966)
    trappist1e = Planet(
        surface_temperature=surface_temperature, planet_mass=planet_mass, surface_radius=5.861e6
    )

    h_kg: float = 2.907214018995482e19
    c_kg: float = 6.822441980162111e20
    o_kg: float = 1.2751814985260409e21

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("O", o_kg),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            TotalPressureConstraint(2.5),
        ]
    )

    system = InteriorAtmosphereSystem(species=species, planet=trappist1e)
    system.solve(constraints=constraints)
