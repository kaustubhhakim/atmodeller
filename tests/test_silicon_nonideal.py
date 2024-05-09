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
"""Tests for non-ideal systems with silicon"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging

import pytest

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ActivityConstraint,
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies, LiquidSpecies
from atmodeller.eos.interfaces import RealGas
from atmodeller.eos.saxena import get_saxena_eos_models
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubility.hydrogen_species import (
    H2_basalt_hirschmann,
    H2O_peridotite_sossi,
)
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_saxena_eos_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


@pytest.mark.skip(reason="with condensed species mass balance another constraint is now required")
def test_SiHO_massSiH_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"])
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    oceans: float = 50
    sih_ratio: float = 1
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)
    si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("Si", si_kg),
            ActivityConstraint(SiO2_l, 1),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2_g": 3955.5461365393257,
        "H2O_g": 238.6969674984362,
        "O2_g": 8.041858631374242e-05,
        "OSi_g": 297.7208780281457,
        "H4Si_g": 3.8854445389482284,
        "O2Si_l": 1.0,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="with condensed species mass balance another constraint is now required")
def test_SiHO_massSiH_solubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"], solubility=H2_basalt_hirschmann())
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    oceans: float = 50
    sih_ratio: float = 1
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)
    si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("Si", si_kg),
            ActivityConstraint(SiO2_l, 1),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2_g": 2225.387378761475,
        "H2O_g": 68.65564430638824,
        "O2_g": 2.2005226675444582e-05,
        "OSi_g": 569.1471756022945,
        "H4Si_g": 4.292996959380682,
        "O2Si_l": 1.0,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_SiHO_massH_logfO2_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"])
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    oceans: float = 50
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ActivityConstraint(SiO2_l, 1),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O_g": 4985.13674243621,
        "H2_g": 3503.0246920397244,
        "H4Si_g": 0.009461878462110699,
        "O2Si_l": 1.0,
        "O2_g": 0.03403441441997074,
        "OSi_g": 14.471993770955361,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_SiHO_massH_logfO2_solubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"], solubility=H2_basalt_hirschmann())
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    oceans: float = 50
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ActivityConstraint(SiO2_l, 1),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2_g": 337.17023763048417,
        "H2O_g": 378.88646918406965,
        "O2_g": 0.025481748164747447,
        "OSi_g": 16.72526080495589,
        "H4Si_g": 9.750334904890755e-05,
        "O2Si_l": 1.0,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_SiHO_totalpressure_logfO2_nosolubility() -> None:
    """Tests H2-H2O and SiO2-SiO-SiH4."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"])
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    planet: Planet = Planet(surface_temperature=3400)

    constraints: SystemConstraints = SystemConstraints(
        [
            TotalPressureConstraint(4000),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ActivityConstraint(SiO2_l, 1),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O_g": 2143.367766318033,
        "H2_g": 1840.8630969974558,
        "H4Si_g": 0.0024461970964935434,
        "O2Si_l": 1.0,
        "O2_g": 0.02877934210919358,
        "OSi_g": 15.737910724463843,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_SiHO_fugacityH2O_logfO2_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"])
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    planet: Planet = Planet(surface_temperature=3400)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2O_g, 5000),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ActivityConstraint(SiO2_l, 1),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O_g": 4999.999999999999,
        "H2_g": 3509.5924241807747,
        "H4Si_g": 0.009503203994822516,
        "O2Si_l": 1.0,
        "O2_g": 0.03406158617252535,
        "OSi_g": 14.466220289280406,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":

    test_SiHO_massSiH_nosolubility()
    test_SiHO_massSiH_solubility()
    test_SiHO_massH_logfO2_nosolubility()
    test_SiHO_massH_logfO2_solubility()
    test_SiHO_totalpressure_logfO2_nosolubility()
    test_SiHO_fugacityH2O_logfO2_nosolubility()
