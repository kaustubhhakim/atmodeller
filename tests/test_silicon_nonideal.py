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

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    SystemConstraints,
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


def test_SiHO_nomass_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4 without solubility."""

    H2_g: GasSpecies = GasSpecies("H2", eos=eos_models["H2"])
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    OSi_g: GasSpecies = GasSpecies("OSi")
    H4Si_g: GasSpecies = GasSpecies("H4Si")
    SiO2_l: LiquidSpecies = LiquidSpecies("SiO2")

    species: Species = Species([H2_g, H2O_g, O2_g, OSi_g, H4Si_g, SiO2_l])

    oceans: float = 50
    # sih_ratio: float = 1
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)
    # si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-1)),
            ElementMassConstraint("H", h_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 1566.940828894344,
        "H2_g": 3687.2119791568507,
        "H4Si_g": 0.08137365652206581,
        "O2Si_l": 1.0,
        "O2_g": 0.0036478802315568536,
        "OSi_g": 44.20457534094316,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_SiHO_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4 without solubility."""

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
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("Si", si_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 441.4119909374954,
        "H2_g": 3453.23970638334,
        "H4Si_g": 0.7158268256935626,
        "O2Si_l": 0.9999999999999976,
        "O2_g": 0.00034647350668364326,
        "OSi_g": 143.4341263480178,
        "degree_of_condensation_Si": 0.48453008967072864,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_SiHO_solubility() -> None:
    """Tests H2-H2O and SiO-SiH4 with solubility."""

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
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("Si", si_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O_g": 147.9110778505931,
        "H2_g": 1204.9127339913045,
        "H4Si_g": 0.09907503881558258,
        "O2Si_l": 0.9999999999999987,
        "O2_g": 0.00031206744561192417,
        "OSi_g": 151.13438225949093,
        "degree_of_condensation_Si": 0.635903850843704,
    }

    system.solve(constraints)
    assert system.solution.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_SiHO_fugacityH2O_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4 without solubility and a fH2O constraint."""

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
            FugacityConstraint(H2O_g, 5000),  # Same as partial pressure since H2O is ideal
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("Si", si_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O_g": 4999.9999999999945,
        "H2_g": 3499.4281733735734,
        "H4Si_g": 0.00938097194089631,
        "O2Si_l": 1.0,
        "O2_g": 0.03428277557575392,
        "OSi_g": 14.419477391994004,
        "degree_of_condensation_Si": 0.9763485181277762,
    }

    system.solve(constraints)
    assert system.solution.isclose(target_pressures, rtol=RTOL, atol=ATOL)
