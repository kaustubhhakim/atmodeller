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
    BufferedFugacityConstraint,
    ElementMassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
from atmodeller.solubility.hydrogen_species import H2O_peridotite_sossi
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_H2O() -> None:
    """Tests H2O (a single species)"""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())

    species: Species = Species([H2O_g])

    oceans: float = 2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 1.0312913336898137,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer.

    A comparable test has been run for FastChem, by tweaking the oxygen logarithmic abundance from
    11.30516874 (this test) to 11.4025 to ensure that the oxygen fugacity (8.7E-8) is the same to
    allow fair comparison. The FastChem element abundance file is (copy-paste):

    # test_H_fO2 from atmodeller
    e-  0.0
    H   12.00
    O   11.4025
    """

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25706483161845733,
        "H2_g": 0.25161113771286514,
        "O2_g": 8.699765393460875e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_holland() -> None:
    """Tests H2-H2O at the IW buffer using thermodynamic data from Holland and Powell"""

    H2O_g: GasSpecies = GasSpecies(
        "H2O",
        solubility=H2O_peridotite_sossi(),
        thermodata_dataset=ThermodynamicDatasetHollandAndPowell(),
    )
    H2_g: GasSpecies = GasSpecies(
        "H2",
        thermodata_dataset=ThermodynamicDatasetHollandAndPowell(),
    )
    O2_g: GasSpecies = GasSpecies(
        "O2",
        thermodata_dataset=ThermodynamicDatasetHollandAndPowell(),
    )

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25705058811429515,
        "H2_g": 0.2539022472323053,
        "O2_g": 8.699766647737794e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_basalt_melt() -> None:
    """Tests H2-H2O at the IW buffer."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet(melt_composition="basalt")
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.09434603056360964,
        "H2_g": 0.09234539760208682,
        "O2_g": 8.699588020866791e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_plus() -> None:
    """Tests H2-H2O at the IW buffer+2.

    A comparable test has been run for FastChem, by tweaking the oxygen logarithmic abundance from
    11.3067 (this test) to 11.65854 to ensure that the oxygen fugacity (8.7E-6) is the same to
    allow fair comparison. The FastChem element abundance file is (copy-paste):

    # test_H_fO2_plus from atmodeller
    e-  0.0
    H   12.00
    O   11.65854
    """

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(2)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25822551891794376,
        "H2_g": 0.025274900256252293,
        "O2_g": 8.699641354691526e-06,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_minus() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.234191482856985,
        "H2_g": 2.292084107842916,
        "O2_g": 8.700876916925329e-10,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 5
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 6.258071518665579,
        "H2_g": 6.123002998015027,
        "O2_g": 8.706308103092035e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_1500K() -> None:
    """Tests H2-H2O at a different temperature."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    planet.surface_temperature = 1500.0  # K

    target: dict[str, float] = {
        "H2O_g": 0.25671119963042033,
        "H2_g": 0.3065119897656826,
        "O2_g": 2.5006714903237476e-12,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25824358142493425,
        "H2_g": 0.2521806525810137,
        "O2_g": 8.740121617121534e-08,
        "CO_g": 59.6819921102523,
        "CO2_g": 13.404792068284909,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C_hill_formula() -> None:
    """Tests H2-H2O and CO-CO2 by changing the order of the chemical formulae for the species"""

    H2O_g: GasSpecies = GasSpecies("OH2", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("OC")
    CO2_g: GasSpecies = GasSpecies("O2C", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint(element="H", value=h_kg),
            ElementMassConstraint(element="C", value=c_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25824358142493425,
        "H2_g": 0.2521806525810137,
        "O2_g": 8.740121617121534e-08,
        "CO_g": 59.6819921102523,
        "CO2_g": 13.404792068284909,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C_total_pressure() -> None:
    """Tests H2-H2O and CO-CO2 with a total pressure constraint."""

    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint(element="H", value=h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            TotalPressureConstraint(value=100),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.25824457265720585,
        "H2_g": 0.2519709032598433,
        "O2_g": 8.754746041831307e-08,
        "CO_g": 81.22997882092373,
        "CO2_g": 18.259805583019286,
    }

    system.solve(constraints, factor=1)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)
