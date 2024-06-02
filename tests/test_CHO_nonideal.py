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
"""Tests for non-ideal C-H-O interior-atmosphere systems"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    PressureConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies
from atmodeller.eos.holland import get_holland_eos_models
from atmodeller.eos.interfaces import RealGas
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
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

eos_holland: dict[str, RealGas] = get_holland_eos_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_pH2_fO2_holland() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the partial pressure of H2.
    """

    H2O_g: GasSpecies = GasSpecies(
        "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
    )
    H2_g: GasSpecies = GasSpecies("H2", eos=eos_holland["H2"])
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            PressureConstraint(H2_g, value=1000),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 1441.892430083237,
        "H2_g": 1000.0,
        "O2_g": 1.0154197223693444e-07,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_fH2_fO2_holland() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """

    H2O_g: GasSpecies = GasSpecies(
        "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
    )
    H2_g: GasSpecies = GasSpecies("H2", eos=eos_holland["H2"])
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=1000),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 989.1708427772087,
        "H2_g": 756.9484278838194,
        "O2_g": 9.71654424299468e-08,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C_holland() -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann(), eos=eos_holland["H2"])
    H2O_g: GasSpecies = GasSpecies(
        "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
    )
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies(formula="CO", eos=eos_holland["CO"])
    CO2_g: GasSpecies = GasSpecies(
        formula="CO2", solubility=CO2_basalt_dixon(), eos=eos_holland["CO2"]
    )
    CH4_g: GasSpecies = GasSpecies(formula="CH4", eos=eos_holland["CH4"])

    species: Species = Species([H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g])

    oceans: float = 10
    planet: Planet = Planet()
    planet.surface_temperature = 2000
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=958),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("C", value=c_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2_g": 695.640114660725,
        "H2O_g": 941.5985234431984,
        "O2_g": 9.86913046974216e-08,
        "CO_g": 277.80794508391784,
        "CO2_g": 66.71386181033674,
        "CH4_g": 10.433657059802755,
    }

    system.solve(constraints)
    assert system.solution.isclose(target, rtol=RTOL, atol=ATOL)
