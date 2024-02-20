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

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    PressureConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies
from atmodeller.eos.holland import get_holland_eos_models
from atmodeller.eos.interfaces import RealGas
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltCO2, BasaltH2, PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()

eos_holland: dict[str, RealGas] = get_holland_eos_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_pH2_fO2_holland() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the partial pressure of H2.
    """

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O(), eos=eos_holland["H2O"]),
            GasSpecies(formula="H2", eos=eos_holland["H2"]),
            GasSpecies(formula="O2"),
        ]
    )

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            PressureConstraint(species="H2", value=1000),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O": 1441.892430083237,
        "H2": 1000.0,
        "O2": 1.0154197223693444e-07,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_fH2_fO2_holland() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O(), eos=eos_holland["H2O"]),
            GasSpecies(formula="H2", eos=eos_holland["H2"]),
            GasSpecies(formula="O2"),
        ]
    )

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=1000),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O": 989.1708427772087,
        "H2": 756.9484278838194,
        "O2": 9.71654424299468e-08,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_H_and_C_holland() -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", solubility=BasaltH2(), eos=eos_holland["H2"]),
            GasSpecies(formula="H2O", solubility=PeridotiteH2O(), eos=eos_holland["H2O"]),
            GasSpecies(formula="O2"),
            GasSpecies(formula="CO", eos=eos_holland["CO"]),
            GasSpecies(formula="CO2", solubility=BasaltCO2(), eos=eos_holland["CO2"]),
            GasSpecies(formula="CH4", eos=eos_holland["CH4"]),
        ]
    )

    oceans: float = 10
    planet: Planet = Planet()
    planet.surface_temperature = 2000
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=958),
            IronWustiteBufferConstraintHirschmann(),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2": 695.640114660725,
        "H2O": 941.5985234431984,
        "O2": 9.86913046974216e-08,
        "CO": 277.80794508391784,
        "CO2": 66.71386181033674,
        "CH4": 10.433657059802755,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)
