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
"""Tests with sulfur"""

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
from atmodeller.core import GasSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
from atmodeller.solubility.hydrogen_species import (
    H2_basalt_hirschmann,
    H2O_basalt_dixon,
)
from atmodeller.solubility.other_species import N2_basalt_libourel
from atmodeller.solubility.sulfur_species import (
    S2_basalt_boulliung,
    S2_sulfate_basalt_boulliung,
    S2_sulfide_basalt_boulliung,
)
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_S2_SO_Sulfide_IW() -> None:
    """Tests S2 and SO Sulfide Solubilities at the IW buffer, 2000 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_sulfide_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([OS_g, S2_g, O2_g])

    planet: Planet = Planet()
    S2_fugacity: float = 1e-5
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(S2_g, S2_fugacity),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 6.018454943818516e-05,
        "S2_g": 1e-05,
        "O2_g": 8.699485217915599e-08,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_Sulfide_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide Solubility at the IW buffer, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_sulfide_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 0.002991983415707734,
        "S2_g": 0.003704873878513222,
        "O2S_g": 0.004839725408460836,
        "O2_g": 1.0269757432683765e-06,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_Sulfate_IW() -> None:
    """Tests SO, S2, and SO2 Sulfate Solubility at the IW buffer, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_sulfate_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 2e-4 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 0.6267850732546694,
        "S2_g": 161.05514086015452,
        "O2S_g": 1.0186830187658191,
        "O2_g": 1.0367593216540548e-06,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at the IW buffer, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 0.00299198314542162,
        "S2_g": 0.0037048732091408782,
        "O2S_g": 0.0048397249712554885,
        "O2_g": 1.0269757432682946e-06,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IWp3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW+3, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(3)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 1.8876140158988506,
        "S2_g": 1.466040005579648,
        "O2S_g": 96.83727235805357,
        "O2_g": 0.0010329892821750164,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IWm3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW-3, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-3)),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 2.9921318707665795e-06,
        "S2_g": 3.7052440315568952e-06,
        "O2S_g": 1.5305309773280557e-07,
        "O2_g": 1.0269750531288816e-09,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_HOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, O and S species at IW, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_basalt_dixon())
    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann())

    species: Species = Species([H2O_g, H2_g, OS_g, S2_g, O2_g, O2S_g])

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            ElementMassConstraint("H", mass_H),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.9880831835505409,
        "H2_g": 0.9394385103704318,
        "OS_g": 0.002991611571529395,
        "S2_g": 0.0037035369886239495,
        "O2_g": 1.0270911157002842e-06,
        "O2S_g": 0.004839395737226402,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_CHONS_Species_IW_MixConstraints() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 2173 K,
    Mix of Fugacity and Mass Constraints."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_basalt_dixon())
    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann())
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())
    N2_g: GasSpecies = GasSpecies("N2", solubility=N2_basalt_libourel())

    species: Species = Species([H2O_g, H2_g, OS_g, S2_g, O2_g, O2S_g, CO_g, CO2_g, N2_g])

    planet: Planet = Planet(surface_temperature=2173)
    S2_fugacity: float = 5e-3
    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_N: float = 0.0000028 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(S2_g, S2_fugacity),
            ElementMassConstraint("H", mass_H),
            ElementMassConstraint("C", mass_C),
            ElementMassConstraint("N", mass_N),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.993868468772402,
        "H2_g": 0.9372769308635419,
        "OS_g": 0.003504432498947422,
        "S2_g": 0.004999999999999999,
        "O2_g": 1.0439522732151121e-06,
        "O2S_g": 0.005715305744187322,
        "CO_g": 229.78209561323393,
        "CO2_g": 47.25770263332629,
        "N2_g": 2.3492111506716515,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_COS_Species_IW() -> None:
    """Tests Sulfur Solubility with C, O and S species at IW, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([OS_g, S2_g, O2_g, O2S_g, CO_g, CO2_g])

    planet: Planet = Planet(surface_temperature=2173)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            ElementMassConstraint("C", mass_C),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "OS_g": 0.003040782781419073,
        "S2_g": 0.0037651132245168973,
        "O2_g": 1.043777676357166e-06,
        "O2S_g": 0.0049587343906308,
        "CO_g": 230.95725097056018,
        "CO2_g": 47.49541654239574,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)


def test_CHOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW-3, 2173 K."""

    OS_g: GasSpecies = GasSpecies("OS")
    S2_g: GasSpecies = GasSpecies("S2", solubility=S2_basalt_boulliung())
    O2_g: GasSpecies = GasSpecies("O2")
    O2S_g: GasSpecies = GasSpecies("O2S")
    H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_basalt_dixon())
    H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann())
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2", solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, OS_g, S2_g, O2_g, O2S_g, CO_g, CO2_g])

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("S", mass_S),
            ElementMassConstraint("H", mass_H),
            ElementMassConstraint("C", mass_C),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2O_g": 0.9938635373137384,
        "H2_g": 0.9373315297419959,
        "OS_g": 0.003040905529668846,
        "S2_g": 0.003765263450757495,
        "O2_g": 1.0438202991344499e-06,
        "O2S_g": 0.004959035809733831,
        "CO_g": 229.93580441594614,
        "CO2_g": 47.286325666730285,
    }

    system.solve(constraints)
    assert system.isclose(target, rtol=RTOL, atol=ATOL)
