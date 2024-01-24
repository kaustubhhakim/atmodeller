"""Tests with sulfur

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.


Tests using the JANAF data for simple interior-atmosphere systems that include sulfur.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, NoSolubility
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import (
    BasaltDixonCO2,
    BasaltDixonH2O,
    BasaltH2,
    BasaltLibourelN2,
    BasaltS2,
    BasaltS2_Sulfate,
    BasaltS2_Sulfide,
)

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_S2_SO_Sulfide_IW() -> None:
    """Tests S2 and SO Sulfide Solubilities at the IW buffer, 2000 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2_Sulfide()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet()
    S2_fugacity: float = 1e-5
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="S2", value=S2_fugacity),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 1e-05,
        "OS": 6.018454943818516e-05,
        "O2": 8.699485217915599e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_AllS_Sulfide_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2_Sulfide()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.003704873878513222,
        "OS": 0.002991983415707734,
        "O2S": 0.004839725408460836,
        "O2": 1.0269757432683765e-06,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_AllS_Sulfate_IW() -> None:
    """Tests SO, S2, and SO2 Sulfate Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2_Sulfate()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 2e-4 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "OS": 0.6267850732546694,
        "S2": 161.05514086015452,
        "O2": 1.0367593216540548e-06,
        "O2S": 1.0186830187658191,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.0037048732091408782,
        "OS": 0.00299198314542162,
        "O2S": 0.0048397249712554885,
        "O2": 1.0269757432682946e-06,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IWp3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW+3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(log10_shift=3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 1.466040005579648,
        "OS": 1.8876140158988506,
        "O2S": 96.83727235805357,
        "O2": 0.0010329892821750164,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_AllS_TotalSolubility_IWm3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW-3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(log10_shift=-3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 3.7052440315568952e-06,
        "OS": 2.9921318707665795e-06,
        "O2S": 1.5305309773280557e-07,
        "O2": 1.0269750531288816e-09,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_HOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, O and S species at IW, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(formula="H2", solubility=BasaltH2()),
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="H", value=mass_H),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.9880831835505409,
        "H2": 0.9394385103704318,
        "OS": 0.002991611571529395,
        "S2": 0.0037035369886239495,
        "O2": 1.0270911157002842e-06,
        "O2S": 0.004839395737226402,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_CHONS_Species_IW_MixConstraints() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 2173 K,
    Mix of Fugacity and Mass Constraints."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(formula="H2", solubility=BasaltH2()),
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="CO", solubility=NoSolubility()),
            GasSpecies(formula="CO2", solubility=BasaltDixonCO2()),
            GasSpecies(formula="N2", solubility=BasaltLibourelN2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    S2_fugacity: float = 5e-3
    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_N: float = 0.0000028 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="S2", value=S2_fugacity),
            MassConstraint(species="H", value=mass_H),
            MassConstraint(species="C", value=mass_C),
            MassConstraint(species="N", value=mass_N),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 229.78209561323393,
        "CO2": 47.25770263332629,
        "H2": 0.9372769308635419,
        "H2O": 0.993868468772402,
        "N2": 2.3492111506716515,
        "O2": 1.0439522732151121e-06,
        "O2S": 0.005715305744187322,
        "OS": 0.003504432498947422,
        "S2": 0.004999999999999999,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_COS_Species_IW() -> None:
    """Tests Sulfur Solubility with C, O and S species at IW, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="CO", solubility=NoSolubility()),
            GasSpecies(formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="C", value=mass_C),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 230.95725097056018,
        "CO2": 47.49541654239574,
        "O2": 1.043777676357166e-06,
        "O2S": 0.0049587343906308,
        "OS": 0.003040782781419073,
        "S2": 0.0037651132245168973,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_CHOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW-3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(formula="H2", solubility=BasaltH2()),
            GasSpecies(formula="OS", solubility=NoSolubility()),
            GasSpecies(formula="S2", solubility=BasaltS2()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
            GasSpecies(formula="O2S", solubility=NoSolubility()),
            GasSpecies(formula="CO", solubility=NoSolubility()),
            GasSpecies(formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="H", value=mass_H),
            MassConstraint(species="C", value=mass_C),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 229.93580441594614,
        "CO2": 47.286325666730285,
        "H2": 0.9373315297419959,
        "H2O": 0.9938635373137384,
        "O2": 1.0438202991344499e-06,
        "O2S": 0.004959035809733831,
        "OS": 0.003040905529668846,
        "S2": 0.003765263450757495,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)
