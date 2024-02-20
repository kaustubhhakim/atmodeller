"""Tests for nonideal gas behaviour

See the LICENSE file for licensing information.
"""

import logging

import pytest

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies, LiquidSpecies
from atmodeller.eos.interfaces import RealGas
from atmodeller.eos.saxena import get_saxena_eos_models
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltH2, PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

rtol: float = 1.0e-8
atol: float = 1.0e-8

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_saxena_eos_models()

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


@pytest.mark.skip(reason="with condensed species mass balance another constraint is now required")
def test_SiHO_massSiH_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", eos=eos_models["H2"]),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="O2"),
            GasSpecies(formula="OSi"),
            GasSpecies(formula="H4Si"),
            LiquidSpecies(formula="SiO2"),
        ]
    )

    oceans: float = 50
    sih_ratio: float = 1
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)
    si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="Si", value=si_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target: dict[str, float] = {
        "H2": 3955.5461365393257,
        "H2O": 238.6969674984362,
        "O2": 8.041858631374242e-05,
        "OSi": 297.7208780281457,
        "H4Si": 3.8854445389482284,
        "SiO2": 1.0,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target, rtol=rtol, atol=atol)


@pytest.mark.skip(reason="with condensed species mass balance another constraint is now required")
def test_SiHO_massSiH_solubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", eos=eos_models["H2"], solubility=BasaltH2()),
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="O2"),
            GasSpecies(formula="OSi"),
            GasSpecies(formula="H4Si"),
            LiquidSpecies(formula="SiO2"),
        ]
    )

    oceans: float = 50
    sih_ratio: float = 1
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)
    si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="Si", value=si_kg),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 2225.387378761475,
        "H2O": 68.65564430638824,
        "O2": 2.2005226675444582e-05,
        "OSi": 569.1471756022945,
        "H4Si": 4.292996959380682,
        "SiO2": 1.0,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_SiHO_massH_logfO2_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", eos=eos_models["H2"]),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="O2"),
            GasSpecies(formula="OSi"),
            GasSpecies(formula="H4Si"),
            LiquidSpecies(formula="SiO2"),
        ]
    )

    oceans: float = 50
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 3552.669457706052,
        "H2O": 4767.286871920151,
        "O2": 0.03382190282100078,
        "OSi": 14.517388181293816,
        "H4Si": 0.008762061109896661,
        "SiO2": 1.0,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_SiHO_massH_logfO2_solubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", eos=eos_models["H2"], solubility=BasaltH2()),
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="O2"),
            GasSpecies(formula="OSi"),
            GasSpecies(formula="H4Si"),
            LiquidSpecies(formula="SiO2"),
        ]
    )

    oceans: float = 50
    planet: Planet = Planet(surface_temperature=3400)
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 337.17023763048417,
        "H2O": 378.88646918406965,
        "O2": 0.025481748164747447,
        "OSi": 16.72526080495589,
        "H4Si": 9.750334904890755e-05,
        "SiO2": 1.0,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_SiHO_totalpressure_logfO2_nosolubility() -> None:
    """Tests H2-H2O and SiO2-SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", eos=eos_models["H2"]),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="O2"),
            GasSpecies(formula="OSi"),
            GasSpecies(
                formula="H4Si",
            ),
            LiquidSpecies(formula="SiO2"),
        ]
    )
    planet: Planet = Planet(surface_temperature=3400)

    constraints: SystemConstraints = SystemConstraints(
        [
            TotalPressureConstraint(value=4000),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 1868.1256657071128,
        "H2O": 2116.1052597622975,
        "O2": 0.02877934211561468,
        "OSi": 15.737910722708122,
        "H4Si": 0.002384364181797566,
        "SiO2": 1.0,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_SiHO_fugacityH2O_logfO2_nosolubility() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(formula="H2", eos=eos_models["H2"]),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="O2"),
            GasSpecies(formula="OSi"),
            GasSpecies(formula="H4Si"),
            LiquidSpecies(formula="SiO2"),
        ]
    )

    planet: Planet = Planet(surface_temperature=3400)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2O", value=5000),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 3668.5524293467392,
        "H2O": 4999.9999999999945,
        "O2": 0.03426380501584421,
        "OSi": 14.42346859651946,
        "H4Si": 0.00939136258814609,
        "SiO2": 1.0,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


if __name__ == "__main__":
    test_SiHO_massSiH_nosolubility()
    test_SiHO_massSiH_solubility()
    test_SiHO_massH_logfO2_nosolubility()
    test_SiHO_massH_logfO2_solubility()
    test_SiHO_totalpressure_logfO2_nosolubility()
    test_SiHO_fugacityH2O_logfO2_nosolubility()
