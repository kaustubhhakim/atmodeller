"""Tests for simple C-H-O interior-atmosphere systems

See the LICENSE file for licensing information.

Tests using the JANAF data for simple C-H-O interior-atmosphere systems.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    IronWustiteBufferConstraintBallhaus,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
    FugacityConstraint,
    TotalPressureConstraint,
)
from atmodeller.interfaces import GasSpecies, NoSolubility, SolidSpecies, ThermodynamicDatasetABC, ThermodynamicDatasetJANAF
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import BasaltDixonCO2, PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 0.25161113771286514,
        "H2O": 0.25706483161845733,
        "O2": 8.699765393460875e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_basalt_melt() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=NoSolubility()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet(melt_composition="basalt")
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 0.09234539760208682,
        "H2O": 0.09434603056360964,
        "O2": 8.699588020866791e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_fO2_plus() -> None:
    """Tests H2-H2O at the IW buffer+2."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 0.025274900256252293,
        "H2O": 0.25822551891794376,
        "O2": 8.699641354691526e-06,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_fO2_minus() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 2.292084107842916,
        "H2O": 0.234191482856985,
        "O2": 8.700876916925329e-10,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 5
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 6.123002998015027,
        "H2O": 6.258071518665579,
        "O2": 8.706308103092035e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_1500K() -> None:
    """Tests H2-H2O at a different temperature."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    planet.surface_temperature = 1500.0  # K

    target_pressures: dict[str, float] = {
        "H2": 0.3065119897656826,
        "H2O": 0.25671119963042033,
        "O2": 2.5006714903237476e-12,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
            GasSpecies(formula="CO", solubility=NoSolubility()),
            GasSpecies(formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 59.65949387634169,
        "CO2": 13.399727189920911,
        "H2": 0.2521808713740826,
        "H2O": 0.2582435801202491,
        "O2": 8.740106362890969e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)


def test_H_and_C_total_pressure() -> None:
    """Tests H2-H2O and CO-CO2 with a total pressure constraint."""

    species: Species = Species(
        [
            GasSpecies(formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(formula="H2", solubility=NoSolubility()),
            GasSpecies(formula="O2", solubility=NoSolubility()),
            GasSpecies(formula="CO", solubility=NoSolubility()),
            GasSpecies(formula="CO2", solubility=BasaltDixonCO2()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
            TotalPressureConstraint(value=100),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "CO": 81.22997882092373,
        "CO2": 18.259805583019286,
        "H2": 0.2519709032598433,
        "H2O": 0.25824457265720585,
        "O2": 8.754746041831307e-08,
    }

    system.solve(constraints, factor=1)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)

def test_graphite() -> None:
    """Tests including graphite."""

    thermodynamic_data: ThermodynamicDatasetABC = ThermodynamicDatasetJANAF()

    species: Species = Species(
        [
            GasSpecies(formula="H2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="H2O", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CO", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CO2", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="CH4", thermodynamic_dataset=thermodynamic_data),
            GasSpecies(formula="O2", thermodynamic_dataset=thermodynamic_data),
            SolidSpecies(
                formula="C",
                thermodynamic_dataset=thermodynamic_data,
                name_in_dataset="graphite",
            ),
        ]
    )

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=44.49334998176607),
        ]
    )

    target_pressures: dict[str, float] = {
        "C": 1.0,
        "CH4": 900.3912797397132,
        "CO": 0.07741709702165529,
        "CO2": 0.0685518295157825,
        "H2": 44.493349981766045,
        "H2O": 14.708340036418534,
        "O2": 1.4458158511932372e-25,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=RTOL, atol=ATOL)