"""Tests for simple SiHO interior-atmosphere systems.

See the LICENSE file for licensing information.

Tests using the JANAF data for simple SiHO interior-atmosphere systems.
"""

from typing import Type

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    FugacityConstraint,
    SystemConstraints,
)
from atmodeller.interfaces import (
    GasSpecies,
    NoSolubility,
    SolidSpecies,
    LiquidSpecies,
    ThermodynamicData,
    ThermodynamicDataBase,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

thermodynamic_data: Type[ThermodynamicDataBase] = ThermodynamicData

rtol: float = 1.0e-8
atol: float = 1.0e-8

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"

def test_H_and_Si_mass() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    sih_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    si_kg: float = sih_ratio * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="Si", value=si_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        'H2O': 0.39363930274450465,
        'H2': 0.3847211799192372,
        'O2': 8.725426165106087e-08,
        'OSi': 46.24328620480727,
        'H4Si': 4.433432437958054e-09
        }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)

def test_H_and_Si_liquid() -> None:
    """Tests H2-H2O and SiO2-SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O"),
            GasSpecies(chemical_formula="H2"),
            GasSpecies(chemical_formula="O2"),
            GasSpecies(chemical_formula="OSi"),
            GasSpecies(chemical_formula="H4Si"),
            LiquidSpecies(
                chemical_formula="O2Si",
                name_in_thermodynamic_data="O2Si(l)",
                #thermodynamic_class=thermodynamic_data,
            ),  # Ideal activity by default.
            #SolidSpecies(chemical_formula="O2Si"),
        ]
    )

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species='H2O', value=1),
            FugacityConstraint(species='OSi', value=1),
            IronWustiteBufferConstraintHirschmann(log10_shift=3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        'H2': 0.03095039977652221,
        'H2O': 1.0,
        'H4Si': 1.9649448270713194e-14,
        'O2': 8.700604070214889e-05,
        'O2Si': 448322.3567819643,
        'OSi': 1.0
        }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)

def test_H_and_Si_fugacity() -> None:
    """Tests H2-H2O and SiO-SiH4."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="OSi", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H4Si", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet()

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species='H2O', value=1),
            FugacityConstraint(species='OSi', value=1),
            IronWustiteBufferConstraintHirschmann(log10_shift=3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        'H2O': 1.0,
        'H2': 0.030950399776522242,
        'O2': 8.700604070214889e-05,
        'OSi': 1.0,
        'H4Si': 1.9649448270713276e-14
        }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)

if __name__ == '__main__':
    test_H_and_Si_mass()
    test_H_and_Si_fugacity()
    test_H_and_Si_liquid()
