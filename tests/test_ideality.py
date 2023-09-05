"""Integration tests.

Tests to ensure that 'correct' values are returned for certain interior-atmosphere systems. 
These are quite rudimentary tests, but at least confirm that nothing fundamental is broken with the
code.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""

from typing import Type

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    IronWustiteBufferConstraintHirschmann,
    SystemConstraint,
    SystemConstraints,
)
from atmodeller.core import Species
from atmodeller.ideality import CorkCH4, CorkCO, CorkH2, CorkSimple, CorkSimpleCO2
from atmodeller.interfaces import (
    ConstantSystemConstraint,
    GasSpecies,
    IdealityConstant,
    NoSolubility,
    ThermodynamicData,
    ThermodynamicDataBase,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet
from atmodeller.solubilities import PeridotiteH2O
from atmodeller.utilities import earth_oceans_to_kg

# Tolerances to compare the test results with target output.
rtol: float = 1.0e-8
atol: float = 1.0e-8

thermodynamic_data: Type[ThermodynamicDataBase] = ThermodynamicData

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def check_simple_Cork_gas(
    gas_type: Type[CorkSimple], expected_V: float, expected_fugacity_coeff: float
) -> None:
    """Checks fugacity and volume for a given gas type using CorkSimple."""
    temperature: float = 2000  # K
    pressure: float = 10  # kbar
    # The class constructor requires no arguments.
    cork: CorkSimple = gas_type()  # type: ignore
    V: float = cork.volume(temperature, pressure)
    fugacity_coeff: float = cork.fugacity_coefficient(temperature, pressure)

    assert V == expected_V
    assert fugacity_coeff == expected_fugacity_coeff


def test_CorkH2() -> None:
    check_simple_Cork_gas(CorkH2, 3.7218446244368684, 4.672042007568433)


def test_CorkCO() -> None:
    check_simple_Cork_gas(CorkCO, 4.6747168815213715, 7.698485559533069)


def test_CorkCH4() -> None:
    check_simple_Cork_gas(CorkCH4, 4.786943829010815, 8.116070626285136)


def test_simple_CorkCO2() -> None:
    check_simple_Cork_gas(CorkSimpleCO2, 4.672048888683978, 7.1335509191383455)


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_class=thermodynamic_data,
                fugacity_coefficient=IdealityConstant(value=2),
            ),
            GasSpecies(
                chemical_formula="H2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                fugacity_coefficient=IdealityConstant(value=2),
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
            ),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        ConstantSystemConstraint(name="mass", species="H", value=h_kg),
        IronWustiteBufferConstraintHirschmann(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = dict(
        [("H2O", 0.19626421729663665), ("H2", 0.19386112601058758), ("O2", 8.69970008669977e-08)]
    )

    system.solve(SystemConstraints(constraints))
    print(system.output)
    assert system.isclose(target_pressures)


def test_H2_with_cork() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_class=thermodynamic_data,
            ),
            GasSpecies(
                chemical_formula="H2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                fugacity_coefficient=CorkH2(),
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
            ),
        ]
    )

    # oceans: float = 1
    planet: Planet = Planet(surface_temperature=2000)
    # h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        # ConstantSystemConstraint(name="mass", species="H", value=h_kg),
        ConstantSystemConstraint(name="fugacity", species="H2", value=1e3),
        IronWustiteBufferConstraintHirschmann(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 747.5737656770727,
        "H2O": 1072.4328856736947,
        "O2": 9.76211086495026e-08,
    }

    system.solve(SystemConstraints(constraints))
    print(system.output)
    assert system.isclose(target_pressures)
