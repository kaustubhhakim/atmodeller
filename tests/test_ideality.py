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

from pytest import approx

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.ideality import (
    CorkCH4,
    CorkCO,
    CorkCorrespondingStates,
    CorkFullABC,
    CorkFullCO2,
    CorkFullH2O,
    CorkH2,
    CorkSimpleCO2,
)
from atmodeller.interfaces import (
    GasSpecies,
    IdealityConstant,
    NoSolubility,
    ThermodynamicData,
    ThermodynamicDataBase,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
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
    temperature: float,
    pressure: float,
    gas_type: Type[CorkCorrespondingStates],
    expected_V: float,
    expected_fugacity_coeff: float,
) -> None:
    """Checks the volume and fugacity cofficient for a given gas type using CorkSimple."""
    # The class constructor requires no arguments.
    cork: CorkCorrespondingStates = gas_type()  # type: ignore
    V: float = cork.volume(temperature, pressure)
    fugacity_coeff: float = cork.fugacity_coefficient(temperature, pressure)

    assert V == approx(expected_V, rtol, atol)
    assert fugacity_coeff == approx(expected_fugacity_coeff, rtol, atol)


def check_full_Cork_gas(
    temperature: float,
    pressure: float,
    gas_type: Type[CorkFullABC],
    expected_fugacity_coeff: float,
) -> None:
    """Checks the fugacity coefficient for a given gas type using CorkFull."""
    # The class constructor requires no arguments.
    cork: CorkFull = gas_type()  # type: ignore
    fugacity_coeff: float = cork.fugacity_coefficient(temperature, pressure)

    assert fugacity_coeff == approx(expected_fugacity_coeff, rtol, atol)


def test_CorkH2() -> None:
    check_simple_Cork_gas(2000, 10, CorkH2, 3.7218446244368684, 4.672042007568433)


def test_CorkCO() -> None:
    check_simple_Cork_gas(2000, 10, CorkCO, 4.6747168815213715, 7.698485559533069)


def test_CorkCH4() -> None:
    check_simple_Cork_gas(2000, 10, CorkCH4, 4.786943829010815, 8.116070626285136)


def test_simple_CorkCO2() -> None:
    check_simple_Cork_gas(2000, 10, CorkSimpleCO2, 4.672048888683978, 7.1335509191383455)


def test_CorkCO2_at_P0() -> None:
    """Below P0 so virial contribution excluded."""
    check_full_Cork_gas(2000, 2, CorkFullCO2, 1.6063624424808558)


def test_CorkCO2_above_P0() -> None:
    """Above P0 so virial contribution included."""
    check_full_Cork_gas(2000, 10, CorkFullCO2, 7.556079199888717)


def test_CorkH2O_above_Tc_below_P0() -> None:
    """Above Tc and below P0."""
    check_full_Cork_gas(2000, 1, CorkFullH2O, 1.048278616058322)


def test_CorkH2O_above_Tc_above_P0() -> None:
    """Above Tc and above P0."""
    check_full_Cork_gas(2000, 5, CorkFullH2O, 1.3444013638026706)


def test_CorkH2O_below_Tc_below_Psat() -> None:
    """Below Tc and below Psat."""
    # Psat = 0.118224 at T = 600 K.
    check_full_Cork_gas(600, 0.1, CorkFullH2O, 0.7910907770688191)


def test_CorkH2O_below_Tc_above_Psat() -> None:
    """Below Tc and above Psat."""
    # Psat = 0.118224 at T = 600 K.
    check_full_Cork_gas(600, 1, CorkFullH2O, 0.14052644311851598)


def test_CorkH2O_below_Tc_above_P0() -> None:
    """Below Tc and above P0."""
    check_full_Cork_gas(600, 10, CorkFullH2O, 0.40066985009753664)


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

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="H", value=h_kg),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

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

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=1e3),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 747.5737656770727,
        "H2O": 1072.4328856736947,
        "O2": 9.76211086495026e-08,
    }

    system.solve(SystemConstraints(constraints))
    print(system.output)
    assert system.isclose(target_pressures)
