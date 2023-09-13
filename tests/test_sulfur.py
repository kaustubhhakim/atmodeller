"""Integration tests.

Tests to ensure that sensible pressures are calculated for certain interior-atmosphere systems.

The target pressures are determined for the combined thermodynamic data, but they are within 1%
of the values for the JANAF thermodynamic data alone.

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

import numpy as np

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.interfaces import (
    GasSpecies,
    NoSolubility,
    ThermodynamicData,
    ThermodynamicDataBase,
    ThermodynamicDataJANAF,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import (
    BasaltDixonCO2,
    BasaltDixonH2O,
    BasaltH2,
    BasaltLibourelN2,
    BasaltS2,
    BasaltS2_Sulfate,
    BasaltS2_Sulfide,
    BasaltSO,
    BasaltSO2,
    BasaltSO2_Sulfate,
    BasaltSO2_Sulfide,
    BasaltSO_Sulfate,
    BasaltSO_Sulfide,
    PeridotiteH2O,
)
from atmodeller.utilities import earth_oceans_to_kg

# Uncomment to test JANAF only. TODO: FIXME: clean up.
# standard_gibbs_free_energy_of_formation: ThermodynamicDataBase = (
#    ThermodynamicDataJANAF()
# )
# Uncomment to test the combined dataset.
standard_gibbs_free_energy_of_formation: Type[ThermodynamicDataBase] = ThermodynamicData

# Both the combined data and JANAF report the same pressures to within 1%.
rtol: float = 1.0e-8
atol: float = 1.0e-8

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_S2_SO_Sulfide_IW() -> None:
    """Tests S2 and SO Sulfide Solubilities at the IW buffer, 2000 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=BasaltSO_Sulfide()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2_Sulfide()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
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
    print("output:", system.output)

    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_Sulfide_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=BasaltSO_Sulfide()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2_Sulfide()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2_Sulfide()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass
    print("S mass:", mass_S)
    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.0007885473892479838,
        "OS": 0.0013803406461261175,
        "O2S": 0.002232789217137583,
        "O2": 1.0269753162299365e-06,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_Sulfate_IW() -> None:
    """Tests SO, S2, and SO2 Sulfate Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=BasaltSO_Sulfate()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2_Sulfate()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2_Sulfate()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass
    print("S mass:", mass_S)
    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 161.03818676537367,
        "OS": 0.6267517706568156,
        "O2S": 1.0186283878331845,
        "O2": 1.0367582919375649e-06,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_TotalSolubility_IW() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at the IW buffer, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass
    print("S mass:", mass_S)
    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.0007885469624182632,
        "OS": 0.0013803402725464258,
        "O2S": 0.002232788612848466,
        "O2": 1.0269753162298525e-06,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_TotalSolubility_IWp3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW+3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass
    print("S mass:", mass_S)
    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(log10_shift=3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 0.35155185366764335,
        "OS": 0.922957016616041,
        "O2S": 47.27780744686448,
        "O2": 0.0010298851822133756,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_AllS_TotalSolubility_IWm3() -> None:
    """Tests SO, S2, and SO2 Sulfide+Sulfate Solubility at IW-3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)
    mass_S: float = 0.0002 * planet.mantle_mass
    print("S mass:", mass_S)
    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            IronWustiteBufferConstraintHirschmann(log10_shift=-3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "S2": 7.885746000191323e-07,
        "OS": 1.3803642849455947e-06,
        "O2S": 7.060819472779197e-08,
        "O2": 1.0269750528534265e-09,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHOS_Species_IW() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
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
        "H2O": 0.9938623599950259,
        "H2": 0.9373400578523818,
        "OS": 0.0014029288354079062,
        "S2": 0.0008014376241378935,
        "O2": 1.0437988324950095e-06,
        "O2S": 0.002287839173668045,
        "CO": 229.64954695306662,
        "CO2": 47.22697115570074,
    }

    # Give a reasonable initial guess
    initial_log10: np.ndarray = np.array([0, 0, -3, -4, -6, -3, 2, 1])
    initial_solution: np.ndarray = 10.0**initial_log10

    system.solve(constraints, initial_solution=initial_solution)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHOS_Species_IWp3() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW+3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
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
            IronWustiteBufferConstraintHirschmann(log10_shift=3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 1.188967762490119,
        "H2": 0.03536836491607379,
        "OS": 0.8721291643917857,
        "S2": 0.30811156610889007,
        "O2": 0.0010492255755840164,
        "O2S": 45.091708496863454,
        "CO": 42.591742711345944,
        "CO2": 277.7000395495448,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHOS_Species_IWm3() -> None:
    """Tests Sulfur Solubility with H, C, O and S species at IW-3, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
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
            IronWustiteBufferConstraintHirschmann(log10_shift=-3),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.4293969992073274,
        "H2": 12.81113199992055,
        "OS": 1.4019595661823213e-06,
        "S2": 8.009115444022984e-07,
        "O2": 1.043041704293421e-09,
        "O2S": 7.227161703487836e-08,
        "CO": 251.48618508527014,
        "CO2": 1.6348618769791592,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHONS_Species_IW() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 2173 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
            GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=2173)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass
    mass_N: float = 0.0000028 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="H", value=mass_H),
            MassConstraint(species="C", value=mass_C),
            MassConstraint(species="N", value=mass_N),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.9938672430188247,
        "H2": 0.9372856248427179,
        "OS": 0.0014031055367537371,
        "S2": 0.0008015385430031337,
        "O2": 1.043930331535086e-06,
        "O2S": 0.002288271457007958,
        "CO": 229.4915238784822,
        "CO2": 47.19744674053677,
        "N2": 2.3491188709942548,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHONS_Species_IW_LowerT() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 1800 K."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
            GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()),
        ]
    )

    planet: Planet = Planet(surface_temperature=1800)

    mass_H: float = 0.00108 * planet.mantle_mass * (2 / 18)
    mass_C: float = 0.00014 * planet.mantle_mass
    mass_S: float = 0.0002 * planet.mantle_mass
    mass_N: float = 0.0000028 * planet.mantle_mass

    constraints: SystemConstraints = SystemConstraints(
        [
            MassConstraint(species="S", value=mass_S),
            MassConstraint(species="H", value=mass_H),
            MassConstraint(species="C", value=mass_C),
            MassConstraint(species="N", value=mass_N),
            IronWustiteBufferConstraintHirschmann(log10_shift=0),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2O": 0.985172841599582,
        "H2": 1.0073714331735775,
        "OS": 4.317740951918956e-05,
        "S2": 7.286429137857455e-05,
        "O2": 2.7714826681743157e-09,
        "O2S": 0.00011508895918623868,
        "CO": 221.91202759714838,
        "CO2": 57.03350797424276,
        "N2": 2.385505258325644,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


def test_CHONS_Species_IW_MixConstraints() -> None:
    """Tests Sulfur Solubility with C, H, N, O, S species at IW, 2173 K,
    Mix of Fugacity and Mass Constraints."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=BasaltDixonH2O()),
            GasSpecies(chemical_formula="H2", solubility=BasaltH2()),
            GasSpecies(chemical_formula="OS", solubility=BasaltSO()),
            GasSpecies(chemical_formula="S2", solubility=BasaltS2()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2S", solubility=BasaltSO2()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
            GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()),
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
        "H2O": 0.9938673900926728,
        "H2": 0.9372852480841496,
        "OS": 0.0035043975980955165,
        "S2": 0.004999999999999999,
        "O2": 1.043931479752966e-06,
        "O2S": 0.005715191906613821,
        "CO": 229.49902517843927,
        "CO2": 47.19901542212745,
        "N2": 2.3491951391673873,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)
