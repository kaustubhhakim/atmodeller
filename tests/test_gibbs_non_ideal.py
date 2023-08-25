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

from atmodeller import __version__
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    MassConstraint,
    SystemConstraint,
    SystemConstraints,
)
from atmodeller.core import InteriorAtmosphereSystem, Planet
from atmodeller.interfaces import NonIdealConstant
from atmodeller.solubilities import PeridotiteH2O
from atmodeller.thermodynamics import (
    GasSpecies,
    NoSolubility,
    Species,
    StandardGibbsFreeEnergyOfFormation,
    StandardGibbsFreeEnergyOfFormationProtocol,
)
from atmodeller.utilities import earth_oceans_to_kg

# Tolerances to compare the test results with target output.
rtol: float = 1.0e-8
atol: float = 1.0e-8

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormation()
)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# region oxygen fugacity


def test_hydrogen_species_oxygen_fugacity_buffer() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                ideality=NonIdealConstant(2),
            ),
            GasSpecies(
                chemical_formula="H2", solubility=NoSolubility(), ideality=NonIdealConstant(2)
            ),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: dict[str, float] = dict(
        [("H2O", 0.19626421729663665), ("H2", 0.19386112601058758), ("O2", 8.69970008669977e-08)]
    )

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures)


# endregion

# def test_hydrogen_species_oxygen_fugacity_buffer_shift_positive() -> None:
#     """Tests H2-H2O at the IW buffer+2."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#     ]

#     oceans: float = 1
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         BufferedFugacityConstraint(log10_shift=2),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array([3.88388984e-02, 8.69972318e-06, 3.93203953e-01])
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# def test_hydrogen_species_oxygen_fugacity_buffer_shift_negative() -> None:
#     """Tests H2-H2O at the IW buffer-2."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#     ]

#     oceans: float = 1
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         BufferedFugacityConstraint(log10_shift=-2),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array([3.35867961e00, 8.70152291e-10, 3.40066982e-01])
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# # endregion

# # region number of oceans


# def test_hydrogen_species_five_oceans() -> None:
#     """Tests H2-H2O for five H oceans."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#     ]

#     oceans: float = 5
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array([9.25738492e00, 8.70975650e-08, 9.37755422e00])
#     system.solve(SystemConstraints(constraints))

#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# def test_hydrogen_species_ten_oceans() -> None:
#     """Tests H2-H2O for ten H oceans."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#     ]

#     oceans: float = 10
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array([3.52710459e01, 8.73871716e-08, 3.57882477e01])
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# # endregion

# # region temperature


# def test_hydrogen_species_temperature() -> None:
#     """Tests H2-H2O at a different temperature."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#     ]

#     oceans: float = 1
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     planet.surface_temperature = 1500.0  # K
#     target_pressures: np.ndarray = np.array([4.69139863e-01, 2.50073390e-12, 3.89671393e-01])
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# # endregion

# # region C over H ratio


# def test_hydrogen_and_carbon_species() -> None:
#     """Tests H2-H2O and CO-CO2."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
#     ]

#     oceans: float = 1
#     ch_ratio: float = 1
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = ch_ratio * h_kg

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         MassConstraint(species="C", value=c_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array(
#         [5.96157589e01, 3.87522780e-01, 8.74014299e-08, 1.32393097e01, 3.93237350e-01]
#     )
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# def test_hydrogen_and_carbon_species_five_ch_ratio() -> None:
#     """Tests H2-H2O and CO-CO2 for C/H=5."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
#     ]

#     oceans: float = 1
#     ch_ratio: float = 5
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = ch_ratio * h_kg

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         MassConstraint(species="C", value=c_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array(
#         [2.99414663e02, 3.83950883e-01, 8.90419112e-08, 6.71143359e01, 3.93252201e-01]
#     )
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# def test_hydrogen_and_carbon_species_ten_ch_ratio() -> None:
#     """Tests H2-H2O and CO-CO2 for C/H=10."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
#     ]

#     oceans: float = 1
#     ch_ratio: float = 10
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = ch_ratio * h_kg

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         MassConstraint(species="C", value=c_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array(
#         [5.98473386e02, 3.79516928e-01, 9.11391785e-08, 1.35719545e02, 3.93261980e-01]
#     )
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# # endregion

# # region methane


# def test_hydrogen_and_carbon_species_with_methane() -> None:
#     """Tests H2-H2O and CO-CO2 and N."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
#         GasSpecies(chemical_formula="CH4", solubility=NoSolubility()),
#     ]

#     oceans: float = 1
#     ch_ratio: float = 1
#     planet: Planet = Planet()
#     planet.surface_temperature = 1500
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = ch_ratio * h_kg

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         MassConstraint(species="C", value=c_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )

#     target_pressures: np.ndarray = np.array(
#         [
#             5.58811958e01,
#             4.71837579e-01,
#             2.51637543e-12,
#             1.79658754e01,
#             3.93135889e-01,
#             6.28583933e-05,
#         ]
#     )
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# # endregion


# # region nitrogen


# def test_hydrogen_and_carbon_species_with_nitrogen() -> None:
#     """Tests H2-H2O and CO-CO2 and N."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
#         GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()),
#     ]

#     oceans: float = 1
#     ch_ratio: float = 1
#     nitrogen_ppmw: float = 2.8
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = ch_ratio * h_kg
#     n_kg: float = nitrogen_ppmw * 1.0e-6 * planet.mantle_mass

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         MassConstraint(species="C", value=c_kg),
#         MassConstraint(species="N", value=n_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )
#     # Order of target pressures: CO, H2, N2, O2, CO2, H2O
#     target_pressures: np.ndarray = np.array(
#         [
#             5.94596836e01,
#             3.87492671e-01,
#             2.35167247e00,
#             8.74133949e-08,
#             1.32055527e01,
#             3.93233711e-01,
#         ]
#     )
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# def test_hydrogen_and_carbon_species_with_NH3() -> None:
#     """Tests H2-H2O and CO-CO2 and NH3."""

#     species: list[ChemicalComponent] = [
#         GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
#         GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
#         GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
#         GasSpecies(chemical_formula="NH3", solubility=NoSolubility()),
#     ]

#     oceans: float = 1
#     ch_ratio: float = 1
#     nitrogen_ppmw: float = 2.8
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = ch_ratio * h_kg
#     n_kg: float = nitrogen_ppmw * 1.0e-6 * planet.mantle_mass

#     constraints: list[SystemConstraint] = [
#         MassConstraint(species="H", value=h_kg),
#         MassConstraint(species="C", value=c_kg),
#         MassConstraint(species="N", value=n_kg),
#         BufferedFugacityConstraint(),
#     ]

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
#     )
#     # Order of target pressures: CO, H2, O2, CO2, H2O, NH3
#     target_pressures: np.ndarray = np.array(
#         [
#             5.80133890e01,
#             3.74858271e-01,
#             8.74172015e-08,
#             1.28846220e01,
#             3.80420404e-01,
#             4.83204103e00,
#         ]
#     )
#     system.solve(SystemConstraints(constraints))
#     assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# # endregion
