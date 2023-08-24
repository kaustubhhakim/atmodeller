"""Integration tests.

Tests to ensure that 'correct' values are returned for certain interior-atmosphere systems. 
These are quite rudimentary tests, but at least confirm that nothing fundamental is broken with the
code.

"""

import numpy as np

from atmodeller import (
    BufferedFugacityConstraint,
    GasSpecies,
    InteriorAtmosphereSystem,
    MassConstraint,
    Planet,
    SystemConstraint,
    __version__,
)
from atmodeller.solubilities import BasaltDixonCO2, BasaltLibourelN2, PeridotiteH2O
from atmodeller.thermodynamics import (
    ChemicalComponent,
    NoSolubility,
    StandardGibbsFreeEnergyOfFormationJANAF,
    StandardGibbsFreeEnergyOfFormationProtocol,
)
from atmodeller.utilities import earth_oceans_to_kg

# Tolerances to compare the test results with target output.
rtol: float = 1.0e-8
atol: float = 1.0e-8

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormationJANAF()
)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# region oxygen fugacity


def test_hydrogen_species_oxygen_fugacity_buffer() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
    ]

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

    target_pressures: np.ndarray = np.array([3.82236039e-01, 8.69991087e-08, 3.90524304e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_oxygen_fugacity_buffer_shift_positive() -> None:
    """Tests H2-H2O at the IW buffer+2."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(log10_shift=2),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([3.84865045e-02, 8.69972299e-06, 3.93206071e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_oxygen_fugacity_buffer_shift_negative() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(log10_shift=-2),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([3.33341150e00, 8.70150928e-10, 3.40600482e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region number of oceans


def test_hydrogen_species_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
    ]

    oceans: float = 5
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([9.17699183e00, 8.70971421e-08, 9.38126343e00])
    system.solve(constraints)

    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_ten_oceans() -> None:
    """Tests H2-H2O for ten H oceans."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
    ]

    oceans: float = 10
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([3.49768773e01, 8.73856897e-08, 3.58146100e01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region temperature


def test_hydrogen_species_temperature() -> None:
    """Tests H2-H2O at a different temperature."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
    ]

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

    planet.surface_temperature = 1500.0  # K
    target_pressures: np.ndarray = np.array([4.65306632e-01, 2.50073310e-12, 3.89710351e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region C over H ratio


def test_hydrogen_and_carbon_species() -> None:
    """Tests H2-H2O and CO-CO2."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [5.95202345e01, 3.84011401e-01, 8.74015968e-08, 1.33684898e01, 3.93244662e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_five_ch_ratio() -> None:
    """Tests H2-H2O and CO-CO2 for C/H=5."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
    ]

    oceans: float = 1
    ch_ratio: float = 5
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [2.98921395e02, 3.80470220e-01, 8.90427869e-08, 6.77663985e01, 3.93259367e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_ten_ch_ratio() -> None:
    """Tests H2-H2O and CO-CO2 for C/H=10."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
    ]

    oceans: float = 1
    ch_ratio: float = 10
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [5.97478399e02, 3.76074435e-01, 9.11410184e-08, 1.37036791e02, 3.93269064e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region methane


def test_hydrogen_and_carbon_species_with_methane() -> None:
    """Tests H2-H2O and CO-CO2 and N."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="CH4", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    planet.surface_temperature = 1500
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [
            5.57453992e01,
            4.67944538e-01,
            2.51638318e-12,
            1.81420742e01,
            3.93144131e-01,
            6.16266832e-05,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion


# region nitrogen


def test_hydrogen_and_carbon_species_with_nitrogen() -> None:
    """Tests H2-H2O and CO-CO2 and N."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    nitrogen_ppmw: float = 2.8
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    n_kg: float = nitrogen_ppmw * 1.0e-6 * planet.mantle_mass

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="N", value=n_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Order of target pressures: CO, H2, N2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            5.93629233e01,
            3.83981547e-01,
            2.35366050e00,
            8.74135623e-08,
            1.33340697e01,
            3.93241005e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_H3N() -> None:
    """Tests H2-H2O and CO-CO2 and H3N."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="H3N", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    nitrogen_ppmw: float = 2.8
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    n_kg: float = nitrogen_ppmw * 1.0e-6 * planet.mantle_mass

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="N", value=n_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Order of target pressures: CO, H2, O2, CO2, H2O, NH3
    target_pressures: np.ndarray = np.array(
        [
            5.79185297e01,
            3.71461655e-01,
            8.74173745e-08,
            1.30099144e01,
            3.80427500e-01,
            4.83625230e00,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region sulfur


def test_hydrogen_and_carbon_species_with_O2S() -> None:
    """Tests H2-H2O and CO-CO2 and S-SO2."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="S", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2S", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    s_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="S", value=s_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, SO2
    target_pressures: np.ndarray = np.array(
        [
            2.21918442e-03,
            5.95425175e01,
            3.84011236e-01,
            8.74019062e-08,
            1.33735184e01,
            3.93245189e-01,
            2.63862045e-02,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and S-H2S."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="S", solubility=NoSolubility()),
        GasSpecies(chemical_formula="H2S", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    s_kg: float = 0.01 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="S", value=s_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            3.94069014e-03,
            5.95446154e01,
            3.83532907e-01,
            8.74033406e-08,
            1.33740993e01,
            3.92758581e-01,
            2.82124297e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_OS_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and SO-H2S."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="OS", solubility=NoSolubility()),
        GasSpecies(chemical_formula="H2S", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    s_kg: float = 0.01 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="S", value=s_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: CO, H2, O2, SO, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            5.95525113e01,
            3.83563317e-01,
            8.74033947e-08,
            2.19644501e-02,
            1.33758769e01,
            3.92789844e-01,
            2.64140283e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region Cl


def test_hydrogen_and_carbon_species_with_HCl() -> None:
    """Tests H2-H2O and CO-CO2 and HCl-Cl."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="Cl", solubility=NoSolubility()),
        GasSpecies(chemical_formula="HCl", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    cl_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="Cl", value=cl_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: Cl, CO, H2, HCl, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            5.03634939e-05,
            5.95239971e01,
            3.83989363e-01,
            2.58128843e-02,
            8.74017652e-08,
            1.33693478e01,
            3.93222473e-01,
        ]
    )

    initial_pressures: np.ndarray = np.array(
        [
            1e-05,
            1e01,
            1e-01,
            1e-02,
            1e-08,
            1e01,
            1e-01,
        ]
    )

    system.solve(constraints, initial_log10_pressures=np.log10(initial_pressures))
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_Cl2() -> None:
    """Tests H2-H2O and CO-CO2 and Cl-Cl2."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="Cl", solubility=NoSolubility()),
        GasSpecies(chemical_formula="Cl2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    cl_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="Cl", value=cl_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: Cl, CO, Cl2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            2.37795934e-02,
            5.95241259e01,
            1.04185642e-03,
            3.84011132e-01,
            8.74017606e-08,
            1.33693764e01,
            3.93244755e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region F


def test_hydrogen_and_carbon_species_with_HF() -> None:
    """Tests H2-H2O and CO-CO2 and HF-F."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="F", solubility=NoSolubility()),
        GasSpecies(chemical_formula="HF", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    f_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="F", value=f_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: F, CO, H2, HF, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            5.19770426e-08,
            5.95072000e01,
            3.83970034e-01,
            4.82490653e-02,
            8.74017751e-08,
            1.33655759e01,
            3.93202701e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_F2() -> None:
    """Tests H2-H2O and CO-CO2 and F2-F."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
        GasSpecies(chemical_formula="F", solubility=NoSolubility()),
        GasSpecies(chemical_formula="F2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(oceans)
    c_kg: float = ch_ratio * h_kg
    f_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="F", value=f_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: F, CO, F2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.82263598e-02,
            5.95059369e01,
            1.08422977e-05,
            3.84010699e-01,
            8.74017669e-08,
            1.33652915e01,
            3.93244325e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
