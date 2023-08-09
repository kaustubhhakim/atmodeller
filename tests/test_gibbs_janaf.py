"""Integration tests.

Tests to ensure that 'correct' values are returned for certain interior-atmosphere systems. 
These are quite rudimentary tests, but at least confirm that nothing fundamental is broken with the
code.

"""

import numpy as np

from atmodeller import (
    OCEAN_MOLES,
    BufferedFugacityConstraint,
    InteriorAtmosphereSystem,
    MassConstraint,
    Molecule,
    Planet,
    SystemConstraint,
    __version__,
)
from atmodeller.thermodynamics import (
    BasaltDixonCO2,
    BasaltLibourelN2,
    NoSolubility,
    PeridotiteH2O,
    StandardGibbsFreeEnergyOfFormation,
    StandardGibbsFreeEnergyOfFormationJANAF,
)
from atmodeller.utilities import MolarMasses

# Tolerances to compare the test results with predefined 'correct' output.
rtol: float = 1.0e-5
atol: float = 1.0e-8

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormation = (
    StandardGibbsFreeEnergyOfFormationJANAF()
)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# region oxygen fugacity


def test_hydrogen_species_oxygen_fugacity_buffer() -> None:
    """Tests H2-H2O at the IW buffer."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([5.67367777e-01, 3.91041669e-08, 3.88629013e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_oxygen_fugacity_buffer_shift_positive() -> None:
    """Tests H2-H2O at the IW buffer+2."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(log10_shift=2),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([5.73875602e-02, 3.91041669e-06, 3.93086668e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_oxygen_fugacity_buffer_shift_negative() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(log10_shift=-2),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([4.57713506e00, 3.91041669e-10, 3.13519300e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region number of oceans


def test_hydrogen_species_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    oceans: float = 5
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([1.33896264e01, 3.91041669e-08, 9.17147134e00])
    system.solve(constraints)

    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_ten_oceans() -> None:
    """Tests H2-H2O for ten H oceans."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    oceans: float = 10
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array([5.01465044e01, 3.91041669e-08, 3.43487722e01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region temperature


def test_hydrogen_species_temperature() -> None:
    """Tests H2-H2O at a different temperature."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.surface_temperature = 1500.0  # K
    target_pressures: np.ndarray = np.array([3.71139878e-01, 3.94917324e-12, 3.90624736e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region C over H ratio


def test_hydrogen_and_carbon_species() -> None:
    """Tests H2-H2O and CO-CO2."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [6.23549375e01, 5.73547762e-01, 3.91041669e-08, 9.36786019e00, 3.92862108e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_five_ch_ratio() -> None:
    """Tests H2-H2O and CO-CO2 for C/H=5."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
    ]

    oceans: float = 1
    ch_ratio: float = 5
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [3.14148712e02, 5.73566897e-01, 3.91041669e-08, 4.71959612e01, 3.92875214e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_ten_ch_ratio() -> None:
    """Tests H2-H2O and CO-CO2 for C/H=10."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
    ]

    oceans: float = 1
    ch_ratio: float = 10
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [6.29010033e02, 5.73569277e-01, 3.91041669e-08, 9.44989808e01, 3.92876845e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region methane


def test_hydrogen_and_carbon_species_with_methane() -> None:
    """Tests H2-H2O and CO-CO2 and N."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="CH4", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    planet: Planet = Planet()
    planet.surface_temperature = 1500
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [
            5.30389045e01,
            3.73708097e-01,
            3.94917324e-12,
            2.16240561e01,
            3.93327787e-01,
            2.98514791e-05,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion


# region nitrogen


def test_hydrogen_and_carbon_species_with_nitrogen() -> None:
    """Tests H2-H2O and CO-CO2 and N."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="N2", solubility=BasaltLibourelN2()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    nitrogen_ppmw: float = 2.8
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    n_kg: float = nitrogen_ppmw * 1.0e-6 * planet.mantle_mass

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="N", value=n_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Order of target pressures: CO, H2, N2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            6.22424654e01,
            5.73543085e-01,
            2.28911406e00,
            3.91041669e-08,
            9.35096301e00,
            3.92858904e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_H3N() -> None:
    """Tests H2-H2O and CO-CO2 and H3N."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="H3N", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    nitrogen_ppmw: float = 2.8
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    n_kg: float = nitrogen_ppmw * 1.0e-6 * planet.mantle_mass

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="N", value=n_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Order of target pressures: CO, H2, O2, CO2, H2O, NH3
    target_pressures: np.ndarray = np.array(
        [
            6.07458355e01,
            5.54851200e-01,
            3.91041669e-08,
            9.12611761e00,
            3.80055553e-01,
            4.69995966e00,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region sulfur


def test_hydrogen_and_carbon_species_with_O2S() -> None:
    """Tests H2-H2O and CO-CO2 and S-SO2."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="S", solubility=NoSolubility()),
        Molecule(name="O2S", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    s_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="S", value=s_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, SO2
    target_pressures: np.ndarray = np.array(
        [
            4.39178677e-03,
            6.23777007e01,
            5.73548706e-01,
            3.91041669e-08,
            9.37128000e00,
            3.92862755e-01,
            2.33629085e-02,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and S-H2S."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="S", solubility=NoSolubility()),
        Molecule(name="H2S", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    s_kg: float = 0.01 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="S", value=s_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            2.57203281e-03,
            6.23886789e01,
            5.72836488e-01,
            3.91041669e-08,
            9.37292931e00,
            3.92374908e-01,
            2.75025364e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_OS_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and SO-H2S."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="OS", solubility=NoSolubility()),
        Molecule(name="H2S", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    s_kg: float = 0.01 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="S", value=s_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: CO, H2, O2, SO, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            6.23927069e01,
            5.72855812e-01,
            3.91041669e-08,
            9.96757132e-03,
            9.37353446e00,
            3.92388144e-01,
            2.67648334e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region Cl


def test_hydrogen_and_carbon_species_with_HCl() -> None:
    """Tests H2-H2O and CO-CO2 and HCl-Cl."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="Cl", solubility=NoSolubility()),
        Molecule(name="HCl", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    cl_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="Cl", value=cl_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: Cl, CO, H2, HCl, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            3.99998655e-05,
            6.23596585e01,
            5.73515469e-01,
            2.50548718e-02,
            3.91041669e-08,
            9.36856944e00,
            3.92839988e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_Cl2() -> None:
    """Tests H2-H2O and CO-CO2 and Cl-Cl2."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="Cl", solubility=NoSolubility()),
        Molecule(name="Cl2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    cl_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="Cl", value=cl_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: Cl, CO, Cl2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            2.31244304e-02,
            6.23597417e01,
            9.85237902e-04,
            5.73547961e-01,
            3.91041669e-08,
            9.36858194e00,
            3.92862244e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region F


def test_hydrogen_and_carbon_species_with_HF() -> None:
    """Tests H2-H2O and CO-CO2 and HF-F."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="F", solubility=NoSolubility()),
        Molecule(name="HF", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    f_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="F", value=f_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: F, CO, H2, HF, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.12670993e-08,
            6.23420939e01,
            5.73486508e-01,
            4.68159908e-02,
            3.91041669e-08,
            9.36593063e00,
            3.92820151e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_F2() -> None:
    """Tests H2-H2O and CO-CO2 and F2-F."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="F", solubility=NoSolubility()),
        Molecule(name="F2", solubility=NoSolubility()),
    ]

    oceans: float = 1
    ch_ratio: float = 1
    # sulfur_ppmw: float = 3.2
    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = ch_ratio * h_kg
    f_kg: float = 0.001 * h_kg

    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        MassConstraint(species="F", value=f_kg),
        BufferedFugacityConstraint(),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: F, CO, F2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.67945465e-02,
            6.23407149e01,
            1.02080515e-05,
            5.73547171e-01,
            3.91041669e-08,
            9.36572346e00,
            3.92861703e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
