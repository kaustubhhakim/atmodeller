"""Integration tests.

Tests to ensure that 'correct' values are returned for certain interior-atmosphere systems. 
These are quite rudimentary tests, but at least confirm that nothing fundamental is broken with the
code.

"""

import numpy as np

from atmodeller import (
    OCEAN_MOLES,
    InteriorAtmosphereSystem,
    MolarMasses,
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
    StandardGibbsFreeEnergyOfFormationHolland,
)

# Tolerances to compare the test results with predefined 'correct' output.
rtol: float = 1.0e-5
atol: float = 1.0e-8

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormation = (
    StandardGibbsFreeEnergyOfFormationHolland()
)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_hydrogen_species_oxygen_fugacity_buffer() -> None:
    # region oxygen fugacity

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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array([5.51582716e-01, 3.90999460e-08, 3.74366503e-01])
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.fo2_shift = 2
    target_pressures: np.ndarray = np.array([5.57750299e-02, 3.90999460e-06, 3.78552523e-01])
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.fo2_shift = -2
    target_pressures: np.ndarray = np.array([4.47072248e00, 3.90999460e-10, 3.03433863e-01])
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array([1.30367078e01, 3.90999460e-08, 8.84818646e00])
    system.solve(constraints, fo2_constraint=True)

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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array([4.89017989e01, 3.90999460e-08, 3.31902993e01])
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.surface_temperature = 1500.0  # K
    target_pressures: np.ndarray = np.array([3.60487977e-01, 3.94851706e-12, 3.76244570e-01])
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [6.24426273e01, 5.57434145e-01, 3.90999460e-08, 9.27500694e00, 3.78337946e-01]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [3.14520086e02, 5.57451622e-01, 3.90999460e-08, 4.67177009e01, 3.78349808e-01]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [6.29733653e02, 5.57453796e-01, 3.90999460e-08, 9.35384088e01, 3.78351283e-01]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    target_pressures: np.ndarray = np.array(
        [
            5.32105507e01,
            3.62910752e-01,
            3.94851706e-12,
            2.14293865e01,
            3.78773243e-01,
            2.82670617e-05,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="N", value=n_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Order of target pressures: CO, H2, N2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            6.23306717e01,
            5.57429769e-01,
            2.28827919e00,
            3.90999460e-08,
            9.25837745e00,
            3.78334976e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_NH3() -> None:
    """Tests H2-H2O and CO-CO2 and NH3."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="NH3", solubility=NoSolubility()),
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="N", value=n_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Order of target pressures: CO, H2, O2, CO2, H2O, NH3
    target_pressures: np.ndarray = np.array(
        [
            6.08313399e01,
            5.39263652e-01,
            3.90999460e-08,
            9.03567201e00,
            3.66005391e-01,
            4.69814663e00,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
