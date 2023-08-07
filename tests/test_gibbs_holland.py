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
    target_pressures: np.ndarray = np.array([5.51553404e-01, 3.91041669e-08, 3.74366814e-01])
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
    target_pressures: np.ndarray = np.array([5.57720222e-02, 3.91041669e-06, 3.78552541e-01])
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
    target_pressures: np.ndarray = np.array([4.47053919e00, 3.91041669e-10, 3.03437800e-01])
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
    target_pressures: np.ndarray = np.array([1.30360539e01, 3.91041669e-08, 8.84822023e00])
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
    target_pressures: np.ndarray = np.array([4.88994995e01, 3.91041669e-08, 3.31905301e01])
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
    target_pressures: np.ndarray = np.array([3.60458283e-01, 3.94917324e-12, 3.76244838e-01])
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
        [6.24423187e01, 5.57404144e-01, 3.91041669e-08, 9.27546172e00, 3.78338003e-01]
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
        [3.14518438e02, 5.57421618e-01, 3.91041669e-08, 4.67199776e01, 3.78349864e-01]
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
        [6.29730340e02, 5.57423793e-01, 3.91041669e-08, 9.35429654e01, 3.78351340e-01]
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
            5.32094850e01,
            3.62880656e-01,
            3.94917324e-12,
            2.14307378e01,
            3.78773301e-01,
            2.82594594e-05,
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
            6.23303573e01,
            5.57399768e-01,
            2.28828685e00,
            3.91041669e-08,
            9.25883047e00,
            3.78335033e-01,
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
            6.08310308e01,
            5.39234629e-01,
            3.91041669e-08,
            9.03611379e00,
            3.66005447e-01,
            4.69816278e00,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
