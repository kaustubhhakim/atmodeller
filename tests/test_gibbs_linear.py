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
    StandardGibbsFreeEnergyOfFormationLinear,
)

# Tolerances to compare the test results with predefined 'correct' output.
rtol: float = 1.0e-5
atol: float = 1.0e-8

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormation = (
    StandardGibbsFreeEnergyOfFormationLinear()
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array([5.48957212e-01, 3.91041669e-08, 3.74394287e-01])
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
    target_pressures: np.ndarray = np.array([5.55056611e-02, 3.91041669e-06, 3.78554139e-01])
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
    target_pressures: np.ndarray = np.array([4.45428703e00, 3.91041669e-10, 3.03786813e-01])
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
    target_pressures: np.ndarray = np.array([1.29781230e01, 3.91041669e-08, 8.85120916e00])
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
    target_pressures: np.ndarray = np.array([4.86957152e01, 3.91041669e-08, 3.32109629e01])
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
    target_pressures: np.ndarray = np.array([3.52806291e-01, 3.94917324e-12, 3.76313649e-01])
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
        [6.23030960e01, 5.54749684e-01, 3.91041669e-08, 9.46764468e00, 3.78344811e-01]
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
        [3.13809096e02, 5.54766936e-01, 3.91041669e-08, 4.76867637e01, 3.78356577e-01]
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
        [6.28312839e02, 5.54769082e-01, 3.91041669e-08, 9.54790867e01, 3.78358041e-01]
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
            [
                5.35104415e01,
                3.55121415e-01,
                3.94917324e-12,
                2.10652229e01,
                3.78783030e-01,
                2.56926226e-05,
            ]
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
            6.21889299e01,
            5.54745249e-01,
            2.29125901e00,
            3.91041669e-08,
            9.45029587e00,
            3.78341787e-01,
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
            6.06923288e01,
            5.36666637e-01,
            3.91041669e-08,
            9.22287077e00,
            3.66011993e-01,
            4.70445471e00,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region sulfur


def test_hydrogen_and_carbon_species_with_SO2() -> None:
    """Tests H2-H2O and CO-CO2 and S-SO2."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="S", solubility=NoSolubility()),
        Molecule(name="SO2", solubility=NoSolubility()),
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="S", value=s_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, SO2
    target_pressures: np.ndarray = np.array(
        [
            4.40345051e-03,
            6.23257907e01,
            5.54750564e-01,
            3.91041669e-08,
            9.47109339e00,
            3.78345411e-01,
            2.33794457e-02,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="S", value=s_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            2.66658922e-03,
            6.23365160e01,
            5.54061841e-01,
            3.91041669e-08,
            9.47272321e00,
            3.77875696e-01,
            2.75211763e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_SO_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and SO-H2S."""

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="SO", solubility=NoSolubility()),
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="S", value=s_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: CO, H2, O2, SO, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            6.23406657e01,
            5.54081091e-01,
            3.91041669e-08,
            1.02932885e-02,
            9.47335381e00,
            3.77888824e-01,
            2.67604173e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="Cl", value=cl_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: Cl, CO, H2, HCl, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.09704538e-05,
            6.23077867e01,
            5.54718442e-01,
            2.50794065e-02,
            3.91041669e-08,
            9.46835748e00,
            3.78323504e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="Cl", value=cl_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: Cl, CO, Cl2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            2.31615458e-02,
            6.23078659e01,
            9.79432105e-04,
            5.54749869e-01,
            3.91041669e-08,
            9.46836952e00,
            3.78344938e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="F", value=f_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: F, CO, H2, HF, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.23737443e-08,
            6.22902323e01,
            5.54690450e-01,
            4.68635634e-02,
            3.91041669e-08,
            9.46568989e00,
            3.78304413e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
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
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
        SystemConstraint(species="F", value=f_kg, field="mass"),
    ]

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )
    # Here the order of target pressures is: F, CO, F2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.68422173e-02,
            6.22888597e01,
            1.01600722e-05,
            5.54749131e-01,
            3.91041669e-08,
            9.46548132e00,
            3.78344434e-01,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
