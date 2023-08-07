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
    target_pressures: np.ndarray = np.array([5.72483121e-01, 3.91041669e-08, 3.88572857e-01])
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
    target_pressures: np.ndarray = np.array([5.79128485e-02, 3.91041669e-06, 3.93083398e-01])
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
    target_pressures: np.ndarray = np.array([4.60871136e00, 3.91041669e-10, 3.12816234e-01])
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
    target_pressures: np.ndarray = np.array([1.35033397e01, 3.91041669e-08, 9.16539041e00])
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
    target_pressures: np.ndarray = np.array([5.05449808e01, 3.91041669e-08, 3.43074003e01])
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
    target_pressures: np.ndarray = np.array([3.74207325e-01, 3.94917324e-12, 3.90596029e-01])
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
        [6.24207531e01, 5.78786802e-01, 3.91041669e-08, 9.27225826e00, 3.92851480e-01]
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
        [3.14496733e02, 5.78806369e-01, 3.91041669e-08, 4.67167534e01, 3.92864761e-01]
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
        [6.29708610e02, 5.78808803e-01, 3.91041669e-08, 9.35397375e01, 3.92866413e-01]
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
            5.31966144e01,
            3.76818044e-01,
            3.94917324e-12,
            2.14255540e01,
            3.93321088e-01,
            3.04645336e-05,
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
            6.23094711e01,
            5.78782110e-01,
            2.28753489e00,
            3.91041669e-08,
            9.25572793e00,
            3.92848296e-01,
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
            6.08116915e01,
            5.59919395e-01,
            3.91041669e-08,
            9.03324105e00,
            3.80045230e-01,
            4.69662215e00,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
