"""Integration tests.

Tests to ensure that 'correct' values are returned for certain interior-atmosphere systems. 
These are quite rudimentary tests, but at least confirm that nothing fundamental is broken with the
code.

"""

import numpy as np

from atmodeller import (
    OCEAN_MOLES,
    BufferedFugacityConstraint,
    GasPhase,
    InteriorAtmosphereSystem,
    MassConstraint,
    Planet,
    SystemConstraint,
    __version__,
)
from atmodeller.thermodynamics import (
    BasaltDixonCO2,
    BasaltLibourelN2,
    NoSolubility,
    PeridotiteH2O,
    PhaseProtocol,
    StandardGibbsFreeEnergyOfFormationJANAF,
    StandardGibbsFreeEnergyOfFormationProtocol,
)
from atmodeller.utilities import MolarMasses

# Tolerances to compare the test results with predefined 'correct' output.
rtol: float = 1.0e-4
atol: float = 1.0e-4

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormationJANAF()
)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# region oxygen fugacity


def test_hydrogen_species_oxygen_fugacity_buffer() -> None:
    """Tests H2-H2O at the IW buffer."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
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

    target_pressures: np.ndarray = np.array([3.82233051e-01, 8.70003606e-08, 3.90520470e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_oxygen_fugacity_buffer_shift_positive() -> None:
    """Tests H2-H2O at the IW buffer+2."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
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

    target_pressures: np.ndarray = np.array([3.84857931e-02, 8.70003606e-06, 3.93202261e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_oxygen_fugacity_buffer_shift_negative() -> None:
    """Tests H2-H2O at the IW buffer-2."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
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

    target_pressures: np.ndarray = np.array([3.33362588e00, 8.70003606e-10, 3.40590418e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region number of oceans


def test_hydrogen_species_five_oceans() -> None:
    """Tests H2-H2O for five H oceans."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
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

    target_pressures: np.ndarray = np.array([9.18186063e00, 8.70003606e-08, 9.38093793e00])
    system.solve(constraints)

    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_species_ten_oceans() -> None:
    """Tests H2-H2O for ten H oceans."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
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

    target_pressures: np.ndarray = np.array([3.50479309e01, 8.70003606e-08, 3.58078256e01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region temperature


def test_hydrogen_species_temperature() -> None:
    """Tests H2-H2O at a different temperature."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
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
    target_pressures: np.ndarray = np.array([4.65306079e-01, 2.50076371e-12, 3.89706477e-01])
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region C over H ratio


def test_hydrogen_and_carbon_species() -> None:
    """Tests H2-H2O and CO-CO2."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
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
        [5.95364892e01, 3.84893955e-01, 8.70003606e-08, 1.33413117e01, 3.93239066e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_five_ch_ratio() -> None:
    """Tests H2-H2O and CO-CO2 for C/H=5."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
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
        [2.99410780e02, 3.84901446e-01, 8.70003606e-08, 6.70938543e01, 3.93246720e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_ten_ch_ratio() -> None:
    """Tests H2-H2O and CO-CO2 for C/H=10."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
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
        [5.99479301e02, 3.84902379e-01, 8.70003606e-08, 1.34335099e02, 3.93247673e-01]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region methane


def test_hydrogen_and_carbon_species_with_methane() -> None:
    """Tests H2-H2O and CO-CO2 and N."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="CH4", solubility=NoSolubility()),
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
            5.57770613e01,
            4.69402598e-01,
            2.50076371e-12,
            1.80957041e01,
            3.93137423e-01,
            6.22418059e-05,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion


# region nitrogen


def test_hydrogen_and_carbon_species_with_nitrogen() -> None:
    """Tests H2-H2O and CO-CO2 and N."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="N2", solubility=BasaltLibourelN2()),
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
            5.93800226e01,
            3.84890326e-01,
            2.35319925e00,
            8.70003606e-08,
            1.33062497e01,
            3.93235359e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_H3N() -> None:
    """Tests H2-H2O and CO-CO2 and H3N."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="H3N", solubility=NoSolubility()),
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
            5.79354315e01,
            3.72348891e-01,
            8.70003606e-08,
            1.29825366e01,
            3.80422006e-01,
            4.83526107e00,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region sulfur


def test_hydrogen_and_carbon_species_with_O2S() -> None:
    """Tests H2-H2O and CO-CO2 and S-SO2."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="S", solubility=NoSolubility()),
        GasPhase(name="O2S", solubility=NoSolubility()),
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
            2.22810683e-03,
            5.95587937e01,
            3.84894470e-01,
            8.70003606e-08,
            1.33463098e01,
            3.93239593e-01,
            2.63700230e-02,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and S-H2S."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="S", solubility=NoSolubility()),
        GasPhase(name="H2S", solubility=NoSolubility()),
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
            3.93076551e-03,
            5.95610100e01,
            3.84418199e-01,
            8.70003606e-08,
            1.33468065e01,
            3.92752995e-01,
            2.82061677e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_OS_H2S() -> None:
    """Tests H2-H2O and CO-CO2 and SO-H2S."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="OS", solubility=NoSolubility()),
        GasPhase(name="H2S", solubility=NoSolubility()),
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
            5.95688788e01,
            3.84448653e-01,
            8.70003606e-08,
            2.18653532e-02,
            1.33485697e01,
            3.92784110e-01,
            2.64166654e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region Cl


def test_hydrogen_and_carbon_species_with_HCl() -> None:
    """Tests H2-H2O and CO-CO2 and HCl-Cl."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="Cl", solubility=NoSolubility()),
        GasPhase(name="HCl", solubility=NoSolubility()),
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
            5.02936024e-05,
            5.95402656e01,
            3.84872238e-01,
            2.58064835e-02,
            8.70003606e-08,
            1.33421579e01,
            3.93216878e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_Cl2() -> None:
    """Tests H2-H2O and CO-CO2 and Cl-Cl2."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="Cl", solubility=NoSolubility()),
        GasPhase(name="Cl2", solubility=NoSolubility()),
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
            2.37740887e-02,
            5.95403939e01,
            1.04137340e-03,
            3.84894045e-01,
            8.70003606e-08,
            1.33421866e01,
            3.93239158e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion

# region F


def test_hydrogen_and_carbon_species_with_HF() -> None:
    """Tests H2-H2O and CO-CO2 and HF-F."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="F", solubility=NoSolubility()),
        GasPhase(name="HF", solubility=NoSolubility()),
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
            5.19053910e-08,
            5.95234645e01,
            3.84852886e-01,
            4.82371395e-02,
            8.70003606e-08,
            1.33383930e01,
            3.93197107e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


def test_hydrogen_and_carbon_species_with_F2() -> None:
    """Tests H2-H2O and CO-CO2 and F2-F."""

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="F", solubility=NoSolubility()),
        GasPhase(name="F2", solubility=NoSolubility()),
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
            4.82144448e-02,
            5.95222004e01,
            1.08370066e-05,
            3.84893624e-01,
            8.70003606e-08,
            1.33381097e01,
            3.93238728e-01,
        ]
    )
    system.solve(constraints)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
