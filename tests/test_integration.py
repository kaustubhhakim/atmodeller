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
    LibourelN2,
    NoSolubility,
    PeridotiteH2O,
)

# Tolerances to compare the test results with predefined 'correct' output.
rtol: float = 1.0e-5
atol: float = 1.0e-8


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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array(
        [0.5489863889689693, 3.9099946023954954e-08, 0.3743939788917565]
    )
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    planet.fo2_shift = 2
    target_pressures: np.ndarray = np.array(
        [0.05550865439977582, 3.909994602395495e-06, 0.37855412085333406]
    )
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    planet.fo2_shift = -2
    target_pressures: np.ndarray = np.array(
        [4.454469880194281, 3.9099946023954873e-10, 0.3037828871188437]
    )
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array(
        [12.978774232084112, 3.9099946023954954e-08, 8.851175591099091]
    )
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    planet.fo2_shift = 0
    target_pressures: np.ndarray = np.array(
        [48.698006951946226, 3.9099946023954954e-08, 33.210733368233136]
    )
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    planet.surface_temperature = 1500.0  # K
    target_pressures: np.ndarray = np.array(
        [0.3528353608781187, 3.9485170552923485e-12, 0.37631338885534227]
    )
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    target_pressures: np.ndarray = np.array(
        [
            62.30341145750454,
            0.5547795429001989,
            3.9099946023954954e-08,
            9.46718162781583,
            0.378344754346716,
        ]
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    target_pressures: np.ndarray = np.array(
        [
            313.8107777311009,
            0.5547967967252911,
            3.9099946023954954e-08,
            47.68444552306492,
            0.37835652099222344,
        ]
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    target_pressures: np.ndarray = np.array(
        [
            628.3162180280887,
            0.5547989433868151,
            3.9099946023954954e-08,
            95.47444701052173,
            0.37835798495775264,
        ]
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)

    target_pressures: np.ndarray = np.array(
        [
            [
                53.51149311593031,
                0.3551508687993751,
                3.9485170552923485e-12,
                21.063886669582402,
                0.3787829730563159,
                2.5699524854819747e-05,
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
        Molecule(name="N2", solubility=LibourelN2()),
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Order of target pressures: CO, H2, N2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            62.189238254089844,
            0.5547751075029279,
            2.2915167502949956,
            3.9099946023954954e-08,
            9.449832682894986,
            0.37834172952485223,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# pylint: disable=invalid-name
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Order of target pressures: CO, H2, O2, CO2, H2O, NH3
    target_pressures: np.ndarray = np.array(
        [
            60.69264462816495,
            0.5366955218828879,
            3.9099946023954954e-08,
            9.222420999518285,
            0.36601193750630423,
            4.7044383283014914,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, SO2
    target_pressures: np.ndarray = np.array(
        [
            4.40383433e-03,
            6.23261060e01,
            5.54780423e-01,
            3.90999460e-08,
            9.47063013e00,
            3.78345354e-01,
            2.33789598e-02,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: S, CO, H2, O2, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            0.0026664373252114925,
            62.33683256651159,
            0.5540916625510903,
            3.9099946023954954e-08,
            9.47226006095088,
            0.3778756384158313,
            0.2752108986618988,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: CO, H2, O2, SO, CO2, H2O, H2S
    target_pressures: np.ndarray = np.array(
        [
            62.34098189874176,
            0.554110911641855,
            3.9099946023954954e-08,
            0.010292182451977246,
            9.472890563854948,
            0.37788876577896124,
            0.2676042609678805,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: Cl, CO, H2, HCl, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.096920265186131e-05,
            62.3081022111629,
            0.5547482993049339,
            0.025079315577599667,
            3.9099946023954954e-08,
            9.467894401254982,
            0.3783234470535277,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: Cl, CO, Cl2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            0.02316146690382876,
            62.3081814216559,
            0.0009794254333854388,
            0.5547797282133187,
            3.9099946023954954e-08,
            9.467906437516012,
            0.37834488072528877,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: F, CO, H2, HF, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            4.237244843244206e-08,
            62.29054758908645,
            0.5547203056288427,
            0.0468633913778612,
            3.9099946023954954e-08,
            9.465226926204767,
            0.37830435611796903,
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

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    # Here the order of target pressures is: F, CO, F2, H2, O2, CO2, H2O
    target_pressures: np.ndarray = np.array(
        [
            0.046842045393683314,
            62.28917450144905,
            1.0159997648474007e-05,
            0.554778990141948,
            3.9099946023954954e-08,
            9.465018281609696,
            0.3783443773804284,
        ]
    )
    system.solve(constraints, fo2_constraint=True)
    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()


# endregion
