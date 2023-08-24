"""Integration tests.

Tests to ensure that sensible pressures are calculated for certain interior-atmosphere systems.

The target pressures are determined for the combined thermodynamic data, but they are within 1%
of the values for the JANAF thermodynamic data alone.

"""

from atmodeller import (
    BufferedFugacityConstraint,
    GasSpecies,
    InteriorAtmosphereSystem,
    MassConstraint,
    Planet,
    SystemConstraint,
    __version__,
)
from atmodeller.solubilities import BasaltDixonCO2, PeridotiteH2O
from atmodeller.thermodynamics import (
    ChemicalComponent,
    NoSolubility,
    StandardGibbsFreeEnergyOfFormation,
    StandardGibbsFreeEnergyOfFormationJANAF,
    StandardGibbsFreeEnergyOfFormationProtocol,
)
from atmodeller.utilities import earth_oceans_to_kg

# Uncomment to test JANAF only.
# standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
#    StandardGibbsFreeEnergyOfFormationJANAF()
# )
# Uncomment to test the combined dataset.
standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormation()
)

# Both the combined data and JANAF report the same pressures to within 1%.
rtol: float = 1.0e-2
atol: float = 1.0e-2


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

    target_pressures: dict[str, float] = {
        "H2": 0.3857055348248646,
        "H2O": 0.390491491329448,
        "O2": 8.699912766341827e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


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

    target_pressures: dict[str, float] = {
        "H2": 0.0388388984114118,
        "H2O": 0.39320395339870556,
        "O2": 8.699723182761213e-06,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


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

    target_pressures: dict[str, float] = {
        "H2": 3.3586796133087784,
        "H2O": 0.3400669822055608,
        "O2": 8.7015229126454e-10,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


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

    target_pressures: dict[str, float] = {
        "H2": 9.257384917231544,
        "H2O": 9.377554217549234,
        "O2": 8.709756497114863e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


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

    target_pressures: dict[str, float] = {
        "H2": 0.46913986286211257,
        "H2O": 0.38967139331200135,
        "O2": 2.5007338977221298e-12,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


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

    target_pressures: dict[str, float] = {
        "CO": 59.615758867959656,
        "CO2": 13.239309714148467,
        "H2": 0.3875227796643467,
        "H2O": 0.3932373500163688,
        "O2": 8.740142990935366e-08,
    }

    system.solve(constraints)
    assert system.isclose(target_pressures, rtol=rtol, atol=atol)


# endregion
