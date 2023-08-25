"""Reproducing the results of Tian and Heng (2023)."""


from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    FugacityConstraint,
    IronWustiteBufferBallhaus,
    SystemConstraints,
)
from atmodeller.core import InteriorAtmosphereSystem, Planet
from atmodeller.interfaces import NoSolubility
from atmodeller.thermodynamics import (
    ChemicalComponent,
    GasSpecies,
    SolidSpecies,
    StandardGibbsFreeEnergyOfFormation,
    StandardGibbsFreeEnergyOfFormationProtocol,
)

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormation()
)

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_graphite() -> None:
    """Tests including graphite."""

    species: list[ChemicalComponent] = [
        GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="H2O", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CO2", solubility=NoSolubility()),
        GasSpecies(chemical_formula="CH4", solubility=NoSolubility()),
        GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        SolidSpecies(chemical_formula="C", common_name="graphite"),  # Ideal activity by default.
    ]

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    # This is comparable to the constraints imposed by Meng.
    constraints: list = [
        BufferedFugacityConstraint(value=IronWustiteBufferBallhaus()),
        FugacityConstraint(species="H2", value=44.49334998176607),
    ]

    system_constraints: SystemConstraints = SystemConstraints(constraints)

    target_pressures: dict[str, float] = {
        "C": 1.0,
        "CH4": 941.9462267908025,
        "CO": 0.0817264668302245,
        "CO2": 0.0715136912730243,
        "H2": 44.493349981766066,
        "H2O": 14.610360605730092,
        "O2": 1.4548505639981167e-25,
    }

    system.solve(system_constraints)
    assert system.isclose(target_pressures)
