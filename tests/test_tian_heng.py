"""Reproducing the results of Tian and Heng (2023)."""

# Tolerances to compare the test results with predefined 'correct' output.
import numpy as np

from atmodeller import __version__
from atmodeller.core import (
    BufferedFugacityConstraint,
    FugacityConstraint,
    InteriorAtmosphereSystem,
    SystemConstraint,
)
from atmodeller.thermodynamics import (
    GasPhase,
    IronWustiteBufferBallhaus,
    NoSolubility,
    Phase,
    Planet,
    SolidPhase,
    StandardGibbsFreeEnergyOfFormation,
    StandardGibbsFreeEnergyOfFormationProtocol,
)

rtol: float = 1.0e-4
atol: float = 1.0e-4

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormation()
)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_graphite() -> None:
    """Tests including graphite."""

    molecules: list[Phase] = [
        GasPhase(chemical_formula="H2", solubility=NoSolubility()),
        GasPhase(chemical_formula="H2O", solubility=NoSolubility()),
        GasPhase(chemical_formula="CO", solubility=NoSolubility()),
        GasPhase(chemical_formula="CO2", solubility=NoSolubility()),
        GasPhase(chemical_formula="CH4", solubility=NoSolubility()),
        GasPhase(chemical_formula="O2", solubility=NoSolubility()),
        SolidPhase(chemical_formula="C", common_name="graphite"),
    ]

    planet: Planet = Planet()
    # Typical temperature considered.
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        molecules=molecules, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    # This is comparable to the constraints imposed by Meng, i.e. an activity of C, an fO2, and a
    # single pressure (for Meng it's the total, but for us we just impose one and solve for the
    # others).
    constraints: list[SystemConstraint] = [
        # Note the buffer excludes pressure-dependence
        BufferedFugacityConstraint(fugacity=IronWustiteBufferBallhaus()),
        FugacityConstraint(species="C", value=1),  # Activity
        # Below are set based on the result of Tian and Heng (2023), and then we compare the
        # output of the other quantities.
        # FugacityConstraint(species="CO2", value=0.06173121847447019),
        FugacityConstraint(species="H2", value=44.49334998176607),
    ]

    target_pressures: np.ndarray = np.array(
        [
            1,
            0.07592516605503474,
            44.493349981766066,
            1.2561211766550179e-25,
            0.0617312184744702,
            13.575444015493717,
            941.7935496182116,
        ]
    )
    system.solve(constraints)

    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()
