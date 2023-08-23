"""Reproducing the results of Tian and Heng (2023)."""


import numpy as np

from atmodeller import __version__
from atmodeller.core import (
    BufferedFugacityConstraint,
    FugacityConstraint,
    InteriorAtmosphereSystem,
    SystemConstraint,
)
from atmodeller.thermodynamics import (
    ChemicalComponent,
    GasSpecies,
    IronWustiteBufferBallhaus,
    NoSolubility,
    Planet,
    SolidSpecies,
    StandardGibbsFreeEnergyOfFormation,
    StandardGibbsFreeEnergyOfFormationProtocol,
)

# Tolerances to compare the test results with predefined 'correct' output.
rtol: float = 1.0e-8
atol: float = 1.0e-8

standard_gibbs_free_energy_of_formation: StandardGibbsFreeEnergyOfFormationProtocol = (
    StandardGibbsFreeEnergyOfFormation()
)


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
        SolidSpecies(chemical_formula="C", common_name="graphite"),
    ]

    planet: Planet = Planet()
    # Typical temperature considered.
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
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
            1.00000000e00,
            8.17264668e-02,
            4.44933500e01,
            1.45485056e-25,
            7.15136913e-02,
            1.46103606e01,
            9.41946227e02,
        ]
    )
    system.solve(constraints)

    assert np.isclose(target_pressures, system.pressures, rtol=rtol, atol=atol).all()
