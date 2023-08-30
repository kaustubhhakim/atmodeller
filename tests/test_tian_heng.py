"""Reproducing the results of Tian and Heng (2023).

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""


from atmodeller import __version__, debug_file_logger, debug_logger
from atmodeller.constraints import (
    IronWustiteBufferConstraintBallhaus,
    SystemConstraints,
)
from atmodeller.core import InteriorAtmosphereSystem, Planet, Species
from atmodeller.interfaces import ConstantSystemConstraint, NoSolubility
from atmodeller.thermodynamics import (
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

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="H2O", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CH4", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
            SolidSpecies(
                chemical_formula="C", common_name="graphite"
            ),  # Ideal activity by default.
        ]
    )

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, gibbs_data=standard_gibbs_free_energy_of_formation, planet=planet
    )

    # This is comparable to the constraints imposed by Meng.
    constraints: list = [
        IronWustiteBufferConstraintBallhaus(),
        ConstantSystemConstraint(name="fugacity", species="H2", value=44.49334998176607),
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
