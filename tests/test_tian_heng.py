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


from typing import Type

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintBallhaus,
    SystemConstraints,
)
from atmodeller.interfaces import (
    GasSpecies,
    SolidSpecies,
    ThermodynamicData,
    ThermodynamicDataBase,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species

thermodynamic_data: Type[ThermodynamicDataBase] = ThermodynamicData

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_graphite() -> None:
    """Tests including graphite."""

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2", thermodynamic_class=thermodynamic_data),
            GasSpecies(chemical_formula="H2O", thermodynamic_class=thermodynamic_data),
            GasSpecies(chemical_formula="CO", thermodynamic_class=thermodynamic_data),
            GasSpecies(chemical_formula="CO2", thermodynamic_class=thermodynamic_data),
            GasSpecies(chemical_formula="CH4", thermodynamic_class=thermodynamic_data),
            GasSpecies(chemical_formula="O2", thermodynamic_class=thermodynamic_data),
            SolidSpecies(
                chemical_formula="C",
                name_in_thermodynamic_data="graphite",
                thermodynamic_class=thermodynamic_data,
            ),  # Ideal activity by default.
        ]
    )

    planet: Planet = Planet()
    planet.surface_temperature = 600 + 273  # K

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    # This is comparable to the constraints imposed by Meng.
    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintBallhaus(),
            FugacityConstraint(species="H2", value=44.49334998176607),
        ]
    )

    target_pressures: dict[str, float] = {
        "C": 1.0,
        "CH4": 941.8712626308082,
        "CO": 0.08171351242186498,
        "CO2": 0.07149671191170186,
        "H2": 44.493349981766066,
        "H2O": 14.609207391097847,
        "O2": 1.4546209065940728e-25,
    }

    system.solve(constraints)
    system.solution_dict
    assert system.isclose(target_pressures)
