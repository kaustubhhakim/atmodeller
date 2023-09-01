"""Core classes and functions for modelling interior-atmosphere systems.

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

from __future__ import annotations

import logging
from collections import UserList
from typing import Union

from atmodeller.interfaces import ChemicalComponent, GasSpecies, SolidSpecies

# FIXME: Creates a circular dependency: from atmodeller.solubilities import composition_solubilities
from atmodeller.utilities import filter_by_type

logger: logging.Logger = logging.getLogger(__name__)


class Species(UserList):
    """Collections of species for an interior-atmosphere system.

    A collection of species. It provides methods to filter species based on their phases (solid,
    gas).

    Args:
        initlist: Initial list of species. Defaults to None.

    Attributes:
        data: List of species contained in the system.
    """

    def __init__(self, initlist: Union[list[ChemicalComponent], None] = None):
        self.data: list[ChemicalComponent]  # For typing.
        super().__init__(initlist)

    @property
    def number(self) -> int:
        """Number of species."""
        return len(self.data)

    @property
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species."""
        return filter_by_type(self, GasSpecies)

    @property
    def number_gas_species(self) -> int:
        """Number of gas species."""
        return len(self.gas_species)

    @property
    def solid_species(self) -> dict[int, SolidSpecies]:
        """Solid species."""
        return filter_by_type(self, SolidSpecies)

    @property
    def number_solid_species(self) -> int:
        """Number of solid species."""
        return len(self.solid_species)

    @property
    def indices(self) -> dict[str, int]:
        """Indices of the species."""
        return {
            chemical_formula: index
            for index, chemical_formula in enumerate(self.chemical_formulas)
        }

    @property
    def chemical_formulas(self) -> list[str]:
        """Chemical formulas of the species."""
        return [species.chemical_formula for species in self.data]

    # FIXME: Creates a circular dependency.
    # def conform_solubilities_to_planet_composition(self, planet: Planet) -> None:
    #     """Ensure that the solubilities of the species are consistent with the planet composition.

    #     Args:
    #         planet: A planet.
    #     """
    #     if planet.melt_composition is not None:
    #         msg: str = (
    #             # pylint: disable=consider-using-f-string
    #             "Setting solubilities to be consistent with the melt composition (%s)"
    #             % planet.melt_composition
    #         )
    #         logger.info(msg)
    #         try:
    #             solubilities: dict[str, Solubility] = composition_solubilities[
    #                 planet.melt_composition.casefold()
    #             ]
    #         except KeyError:
    #             logger.error("Cannot find solubilities for %s", planet.melt_composition)
    #             raise

    #         for species in self.gas_species.values():
    #             try:
    #                 species.solubility = solubilities[species.chemical_formula]
    #                 logger.info(
    #                     "Found solubility law for %s: %s",
    #                     species.chemical_formula,
    #                     species.solubility.__class__.__name__,
    #                 )
    #             except KeyError:
    #                 logger.info("No solubility law for %s", species.chemical_formula)
    #                 species.solubility = NoSolubility()

    def _species_sorter(self, species: ChemicalComponent) -> tuple[int, str]:
        """Sorter for the species.

        Sorts first by species complexity and second by species name.

        Args:
            species: Species.

        Returns:
            A tuple to sort first by number of elements and second by species name.
        """
        return (species.formula.atoms, species.chemical_formula)
