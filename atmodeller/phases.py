#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Phases"""

import logging
from collections.abc import Iterable
from typing import TypeVar

import numpy as np
from molmass import Formula

from atmodeller.constants import GAS_STATE, LIQUID_STATE, SOLID_STATE
from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.type_aliases import NpFloat

logger: logging.Logger = logging.getLogger(__name__)
TSpecies = TypeVar("TSpecies", bound=SpeciesProtocol)


class GasPhase(SpeciesCollection[ChemicalSpecies]):
    """Multicomponent gas mixture"""

    O2_index: NpFloat
    """Index of O2 or np.nan if not present"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        super().__init__(species)
        self.O2_index = self.get_O2_index()
        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self)}"
        )

    @classmethod
    def create(cls, species: Iterable[str]) -> "GasPhase":
        """Creates an instance

        Args:
            species: An iterable of gas species names

        Returns
            An instance
        """
        species_list: list[ChemicalSpecies] = []

        for species_ in species:
            hill_formula = Formula(species_).formula
            species_to_add: ChemicalSpecies = ChemicalSpecies.create_gas(
                hill_formula, state=GAS_STATE
            )
            species_list.append(species_to_add)

        return cls(species_list)

    def get_O2_index(self) -> NpFloat:
        """Gets the species index corresponding to diatomic oxygen.

        Note:
            This returns a float array for type consistency.

        Returns:
            Index of diatomic oxygen, or np.nan if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.species):
            if species_.data.hill_formula == "O2":
                # logger.debug("Found O2 at index = %d", nn)
                return np.array(nn, dtype=float)

        return np.array(np.nan, dtype=float)


class MeltPhase(SpeciesCollection[SpeciesProtocol]):
    """Multicomponent silicate melt with optionally dissolved volatiles"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        super().__init__(species)
        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self)}"
        )

    @classmethod
    def create(cls, species: Iterable[str]) -> "MeltPhase":
        """Creates an instance

        Args:
            species: An iterable of melt species names

        Returns
            An instance
        """
        species_list: list[ChemicalSpecies] = []

        for species_ in species:
            hill_formula = Formula(species_).formula
            species_to_add: ChemicalSpecies = ChemicalSpecies.create_condensed(
                hill_formula, state=LIQUID_STATE
            )
            species_list.append(species_to_add)

        return cls(species_list)


class SolidPhase(SpeciesCollection[ChemicalSpecies]):
    """Multicomponent silicate solid"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        del species
        raise NotImplementedError("SolidPhase is not implemented yet")


class PurePhase(SpeciesCollection[ChemicalSpecies]):
    """Pure, unity-activity phases"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        super().__init__(species)
        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self)}"
        )

    @classmethod
    def create(cls, species: str, state: str = SOLID_STATE) -> "PurePhase":
        """Creates an instance

        Args:
            species: Species
            state: State of aggregation. Defaults to SOLID_STATE.

        Returns
            An instance
        """
        hill_formula = Formula(species).formula
        species_: ChemicalSpecies = ChemicalSpecies.create_condensed(hill_formula, state=state)

        return cls((species_,))
