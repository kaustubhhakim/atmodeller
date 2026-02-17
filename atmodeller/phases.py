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
"""Phase container classes for thermodynamic equilibrium calculations.

This module defines high-level phase abstractions used in the equilibrium solver, including gas
mixtures, silicate melts, solids, and pure unity-activity phases.

Each phase is represented as a `SpeciesCollection` of thermodynamic species (formula + state of
aggregation). Species are constructed from their Hill formulas and assigned an aggregation state
consistent with the JANAF/NASA convention:

    - "g" : gas
    - "l" : liquid
    - "s" : solid

These phase classes provide a structured interface for grouping species by thermodynamic role (gas,
melt, solid, pure phase) while keeping the underlying species-level thermodynamic data separate.

They are used by the equilibrium solver to:
    - distinguish gas and condensed phases,
    - apply appropriate activity conventions,
    - track phase-specific properties (e.g., O2 index in the gas phase),
    - manage multicomponent phase assemblages.
"""

import logging
from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np
from molmass import Formula

from atmodeller.constants import GAS_STATE, LIQUID_STATE, SOLID_STATE
from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.type_aliases import NpFloat

logger: logging.Logger = logging.getLogger(__name__)

TSpecies = TypeVar("TSpecies", bound=SpeciesProtocol)


def _build_species_list(
    species: str | Iterable[str], factory: Callable[[str], ChemicalSpecies]
) -> list[ChemicalSpecies]:
    """Normalize input and build a species list using a factory."""

    if isinstance(species, str):
        species = [species]

    species_list: list[ChemicalSpecies] = []

    for species_ in species:
        hill_formula: str = Formula(species_).formula
        species_list.append(factory(hill_formula))

    return species_list


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
    def create(cls, species: str | Iterable[str]) -> "GasPhase":
        """Creates an instance

        Args:
            species: A single gas species name or iterable of names

        Returns
            An instance
        """
        species_list = _build_species_list(
            species, lambda hill: ChemicalSpecies.create_gas(hill, state=GAS_STATE)
        )

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
    def create(cls, species: str | Iterable[str]) -> "MeltPhase":
        """Creates an instance

        Args:
            species: A single melt species name or iterable of names

        Returns
            An instance
        """
        species_list = _build_species_list(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=LIQUID_STATE)
        )

        return cls(species_list)


class SolidPhase(SpeciesCollection[ChemicalSpecies]):
    """Multicomponent silicate solid"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        super().__init__(species)
        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self)}"
        )

    @classmethod
    def create(cls, species: str | Iterable[str]) -> "SolidPhase":
        """Creates an instance

        Args:
            species: A single solid species name or iterable of names

        Returns
            An instance
        """
        species_list = _build_species_list(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=SOLID_STATE)
        )

        return cls(species_list)


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
            state: State of aggregation. Defaults to :const:`~atmodeller.constants.SOLID_STATE`.

        Returns
            An instance
        """
        species_list = _build_species_list(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=state)
        )

        return cls(species_list)
