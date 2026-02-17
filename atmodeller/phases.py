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
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

import equinox as eqx
import numpy as np
from molmass import Formula

from atmodeller.constants import CONDENSED_STATE, DISSOLVED_STATE, GAS_STATE
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.type_aliases import NpFloat

logger: logging.Logger = logging.getLogger(__name__)
TSpecies = TypeVar("TSpecies", bound=SpeciesProtocol)


class BasePhase(eqx.Module, Generic[TSpecies]):
    """Base class for a phase"""

    species: tuple[TSpecies, ...]
    """Species in the phase"""
    molar_masses: NpFloat
    """Molar masses"""
    species_names: tuple[str, ...]
    """Unique names of all species"""

    def __init__(self, species: Iterable[TSpecies]):
        logger.info(
            f"Creating {self.__class__.__name__} with species: {', '.join(str(species_) for species_ in species)}"
        )
        self.species = tuple(species)
        self.species_names = tuple([species_.data.name for species_ in self])
        self.molar_masses = np.array([species_.data.molar_mass for species_ in self], dtype=float)

    @property
    def number_species(self) -> int:
        """Number of species"""
        return self.__len__()

    def __getitem__(self, index: int) -> TSpecies:
        return self.species[index]

    def __iter__(self) -> Iterator[TSpecies]:
        return iter(self.species)

    def __len__(self) -> int:
        return len(self.species)

    def __str__(self) -> str:
        return str(tuple(str(species) for species in self.species))


class GasPhase(BasePhase[ChemicalSpecies]):
    """Gas phase"""

    O2_index: NpFloat
    """Index of O2 or np.nan if not present"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        super().__init__(species)
        self.O2_index = self.get_O2_index()

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


class MeltPhase(BasePhase[SpeciesProtocol]):
    """Melt phase

    The melt phase can contain both condensed (liquid) and dissolved species, but not gas species.
    """

    def __init__(self, species: Iterable[SpeciesProtocol]):
        super().__init__(species)

    @classmethod
    def create(cls, species: Iterable[str]) -> "MeltPhase":
        """Creates an instance

        Args:
            species: An iterable of species names with a state suffix

        Returns
            An instance
        """
        species_list: list[SpeciesProtocol] = []

        for species_ in species:
            formula, state = species_.split("_")
            hill_formula = Formula(formula).formula
            if state == CONDENSED_STATE:
                species_to_add = ChemicalSpecies.create_condensed(hill_formula, state=state)
            elif state == DISSOLVED_STATE:
                species_to_add = ReservoirSpecies.create_dissolved(hill_formula, state=state)
            else:
                raise ValueError(
                    f"State must be '{CONDENSED_STATE}' or '{DISSOLVED_STATE}', got {state}"
                )
            species_list.append(species_to_add)

        return MeltPhase(species_list)


class SolidPhase(BasePhase):
    """Solid phase"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        del species
        raise NotImplementedError("SolidPhase is not implemented yet")
