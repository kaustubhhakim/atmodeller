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
"""Core"""

from __future__ import annotations

import logging
import sys
from collections import UserList
from typing import Mapping

import numpy as np
import numpy.typing as npt

from atmodeller.eos.interfaces import IdealGas, RealGasProtocol
from atmodeller.interfaces import ChemicalSpecies, CondensedSpecies
from atmodeller.solubility.compositions import composition_solubilities
from atmodeller.solubility.interfaces import NoSolubility, SolubilityProtocol
from atmodeller.utilities import filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class GasSpecies(ChemicalSpecies):
    """A gas species

    Args:
        formula: Chemical formula (e.g. CO2, C, CH4, etc.)
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0.
        solubility: Solubility model. Defaults to no solubility
        eos: A gas equation of state. Defaults to an ideal gas.

    Attributes:
        solubility: Solubility model
    """

    @override
    def __init__(
        self,
        formula: str,
        *,
        solid_melt_distribution_coefficient: float = 0,
        solubility: SolubilityProtocol = NoSolubility(),
        eos: RealGasProtocol = IdealGas(),
        **kwargs,
    ):
        super().__init__(formula, "g", **kwargs)
        self._solid_melt_distribution_coefficient: float = solid_melt_distribution_coefficient
        self.solubility: SolubilityProtocol = solubility
        self._eos: RealGasProtocol = eos

    @property
    def eos(self) -> RealGasProtocol:
        """A gas equation of state"""
        return self._eos

    @property
    def solid_melt_distribution_coefficient(self) -> float:
        """Distribution coefficient between solid and melt"""
        return self._solid_melt_distribution_coefficient


class SolidSpecies(CondensedSpecies):
    """A solid species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        activity: Activity model. Defaults to unity for a pure component.
    """

    @override
    def __init__(self, formula: str, **kwargs):
        super().__init__(formula, "cr", **kwargs)


class LiquidSpecies(CondensedSpecies):
    """A liquid species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        activity: Activity model. Defaults to unity for a pure component.
    """

    @override
    def __init__(self, formula: str, **kwargs):
        super().__init__(formula, "l", **kwargs)


class Species(UserList):
    """A list of species

    Args:
        initlist: Initial list of species. Defaults to None.
    """

    # UserList itself is not a generic class, so this is for typing:
    data: list[ChemicalSpecies]
    """List of species"""

    @property
    def elements(self) -> list[str]:
        elements: list[str] = []
        for species in self.data:
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))

        return unique_elements

    @property
    def number(self) -> int:
        """Number of species"""
        return len(self.data)

    @property
    def number_elements(self) -> int:
        """Number of elements"""
        return len(self.elements)

    @property
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species"""
        return filter_by_type(self, GasSpecies)

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return len(self.gas_species)

    @property
    def condensed_species(self) -> dict[int, CondensedSpecies]:
        """Condensed species"""
        return filter_by_type(self, CondensedSpecies)

    @property
    def condensed_elements(self) -> list[str]:
        """Elements in condensed species"""
        elements: list[str] = []
        for species in self.condensed_species.values():
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))

        return unique_elements

    @property
    def number_condensed_species(self) -> int:
        """Number of condensed species"""
        return len(self.condensed_species)

    def find_species(self, find_species: ChemicalSpecies) -> int:
        """Finds a species and returns its index.

        Args:
            find_species: Species to find

        Returns:
            Index of the species

        Raises:
            ValueError: The species is not in the species list
        """

        for index, species in enumerate(self):
            if species is find_species:
                return index

        raise ValueError(f"{find_species.name} is not in the species list")

    def check_species_present(self, find_species: ChemicalSpecies) -> bool:
        """Checks if a species is present

        Args:
            species: Species to find

        Returns:
            True if the species is present, otherwise False
        """
        for species in self:
            if species is find_species:
                logger.debug("Found %s in the species list", find_species.name)
                return True

        logger.debug("%s not found in the species list", find_species.name)

        return False

    @property
    def names(self) -> list[str]:
        """Unique names of the species"""
        return [species.name for species in self.data]

    def conform_solubilities_to_composition(self, melt_composition: str | None = None) -> None:
        """Conforms the solubilities of the gas species to the planet composition.

        Args:
            melt_composition: Composition of the melt. Defaults to None.
        """
        if melt_composition is not None:
            logger.info(
                "Setting solubilities to be consistent with the melt composition (%s)",
                melt_composition,
            )
            try:
                solubilities: Mapping[str, SolubilityProtocol] = composition_solubilities[
                    melt_composition.casefold()
                ]
            except KeyError as exc:
                raise ValueError(f"Cannot find solubilities for {melt_composition}") from exc

            for species in self.gas_species.values():
                try:
                    species.solubility = solubilities[species.hill_formula]
                    logger.info(
                        "Found solubility law for %s: %s",
                        species.hill_formula,
                        species.solubility.__class__.__name__,
                    )
                except KeyError:
                    logger.info("No solubility law for %s", species.hill_formula)
                    species.solubility = NoSolubility()

    def composition_matrix(self) -> npt.NDArray:
        """Creates a matrix where species (rows) are split into their element counts (columns).

        Returns:
            A matrix of element counts
        """
        matrix: npt.NDArray[np.int_] = np.zeros((self.number, self.number_elements), dtype=np.int_)
        for species_index, species in enumerate(self.data):
            for element_index, element in enumerate(self.elements):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                matrix[species_index, element_index] = count

        return matrix
