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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Type

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

if TYPE_CHECKING:
    from atmodeller.constraints import SystemConstraints

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
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species"""
        return filter_by_type(self, GasSpecies)

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return self.number_species(GasSpecies)

    @property
    def elements_in_gas_species(self) -> list[str]:
        """Elements in gas species"""
        return self.elements(GasSpecies)

    @property
    def condensed_species(self) -> dict[int, CondensedSpecies]:
        """Condensed species"""
        return filter_by_type(self, CondensedSpecies)

    @property
    def number_condensed_species(self) -> int:
        """Number of condensed species"""
        return self.number_species(CondensedSpecies)

    @property
    def elements_in_condensed_species(self) -> list[str]:
        """Elements in condensed species"""
        return self.elements(CondensedSpecies)

    @property
    def names(self) -> list[str]:
        """Unique names of the species"""
        return [species.name for species in self.data]

    def number_species(self, species_type: Type[ChemicalSpecies] = ChemicalSpecies) -> int:
        """Number of species

        Args:
            species_type: Filter by species type. Defaults to ChemicalSpecies (i.e. return all).

        Returns:
            Number of species
        """
        filtered_species: dict[int, ChemicalSpecies] = filter_by_type(self, species_type)

        return len(filtered_species)

    def elements(self, species_type: Type[ChemicalSpecies] = ChemicalSpecies) -> list[str]:
        """Unique elements in the species.

        Args:
            species_type: Filter by species type. Defaults to ChemicalSpecies (i.e. return all).

        Returns:
            A list of unique elements
        """
        elements: list[str] = []
        filtered_species: dict[int, ChemicalSpecies] = filter_by_type(self, species_type)
        for species in filtered_species.values():
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))

        return unique_elements

    def number_elements(self, species_type: Type[ChemicalSpecies] = ChemicalSpecies) -> int:
        """Number of elements

        Args:
            species_type: Filter by species type. Defaults to ChemicalSpecies (i.e. return all).

        Returns:
            Number of elements in species
        """
        return len(self.elements(species_type))

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
        matrix: npt.NDArray[np.int_] = np.zeros(
            (self.number_species(), self.number_elements()), dtype=np.int_
        )
        for species_index, species in enumerate(self.data):
            for element_index, element in enumerate(self.elements()):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                matrix[species_index, element_index] = count

        return matrix


@dataclass
class Solution:
    """The solution"""

    species: Species
    constraints: SystemConstraints
    _data: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Sets the order/indexing of the arrays"""
        # Work using species to set up ordering and indexing
        self.data = np.zeros_like(self.species, dtype=np.float_)
        # Create ordered dict
        for species in self.species:
            self._data[species] = 0
        for elements in self.species.elements(CondensedSpecies):
            self._data[elements] = 0

    # TODO: Rename, since some terms not log.
    @property
    def log_solution(self) -> npt.NDArray:
        """The solution.

        For gas species and condensed species the solution is the log10 activity and log10 partial
        pressure, respectively. Subsequent entries in the solution array relate to the degree of
        condensation for elements in condensed species, if applicable.
        """
        return np.array(list(self._data.values()))

    @log_solution.setter
    def log_solution(self, value: npt.NDArray) -> None:
        """The solution."""
        for nn, key in enumerate(self._data):
            self._data[key] = value[nn]

    @property
    def solution(self) -> npt.NDArray:
        """Solution"""
        return 10**self.log_solution

    @property
    def assemble_reaction(self) -> npt.NDArray:
        """Assembles modified activities, partial pressures, and lamdba factors"""

    @property
    def assemble_auxilliary(self) -> npt.NDArray:
        """Assembles the auxilliary equations"""

    @property
    def activities_dict(self) -> dict[str, float]:
        """Gets the activities"""
        # Compute from activities_pressures which contains the modified activities

    @property
    def log10_activities_dict(self) -> dict[str, float]:
        """Gets the log10 activities"""
        # Compute from activities_pressures which contraints the modified activities

    @property
    def degree_of_condensation_dict(self) -> dict[str, float]:
        """Gets the degree of condensation"""
        # Compute from betas

    @property
    def pressures_dict(self) -> dict[str, float]:
        """Gets the pressures"""
        # Compute from activities_pressures which contains the modified pressures

    @property
    def log10_pressures_dict(self) -> dict[str, float]:
        """Gets the log10 pressures"""
        # Compute from activities_pressures the log10 pressures

    @property
    def degree_of_condensation_number(self) -> int:
        """Number of elements to solve for the degree of condensation"""
        return len(self.degree_of_condensation_elements)

    @property
    def degree_of_condensation_elements(self) -> list[str]:
        """Elements to solve for the degree of condensation

        The elements for which to calculate the degree of condensation depends on both which
        elements are in condensed species and which mass constraints are applied.
        """
        # condensation: list[str] = list(self.species.elements(CondensedSpecies))
        # FIXME: Hack to remove O for testing
        # if "O" in condensation:
        #    condensation.remove("O")

        condensation: list[str] = []
        for constraint in self.constraints.mass_constraints:
            if constraint.element in self.species.elements_in_condensed_species:
                condensation.append(constraint.element)

        return condensation

    def solution_dict(self) -> dict[str, float]:
        """Solution for all species in a dictionary

        This is convenient for a quick check of the solution, but in general you will use
        self.output to return a dictionary of all the data or export the data to Excel or a
        DataFrame.
        """
        output: dict[str, float] = {}
        # Gas species partial pressures
        for name, solution in zip(
            self.species.names, self.solution[: self.species.number_species()]
        ):
            output[name] = solution
        # Degree of condensation for elements in condensed species
        for degree_of_condensation, solution in zip(
            self.degree_of_condensation_elements, self.solution[self.species.number_species() :]
        ):
            key: str = f"degree_of_condensation_{degree_of_condensation}"
            output[key] = solution / (1 + solution)

        return output

    # @property
    # def fugacities_dict(self) -> dict[str, float]:
    #     """Fugacities of all species in a dictionary."""
    #     output: dict[str, float] = {}
    #     for key, value in self.log10_fugacity_coefficients_dict.items():
    #         # TODO: Not clean to append _g suffix to denote gas phase.
    #         output[key] = 10 ** (np.log10(self.solution_dict()[f"{key}_g"]) + value)

    #     return output
