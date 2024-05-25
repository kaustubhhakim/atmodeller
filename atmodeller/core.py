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
    def gas_species(self) -> list[GasSpecies]:
        """Gas species"""
        return list(filter_by_type(self, GasSpecies).values())

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return self.number_species(GasSpecies)

    @property
    def elements_in_gas_species(self) -> list[str]:
        """Elements in gas species"""
        return self.elements(GasSpecies)

    @property
    def condensed_species(self) -> list[CondensedSpecies]:
        """Condensed species"""
        return list(filter_by_type(self, CondensedSpecies).values())

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

            for species in self.gas_species:
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
    """The solution

    Stores and updates the solution and assembles the appropriate vectors to solve the coupled
    reaction network and mass balance system.

    The ordering of the solution vector must be maintained for consistency and is organised as
    follows:

    # FIXME: Note log10 or not quantities

        1. Species activities and fugacities, ordered according to the input species list
        2. Lambda factors for condensed phase species
        3. Beta factors for elements in condensed phases

    Args:
        _species: Species
        _constraints: Constraints
        _temperature: Temperature
    """

    _species: Species
    _constraints: SystemConstraints
    _temperature: float
    _species_solution: dict[ChemicalSpecies, float] = field(init=False, default_factory=dict)
    _lambda_solution: dict[CondensedSpecies, float] = field(init=False, default_factory=dict)
    _beta_solution: dict[str, float] = field(init=False, default_factory=dict)

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return (
            self._species.number_species()
            + self._species.number_condensed_species
            + self.number_condensed_elements
        )

    def set_data(self, value: npt.NDArray) -> None:
        """Sets the solution dictionaries

        Args:
            value: A vector, which is usually passed by the solver
        """
        species_index: int = 0
        lambda_index: int = 0
        start_index: int = 0
        for species_index, species in enumerate(self._species):
            self._species_solution[species] = value[start_index + species_index]
        start_index += species_index + 1
        for lambda_index, species in enumerate(self._species.condensed_species):
            self._lambda_solution[species] = value[start_index + lambda_index]
        start_index += lambda_index + 1
        for beta_index, element in enumerate(self.condensed_elements):
            self._beta_solution[element] = value[start_index + beta_index]

    @property
    def species_array(self) -> npt.NDArray:
        return np.array(list(self._species_solution.values()))

    @property
    def lambda_array(self) -> npt.NDArray:
        lambda_array: npt.NDArray = np.zeros(self._species.number_species(), dtype=float)
        for species in self._species.condensed_species:
            index: int = self._species.find_species(species)
            lambda_array[index] = self._lambda_solution[species]

        return lambda_array

    @property
    def beta_array(self) -> npt.NDArray:
        return np.array(list(self._beta_solution.values()))

    @property
    def log10_gas_pressures(self) -> dict[GasSpecies, float]:
        """Log10 gas pressures"""
        return {species: self._species_solution[species] for species in self._species.gas_species}

    @property
    def gas_pressures(self) -> dict[GasSpecies, float]:
        """Gas pressures"""
        return {key: 10**value for key, value in self.log10_gas_pressures.items()}

    @property
    def total_pressure(self) -> float:
        """Total pressure"""
        return sum(self.gas_pressures.values())

    @property
    def gas_molar_mass(self) -> float:
        """Molar mass of the gas"""
        mass: float = 0
        for species in self._species.gas_species:
            mass += species.molar_mass * self.gas_pressures[species]
        mass /= self.total_pressure

        return mass

    @property
    def log10_fugacity_coefficients(self) -> dict[GasSpecies, float]:
        """Log10 fugacity coefficients"""
        log10_coefficients: dict[GasSpecies, float] = {}
        for species in self._species.gas_species:
            log10_coefficients[species] = np.log10(
                species.eos.fugacity_coefficient(self._temperature, self.total_pressure)
            )

        return log10_coefficients

    @property
    def fugacity_coefficients(self) -> dict[GasSpecies, float]:
        """Fugacity coefficients"""
        return {key: 10**value for key, value in self.log10_fugacity_coefficients.items()}

    @property
    def log10_gas_fugacities(self) -> dict[GasSpecies, float]:
        """Log10 gas fugacities"""
        log10_fugacities: dict[GasSpecies, float] = {}
        for species in self._species.gas_species:
            log10_fugacities[species] = (
                self.log10_gas_pressures[species] + self.log10_fugacity_coefficients[species]
            )

        return log10_fugacities

    @property
    def gas_fugacities(self) -> dict[GasSpecies, float]:
        """Gas fugacities"""
        return {key: 10**value for key, value in self.log10_gas_fugacities.items()}

    @property
    def gas_fugacities_by_hill_formula(self) -> dict[str, float]:
        """Gas fugacities by hill formula"""
        return {key.hill_formula: value for key, value in self.gas_fugacities.items()}

    @property
    def modified_activities(self) -> dict[CondensedSpecies, float]:
        """Modified activities"""
        return {
            species: self._species_solution[species] for species in self._species.condensed_species
        }

    @property
    def log10_activities(self) -> dict[CondensedSpecies, float]:
        """Log10 activities"""
        activities: dict[CondensedSpecies, float] = {}
        for species in self._species.condensed_species:
            activities[species] = (
                self.modified_activities[species] - self._lambda_solution[species]
            )

        return activities

    @property
    def activities(self) -> dict[CondensedSpecies, float]:
        """Activities"""
        return {species: 10**value for species, value in self.log10_activities.items()}

    @property
    def degree_of_condensation(self) -> dict[str, float]:
        """Degree of condensation for elements"""
        return {
            element: 10**value / (1 + 10**value) for element, value in self._beta_solution.items()
        }

    # @property
    # def assemble_reaction(self) -> npt.NDArray:
    #     """Assembles modified activities, partial pressures, and lamdba factors"""

    # @property
    # def assemble_auxilliary(self) -> npt.NDArray:
    #     """Assembles the auxilliary equations"""

    @property
    def number_condensed_elements(self) -> int:
        """Number of elements that are present in a condensed species"""
        return len(self.condensed_elements)

    @property
    def condensed_elements(self) -> list[str]:
        """Elements in condensed species that should adhere to mass balance

        The elements for which to calculate the degree of condensation depends on both which
        elements are in condensed species and which mass constraints are applied.
        """
        condensation: list[str] = []
        for constraint in self._constraints.mass_constraints:
            if constraint.element in self._species.elements_in_condensed_species:
                condensation.append(constraint.element)

        return condensation

    def solution_dict(self) -> dict[str, float]:
        """Solution in a dictionary"""
        output: dict[str, float] = {}
        for species, pressure in self.gas_pressures.items():
            output[species.name] = pressure
        for species, activity in self.activities.items():
            output[species.name] = activity
        for element, degree_of_condensation in self.degree_of_condensation.items():
            output[f"degree_of_condensation_{element}"] = degree_of_condensation

        return output

    def isclose(
        self, target_dict: dict[str, float], rtol: float = 1.0e-5, atol: float = 1.0e-8
    ) -> np.bool_:
        """Determines if the solution pressures are close to target values within a tolerance.

        Args:
            target_dict: Dictionary of the target values
            rtol: Relative tolerance
            atol: Absolute tolerance

        Returns:
            True if the solution is close to the target, otherwise False
        """

        if len((self.solution_dict())) != len(target_dict):
            return np.bool_(False)

        target_values: list = list(dict(sorted(target_dict.items())).values())
        solution_values: list = list(dict(sorted(self.solution_dict().items())).values())
        isclose: np.bool_ = np.isclose(target_values, solution_values, rtol=rtol, atol=atol).all()

        return isclose

    def isclose_tolerance(self, target_dict: dict[str, float], message: str = "") -> float | None:
        """Writes a log message with the tightest tolerance that is satisfied.

        Args:
            target_dict: Dictionary of the target values
            message: Message prefix to write to the logger when a tolerance is satisfied

        Returns:
            The tightest tolerance satisfied
        """
        for log_tolerance in (-6, -5, -4, -3, -2, -1):
            tol: float = 10**log_tolerance
            if self.isclose(target_dict, rtol=tol, atol=tol):
                logger.info("%s (tol = %f)".lstrip(), message, tol)
                return tol

        logger.info("%s (no tolerance < 0.1 satisfied)".lstrip(), message)
