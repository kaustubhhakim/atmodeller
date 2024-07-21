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
"""Solution"""

from __future__ import annotations

import logging
import sys
from collections import UserDict
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from molmass import Formula

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR
from atmodeller.core import GasSpecies, Species
from atmodeller.interfaces import ChemicalSpecies, CondensedSpecies
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

T = TypeVar("T", bound=ChemicalSpecies)

ACTIVITY_PREFIX: str = "activity_"
"""Prefix for the dictionary key for the activity of condensed species"""
STABILITY_PREFIX: str = "stability_"
"""Prefix for the dictionary key for the stability of condensed species"""
CONDENSED_MASS_PREFIX: str = "mass_"
"""Prefix for the dictionary key for the mass of condensed species"""

logger: logging.Logger = logging.getLogger(__name__)


class SolutionComponent(Generic[T]):
    """A solution component"""

    def __init__(self, species: T, prefix: str = ""):
        self._species: T = species
        self._name: str = f"{prefix}{species.name}"
        self._value: float

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value

    @property
    def physical(self) -> float:
        return 10**self.value

    @property
    def species(self) -> T:
        return self._species

    @property
    def name(self) -> str:
        return self._name

    def clip(self, minimum_value: float | None = None, maximum_value: float | None = None) -> None:
        """clips the value.

        Args:
            minimum_value: Minimum value. Defaults to None, meaning do not clip.
            maximum_value: Maximum value. Defaults to None, meaning do not clip.
        """
        self.value = np.clip(self.value, minimum_value, maximum_value)

    def fill(self, fill_value: float) -> None:
        """Fills value if it is missing.

        Args:
            fill_value: The fill value
        """
        if not hasattr(self, "_value"):
            self.value = fill_value

    def perturb(self, perturb: float = 0) -> None:
        """Perturbs the value.

        Args:
            perturb: Maximum log10 value to perturb the values. Defaults to 0.
        """
        self.value += perturb * (2 * np.random.rand() - 1)

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        return {self.name: self.value}

    def solution_dict(self, *args, **kwargs) -> dict[str, float]:
        """Solution in a dictionary

        Args:
            *args: Unused positional arguments
            **kwargs: Unused keyword arguments

        Returns:
            Solution dictionary
        """
        del args
        del kwargs

        return {self.name: self.physical}


class NumberDensitySolution(SolutionComponent[T]):
    """A number density solution component"""

    def number_density(self, element: str | None = None) -> float:
        """Number density of the species or element.

        Args:
            element: Element to compute the number density for, or None to compute for the species.
                Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        if element is not None:
            try:
                count: int = self.species.composition()[element].count
            except KeyError:
                # Element not in formula
                count = 0
        else:
            count = 1

        return count * self.physical

    def mass(self, gas_volume: float, element: str | None = None) -> float:
        """Mass

        Args:
            gas_volume: Gas volume in m :sup:`3`
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        if element is not None:
            molar_mass: float = UnitConversion().g_to_kg(Formula(element).mass)
        else:
            molar_mass = self.species.molar_mass

        return self.moles(gas_volume, element) * molar_mass

    def molecules(self, gas_volume: float, element: str | None = None) -> float:
        """Number of molecules

        Args:
            gas_volume: Gas volume in m :sup:`3`
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or `element` if not None.
        """
        return self.number_density(element) * gas_volume

    def moles(self, gas_volume: float, element: str | None = None) -> float:
        """Number of moles

        Args:
            gas_volume: Gas volume in m :sup:`3`
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self.molecules(gas_volume, element) / AVOGADRO


class CondensedNumberDensity(NumberDensitySolution[CondensedSpecies]):
    """The condensed number density solution"""

    @override
    def solution_dict(self, gas_volume: float) -> dict[str, float]:
        """Solution in a dictionary

        Args:
            gas_volume: Gas volume in m :sup:`3`

        Returns:
            Mass in kg in a dictionary
        """
        return {self.name: self.mass(gas_volume)}


class GasNumberDensity(NumberDensitySolution[GasSpecies]):
    """The gas number density solution"""

    def pressure(self, gas_temperature: float) -> float:
        """Pressure in bar

        Args:
            gas_temperature: Gas temperature in K

        Returns:
            Pressure in bar
        """
        return self.number_density() * BOLTZMANN_CONSTANT_BAR * gas_temperature

    def log10_pressure(self, gas_temperature: float) -> float:
        """Log10 pressures

        Args:
            gas_temperature: Gas temperature in K

        Returns:
            Pressure in bar
        """
        return np.log10(self.pressure(gas_temperature))

    def density(self, gas_volume: float) -> float:
        """Density

        Args:
            gas_volume: Gas volume in m :sup:`3`

        Returns:
            Density
        """
        return self.mass(gas_volume) / gas_volume

    def log10_fugacity_coefficient(self, gas_temperature: float, gas_pressure: float) -> float:
        """Log10 fugacity coefficients

        Args:
            gas_temperature: Gas temperature in K
            gas_pressure: Gas pressure in bar

        Returns:
            Log10 fugacity coefficient
        """
        return np.log10(self.fugacity_coefficient(gas_temperature, gas_pressure))

    def fugacity_coefficient(self, gas_temperature: float, gas_pressure: float) -> float:
        """Fugacity coefficients

        Args:
            gas_temperature: Gas temperature in K
            gas_pressure: Gas pressure in bar

        Returns:
            Fugacity coefficient
        """
        return self.species.eos.fugacity_coefficient(gas_temperature, gas_pressure)

    def log10_fugacity(self, gas_temperature: float, gas_pressure: float) -> float:
        """Log10 fugacity

        Args:
            gas_temperature: Gas temperature in K
            gas_pressure: Gas pressure in bar

        Returns:
            Log10 fugacity
        """
        return self.log10_pressure(gas_temperature) + self.log10_fugacity_coefficient(
            gas_temperature, gas_pressure
        )

    def fugacity(self, gas_temperature: float, gas_pressure: float) -> float:
        """Fugacity

        Args:
            gas_temperature: Gas temperature in K
            gas_pressure: Gas pressure in bar

        Returns:
            Fugacity
        """
        return 10 ** self.log10_fugacity(gas_temperature, gas_pressure)

    def fugacity_by_hill_formula(
        self, gas_temperature: float, gas_pressure: float
    ) -> dict[str, float]:
        """Fugacity by hill formula

        Args:
            gas_temperature: Gas temperature in K
            gas_pressure: Gas pressure in bar

        Returns:
            Fugacity by hill formula
        """
        return {self.species.hill_formula: self.fugacity(gas_temperature, gas_pressure)}

    @override
    def solution_dict(self, gas_temperature: float) -> dict[str, float]:
        """Solution in a dictionary

        Args:
            gas_temperature: Gas temperature in K

        Returns:
            Pressure in bar in a dictionary
        """
        return {self.name: self.pressure(gas_temperature)}


class MeltNumberDensity(NumberDensitySolution[GasSpecies]): ...


CondensedSolutionComponent = SolutionComponent[CondensedSpecies]


@dataclass
class CondensedCollection:
    """Condensed collection"""

    mass: CondensedNumberDensity
    activity: CondensedSolutionComponent
    stability: CondensedSolutionComponent

    @classmethod
    def initialise_with_species(cls, species: CondensedSpecies) -> CondensedCollection:
        """Initialise the condensed collection with a species.

        Args:
            species: Species to initialise the collection

        Returns:
            An instance of CondensedCollection initialised with the given species
        """
        mass: CondensedNumberDensity = CondensedNumberDensity(species, CONDENSED_MASS_PREFIX)
        activity: CondensedSolutionComponent = CondensedSolutionComponent(species, ACTIVITY_PREFIX)
        stability: CondensedSolutionComponent = CondensedSolutionComponent(
            species, STABILITY_PREFIX
        )

        return cls(mass, activity, stability)

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        return (
            self.mass.raw_solution_dict()
            | self.activity.raw_solution_dict()
            | self.stability.raw_solution_dict()
        )

    def solution_dict(self, gas_volume: float) -> dict[str, float]:
        """Solution in a dictionary"""
        return self.mass.solution_dict(gas_volume) | self.activity.solution_dict()


class CondensedSolution(UserDict[CondensedSpecies, CondensedCollection]):
    """Condensed solution"""

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return 3 * len(self.data)

    @classmethod
    def initialise_with_species(
        cls, condensed_species: list[CondensedSpecies]
    ) -> CondensedSolution:
        """Initialise the condensed solution with a list of condensed species.

        Args:
            condensed_species: List of condensed species to initialise the solution

        Returns:
            An instance of CondensedSolution initialised with the given species
        """
        init_dict: dict[CondensedSpecies, CondensedCollection] = {}
        for species in condensed_species:
            init_dict[species] = CondensedCollection.initialise_with_species(species)

        return cls(init_dict)

    def raw_solution_dict(self):
        """Raw solution in a dictionary"""
        raw_solution_dict: dict[str, float] = {}
        for condensed_collection in self.data.values():
            raw_solution_dict |= condensed_collection.raw_solution_dict()

        return raw_solution_dict

    def solution_dict(self, gas_volume: float):
        """Solution in a dictionary"""
        solution_dict: dict[str, float] = {}
        for condensed_collection in self.data.values():
            solution_dict |= condensed_collection.solution_dict(gas_volume)

        return solution_dict


class GasSolution(UserDict[GasSpecies, GasNumberDensity]):
    """Gas solution"""

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return len(self.data)

    @property
    def mean_molar_mass(self) -> float:
        """Mean molar mass"""
        return sum(
            [
                solution.species.molar_mass * solution.number_density() / self.gas_number_density
                for solution in self.data.values()
            ]
        )

    @classmethod
    def initialise_with_species(cls, gas_species: list[GasSpecies]) -> GasSolution:
        """Initialise the gas solution with a list of gas species.

        Args:
            gas_species: List of gas species to initialise the solution

        Returns:
            An instance of GasSolution initialised with the given species
        """
        init_dict: dict[GasSpecies, GasNumberDensity] = {}
        for species in gas_species:
            init_dict[species] = GasNumberDensity(species)

        return cls(init_dict)

    def fugacities_by_hill_formula(
        self, gas_temperature: float, gas_pressure: float
    ) -> dict[str, float]:
        """Fugacities by Hill formula

        Args:
            gas_temperature: Temperature in K
            gas_pressure: Gas pressure in bar

        Returns:
            Fugacities by Hill formula
        """
        fugacities: dict[str, float] = {}
        for solution in self.data.values():
            fugacities |= solution.fugacity_by_hill_formula(gas_temperature, gas_pressure)

        return fugacities

    @property
    def gas_number_density(self) -> float:
        """Number density"""
        return sum(solution.number_density() for solution in self.data.values())

    def gas_pressure(self, gas_temperature: float) -> float:
        """Pressure"""
        return sum(gas.pressure(gas_temperature) for gas in self.data.values())

    def raw_solution_dict(self):
        """Raw solution in a dictionary"""
        raw_solution_dict: dict[str, float] = {}
        for gas_solution in self.data.values():
            raw_solution_dict |= gas_solution.raw_solution_dict()

        return raw_solution_dict

    def solution_dict(self, gas_temperature: float):
        """Solution in a dictionary"""
        solution_dict: dict[str, float] = {}
        for gas_solution in self.data.values():
            solution_dict |= gas_solution.solution_dict(gas_temperature)

        return solution_dict


class Solution:
    """The solution

    Args:
        species: Species
    """

    def __init__(self, species: Species):
        self._species: Species = species
        self.gas: GasSolution = GasSolution.initialise_with_species(species.gas_species)
        self.condensed: CondensedSolution = CondensedSolution.initialise_with_species(
            species.condensed_species
        )

    @property
    def number(self) -> int:
        """Total number of solution quantities"""
        return self.gas.number + self.condensed.number

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """The solution as an array for the solver"""
        data: npt.NDArray[np.float_] = np.zeros(self.number, dtype=np.float_)
        index: int = 0
        for solution in self.gas.values():
            data[index] = solution.value
            index += 1
        for solution in self.condensed.values():
            data[index] = solution.activity.value
            index += 1
            data[index] = solution.mass.value
            index += 1
            data[index] = solution.stability.value
            index += 1

        return data

    @data.setter
    def data(self, value: npt.NDArray[np.float_]) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        index: int = 0
        for solution in self.gas.values():
            solution.value = value[index]
            index += 1
        for solution in self.condensed.values():
            solution.activity.value = value[index]
            index += 1
            solution.mass.value = value[index]
            index += 1
            solution.stability.value = value[index]
            index += 1

    # FIXME: To refresh
    # def merge(self, other: Solution) -> None:
    #     """Merges the data from another solution

    #     Args:
    #         other: The other solution to merge data from
    #     """
    #     self.gas.data |= other.gas.data
    #     self.activity.data |= other.activity.data
    #     self.condensed.data |= other.condensed.data
    #     self.stability.data |= other.stability.data

    def reaction_array(self) -> npt.NDArray[np.float_]:
        """The reaction array

        Returns:
            The reaction array
        """
        reaction_array: npt.NDArray = np.zeros(self._species.number, dtype=np.float_)
        index: int = 0
        for solution in self.gas.values():
            reaction_array[index] = solution.value
            index += 1
        for solution in self.condensed.values():
            reaction_array[index] = solution.activity.value
            index += 1

        return reaction_array

    def stability_array(self) -> npt.NDArray[np.float_]:
        """The condensate stability array

        Returns:
            The condensate stability array
        """
        stability_array: npt.NDArray = np.zeros(self._species.number, dtype=np.float_)
        for species, solution in self.condensed.items():
            index: int = self._species.species_index(species)
            stability_array[index] = solution.stability.value

        return stability_array

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        return self.gas.raw_solution_dict() | self.condensed.raw_solution_dict()

    def solution_dict(self, gas_temperature: float, gas_volume: float) -> dict[str, float]:
        """Solution as a dictionary

        Args:
            gas_temperature: Gas temperature in K
            gas_volume: Gas volume in m :sup:`3`

        Returns:
            Solution in a dictionary
        """
        output: dict[str, float] = self.gas.solution_dict(
            gas_temperature
        ) | self.condensed.solution_dict(gas_volume)

        return output
