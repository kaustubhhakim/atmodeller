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
from abc import ABC, abstractmethod
from collections import UserDict
from typing import Any, Generic, Protocol, Type, TypeVar

import numpy as np
import numpy.typing as npt
from molmass import Formula

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.core import GasSpecies, Planet, Species
from atmodeller.interfaces import ChemicalSpecies, CondensedSpecies
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

T_co = TypeVar("T_co", bound=ChemicalSpecies, covariant=True)

ACTIVITY_PREFIX: str = "activity_"
"""Name prefix for the activity of condensed species"""
STABILITY_PREFIX: str = "stability_"
"""Name prefix for the stability of condensed species"""

logger: logging.Logger = logging.getLogger(__name__)


class SolutionComponentProtocol(Protocol):
    """Solution component protocol"""

    _species: ChemicalSpecies
    _solution: Solution

    @property
    def value(self) -> float: ...


class NumberDensityMixin(SolutionComponentProtocol):
    """Number density mixin"""

    def number_density(self, *, element: str | None = None) -> float:
        """Number density of the species or element

        Args:
            element: Element to compute the number density for, or None to compute for the species.
                Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        if element is not None:
            try:
                count: int = self._species.composition()[element].count
            except KeyError:
                # Element not in formula
                count = 0
        else:
            count = 1

        return count * 10**self.value

    def mass(self, *, element: str | None = None) -> float:
        """Mass of the species or element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        if element is not None:
            molar_mass: float = UnitConversion().g_to_kg(Formula(element).mass)
        else:
            molar_mass = self._species.molar_mass

        return self.moles(element=element) * molar_mass

    def molecules(self, *, element: str | None = None) -> float:
        """Number of molecules of the species or element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or `element` if not None.
        """
        return self.number_density(element=element) * self._solution.gas.gas_volume()

    def moles(self, *, element: str | None = None) -> float:
        """Number of moles of the species or element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self.molecules(element=element) / AVOGADRO


class ValueSetterMixin:
    """Mixin to set value"""

    _value: float

    @property
    def value(self) -> float:
        """Gets the value."""
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        """Sets the value.

        Args:
            value: Value to set
        """
        self._value = value


class NumberDensitySolution(NumberDensityMixin, Generic[T_co]):
    """A number density solution component

    Args:
        species: A chemical species
        solution: The solution
    """

    # pylint: disable=super-init-not-called
    def __init__(self, species: T_co, solution: Solution):
        self._species: T_co = species
        self._solution: Solution = solution


class NumberDensitySolutionWithSetter(ValueSetterMixin, NumberDensitySolution[T_co]):
    """A number density solution component with a setter"""


class CondensedNumberDensity(NumberDensitySolutionWithSetter[CondensedSpecies]):
    """A number density solution component with a setter for a condensed species"""


class GasNumberDensity(NumberDensitySolutionWithSetter[GasSpecies]):
    """A number density solution for a gas species"""

    def pressure(self) -> float:
        """Pressure in bar"""
        return (
            self.number_density() * BOLTZMANN_CONSTANT_BAR * self._solution.gas.gas_temperature()
        )

    def log10_pressure(self) -> float:
        """Log10 pressure"""
        return np.log10(self.pressure())

    def density(self) -> float:
        """Density"""
        return self.mass() / self._solution.gas.gas_volume()

    def log10_fugacity_coefficient(self) -> float:
        """Log10 fugacity coefficient"""
        return np.log10(self.fugacity_coefficient())

    def fugacity_coefficient(self) -> float:
        """Fugacity coefficient"""
        return self._species.eos.fugacity_coefficient(
            self._solution.gas.gas_temperature(), self._solution.gas.gas_pressure()
        )

    def log10_fugacity(self) -> float:
        """Log10 fugacity"""
        return self.log10_pressure() + self.log10_fugacity_coefficient()

    def fugacity(self) -> float:
        """Fugacity"""
        return 10 ** self.log10_fugacity()


class InteriorNumberDensity(ABC, NumberDensitySolutionWithSetter[GasSpecies]):
    """A number density of a species in an interior reservoir

    Args:
        species: A gas species
        solution: The solution
    """

    ppmw: float

    @property
    @abstractmethod
    def reservoir_mass(self) -> float:
        """Reservoir mass"""

    def set_all_from_ppmw(self, ppmw: float) -> None:
        """Sets the state from the ppmw in the reservoir.

        Args:
            ppmw: Parts-per-million by weight
        """
        self.ppmw = ppmw
        number_density: float = (
            UnitConversion.ppm_to_fraction(self.ppmw)
            * AVOGADRO
            / self._species.molar_mass
            * self.reservoir_mass
            / self._solution.gas.gas_volume()
        )
        self.value = np.log10(number_density)


class DissolvedInteriorNumberDensity(InteriorNumberDensity):
    """A number density of a species dissolved in melt

    Args:
        species: A gas species
        solution: The solution
    """

    @property
    def reservoir_mass(self) -> float:
        return self._solution.planet.mantle_melt_mass


class TrappedInteriorNumberDensity(InteriorNumberDensity):
    """A number density of a species trapped in solids

    Args:
        species: A gas species
        solution: The solution
    """

    @property
    def reservoir_mass(self) -> float:
        return self._solution.planet.mantle_solid_mass


class GasCollection(NumberDensitySolution[GasSpecies]):
    """Gas collection

    Args:
        species: A gas species
        solution: The solution

    Attributes:
        species: The gas species
        gas: Solution for the species in the gas phase
        dissolved: Solution for the species dissolved in melt
        trapped: Solution for the species trapped in solids
    """

    NUMBER: int = 1
    """Number of solution quantities

    The number density of the species in the gas phase is the only true solution quantity,
    since the number density in other reservoirs can be determined from the aforementioned
    number density.
    """

    @override
    def __init__(self, species: GasSpecies, solution: Solution):
        self._species: GasSpecies = species
        self._solution: Solution = solution
        super().__init__(species, solution)
        self.gas_abundance: GasNumberDensity = GasNumberDensity(species, solution)
        self.dissolved_abundance: InteriorNumberDensity = DissolvedInteriorNumberDensity(
            species, solution
        )
        self.trapped_abundance: InteriorNumberDensity = TrappedInteriorNumberDensity(
            species, solution
        )

    @property
    def species(self) -> GasSpecies:
        return self._species

    @property
    def value(self) -> float:
        return np.log10(
            10**self.gas_abundance.value
            + 10**self.dissolved_abundance.value
            + 10**self.trapped_abundance.value
        )


class CondensedSolutionComponent(ValueSetterMixin):
    """A condensed solution component

    Args:
        species: A condensed species
        solution: The solution
    """

    def __init__(self, species: CondensedSpecies, solution: Solution):
        self._species: CondensedSpecies = species
        self._solution: Solution = solution


class CondensedCollection(NumberDensitySolution[CondensedSpecies]):
    """Condensed collection

    Args:
        species: A condensed species
        solution: The Solution

    Attributes:
        number_density: Solution for the fictitous number density of the condensate
        activity: Solution for the activity of the condensate
        stability: Solution for the stability of the condensate
    """

    NUMBER: int = 3
    """Number of solution quantities"""

    @override
    def __init__(self, species: CondensedSpecies, solution: Solution):
        self._species: CondensedSpecies = species
        self._solution: Solution = solution
        super().__init__(species, solution)
        self.condensed_abundance: CondensedNumberDensity = CondensedNumberDensity(
            species, solution
        )
        self.activity: CondensedSolutionComponent = CondensedSolutionComponent(species, solution)
        self.stability: CondensedSolutionComponent = CondensedSolutionComponent(species, solution)

    @property
    def species(self) -> CondensedSpecies:
        return self._species

    @property
    def value(self) -> float:
        return self.condensed_abundance.value


U = TypeVar("U", GasSpecies, CondensedSpecies)
V = TypeVar("V", GasCollection, CondensedCollection)


class SolutionDict(ABC, UserDict[U, V]):
    """Solution dictionary

    Args:
        species_list: A list of chemical species
        solution: The solution
        planet: The planet
    """

    collection_class: Any

    def __init__(self, init_dict=None, /, **kwargs):
        super().__init__(init_dict, **kwargs)
        self._solution: Solution
        self._planet: Planet

    @classmethod
    def from_species(cls, species_list: list[U], solution: Solution, planet: Planet) -> Self:
        init_dict: dict[U, V] = {}
        for species in species_list:
            init_dict[species] = cls.collection_class(species, solution)
        obj: Self = cls(init_dict)
        obj._solution = solution
        obj._planet = planet

        return obj

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return sum(collection.NUMBER for collection in self.values())

    def number_density(self, *, element: str | None = None) -> float:
        """Total number density of the species or element

        Args:
            element: Element to compute the number density for, or None to compute for the species.
                Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        return sum(collection.number_density(element=element) for collection in self.values())

    def mass(self, *, element: str | None = None) -> float:
        """Total mass of the species or element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        return sum(collection.mass(element=element) for collection in self.values())

    def molecules(self, *, element: str | None = None) -> float:
        """Total number of molecules of the species or element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or `element` if not None.
        """
        return sum(collection.molecules(element=element) for collection in self.values())

    def moles(self, *, element: str | None = None) -> float:
        """Total number of moles of the species or element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return sum(collection.moles(element=element) for collection in self.values())


class CondensedSolutionDict(SolutionDict[CondensedSpecies, CondensedCollection]):
    """Condensed solution

    Args:
        species: Species
        planet: Planet
        solution: The solution
    """

    collection_class: Type[CondensedCollection] = CondensedCollection


class GasSolutionDict(SolutionDict[GasSpecies, GasCollection]):
    """Gas solution

    Args:
        species: Species
        planet: Planet
        solution: The solution
    """

    collection_class: Type[GasCollection] = GasCollection

    def fugacities_by_hill_formula(self) -> dict[str, float]:
        """Fugacities by Hill formula

        Returns:
            Fugacities by Hill formula
        """
        fugacities: dict[str, float] = {}
        for collection in self.values():
            fugacities[collection.species.hill_formula] = collection.gas_abundance.fugacity()

        return fugacities

    def gas_mean_molar_mass(self) -> float:
        """Mean molar mass"""
        return sum(
            [
                collection.species.molar_mass
                * collection.gas_abundance.number_density()
                / self.gas_number_density()
                for collection in self.values()
            ]
        )

    def gas_number_density(self) -> float:
        """Gas number density"""
        return sum(collection.gas_abundance.number_density() for collection in self.values())

    def gas_pressure(self) -> float:
        """Gas pressure"""
        return sum(collection.gas_abundance.pressure() for collection in self.values())

    def gas_temperature(self) -> float:
        """Gas temperature"""
        return self._planet.surface_temperature

    def gas_volume(self) -> float:
        """Total volume of the atmosphere

        Derived using the mechanical pressure balance due to the weight of the atmosphere and the
        ideal gas equation of state.
        """
        volume: float = self._planet.surface_area / self._planet.surface_gravity
        volume *= GAS_CONSTANT * self.gas_temperature() / self.gas_mean_molar_mass()

        return volume


class Solution:
    """The solution

    Args:
        species: Species
        planet: Planet
    """

    def __init__(self, species: Species, planet: Planet):
        self._species: Species = species
        self._planet: Planet = planet
        self.gas: GasSolutionDict = GasSolutionDict.from_species(species.gas_species, self, planet)
        self.condensed: CondensedSolutionDict = CondensedSolutionDict.from_species(
            species.condensed_species, self, planet
        )

    @property
    def number(self) -> int:
        """Total number of solution quantities"""
        return self.gas.number + self.condensed.number

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self._planet

    @property
    def species(self) -> Species:
        """Species"""
        return self._species

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """The solution as an array for the solver"""
        data: npt.NDArray[np.float_] = np.zeros(self.number, dtype=np.float_)
        index: int = 0
        for solution in self.gas.values():
            data[index] = solution.gas_abundance.value
            index += 1
        for solution in self.condensed.values():
            data[index] = solution.activity.value
            index += 1
            data[index] = solution.condensed_abundance.value
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
            solution.gas_abundance.value = value[index]
            index += 1
        for solution in self.condensed.values():
            solution.activity.value = value[index]
            index += 1
            solution.condensed_abundance.value = value[index]
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
            reaction_array[index] = solution.gas_abundance.value
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

    def to_dict(self):
        return self.gas | self.condensed

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        output: dict[str, float] = {}
        for solution in self.gas.values():
            species_name: str = solution.species.name
            output[species_name] = solution.gas_abundance.value
        for solution in self.condensed.values():
            species_name: str = solution.species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = solution.activity.value
            output[species_name] = solution.condensed_abundance.value
            output[f"{STABILITY_PREFIX}{species_name}"] = solution.stability.value

        return output

    def solution_dict(self) -> dict[str, float]:
        """Solution as a dictionary"""
        output: dict[str, float] = {}
        for solution in self.gas.values():
            species_name: str = solution.species.name
            output[species_name] = solution.gas_abundance.pressure()
        for solution in self.condensed.values():
            species_name: str = solution.species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = 10**solution.activity.value
            output[f"mass_{species_name}"] = solution.condensed_abundance.mass()

        return output
