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
from typing import Generic, Protocol, TypeVar

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

T_co = TypeVar("T_co", bound=ChemicalSpecies, covariant=True)

ACTIVITY_PREFIX: str = "activity_"
"""Name prefix for the activity of condensed species"""
STABILITY_PREFIX: str = "stability_"
"""Name prefix for the stability of condensed species"""

logger: logging.Logger = logging.getLogger(__name__)


class SolutionComponentProtocol(Protocol[T_co]):
    """Solution protocol"""

    @property
    def species(self) -> T_co: ...

    @property
    def value(self) -> float: ...


class NumberDensitySolution(SolutionComponentProtocol, Generic[T_co]):
    """A number density solution component

    Args:
        species: A Chemical species
    """

    def __init__(self, species: T_co):
        self._species: T_co = species

    @property
    def species(self) -> T_co:
        """Species"""
        return self._species

    def number_density(self, *, element: str | None = None) -> float:
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

        return count * 10**self.value

    def mass(self, gas_volume: float, *, element: str | None = None) -> float:
        """Mass of the species or element.

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

        return self.moles(gas_volume, element=element) * molar_mass

    def molecules(self, gas_volume: float, *, element: str | None = None) -> float:
        """Number of molecules of the species or element.

        Args:
            gas_volume: Gas volume in m :sup:`3`
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or `element` if not None.
        """
        return self.number_density(element=element) * gas_volume

    def moles(self, gas_volume: float, *, element: str | None = None) -> float:
        """Number of moles of the species or element.

        Args:
            gas_volume: Gas volume in m :sup:`3`
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self.molecules(gas_volume, element=element) / AVOGADRO


class ValueSetterMixin:
    """Mixin to set the value"""

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


class NumberDensitySolutionWithSetter(ValueSetterMixin, NumberDensitySolution[T_co]):
    """A number density solution component with a setter"""


class CondensedNumberDensity(NumberDensitySolutionWithSetter[CondensedSpecies]):
    """A number density solution component with a setter for a condensed species"""


class GasNumberDensity(NumberDensitySolutionWithSetter[GasSpecies]):
    """A number density solution for a gas"""

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
        return self._species.eos.fugacity_coefficient(gas_temperature, gas_pressure)

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


class InteriorNumberDensity(NumberDensitySolutionWithSetter[GasSpecies]):
    """An interior reservoir number density

    Args:
        species: Species
    """

    @override
    def __init__(self, species: GasSpecies):
        super().__init__(species)
        self._ppmw: float

    @property
    def ppmw(self) -> float:
        return self._ppmw

    @ppmw.setter
    def ppmw(self, value: float) -> None:
        self._ppmw = value

    def set_all(self, ppmw: float, reservoir_mass: float, gas_volume: float) -> None:
        """Sets the state

        Args:
            ppmw: Parts-per-million by weight
            reservoir_mass: Mass of the reservoir in kg
            gas_volume: Gas volume in m :sup:`3`
        """
        self.ppmw = ppmw
        number_density: float = (
            UnitConversion.ppm_to_fraction(self.ppmw)
            * AVOGADRO
            / self.species.molar_mass
            * reservoir_mass
            / gas_volume
        )
        self.value = np.log10(number_density)


class GasCollection(NumberDensitySolution[GasSpecies]):
    """Gas collection

    Args:
        species: A gas species

    Attributes:
        species: The gas species
        gas: Solution for the species in the gas phase
        dissolved: Solution for the species dissolved in melt
        trapped: Solution for the species trapped in solids
    """

    @override
    def __init__(self, species: GasSpecies):
        super().__init__(species)
        self.gas: GasNumberDensity = GasNumberDensity(species)
        self.dissolved: InteriorNumberDensity = InteriorNumberDensity(species)
        self.trapped: InteriorNumberDensity = InteriorNumberDensity(species)

    @property
    def value(self) -> float:
        return np.log10(10**self.gas.value + 10**self.dissolved.value + 10**self.trapped.value)


class CondensedSolutionComponent(ValueSetterMixin):

    def __init__(self, species: CondensedSpecies):
        self._species: CondensedSpecies = species


class CondensedCollection(NumberDensitySolution[CondensedSpecies]):
    """Condensed collection

    Args:
        species: A condensed species

    Attributes:
        species: The condensed species
        number_density: Solution for the fictitous number density of the condensate
        activity: Solution for the activity of the condensate
        stability: Solution for the stability of the condensate
    """

    @override
    def __init__(self, species: CondensedSpecies):
        super().__init__(species)
        self.number_density: CondensedNumberDensity = CondensedNumberDensity(species)
        self.activity: CondensedSolutionComponent = CondensedSolutionComponent(species)
        self.stability: CondensedSolutionComponent = CondensedSolutionComponent(species)

    @property
    def value(self) -> float:
        return self.number_density.value


class CondensedSolution(UserDict[CondensedSpecies, CondensedCollection]):
    """Condensed solution"""

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return 3 * len(self)

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
            init_dict[species] = CondensedCollection(species)

        return cls(init_dict)


class GasSolution(UserDict[GasSpecies, GasCollection], NumberDensitySolution):
    """Gas solution"""

    @property
    def value(self) -> float:
        return np.log10(sum(10**collection.value for collection in self.values()))

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return len(self)

    @property
    def mean_molar_mass(self) -> float:
        """Mean molar mass"""
        return sum(
            [
                collection.species.molar_mass
                * collection.gas.number_density()
                / self.gas_number_density()
                for collection in self.values()
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
        init_dict: dict[GasSpecies, GasCollection] = {}
        for species in gas_species:
            init_dict[species] = GasCollection(species)

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
        for collection in self.values():
            fugacities[collection.species.hill_formula] = collection.gas.fugacity(
                gas_temperature, gas_pressure
            )

        return fugacities

    def gas_number_density(self) -> float:
        """Gas number density"""
        return sum(collection.gas.number_density() for collection in self.values())

    def gas_pressure(self, gas_temperature: float) -> float:
        """Gas pressure"""
        return sum(collection.gas.pressure(gas_temperature) for collection in self.values())


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
            data[index] = solution.gas.value
            index += 1
        for solution in self.condensed.values():
            data[index] = solution.activity.value
            index += 1
            data[index] = solution.number_density.value
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
            solution.gas.value = value[index]
            index += 1
        for solution in self.condensed.values():
            solution.activity.value = value[index]
            index += 1
            solution.number_density.value = value[index]
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
            reaction_array[index] = solution.gas.value
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
        output: dict[str, float] = {}
        for solution in self.gas.values():
            species_name: str = solution.species.name
            output[species_name] = solution.gas.value
        for solution in self.condensed.values():
            species_name: str = solution.species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = solution.activity.value
            output[species_name] = solution.number_density.value
            output[f"{STABILITY_PREFIX}{species_name}"] = solution.stability.value

        return output

    def solution_dict(self, gas_temperature: float, gas_volume: float) -> dict[str, float]:
        """Solution as a dictionary

        Args:
            gas_temperature: Gas temperature in K
            gas_volume: Gas volume in m :sup:`3`

        Returns:
            Solution in a dictionary
        """
        output: dict[str, float] = {}
        for solution in self.gas.values():
            species_name: str = solution.species.name
            output[species_name] = solution.gas.pressure(gas_temperature)
        for solution in self.condensed.values():
            species_name: str = solution.species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = 10**solution.activity.value
            output[f"mass_{species_name}"] = solution.number_density.mass(gas_volume)

        return output
