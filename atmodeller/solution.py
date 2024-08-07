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
from collections import Counter, UserDict
from typing import Generic, Protocol, TypeVar, cast

import jax.numpy as jnp

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.core import GasSpecies, Planet, Species
from atmodeller.interfaces import (
    ChemicalSpecies,
    CondensedSpecies,
    TypeChemicalSpecies,
    TypeChemicalSpecies_co,
)
from atmodeller.utilities import UnitConversion, get_molar_mass

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


logger: logging.Logger = logging.getLogger(__name__)

ACTIVITY_PREFIX: str = "activity_"
"""Name prefix for the activity of condensed species"""
ELEMENT_PREFIX: str = "element_"
"""Name prefix for the output keys of elements"""
SPECIES_PREFIX: str = ""
"""Name prefix for the output keys of species"""
STABILITY_PREFIX: str = "stability_"
"""Name prefix for the stability of condensed species"""
TAU: float = 1e-15
"""Tau factor for the stability of condensed species"""
LOG10_TAU: jnp.ndarray = jnp.log10(TAU)
"""Log10 of the tau factor"""


class ComponentProtocol(Protocol[TypeChemicalSpecies_co]):
    """Solution component protocol"""

    _species: TypeChemicalSpecies_co
    _solution: Solution

    @property
    def value(self) -> float: ...


class ComponentWithSetterProtocol(ComponentProtocol, Protocol):
    """Solution component with setter protocol"""

    @property
    def value(self) -> float: ...

    @value.setter
    def value(self, value: float) -> None: ...


class _ValueSetterMixin:
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


class _NumberDensity(ABC, Generic[TypeChemicalSpecies_co]):
    """A number density solution

    Args:
        species: A chemical species
        solution: The solution
    """

    output_prefix: str = ""
    """Prefix for the keys in the output dictionary"""

    def __init__(self, species: TypeChemicalSpecies_co, solution: Solution):
        self._species: TypeChemicalSpecies_co = species
        self._solution: Solution = solution

    @property
    @abstractmethod
    def value(self) -> float:
        """Log10 of the number density"""

    def elements(self) -> float:
        """Total number of elements"""
        return self._species.atoms * self.molecules()

    def element_moles(self) -> float:
        """Total number of moles of elements"""
        return self._species.atoms * self.moles()

    def element_number_density(self) -> float:
        """Number density of all elements"""
        return self._species.atoms * self.number_density()

    def number_density(self, *, element: str | None = None) -> float:
        """Number density of the species or an individual element

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
        """Mass of the species or an individual element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        if element is not None:
            molar_mass: float = get_molar_mass(element)
        else:
            molar_mass = self._species.molar_mass

        return self.moles(element=element) * molar_mass

    def molecules(self, *, element: str | None = None) -> float:
        """Number of molecules of the species or number of an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return self.number_density(element=element) * self._solution.atmosphere.volume()

    def moles(self, *, element: str | None = None) -> float:
        """Number of moles of the species or element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self.molecules(element=element) / AVOGADRO

    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter().

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output_dict: dict[str, float] = {}
        output_dict[f"{self.output_prefix}number_density"] = self.number_density(element=element)
        output_dict[f"{self.output_prefix}mass"] = self.mass(element=element)
        output_dict[f"{self.output_prefix}molecules"] = self.molecules(element=element)
        output_dict[f"{self.output_prefix}moles"] = self.moles(element=element)

        return output_dict

    def __repr__(self) -> str:
        return f"number_density={self.number_density():.2e}"


TypeNumberDensity = TypeVar("TypeNumberDensity", bound=_NumberDensity)


class _NumberDensityWithSetter(_ValueSetterMixin, _NumberDensity[TypeChemicalSpecies_co]):
    """A number density solution component with a setter"""


class _CondensedNumberDensity(_NumberDensityWithSetter[CondensedSpecies]):
    """A number density solution component with a setter for a condensed species

    Args:
        species: A gas species
        solution: The solution
    """


class _GasNumberDensity(_NumberDensityWithSetter[GasSpecies]):
    """A number density solution for a gas species

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "atmosphere_"
    """Prefix for the keys in the output dictionary"""

    def pressure(self) -> float:
        """Pressure in bar"""
        return (
            self.number_density()
            * BOLTZMANN_CONSTANT_BAR
            * self._solution.atmosphere.temperature()
        )

    def log10_pressure(self) -> jnp.ndarray:
        """Log10 pressure"""
        return jnp.log10(self.pressure())

    def density(self) -> float:
        """Density"""
        return self.mass() / self._solution.atmosphere.volume()

    def log10_fugacity_coefficient(self) -> jnp.ndarray:
        """Log10 fugacity coefficient"""
        return jnp.log10(self.fugacity_coefficient())

    def fugacity_coefficient(self) -> float:
        """Fugacity coefficient"""
        return self._species.eos.fugacity_coefficient(
            self._solution.atmosphere.temperature(), self._solution.atmosphere.pressure()
        )

    def log10_fugacity(self) -> jnp.ndarray:
        """Log10 fugacity"""
        return self.log10_pressure() + self.log10_fugacity_coefficient()

    def fugacity(self) -> jnp.ndarray:
        """Fugacity"""
        return 10 ** self.log10_fugacity()

    def volume_mixing_ratio(self) -> float:
        """Volume mixing ratio"""
        return self.number_density() / self._solution.atmosphere.number_density()

    @override
    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output_dict: dict[str, float] = super().output_dict(element=element)

        if element is None:
            output_dict["pressure"] = self.pressure()
            output_dict["fugacity_coefficient"] = self.fugacity_coefficient()
            output_dict["fugacity"] = self.fugacity().item()
            output_dict["volume_mixing_ratio"] = self.volume_mixing_ratio()

        return output_dict


class _DissolvedNumberDensity(_NumberDensity[GasSpecies]):
    """A number density of a species dissolved in melt

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "dissolved_"
    """Prefix for the keys in the output dictionary"""

    def ppmw(self) -> float:
        """Parts-per-million by weight of the volatile"""
        return self._species.solubility.concentration(
            fugacity=self._solution.gas_solution[self._species].gas_abundance.fugacity(),
            temperature=self._solution.atmosphere.temperature(),
            pressure=self._solution.atmosphere.pressure(),
            **self._solution.fugacities_by_hill_formula(),
        )

    @property
    def reservoir_mass(self) -> float:
        """Mass of the reservoir"""
        return self._solution.planet.mantle_melt_mass

    @property
    def value(self) -> jnp.ndarray:
        """Log10 of the number density"""
        number_density: jnp.ndarray = (
            UnitConversion.ppm_to_fraction(self.ppmw())
            * AVOGADRO
            / self._species.molar_mass
            * self.reservoir_mass
            / self._solution.atmosphere.volume()
        )

        return jnp.log10(number_density)

    @override
    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output_dict: dict[str, float] = super().output_dict(element=element)
        if element is None:
            output_dict[f"{self.output_prefix}ppmw"] = self.ppmw()

        return output_dict


class _TrappedNumberDensity(_DissolvedNumberDensity):
    """A number density of a species trapped in solids

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "trapped_"
    """Prefix for the keys in the output dictionary"""

    @override
    def ppmw(self) -> float:
        dissolved_ppmw: float = super().ppmw()
        trapped_ppmw: float = dissolved_ppmw * self._species.solid_melt_distribution_coefficient

        return trapped_ppmw

    @property
    def reservoir_mass(self) -> float:
        return self._solution.planet.mantle_solid_mass


class _CondensedSolutionComponent(_ValueSetterMixin):
    """A condensed solution component

    This is used for the activity and stability of condensed species.

    Args:
        species: A condensed species
        solution: The solution
    """

    def __init__(self, species: CondensedSpecies, solution: Solution):
        self._species: CondensedSpecies = species
        self._solution: Solution = solution


class _TauC(_CondensedSolutionComponent):
    """Tauc factor for the calculation of condensate stability :citep:`{e.g.,}KSP24{Equation 19}`

    Args:
        species: A condensed species
        solution: The solution
    """

    @property
    def value(self) -> jnp.ndarray:
        element_number_densities: list[float] = [
            self._solution.number_density(element=element) for element in self._species.elements
        ]
        log10_tauc: jnp.ndarray = LOG10_TAU + jnp.log10(
            jnp.min(jnp.array(element_number_densities))
        )
        logger.debug("log10_tau (%s) = %f", self._species.name, log10_tauc)

        return log10_tauc


class _GasCollection(_NumberDensity[GasSpecies]):
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
    output_prefix: str = "total_"
    """Prefix for the keys in the output dictionary"""

    @override
    def __init__(self, species: GasSpecies, solution: Solution):
        self._species: GasSpecies = species
        self._solution: Solution = solution
        super().__init__(species, solution)
        self.gas_abundance: _GasNumberDensity = _GasNumberDensity(species, solution)
        self.dissolved_abundance: _DissolvedNumberDensity = _DissolvedNumberDensity(
            species, solution
        )
        self.trapped_abundance: _TrappedNumberDensity = _TrappedNumberDensity(species, solution)

    @property
    def value(self) -> jnp.ndarray:
        return jnp.log10(
            10**self.gas_abundance.value
            + 10**self.dissolved_abundance.value
            + 10**self.trapped_abundance.value
        )

    @override
    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output_dict: dict[str, float] = (
            self.gas_abundance.output_dict(element=element)
            | self.dissolved_abundance.output_dict(element=element)
            | self.trapped_abundance.output_dict(element=element)
        )
        if element is None:
            output_dict |= super().output_dict()
            output_dict["molar_mass"] = self._species.molar_mass

        return output_dict

    def __repr__(self) -> str:
        return f"{self.gas_abundance!r}"


class _CondensedCollection(_NumberDensity[CondensedSpecies]):
    """Condensed collection

    Args:
        species: A condensed species
        solution: The Solution

    Attributes:
        number_density: Solution for the fictitous number density of the condensate
        activity: Solution for the activity of the condensate
        stability: Solution for the stability of the condensate
        tauc: Tauc for condensate stability
    """

    NUMBER: int = 3
    """Number of solution quantities"""
    output_prefix: str = "condensed_"
    """Prefix for the keys in the output dictionary"""

    @override
    def __init__(self, species: CondensedSpecies, solution: Solution):
        self._species: CondensedSpecies = species
        self._solution: Solution = solution
        super().__init__(species, solution)
        self.condensed_abundance: _CondensedNumberDensity = _CondensedNumberDensity(
            species, solution
        )
        self.activity: _CondensedSolutionComponent = _CondensedSolutionComponent(species, solution)
        self.stability: _CondensedSolutionComponent = _CondensedSolutionComponent(
            species, solution
        )
        self.tauc: _TauC = _TauC(species, solution)

    @property
    def species(self) -> CondensedSpecies:
        return self._species

    @property
    def value(self) -> float:
        return self.condensed_abundance.value

    def __repr__(self) -> str:
        base_repr: str = super().__repr__().rstrip(")")
        repr_str: str = f"{base_repr}, activity={10**self.activity.value}, "
        repr_str += f"stability={10**self.stability.value}"

        return repr_str

    @override
    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        For element = None this must return a dictionary with entries that can be summed using
        Counter() when an element is specified.

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output_dict: dict[str, float] = super().output_dict(element=element)
        if element is None:
            output_dict["activity"] = 10**self.activity.value
            output_dict["molar_mass"] = self.species.molar_mass

        return output_dict


class _SolutionContainer(UserDict[TypeChemicalSpecies, TypeNumberDensity]):
    """A container for the solution"""

    def _sum_values(self, method_name: str, *args, **kwargs) -> float:
        """Helper method to sum values from the dictionary based on a given method name."""
        return sum(getattr(value, method_name)(*args, **kwargs) for value in self.values())

    def number_density(self, *, element: str | None = None) -> float:
        """Total number density of the species or an individual element

        Args:
            element: Element to compute the number density for, or None to compute for the species.
                Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        return self._sum_values("number_density", element=element)

    def element_number_density(self) -> float:
        """Number density of all elements"""
        return self._sum_values("element_number_density")

    def mass(self, *, element: str | None = None) -> float:
        """Total mass of the species or an individual element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        return self._sum_values("mass", element=element)

    def molecules(self, *, element: str | None = None) -> float:
        """Total number of molecules of the species or an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return self._sum_values("molecules", element=element)

    def elements(self) -> float:
        """Total number of elements"""
        return self._sum_values("elements")

    def moles(self, *, element: str | None = None) -> float:
        """Total number of moles of the species or an individual element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self._sum_values("moles", element=element)

    def element_moles(self) -> float:
        """Total number of moles of elements"""
        return self._sum_values("element_moles")


class _Atmosphere(_SolutionContainer[GasSpecies, _GasNumberDensity]):
    """Bulk properties of the atmosphere"""

    planet: Planet
    """Planet"""

    def molar_mass(self) -> float:
        """Molar mass"""
        return (
            sum([species.molar_mass * value.number_density() for species, value in self.items()])
            / self.number_density()
        )

    def pressure(self) -> float:
        """Pressure"""
        return self._sum_values("pressure")

    def temperature(self) -> float:
        """Temperature

        This is the same as the surface temperature of the planet
        """
        return self.planet.surface_temperature

    def volume(self) -> float:
        """Volume

        Derived using the mechanical pressure balance due to the weight of the atmosphere and the
        ideal gas equation of state.

        TODO: Should a correction be applied to the volume term for a non-ideal atmosphere?
        """
        volume: float = self.planet.surface_area / self.planet.surface_gravity
        volume *= GAS_CONSTANT * self.temperature() / self.molar_mass()

        return volume

    def output_dict(self) -> dict[str, float]:
        """Output dictionary

        Returns:
            Output dictionary
        """
        output_dict: dict[str, float] = {}
        output_dict["mass"] = self.mass()
        output_dict["molecules"] = self.molecules()
        output_dict["moles"] = self.moles()
        output_dict["number_density"] = self.number_density()
        output_dict["molar_mass"] = self.molar_mass()
        output_dict["elements"] = self.elements()
        output_dict["element_moles"] = self.element_moles()
        output_dict["pressure"] = self.pressure()
        output_dict["temperature"] = self.temperature()
        output_dict["volume"] = self.volume()

        return output_dict


class Solution(_SolutionContainer[ChemicalSpecies, _GasCollection | _CondensedCollection]):
    """The solution

    Since this solution class is also used for the initial solution, which does not require the
    :class:`Planet`, :attr:`planet` is a property with a setter.

    Args:
        init_dict: Initial dictionary
    """

    _atmosphere: _Atmosphere

    @classmethod
    def create_from_species(cls, *, species: Species) -> Self:
        """Creates a Solution instance

        Args:
            species: Species
        """
        solution: Self = cls()
        for gas_species in species.gas_species().values():
            solution[gas_species] = _GasCollection(gas_species, solution)
        for condensed_species in species.condensed_species().values():
            solution[condensed_species] = _CondensedCollection(condensed_species, solution)

        init_dict: dict[GasSpecies, _GasNumberDensity] = {
            species: collection.gas_abundance
            for species, collection in solution.gas_solution.items()
        }
        # Only need to set these attributes once so pylint: disable=protected-access
        solution._atmosphere = _Atmosphere(init_dict)

        return solution

    @property
    def atmosphere(self) -> _Atmosphere:
        return self._atmosphere

    @property
    def condensed_solution(self) -> dict[CondensedSpecies, _CondensedCollection]:
        return cast(
            dict[CondensedSpecies, _CondensedCollection],
            {key: value for key, value in self.items() if isinstance(key, CondensedSpecies)},
        )

    @property
    def gas_solution(self) -> dict[GasSpecies, _GasCollection]:
        return cast(
            dict[GasSpecies, _GasCollection],
            {key: value for key, value in self.items() if isinstance(key, GasSpecies)},
        )

    @property
    def number(self) -> int:
        """Total number of solution quantities"""
        return sum(collection.NUMBER for collection in self.values())

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self._atmosphere.planet

    @planet.setter
    def planet(self, value: Planet) -> None:
        """Planet setter, which sets planet on :attr:`_atmosphere`"""
        self._atmosphere.planet = value

    @property
    def value(self) -> jnp.ndarray:
        """The solution as an array for the solver"""
        value_list: list[float] = []
        for gas_collection in self.gas_solution.values():
            value_list.append(gas_collection.gas_abundance.value)
        for condensed_collection in self.condensed_solution.values():
            value_list.append(condensed_collection.activity.value)
            value_list.append(condensed_collection.condensed_abundance.value)
            value_list.append(condensed_collection.stability.value)

        return jnp.array(value_list)

    @value.setter
    def value(self, value: jnp.ndarray) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        index: int = 0
        for gas_collection in self.gas_solution.values():
            gas_collection.gas_abundance.value = value[index]
            index += gas_collection.NUMBER
        for condensed_collection in self.condensed_solution.values():
            condensed_collection.activity.value = value[index]
            condensed_collection.condensed_abundance.value = value[index + 1]
            condensed_collection.stability.value = value[index + 2]
            index += condensed_collection.NUMBER

    def elements(self) -> list[str]:
        """Unique elements in the species

        Returns:
            A list of unique elements
        """
        elements: list[str] = []
        for species in self:
            elements.extend(species.elements)

        return list(set(elements))

    def merge(self, other: Solution) -> None:
        """Merges the data from another solution

        Args:
            other: The other solution to merge data from
        """
        self.data |= other.data

    def fugacities_by_hill_formula(self) -> dict[str, jnp.ndarray]:
        """Fugacities by Hill formula

        Returns:
            Fugacities by Hill formula
        """
        fugacities: dict[str, jnp.ndarray] = {}
        for gas_species, collection in self.gas_solution.items():
            fugacities[gas_species.hill_formula] = collection.gas_abundance.fugacity()

        return fugacities

    def total_moles_hydrogen(self) -> float | None:
        """Total moles of hydrogen"""
        moles_of_hydrogen: float = self.moles(element="H")
        if moles_of_hydrogen <= 0:
            return None
        else:
            return moles_of_hydrogen

    def _output_elements(self) -> dict[str, dict[str, float]]:
        """Output for elements"""
        output_dict: dict[str, dict[str, float]] = {}
        for element in self.elements():
            element_dict: dict[str, float] = output_dict.setdefault(
                f"{ELEMENT_PREFIX}{element}", {}
            )
            element_dict["total_mass"] = self.mass(element=element)
            total_moles = self.moles(element=element)
            element_dict["total_moles"] = total_moles
            total_moles_hydrogen: float | None = self.total_moles_hydrogen()
            if total_moles_hydrogen is not None:
                element_dict["logarithmic_abundance"] = (
                    jnp.log10(total_moles / total_moles_hydrogen).item() + 12
                )
            counter: Counter = Counter()
            for collection in self.values():
                output: dict[str, float] = collection.output_dict(element=element)
                counter += Counter(output)
            element_dict |= dict(counter)
            try:
                element_dict["degree_of_condensation"] = (
                    element_dict["condensed_moles"] / element_dict["total_moles"]
                )
            except KeyError:
                # No condensed species for this element
                pass
            element_dict["volume_mixing_ratio"] = (
                element_dict["atmosphere_moles"] / self.atmosphere.element_moles()
            )
            element_dict["molar_mass"] = get_molar_mass(element)

        return output_dict

    def output_full(self) -> dict[str, dict[str, float]]:
        """Full output"""
        output_dict: dict[str, dict[str, float]] = {}
        for species, collection in self.items():
            output_dict[f"{SPECIES_PREFIX}{species.name}"] = collection.output_dict()
        output_dict |= self._output_elements()
        output_dict["atmosphere"] = self.atmosphere.output_dict()
        output_dict["planet"] = self.planet.output_dict()
        output_dict["raw_solution"] = self.output_raw_solution()
        output_dict["solution"] = self.output_solution()

        return output_dict

    def output_raw_solution(self) -> dict[str, float]:
        """Outputs the raw solution as seen by the solver

        Returns:
            A dictionary with formatted keys and float values representing the raw solution.
        """
        output: dict[str, float] = {}
        for gas_species, collection in self.gas_solution.items():
            output[gas_species.name] = collection.gas_abundance.value
        for condensed_species, collection in self.condensed_solution.items():
            species_name: str = condensed_species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = collection.activity.value
            output[species_name] = collection.condensed_abundance.value
            output[f"{STABILITY_PREFIX}{species_name}"] = collection.stability.value

        return output

    def output_solution(self) -> dict[str, float]:
        """Outputs the solution in a convenient form for comparison and benchmarking"""
        output: dict[str, float] = {}
        for gas_species, collection in self.gas_solution.items():
            output[gas_species.name] = collection.gas_abundance.pressure()
        for condensed_species, collection in self.condensed_solution.items():
            species_name: str = condensed_species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = 10**collection.activity.value
            output[f"mass_{species_name}"] = collection.condensed_abundance.mass()

        return output
