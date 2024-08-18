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
from collections import Counter
from typing import Generic, Protocol, TypeVar, cast

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.core import GasSpecies, Planet, Species
from atmodeller.interfaces import (
    ChemicalSpecies,
    CondensedSpecies,
    ImmutableDict,
    TypeChemicalSpecies,
    TypeChemicalSpecies_co,
)
from atmodeller.utilities import UnitConversion, get_molar_mass, logsumexp_base10

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
    def value(self) -> Array: ...


class ComponentSetterProtocol(ComponentProtocol, Protocol):
    """Solution component with setter protocol"""

    @property
    def value(self) -> Array: ...

    @value.setter
    def value(self, value: ArrayLike) -> None: ...


class _ValueSetterMixin:
    """Mixin to set value"""

    _value: ArrayLike

    @property
    def value(self) -> Array:
        """Gets the value."""
        return jnp.asarray(self._value)

    @value.setter
    def value(self, value: ArrayLike) -> None:
        """Sets the value.

        Args:
            value: Value to set
        """
        self._value = value


class _NumberDensitySpecies(ABC, Generic[TypeChemicalSpecies_co]):
    """A number density for a species

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
    def value(self) -> Array:
        """Log10 of the number density"""

    @property
    def species(self) -> TypeChemicalSpecies_co:
        return self._species

    def elements(self) -> Array:
        """Total number of elements"""
        return self._species.atoms * self.molecules()

    def element_moles(self) -> Array:
        """Total number of moles of elements"""
        return self._species.atoms * self.moles()

    def element_number_density(self) -> Array:
        """Number density of all elements"""
        return self._species.atoms * self.number_density()

    def log10_number_density(self, *, element: str | None = None) -> Array:
        """Log10 number density of the species or an individual element

        Args:
            element: Element to compute the log10 number density for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        if element is not None:
            assert element in self._species.composition()
            count: int = self._species.composition()[element].count
        else:
            count = 1

        return jnp.log10(count) + self.value

    def number_density(self, *, element: str | None = None) -> Array:
        """Number density of the species or an individual element

        Args:
            element: Element to compute the number density for, or None to compute for the species.
                Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        if element is not None:
            if element in self._species.composition():
                return jnp.power(10, self.log10_number_density(element=element))
            else:
                return jnp.array(0)

        return jnp.power(10, self.log10_number_density())

    def mass(self, *, element: str | None = None) -> Array:
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

    def molecules(self, *, element: str | None = None) -> Array:
        """Number of molecules of the species or number of an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return self.number_density(element=element) * self._solution.atmosphere.volume()

    def moles(self, *, element: str | None = None) -> Array:
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
        output: dict[str, float] = {}
        output[f"{self.output_prefix}number_density"] = self.number_density(element=element).item()
        output[f"{self.output_prefix}mass"] = self.mass(element=element).item()
        output[f"{self.output_prefix}molecules"] = self.molecules(element=element).item()
        output[f"{self.output_prefix}moles"] = self.moles(element=element).item()

        return output

    def __repr__(self) -> str:
        return f"number_density={self.number_density():.2e}"


TypeNumberDensitySpecies = TypeVar("TypeNumberDensitySpecies", bound=_NumberDensitySpecies)


class _NumberDensitySpeciesSetter(
    _ValueSetterMixin, _NumberDensitySpecies[TypeChemicalSpecies_co]
):
    """A number density with a setter"""


class _CondensedNumberDensitySpeciesSetter(_NumberDensitySpeciesSetter[CondensedSpecies]):
    """A number density with a setter for a condensed species

    Args:
        species: A gas species
        solution: The solution
    """


class _GasNumberDensitySpeciesSetter(_NumberDensitySpeciesSetter[GasSpecies]):
    """A number density with a setter for a gas species

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "atmosphere_"
    """Prefix for the keys in the output dictionary"""

    def pressure(self) -> Array:
        """Pressure in bar"""
        return jnp.power(10, self.log10_pressure())

    def log10_pressure(self) -> Array:
        """Log10 pressure"""
        return (
            self.log10_number_density()
            + jnp.log10(BOLTZMANN_CONSTANT_BAR)
            + jnp.log10(self._solution.atmosphere.temperature())
        )

    def density(self) -> Array:
        """Density"""
        return self.mass() / self._solution.atmosphere.volume()

    def log10_fugacity_coefficient(self) -> Array:
        """Log10 fugacity coefficient"""
        return jnp.log10(self.fugacity_coefficient())

    def fugacity_coefficient(self) -> Array:
        """Fugacity coefficient"""
        return jnp.array(
            self._species.eos.fugacity_coefficient(
                self._solution.atmosphere.temperature(), self._solution.atmosphere.pressure()
            )
        )

    def log10_fugacity(self) -> Array:
        """Log10 fugacity"""
        return self.log10_pressure() + self.log10_fugacity_coefficient()

    def fugacity(self) -> Array:
        """Fugacity"""
        return jnp.power(10, self.log10_fugacity())

    def volume_mixing_ratio(self) -> Array:
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
        output: dict[str, float] = super().output_dict(element=element)

        if element is None:
            output["pressure"] = self.pressure().item()
            output["fugacity_coefficient"] = self.fugacity_coefficient().item()
            output["fugacity"] = self.fugacity().item()
            output["volume_mixing_ratio"] = self.volume_mixing_ratio().item()

        return output


class _DissolvedNumberDensitySpecies(_NumberDensitySpecies[GasSpecies]):
    """A number density of a species dissolved in melt

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "dissolved_"
    """Prefix for the keys in the output dictionary"""

    def ppmw(self) -> Array:
        """Parts-per-million by weight of the volatile"""
        return jnp.asarray(
            self._species.solubility.concentration(
                fugacity=self._solution.gas_solution[self._species].abundance.fugacity(),
                temperature=self._solution.atmosphere.temperature(),
                pressure=self._solution.atmosphere.pressure(),
                **self._solution.fugacities_by_hill_formula(),
            )
        )

    @property
    def reservoir_mass(self) -> float:
        """Mass of the reservoir"""
        return self._solution.planet.mantle_melt_mass

    @property
    def value(self) -> ArrayLike:
        """Log10 of the number density

        If there is no solubility or no reservoir mass then -jnp.inf is returned
        """
        ppmw_value: Array = self.ppmw()

        number_density: Array = jnp.where(
            (ppmw_value > 0) & (self.reservoir_mass > 0),
            UnitConversion.ppm_to_fraction(ppmw_value)
            * AVOGADRO
            / self._species.molar_mass
            * self.reservoir_mass
            / self._solution.atmosphere.volume(),
            -jnp.inf,
        )

        return jnp.where(number_density == -jnp.inf, -jnp.inf, jnp.log10(number_density))

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
        output: dict[str, float] = super().output_dict(element=element)
        if element is None:
            output[f"{self.output_prefix}ppmw"] = self.ppmw().item()

        return output


class _TrappedNumberDensitySpecies(_DissolvedNumberDensitySpecies):
    """A number density of a species trapped in solids

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "trapped_"
    """Prefix for the keys in the output dictionary"""

    @override
    def ppmw(self) -> Array:
        dissolved_ppmw: Array = super().ppmw()
        trapped_ppmw: Array = dissolved_ppmw * self._species.solid_melt_distribution_coefficient

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
    def value(self) -> Array:
        element_number_densities: list[Array] = [
            self._solution.number_density(element=element) for element in self._species.elements
        ]
        log10_tauc: Array = LOG10_TAU + jnp.log10(jnp.min(jnp.array(element_number_densities)))

        return log10_tauc


class _GasSpeciesContainer(_NumberDensitySpecies[GasSpecies]):
    """Gas species container

    Args:
        species: A gas species
        solution: The solution

    Attributes:
        # TODO: Consistency between this gas container and the condensed container
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
        self.abundance: _GasNumberDensitySpeciesSetter = _GasNumberDensitySpeciesSetter(
            species, solution
        )
        # FIXME: This raises NaN issues for Optimistix
        # self.dissolved_abundance: _DissolvedNumberDensitySpecies = _DissolvedNumberDensitySpecies(
        #    species, solution
        # )
        # self.trapped_abundance: _TrappedNumberDensitySpecies = _TrappedNumberDensitySpecies(species, solution)

    @property
    def value(self) -> Array:
        # FIXME: To reinstate
        # gas_value: Array = self.abundance.value
        # dissolved_value: ArrayLike = self.dissolved_abundance.value
        # trapped_value: ArrayLike = self.trapped_abundance.value

        # log10_values: Array = jnp.asarray((gas_value, dissolved_value, trapped_value))

        # return logsumexp_base10(log10_values)

        return self.abundance.value

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
        output: dict[str, float] = (
            self.abundance.output_dict(element=element)
            | self.dissolved_abundance.output_dict(element=element)
            | self.trapped_abundance.output_dict(element=element)
        )
        if element is None:
            output |= super().output_dict()
            output["molar_mass"] = self._species.molar_mass

        return output

    def __repr__(self) -> str:
        return f"{self.abundance!r}"


class _CondensedSpeciesContainer(_NumberDensitySpecies[CondensedSpecies]):
    """Condensed species container

    Args:
        species: A condensed species
        solution: The Solution

    Attributes:
        # TODO: Consistency between this gas container and the condensed container
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
        self.abundance: _CondensedNumberDensitySpeciesSetter = (
            _CondensedNumberDensitySpeciesSetter(species, solution)
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
    def value(self) -> Array:
        return self.abundance.value

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
        output: dict[str, float] = super().output_dict(element=element)
        if element is None:
            output["activity"] = 10 ** self.activity.value.item()
            output["molar_mass"] = self.species.molar_mass

        return output


class _Collection(ImmutableDict[TypeChemicalSpecies, TypeNumberDensitySpecies]):
    """A container for the solution"""

    output_prefix: str = ""
    """Prefix for the keys in the output dictionary"""

    # TODO: This assumes that the data are contained in self.data

    def _sum_values(self, method_name: str, *args, **kwargs) -> Array:
        """Helper method to sum values from the dictionary based on a given method name."""
        return jnp.sum(
            jnp.array(
                [getattr(value, method_name)(*args, **kwargs) for value in self.data.values()]
            )
        )

    def log10_number_density(self, *, element: str | None = None) -> Array:
        """Log10 number density of all species or an individual element

        Args:
            element: Element to compute the log10 number density for, or None to compute for all
                species. Defaults to None.

        Returns:
            Number density for all species or `element` if not None.
        """
        if element is not None:
            log10_number_densities: Array = jnp.array(
                [
                    value.log10_number_density(element=element)
                    for value in self.data.values()
                    if element in value.species.composition()
                ]
            )
        else:
            log10_number_densities = jnp.array(
                [value.log10_number_density() for value in self.data.values()]
            )

        # For Optimistix/JAX debugging
        # jax.debug.print("log10_number_densities = {out}", out=log10_number_densities)

        return logsumexp_base10(log10_number_densities)

    def number_density(self, *, element: str | None = None) -> Array:
        """Number density of all species or an individual element

        Args:
            element: Element to compute the number density for, or None to compute for all species.
                Defaults to None.

        Returns:
            Number density for all species or `element` if not None.
        """
        return self._sum_values("number_density", element=element)

    def element_number_density(self) -> Array:
        """Number density of all elements"""
        return self._sum_values("element_number_density")

    def mass(self, *, element: str | None = None) -> Array:
        """Total mass of the species or an individual element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        return self._sum_values("mass", element=element)

    def molecules(self, *, element: str | None = None) -> Array:
        """Total number of molecules of the species or an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return self._sum_values("molecules", element=element)

    def elements(self) -> Array:
        """Total number of elements"""
        return self._sum_values("elements")

    def moles(self, *, element: str | None = None) -> Array:
        """Total number of moles of the species or an individual element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self._sum_values("moles", element=element)

    def element_moles(self) -> Array:
        """Total number of moles of elements"""
        return self._sum_values("element_moles")

    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = {}
        output[f"{self.output_prefix}number_density"] = self.number_density(element=element).item()
        output[f"{self.output_prefix}mass"] = self.mass(element=element).item()
        output[f"{self.output_prefix}molecules"] = self.molecules(element=element).item()
        output[f"{self.output_prefix}moles"] = self.moles(element=element).item()
        output[f"{self.output_prefix}elements"] = self.elements().item()
        output[f"{self.output_prefix}element_moles"] = self.element_moles().item()

        return output


class _GasCollection(_Collection[GasSpecies, _GasSpeciesContainer]):
    """A collection of gas species"""

    @override
    def __init__(self, data: dict[GasSpecies, _GasSpeciesContainer] | None = None):
        if data is None:
            self.data: dict[T, U] = {}
        else:
            self.data = data

    @override
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Instantiating a gas collection should also instantiate the atmosphere
        self.atmosphere: _Collection[GasSpecies, _GasNumberDensitySpeciesSetter]

    @classmethod
    def create(cls, species: Species) -> Self:
        """Creates an instance from species

        Args:
            species: Species
        """
        gas_collection: Self = cls()
        for gas_species in species.gas_species().values():
            gas_collection.data[gas_species] = _GasSpeciesContainer(
                gas_species,
            )


class _CondensedCollection(_Collection[CondensedSpecies, _CondensedSpeciesContainer]):
    """A collection of condensed species"""


class _Atmosphere(_Collection[GasSpecies, _GasNumberDensitySpeciesSetter]):
    """Bulk properties of the atmosphere"""

    @override
    def __init__(self, data: dict[GasSpecies, _GasNumberDensitySpeciesSetter], planet: Planet):
        super().__init__(data)
        self.planet: Planet = planet

    def log10_molar_mass(self) -> Array:
        """Log10 molar mass"""
        molar_masses: Array = jnp.array(
            [value._species.molar_mass for value in self.data.values()]
        )
        log10_number_densities: Array = jnp.array([value.value for value in self.data.values()])

        molar_mass: Array = (
            logsumexp_base10(log10_number_densities, molar_masses) - self.log10_number_density()
        )

        return molar_mass

    def molar_mass(self) -> Array:
        """Molar mass"""
        return jnp.power(10, self.log10_molar_mass())

    def pressure(self) -> Array:
        """Pressure"""
        log10_pressure: Array = logsumexp_base10(
            jnp.array([value.log10_pressure() for value in self.data.values()])
        )
        return jnp.power(10, log10_pressure)

    def temperature(self) -> float:
        """Temperature

        This is the same as the surface temperature of the planet
        """
        return self.planet.surface_temperature

    def log10_volume(self) -> Array:
        """Log10 volume

        Derived using the mechanical pressure balance due to the weight of the atmosphere and the
        ideal gas equation of state.

        TODO: Should a correction be applied to the volume term for a non-ideal atmosphere?
        """
        log10_volume: Array = (
            jnp.log10(GAS_CONSTANT)
            + jnp.log10(self.temperature())
            - self.log10_molar_mass()
            + jnp.log10(self.planet.surface_area)
            - jnp.log10(self.planet.surface_gravity)
        )

        return log10_volume

    def volume(self) -> Array:
        """Volume"""
        return jnp.power(10, self.log10_volume())

    def output_dict(self) -> dict[str, float]:
        """Output dictionary

        Returns:
            Output dictionary
        """
        output: dict[str, float] = super().output_dict()
        output[f"{self.output_prefix}pressure"] = self.pressure().item()
        output[f"{self.output_prefix}temperature"] = self.temperature()
        output[f"{self.output_prefix}volume"] = self.volume().item()

        return output


class Solution(_Collection[ChemicalSpecies, _GasSpeciesContainer | _CondensedSpeciesContainer]):
    """The solution

    Since this solution class is also used for the initial solution, which does not require the
    :class:`Planet`, :attr:`planet` is a property with a setter.

    Args:
        init_dict: Initial dictionary
    """

    _atmosphere: _Atmosphere

    @classmethod
    def create_from_species(cls, species: Species) -> Self:
        """Creates a Solution instance

        Args:
            species: Species
        """
        solution: Self = cls()
        for gas_species in species.gas_species().values():
            solution.data[gas_species] = _GasSpeciesContainer(gas_species, solution)
        for condensed_species in species.condensed_species().values():
            solution.data[condensed_species] = _CondensedSpeciesContainer(
                condensed_species, solution
            )

        # # TODO: Moved below
        # init_dict: dict[GasSpecies, _GasNumberDensitySpeciesSetter] = {
        #     species: collection.abundance for species, collection in solution.gas_solution.items()
        # }
        # # Only need to set these attributes once so pylint: disable=protected-access
        # solution._atmosphere = _Atmosphere(init_dict)

        return solution

    @classmethod
    def create_from_species_and_planet(cls, species: Species, planet: Planet) -> Self:
        """Creates a Solution instance

        Args:
            species: Species
            planet: Planet
        """
        solution: Self = cls.create_from_species(species)
        init_dict: dict[GasSpecies, _GasNumberDensitySpeciesSetter] = {
            species: collection.abundance for species, collection in solution.gas_solution.items()
        }
        # Only need to set these attributes once so pylint: disable=protected-access
        solution._atmosphere = _Atmosphere(init_dict, planet)

        return solution

    @property
    def atmosphere(self) -> _Atmosphere:
        return self._atmosphere

    @property
    def condensed_solution(self) -> dict[CondensedSpecies, _CondensedSpeciesContainer]:
        return cast(
            dict[CondensedSpecies, _CondensedSpeciesContainer],
            {key: value for key, value in self.items() if isinstance(key, CondensedSpecies)},
        )

    @property
    def gas_solution(self) -> dict[GasSpecies, _GasSpeciesContainer]:
        return cast(
            dict[GasSpecies, _GasSpeciesContainer],
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

    def pressure(self) -> Array:
        return self.atmosphere.pressure()

    def temperature(self) -> float:
        return self.atmosphere.temperature()

    @property
    def value(self) -> Array:
        """The solution as an array for the solver"""
        value_list: list[Array] = []
        for gas_collection in self.gas_solution.values():
            value_list.append(gas_collection.abundance.value)
        for condensed_collection in self.condensed_solution.values():
            value_list.append(condensed_collection.activity.value)
            value_list.append(condensed_collection.abundance.value)
            value_list.append(condensed_collection.stability.value)

        return jnp.array(value_list)

    @value.setter
    def value(self, value: Array) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        index: int = 0
        for gas_collection in self.gas_solution.values():
            gas_collection.abundance.value = value[index]
            index += gas_collection.NUMBER
        for condensed_collection in self.condensed_solution.values():
            condensed_collection.activity.value = value[index]
            condensed_collection.abundance.value = value[index + 1]
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

    # TODO: This should move to the gas collection only
    def fugacities_by_hill_formula(self) -> dict[str, Array]:
        """Fugacities by Hill formula

        Returns:
            Fugacities by Hill formula
        """
        fugacities: dict[str, Array] = {}
        for gas_species, collection in self.gas_solution.items():
            fugacities[gas_species.hill_formula] = collection.abundance.fugacity()

        return fugacities

    def get_reaction_array(self) -> Array:
        """Gets the reaction array

        Returns:
            The reaction array
        """
        reaction_list: list = []
        for collection in self.gas_solution.values():
            reaction_list.append(collection.abundance.value)
        for collection in self.condensed_solution.values():
            reaction_list.append(collection.activity.value)

        return jnp.array(reaction_list, dtype=jnp.float_)

    def total_moles_hydrogen(self) -> Array | None:
        """Total moles of hydrogen"""
        moles_of_hydrogen: Array = self.moles(element="H")
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
            element_dict["total_mass"] = self.mass(element=element).item()
            total_moles = self.moles(element=element)
            element_dict["total_moles"] = total_moles.item()
            total_moles_hydrogen: Array | None = self.total_moles_hydrogen()
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
                element_dict["atmosphere_moles"] / self.atmosphere.element_moles().item()
            )
            element_dict["molar_mass"] = get_molar_mass(element)

        return output_dict

    def output_full(self) -> dict[str, dict[str, float]]:
        """Full output"""
        output: dict[str, dict[str, float]] = {}
        for species, collection in self.items():
            output[f"{SPECIES_PREFIX}{species.name}"] = collection.output_dict()
        output |= self._output_elements()
        output["atmosphere"] = self.atmosphere.output_dict()
        output["planet"] = self.planet.output_dict()
        output["raw_solution"] = self.output_raw_solution()
        output["solution"] = self.output_solution()

        return output

    def output_raw_solution(self) -> dict[str, float]:
        """Outputs the raw solution as seen by the solver

        Returns:
            A dictionary with formatted keys and float values representing the raw solution.
        """
        output: dict[str, float] = {}
        for gas_species, collection in self.gas_solution.items():
            output[gas_species.name] = collection.abundance.value.item()
        for condensed_species, collection in self.condensed_solution.items():
            species_name: str = condensed_species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = collection.activity.value.item()
            output[species_name] = collection.abundance.value.item()
            output[f"{STABILITY_PREFIX}{species_name}"] = collection.stability.value.item()

        return output

    def output_solution(self) -> dict[str, float]:
        """Outputs the solution in a convenient form for comparison and benchmarking"""
        output: dict[str, float] = {}
        for gas_species, collection in self.gas_solution.items():
            output[gas_species.name] = collection.abundance.pressure().item()
        for condensed_species, collection in self.condensed_solution.items():
            species_name: str = condensed_species.name
            output[f"{ACTIVITY_PREFIX}{species_name}"] = 10 ** collection.activity.value.item()
            output[f"mass_{species_name}"] = collection.abundance.mass().item()

        return output


# TODO: Still required?  Move to test conf?
# def isclose_tolerance(self, target_dict: dict[str, float], message: str = "") -> float | None:
#     """Writes a log message with the tightest tolerance that is satisfied.

#     Args:
#         target_dict: Dictionary of the target values, which should adhere to the format of
#             :meth:`output_solution()`
#         message: Message prefix to write to the logger when a tolerance is satisfied. Defaults
#             to an empty string.

#     Returns:
#         The tightest tolerance satisfied
#     """
#     for log_tolerance in (-6, -5, -4, -3, -2, -1):
#         tol: float = 10**log_tolerance
#         if self.isclose(target_dict, rtol=tol, atol=tol):
#             logger.info("%s (tol = %f)".lstrip(), message, tol)
#             return tol

#     logger.info("%s (no tolerance < 0.1 satisfied)".lstrip(), message)
