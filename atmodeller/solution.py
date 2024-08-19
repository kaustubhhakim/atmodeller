#!/usr/bin/env python3
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
from collections import ChainMap, Counter
from typing import Generic, Protocol, TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.core import GasSpecies, Planet, Species
from atmodeller.interfaces import (
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

# Individual species


class SpeciesComponentProtocol(Protocol[TypeChemicalSpecies_co]):
    """Species component protocol"""

    _species: TypeChemicalSpecies_co

    @property
    def value(self) -> Array: ...


class SpeciesComponentSetterProtocol(SpeciesComponentProtocol, Protocol):
    """Species component with setter protocol"""

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


class NumberDensitySpecies(ABC, Generic[TypeChemicalSpecies]):
    """A number density for a species

    Args:
        species: A chemical species
    """

    output_prefix: str = ""
    """Prefix for the keys in the output dictionary"""

    def __init__(self, species: TypeChemicalSpecies):
        self._species: TypeChemicalSpecies = species

    @property
    @abstractmethod
    def value(self) -> Array:
        """Log10 of the number density"""

    @property
    def species(self) -> TypeChemicalSpecies:
        return self._species

    def elements(self, atmosphere: Atmosphere) -> Array:
        """Total number of elements

        Args:
            atmosphere: Atmosphere
        """
        return self._species.atoms * self.molecules(atmosphere)

    def element_moles(self, atmosphere: Atmosphere) -> Array:
        """Total number of moles of elements

        Args:
            atmosphere: Atmosphere
        """
        return self._species.atoms * self.moles(atmosphere)

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

    def mass(self, atmosphere: Atmosphere, *, element: str | None = None) -> Array:
        """Mass of the species or an individual element

        Args:
            atmosphere: Atmosphere
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        if element is not None:
            molar_mass: float = get_molar_mass(element)
        else:
            molar_mass = self._species.molar_mass

        return self.moles(atmosphere, element=element) * molar_mass

    def molecules(self, atmosphere: Atmosphere, *, element: str | None = None) -> Array:
        """Number of molecules of the species or number of an individual element

        Args:
            atmosphere: Atmosphere
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return self.number_density(element=element) * atmosphere.volume()

    def moles(self, atmosphere: Atmosphere, *, element: str | None = None) -> Array:
        """Number of moles of the species or element

        Args:
            atmosphere: Atmosphere
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self.molecules(atmosphere, element=element) / AVOGADRO

    def output_dict(
        self, atmosphere: Atmosphere, *, element: str | None = None
    ) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter().

        Args:
            atmosphere: Atmosphere
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = {}
        output[f"{self.output_prefix}number_density"] = self.number_density(element=element).item()
        output[f"{self.output_prefix}mass"] = self.mass(atmosphere, element=element).item()
        output[f"{self.output_prefix}molecules"] = self.molecules(
            atmosphere, element=element
        ).item()
        output[f"{self.output_prefix}moles"] = self.moles(atmosphere, element=element).item()

        return output

    def __repr__(self) -> str:
        return f"number_density={self.number_density():.2e}"


TypeNumberDensitySpecies = TypeVar("TypeNumberDensitySpecies", bound=NumberDensitySpecies)
TypeNumberDensitySpecies_co = TypeVar(
    "TypeNumberDensitySpecies_co", bound=NumberDensitySpecies, covariant=True
)


class _NumberDensitySpeciesSetter(_ValueSetterMixin, NumberDensitySpecies[TypeChemicalSpecies]):
    """A number density with a setter"""


class _CondensedNumberDensitySpeciesSetter(_NumberDensitySpeciesSetter[CondensedSpecies]):
    """A number density with a setter for a condensed species

    Args:
        species: A gas species
        solution: The solution
    """


class GasNumberDensitySpeciesSetter(_NumberDensitySpeciesSetter[GasSpecies]):
    """A number density with a setter for a gas species

    Args:
        species: A gas species
        solution: The solution
    """

    output_prefix: str = "atmosphere_"
    """Prefix for the keys in the output dictionary"""

    def pressure(self, atmosphere: Atmosphere) -> Array:
        """Pressure in bar

        Args:
            atmosphere: Atmosphere
        """
        return jnp.power(10, self.log10_pressure(atmosphere))

    def log10_pressure(self, atmosphere: Atmosphere) -> Array:
        """Log10 pressure

        Args:
            atmosphere: Atmosphere
        """
        return (
            self.log10_number_density()
            + jnp.log10(BOLTZMANN_CONSTANT_BAR)
            + jnp.log10(atmosphere.temperature())
        )

    def density(self, atmosphere: Atmosphere) -> Array:
        """Density

        Args:
            atmosphere: Atmosphere
        """
        return self.mass(atmosphere) / atmosphere.volume()

    def log10_fugacity_coefficient(self, atmosphere: Atmosphere) -> Array:
        """Log10 fugacity coefficient

        Args:
            atmosphere: Atmosphere
        """
        return jnp.log10(self.fugacity_coefficient(atmosphere))

    def fugacity_coefficient(self, atmosphere: Atmosphere) -> Array:
        """Fugacity coefficient

        Args:
            atmosphere: Atmosphere
        """
        return jnp.array(
            self._species.eos.fugacity_coefficient(atmosphere.temperature(), atmosphere.pressure())
        )

    def log10_fugacity(self, atmosphere: Atmosphere) -> Array:
        """Log10 fugacity

        Args:
            atmosphere: Atmosphere
        """
        return self.log10_pressure(atmosphere) + self.log10_fugacity_coefficient(atmosphere)

    def fugacity(self, atmosphere: Atmosphere) -> Array:
        """Fugacity

        Args:
            atmosphere: Atmosphere
        """
        return jnp.power(10, self.log10_fugacity(atmosphere))

    def volume_mixing_ratio(self, atmosphere: Atmosphere) -> Array:
        """Volume mixing ratio

        Args:
            atmosphere: Atmosphere
        """
        return self.number_density() / atmosphere.number_density()

    @override
    def output_dict(
        self, atmosphere: Atmosphere, *, element: str | None = None
    ) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            atmosphere: Atmosphere
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = super().output_dict(atmosphere, element=element)

        if element is None:
            output["pressure"] = self.pressure(atmosphere).item()
            output["fugacity_coefficient"] = self.fugacity_coefficient(atmosphere).item()
            output["fugacity"] = self.fugacity(atmosphere).item()
            output["volume_mixing_ratio"] = self.volume_mixing_ratio(atmosphere).item()

        return output


class DissolvedNumberDensitySpecies(NumberDensitySpecies[GasSpecies]):
    """A number density of a species dissolved in melt

    Args:
        species: A gas species
    """

    output_prefix: str = "dissolved_"
    """Prefix for the keys in the output dictionary"""

    _value: Array

    def ppmw(self, atmosphere: Atmosphere) -> Array:
        """Parts-per-million by weight of the volatile

        Args:
            atmosphere: Atmosphere

        Returns:
            Parts-per-million by weight of the volatile
        """
        return jnp.asarray(
            self._species.solubility.concentration(
                fugacity=atmosphere[self._species].fugacity(atmosphere),
                temperature=atmosphere.temperature(),
                pressure=atmosphere.pressure(),
                **atmosphere.fugacities_by_hill_formula(),
            )
        )

    def reservoir_mass(self, atmosphere: Atmosphere) -> float:
        """Mass of the reservoir

        Args:
            atmosphere: Atmosphere

        Returns:
            Mass of the melt reservoir
        """
        return atmosphere.planet.mantle_melt_mass

    def set_value(self, atmosphere: Atmosphere) -> None:
        """Sets the number density.

        Args:
            atmosphere: Atmosphere
        """
        ppmw_value: Array = self.ppmw(atmosphere)
        reservoir_mass: float = self.reservoir_mass(atmosphere)

        self._value: Array = jnp.where(
            (ppmw_value > 0) & (reservoir_mass > 0),
            jnp.log10(
                UnitConversion.ppm_to_fraction(ppmw_value)
                * AVOGADRO
                / self._species.molar_mass
                * self.reservoir_mass(atmosphere)
                / atmosphere.volume()
            ),
            -jnp.inf,
        )

    @property
    def value(self) -> ArrayLike:
        return self._value

    @override
    def output_dict(
        self, atmosphere: Atmosphere, *, element: str | None = None
    ) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            atmosphere: Atmosphere
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = super().output_dict(atmosphere, element=element)
        if element is None:
            output[f"{self.output_prefix}ppmw"] = self.ppmw(atmosphere).item()

        return output


class TrappedNumberDensitySpecies(DissolvedNumberDensitySpecies):
    """A number density of a species trapped in solids

    Args:
        species: A gas species
    """

    output_prefix: str = "trapped_"
    """Prefix for the keys in the output dictionary"""

    @override
    def ppmw(self, atmosphere: Atmosphere) -> Array:
        dissolved_ppmw: Array = super().ppmw(atmosphere)
        trapped_ppmw: Array = dissolved_ppmw * self._species.solid_melt_distribution_coefficient

        return trapped_ppmw

    @override
    def reservoir_mass(self, atmosphere: Atmosphere) -> float:
        """Mass of the reservoir

        Args:
            atmosphere: Atmosphere

        Returns:
            Mass of the solid reservoir
        """
        return atmosphere.planet.mantle_solid_mass


# class _CondensedSolutionComponent(_ValueSetterMixin):
#     """A condensed solution component

#     This is used for the activity and stability of condensed species.

#     Args:
#         species: A condensed species
#         solution: The solution
#     """

#     def __init__(self, species: CondensedSpecies, solution: Solution):
#         self._species: CondensedSpecies = species
#         self._solution: Solution = solution


# class _TauC(_CondensedSolutionComponent):
#     """Tauc factor for the calculation of condensate stability :citep:`{e.g.,}KSP24{Equation 19}`

#     Args:
#         species: A condensed species
#         solution: The solution
#     """

#     @property
#     def value(self) -> Array:
#         element_number_densities: list[Array] = [
#             self._solution.number_density(element=element) for element in self._species.elements
#         ]
#         log10_tauc: Array = LOG10_TAU + jnp.log10(jnp.min(jnp.array(element_number_densities)))

#         return log10_tauc


# Containers for the (potentially) multiple reservoirs that an individual species is contained
# within.


class GasSpeciesContainer(NumberDensitySpecies[GasSpecies]):
    """Gas species container

    Args:
        species: A gas species

    Attributes:
        species: The gas species
        gas: Species in the gas phase
        dissolved: Species dissolved in melt
        trapped: Species trapped in solids
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
    def __init__(self, species: GasSpecies):
        super().__init__(species)
        self.abundance: GasNumberDensitySpeciesSetter = GasNumberDensitySpeciesSetter(species)
        self.dissolved_abundance: DissolvedNumberDensitySpecies = DissolvedNumberDensitySpecies(
            species
        )
        self.trapped_abundance: TrappedNumberDensitySpecies = TrappedNumberDensitySpecies(species)

    def set_interior_values(self, gas_abundance: Array, atmosphere: Atmosphere) -> None:
        """Sets the number density of interior reservoirs

        Args:
            gas_abundance: Log10 number density in the gas
            atmosphere: Atmosphere
        """
        # TODO: This must be uncomment to work. Unsure why.
        self.abundance.value = gas_abundance
        # jax.debug.print("now here = {out}", out=self.abundance.value)
        self.dissolved_abundance.set_value(atmosphere)
        self.trapped_abundance.set_value(atmosphere)

    @property
    def value(self) -> Array:
        gas_value: Array = self.abundance.value
        dissolved_value: ArrayLike = self.dissolved_abundance.value
        trapped_value: ArrayLike = self.trapped_abundance.value
        log10_values: Array = jnp.asarray((gas_value, dissolved_value, trapped_value))

        return logsumexp_base10(log10_values)

    @override
    def output_dict(
        self, atmosphere: Atmosphere, *, element: str | None = None
    ) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            atmosphere: Atmosphere
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = (
            self.abundance.output_dict(atmosphere, element=element)
            | self.dissolved_abundance.output_dict(atmosphere, element=element)
            | self.trapped_abundance.output_dict(atmosphere, element=element)
        )
        if element is None:
            output |= super().output_dict(atmosphere)
            output["molar_mass"] = self._species.molar_mass

        return output

    def __repr__(self) -> str:
        return f"{self.abundance!r}"


# class _CondensedSpeciesContainer(NumberDensitySpecies[CondensedSpecies]):
#     """Condensed species container

#     Args:
#         species: A condensed species

#     Attributes:
#         # TODO: Consistency between this gas container and the condensed container
#         number_density: Solution for the fictitous number density of the condensate
#         activity: Solution for the activity of the condensate
#         stability: Solution for the stability of the condensate
#         tauc: Tauc for condensate stability
#     """

#     NUMBER: int = 3
#     """Number of solution quantities"""
#     output_prefix: str = "condensed_"
#     """Prefix for the keys in the output dictionary"""

#     @override
#     def __init__(self, species: CondensedSpecies):
#         super().__init__(species)
#         self.abundance: _CondensedNumberDensitySpeciesSetter = (
#             _CondensedNumberDensitySpeciesSetter(species)
#         )
#         self.activity: _CondensedSolutionComponent = _CondensedSolutionComponent(species)
#         self.stability: _CondensedSolutionComponent = _CondensedSolutionComponent(species)
#         self.tauc: _TauC = _TauC(species)

#     @property
#     def species(self) -> CondensedSpecies:
#         return self._species

#     @property
#     def value(self) -> Array:
#         return self.abundance.value

#     def __repr__(self) -> str:
#         base_repr: str = super().__repr__().rstrip(")")
#         repr_str: str = f"{base_repr}, activity={10**self.activity.value}, "
#         repr_str += f"stability={10**self.stability.value}"

#         return repr_str

#     @override
#     def output_dict(
#         self, atmosphere: Atmosphere, *, element: str | None = None
#     ) -> dict[str, float]:
#         """Output dictionary

#         For element = None this must return a dictionary with entries that can be summed using
#         Counter() when an element is specified.

#         Args:
#             atmosphere: Atmosphere
#             element: Element to compute the output for, or None to compute for the species.
#                 Defaults to None.

#         Returns:
#             Output dictionary
#         """
#         output: dict[str, float] = super().output_dict(atmosphere, element=element)
#         if element is None:
#             output["activity"] = 10 ** self.activity.value.item()
#             output["molar_mass"] = self.species.molar_mass

#         return output


class CollectionMixin(ABC, Generic[TypeChemicalSpecies, TypeNumberDensitySpecies]):
    """TODO: Mixin"""

    output_prefix: str = ""
    """Prefix for the keys in the output dictionary"""

    data: dict[TypeChemicalSpecies, TypeNumberDensitySpecies]

    @property
    @abstractmethod
    def atmosphere(self) -> Atmosphere: ...

    def log10_number_density(self, element: str | None = None) -> Array:
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

    def number_density(self, element: str | None = None) -> Array:
        """Number density of all species or an individual element

        Args:
            element: Element to compute the number density for, or None to compute for all species.
                Defaults to None.

        Returns:
            Number density for all species or `element` if not None.
        """
        return jnp.power(10, self.log10_number_density(element=element))

    def element_number_density(self) -> Array:
        """Number density of all elements"""
        return jnp.sum(jnp.array([value.element_number_density() for value in self.data.values()]))

    def mass(self, element: str | None = None) -> Array:
        """Total mass of the species or an individual element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        return jnp.sum(
            jnp.array(
                [value.mass(self.atmosphere, element=element) for value in self.data.values()]
            )
        )

    def molecules(self, element: str | None = None) -> Array:
        """Total number of molecules of the species or an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return jnp.sum(
            jnp.array(
                [value.molecules(self.atmosphere, element=element) for value in self.data.values()]
            )
        )

    def elements(self) -> Array:
        """Total number of elements"""
        return jnp.sum(
            jnp.array([value.elements(self.atmosphere) for value in self.data.values()])
        )

    def moles(self, element: str | None = None) -> Array:
        """Total number of moles of the species or an individual element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return jnp.sum(
            jnp.array(
                [value.moles(self.atmosphere, element=element) for value in self.data.values()]
            )
        )

    def element_moles(self) -> Array:
        """Total number of moles of elements"""
        return jnp.sum(
            jnp.array([value.element_moles(self.atmosphere) for value in self.data.values()])
        )

    def output_dict(self, *, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = {}
        output[f"{self.output_prefix}number_density"] = self.number_density(element).item()
        output[f"{self.output_prefix}mass"] = self.mass(element).item()
        output[f"{self.output_prefix}molecules"] = self.molecules(element).item()
        output[f"{self.output_prefix}moles"] = self.moles(element).item()
        output[f"{self.output_prefix}elements"] = self.elements().item()
        output[f"{self.output_prefix}element_moles"] = self.element_moles().item()

        return output


# class _Collection(ImmutableDict[TypeChemicalSpecies, TypeNumberDensitySpecies_co]):
#     """A container for the solution"""

#     output_prefix: str = ""
#     """Prefix for the keys in the output dictionary"""

#     @override
#     def __init__(
#         self,
#         data: dict[TypeChemicalSpecies, TypeNumberDensitySpecies_co] | None = None,
#         *,
#         planet: Planet,
#     ):
#         super().__init__(data)
#         self._planet: Planet = planet

#     @property
#     def planet(self) -> Planet:
#         return self._planet


class Atmosphere(
    ImmutableDict[GasSpecies, GasNumberDensitySpeciesSetter],
    CollectionMixin[GasSpecies, GasNumberDensitySpeciesSetter],
):
    """Bulk properties of the atmosphere"""

    @override
    def __init__(self, data: dict[GasSpecies, GasNumberDensitySpeciesSetter], planet: Planet):
        super().__init__(data)
        self._planet: Planet = planet

    @property
    def atmosphere(self) -> Atmosphere:
        return self

    @property
    def planet(self) -> Planet:
        return self._planet

    @override
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

    def fugacities_by_hill_formula(self) -> dict[str, Array]:
        """Fugacities by Hill formula

        Returns:
            Fugacities by Hill formula
        """
        fugacities: dict[str, Array] = {}
        for gas_species, value in self.data.items():
            fugacities[gas_species.hill_formula] = value.fugacity(self)

        return fugacities

    def log10_molar_mass(self) -> Array:
        """Log10 molar mass"""
        molar_masses: Array = jnp.array([value.species.molar_mass for value in self.data.values()])
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
            jnp.array([value.log10_pressure(self) for value in self.data.values()])
        )
        return jnp.power(10, log10_pressure)

    def temperature(self) -> float:
        """Temperature"""
        return self._planet.surface_temperature

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
            + jnp.log10(self._planet.surface_area)
            - jnp.log10(self._planet.surface_gravity)
        )

        return log10_volume

    def volume(self) -> Array:
        """Volume"""
        return jnp.power(10, self.log10_volume())


class GasCollection(
    ImmutableDict[GasSpecies, GasSpeciesContainer],
    CollectionMixin[GasSpecies, GasSpeciesContainer],
):
    """A collection of gas species"""

    @override
    def __init__(self, data: dict[GasSpecies, GasSpeciesContainer], planet: Planet):
        super().__init__(data)
        self._planet: Planet = planet
        self._atmosphere: Atmosphere = Atmosphere(
            {species: collection.abundance for species, collection in self.items()}, planet
        )

    @classmethod
    def create(cls, species: Species, planet: Planet) -> Self:
        """Creates a gas collection from species.

        Args:
            species: Species
            planet: Planet
        """
        init_dict: dict[GasSpecies, GasSpeciesContainer] = {
            gas_species: GasSpeciesContainer(gas_species)
            for gas_species in species.gas_species().values()
        }

        return cls(init_dict, planet)

    @property
    def atmosphere(self) -> Atmosphere:
        return self._atmosphere

    @property
    def value(self) -> Array:
        """The solution as an array for the solver"""
        value_list: list[Array] = []
        for container in self.data.values():
            value_list.append(container.abundance.value)

        return jnp.array(value_list)

    @value.setter
    def value(self, value: Array) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        # Must set all gas species first because potentially multiple gas species are required to
        # compute the interior dissolved and trapped content.
        index: int = 0
        for container in self.data.values():
            container.abundance.value = value[index]
            jax.debug.print("here = {out}", out=container.abundance.value)

        # jax.debug.print("{out}", out=self.atmosphere)

        index = 0
        for container in self.data.values():
            container.set_interior_values(value[index], self.atmosphere)
            index += container.NUMBER

    def output_raw_solution(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for species, container in self.data.items():
            output[species.name] = container.abundance.value.item()

        return output

    def output_solution(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for species, container in self.data.items():
            output[species.name] = container.abundance.pressure(self.atmosphere).item()

        return output


# class CondensedCollection(
#     _Collection[CondensedSpecies, _CondensedSpeciesContainer], _CollectionMixin
# ):
#     """A collection of condensed species"""

#     @classmethod
#     def create_from_species(cls, species: Species, planet: Planet) -> Self:
#         """Creates a condensed collection from species.

#         Args:
#             species: Species
#             planet: Planet
#         """
#         init_dict: dict[CondensedSpecies, _CondensedSpeciesContainer] = {
#             condensed_species: _CondensedSpeciesContainer(condensed_species)
#             for condensed_species in species.condensed_species().values()
#         }

#         return cls(init_dict, planet=planet)


class Solution(CollectionMixin):
    """The solution

    # TODO: Update all
    """

    def __init__(
        self,
        gas_collection: GasCollection,
        # condensed_collection: CondensedCollection,
        planet: Planet,
    ):
        self.gas: GasCollection = gas_collection
        # self.condensed: CondensedCollection = condensed_collection
        self._planet: Planet = planet

    @property
    def data(self):
        return ChainMap(self.gas.data)  # , self.condensed.data)

    @property
    def atmosphere(self) -> Atmosphere:
        return self.gas.atmosphere

    @property
    def value(self) -> Array:
        """The solution as an array for the solver"""
        return self.gas.value

    @value.setter
    def value(self, value: Array) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        self.gas.value = value

    @classmethod
    def create(cls, species: Species, planet: Planet) -> Self:
        gas_collection: GasCollection = GasCollection.create(species, planet)
        # condensed_collection: CondensedCollection = CondensedCollection.create_from_species(
        #    species, planet
        # )

        return cls(gas_collection, planet)  # , condensed_collection, planet)

    def __repr__(self) -> str:
        return self.data.__repr__()

    #     # _atmosphere: Atmosphere

    #     # @classmethod
    #     # def create_from_species(cls, species: Species) -> Self:
    #     #     """Creates a Solution instance

    #     #     Args:
    #     #         species: Species
    #     #     """
    #     #     solution: Self = cls()
    #     #     for gas_species in species.gas_species().values():
    #     #         solution.data[gas_species] = GasSpeciesContainer(gas_species, solution)
    #     #     for condensed_species in species.condensed_species().values():
    #     #         solution.data[condensed_species] = _CondensedSpeciesContainer(
    #     #             condensed_species, solution
    #     #         )

    #     #     return solution

    #     # @classmethod
    #     # def create_from_species_and_planet(cls, species: Species, planet: Planet) -> Self:
    #     #     """Creates a Solution instance

    #     #     Args:
    #     #         species: Species
    #     #         planet: Planet
    #     #     """
    #     #     solution: Self = cls.create_from_species(species)
    #     #     init_dict: dict[GasSpecies, GasNumberDensitySpeciesSetter] = {
    #     #         species: collection.abundance for species, collection in solution.gas_solution.items()
    #     #     }
    #     #     # Only need to set these attributes once so pylint: disable=protected-access
    #     #     solution._atmosphere = Atmosphere(init_dict, planet)

    #     #     return solution

    #     # def create_from_species_and_planet(cls, species: Species, planet: Planet) -> Self:
    #     #     solution: Self

    #     # # TODO: Now access directly from gas collection.
    #     # # @property
    #     # # def atmosphere(self) -> Atmosphere:
    #     # #     return self._atmosphere

    #     # @property
    #     # def condensed_solution(self) -> dict[CondensedSpecies, _CondensedSpeciesContainer]:
    #     #     return self.condensed_solution
    #     #     # return cast(
    #     #     #     dict[CondensedSpecies, _CondensedSpeciesContainer],
    #     #     #     {key: value for key, value in self.items() if isinstance(key, CondensedSpecies)},
    #     #     # )

    @property
    def condensed_solution(self) -> dict:
        return {}

    @property
    def gas_solution(self) -> GasCollection:  # dict[GasSpecies, GasSpeciesContainer]:
        return self.gas

    @property
    def number(self) -> int:
        """Total number of solution quantities"""
        return sum(collection.NUMBER for collection in self.data.values())

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self._planet

    def pressure(self) -> Array:
        return self.atmosphere.pressure()

    def temperature(self) -> float:
        return self.atmosphere.temperature()

    #     @property
    #     def value(self) -> Array:
    #         """The solution as an array for the solver"""
    #         value_list: list[Array] = []
    #         for gas_collection in self.gas_solution.values():
    #             value_list.append(gas_collection.abundance.value)
    #         for condensed_collection in self.condensed_solution.values():
    #             value_list.append(condensed_collection.activity.value)
    #             value_list.append(condensed_collection.abundance.value)
    #             value_list.append(condensed_collection.stability.value)

    #         return jnp.array(value_list)

    #     @value.setter
    #     def value(self, value: Array) -> None:
    #         """Sets the solution from an array.

    #         Args:
    #             value: An array, which is usually passed by the solver.
    #         """
    #         index: int = 0
    #         for gas_collection in self.gas_solution.values():
    #             gas_collection.abundance.value = value[index]
    #             index += gas_collection.NUMBER
    #         for condensed_collection in self.condensed_solution.values():
    #             condensed_collection.activity.value = value[index]
    #             condensed_collection.abundance.value = value[index + 1]
    #             condensed_collection.stability.value = value[index + 2]
    #             index += condensed_collection.NUMBER

    #     # FIXME: Same name as method in base class. Conflict?
    #     def elements(self) -> list[str]:
    #         """Unique elements in the species

    #         Returns:
    #             A list of unique elements
    #         """
    #         elements: list[str] = []
    #         for species in self:
    #             elements.extend(species.elements)

    #         return list(set(elements))

    #     # TODO: Probably not required anymore.
    #     # def merge(self, other: Solution) -> None:
    #     #     """Merges the data from another solution

    #     #     Args:
    #     #         other: The other solution to merge data from
    #     #     """
    #     #     self.data |= other.data

    def get_reaction_array(self) -> Array:
        """Gets the reaction array

        Returns:
            The reaction array
        """
        reaction_list: list = []
        for collection in self.gas.values():
            reaction_list.append(collection.abundance.value)
        # for collection in self.condensed.values():
        #    reaction_list.append(collection.activity.value)

        return jnp.array(reaction_list, dtype=jnp.float_)

    #     def total_moles_hydrogen(self) -> Array | None:
    #         """Total moles of hydrogen"""
    #         moles_of_hydrogen: Array = self.moles(element="H")
    #         if moles_of_hydrogen <= 0:
    #             return None
    #         else:
    #             return moles_of_hydrogen

    #     def _output_elements(self) -> dict[str, dict[str, float]]:
    #         """Output for elements"""
    #         output_dict: dict[str, dict[str, float]] = {}
    #         for element in self.elements():
    #             element_dict: dict[str, float] = output_dict.setdefault(
    #                 f"{ELEMENT_PREFIX}{element}", {}
    #             )
    #             element_dict["total_mass"] = self.mass(element=element).item()
    #             total_moles = self.moles(element=element)
    #             element_dict["total_moles"] = total_moles.item()
    #             total_moles_hydrogen: Array | None = self.total_moles_hydrogen()
    #             if total_moles_hydrogen is not None:
    #                 element_dict["logarithmic_abundance"] = (
    #                     jnp.log10(total_moles / total_moles_hydrogen).item() + 12
    #                 )
    #             counter: Counter = Counter()
    #             for collection in self.values():
    #                 output: dict[str, float] = collection.output_dict(element=element)
    #                 counter += Counter(output)
    #             element_dict |= dict(counter)
    #             try:
    #                 element_dict["degree_of_condensation"] = (
    #                     element_dict["condensed_moles"] / element_dict["total_moles"]
    #                 )
    #             except KeyError:
    #                 # No condensed species for this element
    #                 pass
    #             element_dict["volume_mixing_ratio"] = (
    #                 element_dict["atmosphere_moles"] / self.atmosphere.element_moles().item()
    #             )
    #             element_dict["molar_mass"] = get_molar_mass(element)

    #         return output_dict

    #     def output_full(self) -> dict[str, dict[str, float]]:
    #         """Full output"""
    #         output: dict[str, dict[str, float]] = {}
    #         for species, collection in self.items():
    #             output[f"{SPECIES_PREFIX}{species.name}"] = collection.output_dict()
    #         output |= self._output_elements()
    #         output["atmosphere"] = self.atmosphere.output_dict()
    #         output["planet"] = self.planet.output_dict()
    #         output["raw_solution"] = self.output_raw_solution()
    #         output["solution"] = self.output_solution()

    #         return output

    def output_raw_solution(self) -> dict[str, float]:
        return self.gas.output_raw_solution()

    def output_solution(self) -> dict[str, float]:
        return self.gas.output_solution()


#     def output_raw_solution(self) -> dict[str, float]:
#         """Outputs the raw solution as seen by the solver

#         Returns:
#             A dictionary with formatted keys and float values representing the raw solution.
#         """
#         output: dict[str, float] = {}
#         for gas_species, collection in self.gas_solution.items():
#             output[gas_species.name] = collection.abundance.value.item()
#         for condensed_species, collection in self.condensed_solution.items():
#             species_name: str = condensed_species.name
#             output[f"{ACTIVITY_PREFIX}{species_name}"] = collection.activity.value.item()
#             output[species_name] = collection.abundance.value.item()
#             output[f"{STABILITY_PREFIX}{species_name}"] = collection.stability.value.item()

#         return output

#     def output_solution(self) -> dict[str, float]:
#         """Outputs the solution in a convenient form for comparison and benchmarking"""
#         output: dict[str, float] = {}
#         for gas_species, collection in self.gas_solution.items():
#             output[gas_species.name] = collection.abundance.pressure().item()
#         for condensed_species, collection in self.condensed_solution.items():
#             species_name: str = condensed_species.name
#             output[f"{ACTIVITY_PREFIX}{species_name}"] = 10 ** collection.activity.value.item()
#             output[f"mass_{species_name}"] = collection.abundance.mass().item()

#         return output


# # TODO: Still required?  Move to test conf?
# # def isclose_tolerance(self, target_dict: dict[str, float], message: str = "") -> float | None:
# #     """Writes a log message with the tightest tolerance that is satisfied.

# #     Args:
# #         target_dict: Dictionary of the target values, which should adhere to the format of
# #             :meth:`output_solution()`
# #         message: Message prefix to write to the logger when a tolerance is satisfied. Defaults
# #             to an empty string.

# #     Returns:
# #         The tightest tolerance satisfied
# #     """
# #     for log_tolerance in (-6, -5, -4, -3, -2, -1):
# #         tol: float = 10**log_tolerance
# #         if self.isclose(target_dict, rtol=tol, atol=tol):
# #             logger.info("%s (tol = %f)".lstrip(), message, tol)
# #             return tol

# #     logger.info("%s (no tolerance < 0.1 satisfied)".lstrip(), message)


# Placeholder to enable code to run
# class Solution: ...


if __name__ == "__main__":

    # Quick testing
    species_: Species = Species([GasSpecies("H2O"), GasSpecies("H2"), GasSpecies("O2")])
    planet_: Planet = Planet()

    # Must create and then set a value
    out = GasCollection.create(species_, planet_)
    out.value = jnp.array([26, 28, 24])

    print(out.atmosphere)
    print(out.atmosphere.moles())
    print(out.atmosphere.number_density())
    out.value = jnp.array([1, 2, 3])
    print(out.atmosphere.number_density(element="H"))

    print("Trying solution")

    solution: Solution = Solution.create(species_, planet_)
    solution.value = jnp.array([6, 7, 8])
    print(solution)
    print(solution.molecules("H"))
