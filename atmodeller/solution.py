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
"""Solution classes

Atmodeller uses number densities as the solution quantity and it is convenient to be able to
compute all other quantities self-consistently once these are set.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from collections import ChainMap, Counter
from collections.abc import Mapping
from typing import Generic, Iterator, Protocol, TypeVar, cast

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT, MACHEPS
from atmodeller.constraints import SystemConstraints
from atmodeller.core import GasSpecies, LiquidSpecies, Species
from atmodeller.interfaces import (
    ChemicalSpecies,
    CondensedSpecies,
    ImmutableDict,
    TypeChemicalSpecies,
    TypeChemicalSpecies_co,
)
from atmodeller.jax_containers import Planet
from atmodeller.utilities import get_molar_mass, logsumexp_base10, unit_conversion

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
LOG10_TAU: Array = jnp.log10(TAU)
"""Log10 of the tau factor"""


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


class NumberDensitySpecies(ABC, Generic[TypeChemicalSpecies_co]):
    """A number density for a species

    This class must be instantiated for all gas species before :class:`Atmosphere` can be
    instantiated. Hence :attr:`atmosphere` has a setter so it can be set later.

    Args:
        species: A chemical species
    """

    output_prefix: str = ""
    """Prefix for the keys in the output dictionary"""

    _atmosphere: Atmosphere
    """Atmosphere"""

    def __init__(self, species: TypeChemicalSpecies_co):
        self._species: TypeChemicalSpecies_co = species

    @property
    @abstractmethod
    def value(self) -> Array:
        """Log10 of the number density"""

    @property
    def atmosphere(self) -> Atmosphere:
        """Atmosphere"""
        return self._atmosphere

    @atmosphere.setter
    def atmosphere(self, value: Atmosphere) -> None:
        self._atmosphere = value

    @property
    def species(self) -> TypeChemicalSpecies_co:
        return self._species

    def elements(self) -> Array:
        """Number of elements"""
        return self._species.atoms * self.molecules()

    def element_moles(self) -> Array:
        """Number of moles of elements"""
        return self._species.atoms * self.moles()

    def element_number_density(self) -> Array:
        """Number density of elements"""
        return self._species.atoms * self.number_density()

    def log10_number_density(self, element: str | None = None) -> Array:
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

    def number_density(self, element: str | None = None) -> Array:
        """Number density of the species or an individual element

        Args:
            element: Element to compute the number density for, or None to compute for the species.
                Defaults to None.

        Returns:
            Number density for the species or `element` if not None.
        """
        if element is not None:
            if element in self._species.composition():
                return jnp.power(10, self.log10_number_density(element))
            else:
                return jnp.array(0)

        return jnp.power(10, self.log10_number_density())

    def mass(self, element: str | None = None) -> Array:
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

        return self.moles(element) * molar_mass

    def molecules(self, element: str | None = None) -> Array:
        """Number of molecules of the species or number of an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return self.number_density(element) * self.atmosphere.volume()

    def moles(self, element: str | None = None) -> Array:
        """Number of moles of the species or element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return self.molecules(element) / AVOGADRO

    def output_dict(self, element: str | None = None) -> dict[str, float]:
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
        output[f"{self.output_prefix}number_density"] = self.number_density(element).item()
        output[f"{self.output_prefix}mass"] = self.mass(element).item()
        output[f"{self.output_prefix}molecules"] = self.molecules(element).item()
        output[f"{self.output_prefix}moles"] = self.moles(element).item()

        return output


TypeNumberDensitySpecies = TypeVar("TypeNumberDensitySpecies", bound=NumberDensitySpecies)
TypeNumberDensitySpecies_co = TypeVar(
    "TypeNumberDensitySpecies_co", bound=NumberDensitySpecies, covariant=True
)


class NumberDensitySpeciesSetter(_ValueSetterMixin, NumberDensitySpecies[TypeChemicalSpecies]):
    """A number density with a setter"""


class CondensedNumberDensitySpeciesSetter(NumberDensitySpeciesSetter[CondensedSpecies]):
    """A number density with a setter for a condensed species"""


class GasNumberDensitySpeciesSetter(NumberDensitySpeciesSetter[GasSpecies]):
    """A number density with a setter for a gas species"""

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
            + jnp.log10(self.atmosphere.temperature())
        )

    def density(self) -> Array:
        """Density"""
        return self.mass() / self.atmosphere.volume()

    def log10_fugacity_coefficient(self) -> Array:
        """Log10 fugacity coefficient"""
        return jnp.log10(self.fugacity_coefficient())

    def fugacity_coefficient(self) -> Array:
        """Fugacity coefficient"""
        return jnp.asarray(
            self._species.eos.fugacity_coefficient(
                self.atmosphere.temperature(), self.atmosphere.pressure()
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
        return self.number_density() / self.atmosphere.number_density()

    @override
    def output_dict(self, element: str | None = None) -> dict[str, float]:
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
        output: dict[str, float] = super().output_dict(element)

        if element is None:
            output["pressure"] = self.pressure().item()
            output["fugacity_coefficient"] = self.fugacity_coefficient().item()
            output["fugacity"] = self.fugacity().item()
            output["volume_mixing_ratio"] = self.volume_mixing_ratio().item()

        return output


class DissolvedNumberDensitySpecies(NumberDensitySpecies[GasSpecies]):
    """A number density of a species dissolved in melt

    Args:
        species: A gas species
    """

    output_prefix: str = "dissolved_"
    """Prefix for the keys in the output dictionary"""

    def ppmw(self) -> Array:
        """Parts-per-million by weight of the volatile"""
        return jnp.asarray(
            self._species.solubility.concentration(
                fugacity=self.atmosphere[self._species].fugacity(),
                temperature=self.atmosphere.temperature(),
                pressure=self.atmosphere.pressure(),
                **self.atmosphere.fugacities_by_hill_formula(),
            )
        )

    def reservoir_mass(self) -> ArrayLike:
        """Mass of the reservoir"""
        return self.atmosphere.planet.mantle_melt_mass

    @property
    def value(self) -> Array:
        # This could be switched for a small number instead of inf if problems arise.
        small_value_for_gradient_stability: Array = jnp.array(-jnp.inf)  # jnp.array(-100.0)

        if isinstance(self.species.solubility, NoSolubility) or self.reservoir_mass() < MACHEPS:
            # Short-cut for no solubility for improved speed since autodiffing solubility is slow.
            out: Array = small_value_for_gradient_stability
        else:
            # Calculating the ppmw seems slow, possibly due to the functional dependencies when
            # autodiffing. Hence this is an obvious candidate for future performance improvements.
            ppmw_value: Array = self.ppmw()
            out = (
                jnp.log10(ppmw_value * unit_conversion.ppm_to_fraction)
                + jnp.log10(AVOGADRO)
                - jnp.log10(self._species.molar_mass)
                + jnp.log10(self.reservoir_mass())
                - jnp.log10(self.atmosphere.volume())
            )

        return out

    @override
    def output_dict(self, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        This must return a dictionary of summable values when the dictionary is converted to a
        Counter() when an element is specified.

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = super().output_dict(element)
        if element is None:
            output[f"{self.output_prefix}ppmw"] = self.ppmw().item()

        return output


class TrappedNumberDensitySpecies(DissolvedNumberDensitySpecies):
    """A number density of a species trapped in solids

    Args:
        species: A gas species
    """

    output_prefix: str = "trapped_"
    """Prefix for the keys in the output dictionary"""

    @override
    def ppmw(self) -> Array:
        dissolved_ppmw: Array = super().ppmw()
        trapped_ppmw: Array = dissolved_ppmw * self._species.solid_melt_distribution_coefficient

        return trapped_ppmw

    @override
    def reservoir_mass(self) -> ArrayLike:
        """Mass of the reservoir"""
        return self.atmosphere.planet.mantle_solid_mass


class CondensedSolutionComponent(_ValueSetterMixin):
    """A condensed solution component

    This is used for the activity and stability of condensed species.

    Args:
        species: A condensed species
    """

    def __init__(self, species: CondensedSpecies):
        self._species: CondensedSpecies = species


class TauC:
    """Tauc factor for the calculation of condensate stability :citep:`{e.g.,}KSP24{Equation 19}`

    Args:
        species: A condensed species
    """

    def __init__(self, species: CondensedSpecies):
        self._species: CondensedSpecies = species

    def get_value(self, constraints: SystemConstraints, log10_atmosphere_volume: Array):
        """Gets the value of tauc.

        This effectively controls the minimum non-zero number density of the unstable condensates.

        Args:
            constraints: Constraints
            log10_atmosphere_volume: Log10 volume of the atmosphere

        Returns:
            Value of tauc
        """
        log10_element_atoms: list[Array] = []

        # Find the minimum number density of an element in the species
        for mass_constraint in constraints.mass_constraints:
            if mass_constraint.element in self._species.composition():
                log10_element_atoms.append(mass_constraint.log10_number_of_molecules)

        log10_element_number_density: Array = (
            jnp.min(jnp.array(log10_element_atoms)) - log10_atmosphere_volume
        )

        return LOG10_TAU + log10_element_number_density


class GasSpeciesContainer(NumberDensitySpeciesSetter[GasSpecies]):
    """Gas species container

    Args:
        species: A gas species

    Attributes:
        species: The gas species
        abundance: Species in the gas phase, which is the primary solution quantity
        dissolved: Species dissolved in melt
        trapped: Species trapped in solids
    """

    NUMBER: int = 1
    """Number of solution quantities

    The number density of the species in the gas phase is the only solution quantity, since the
    number density in other reservoirs are directly determined from the gas number densities.
    """
    output_prefix: str = "total_"
    """Prefix for the keys in the output dictionary"""

    @override
    def __init__(self, species: GasSpecies):
        super().__init__(species)
        self.abundance: GasNumberDensitySpeciesSetter = GasNumberDensitySpeciesSetter(species)
        self.dissolved: DissolvedNumberDensitySpecies = DissolvedNumberDensitySpecies(species)
        self.trapped: TrappedNumberDensitySpecies = TrappedNumberDensitySpecies(species)

    @property
    def value(self) -> Array:
        """Total abundance across all reservoirs"""
        values: Array = jnp.asarray(
            (self.abundance.value, self.dissolved.value, self.trapped.value)
        )

        return logsumexp_base10(values)

    @value.setter
    def value(self, gas_abundance: Array) -> None:
        """Sets the gas abundance

        Only the gas abundance needs to be set because the dissolved and trapped reservoirs are
        a function of all the gas abundances.

        Args:
            gas_abundance: Gas abundance
        """
        self.abundance.value = gas_abundance

    @NumberDensitySpeciesSetter.atmosphere.setter
    def atmosphere(self, value: Atmosphere) -> None:
        """Sets and propagates the atmosphere through all reservoirs"""
        # Call super() atmosphere setter
        NumberDensitySpeciesSetter.atmosphere.__set__(
            cast(NumberDensitySpeciesSetter, self), value
        )
        self.abundance.atmosphere = value
        self.dissolved.atmosphere = value
        self.trapped.atmosphere = value

    @override
    def output_dict(self, element: str | None = None) -> dict[str, float]:
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
            self.abundance.output_dict(element)
            | self.dissolved.output_dict(element)
            | self.trapped.output_dict(element)
        )
        if element is None:
            output |= super().output_dict()
            output["molar_mass"] = self._species.molar_mass

        return output


class CondensedSpeciesContainer(NumberDensitySpecies[CondensedSpecies]):
    """Condensed species container

    Args:
        species: A condensed species

    Attributes:
        abundance: Fictitious number density of the condensate
        activity: Activity of the condensate
        stability: Stability of the condensate
        tauc: Tauc for condensate stability
    """

    NUMBER: int = 3
    """Number of solution quantities"""
    output_prefix: str = "condensed_"
    """Prefix for the keys in the output dictionary"""

    @override
    def __init__(self, species: CondensedSpecies):
        super().__init__(species)
        self.abundance: CondensedNumberDensitySpeciesSetter = CondensedNumberDensitySpeciesSetter(
            species
        )
        self.activity: CondensedSolutionComponent = CondensedSolutionComponent(species)
        self.stability: CondensedSolutionComponent = CondensedSolutionComponent(species)
        self.tauc: TauC = TauC(species)

    @property
    def value(self) -> Array:
        """Total abundance across all the reservoirs"""
        return self.abundance.value

    # Value setting is done by the outer container since abundance, activity, and stability all
    # need updating (unlike for the GasSpeciesContainer, where only the gas abundance is a
    # solution quantity).

    @NumberDensitySpeciesSetter.atmosphere.setter
    def atmosphere(self, value: Atmosphere) -> None:
        """Sets and propagates the atmosphere through all reservoirs"""
        # Call super() atmosphere setter
        NumberDensitySpeciesSetter.atmosphere.__set__(
            cast(NumberDensitySpeciesSetter, self), value
        )
        self.abundance.atmosphere = value

    # TODO: Can mess up JAX.
    # def __repr__(self) -> str:
    #     base_repr: str = super().__repr__().rstrip(")")
    #     repr_str: str = f"{base_repr}, activity={10**self.activity.value}, "
    #     repr_str += f"stability={10**self.stability.value}"

    #     return repr_str

    @override
    def output_dict(self, element: str | None = None) -> dict[str, float]:
        """Output dictionary

        For element = None this must return a dictionary with entries that can be summed using
        Counter() when an element is specified.

        Args:
            element: Element to compute the output for, or None to compute for the species.
                Defaults to None.

        Returns:
            Output dictionary
        """
        output: dict[str, float] = super().output_dict(element)
        if element is None:
            output["activity"] = 10 ** self.activity.value.item()
            output["molar_mass"] = self.species.molar_mass

        return output


class CollectionMixin(ABC, Generic[TypeChemicalSpecies_co, TypeNumberDensitySpecies_co]):
    """Collection mixin to compute aggregated quantities"""

    output_prefix: str = ""
    """Prefix for the keys in the output dictionary"""

    data: dict[TypeChemicalSpecies_co, TypeNumberDensitySpecies_co]
    _atmosphere: Atmosphere
    _planet: Planet

    @property
    def atmosphere(self) -> Atmosphere:
        """Atmosphere"""
        return self._atmosphere

    @atmosphere.setter
    def atmosphere(self, value: Atmosphere) -> None:
        """Sets the atmosphere to self and all data values consistently"""
        self._atmosphere = value
        for container in self.data.values():
            container.atmosphere = value

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self._planet

    def log10_number_density(self, element: str | None = None) -> Array:
        """Log10 number density of all species or an individual element

        Args:
            element: Element to compute the log10 number density for, or None to compute for all
                species. Defaults to None.

        Returns:
            Number density for all species or `element` if not None.
        """
        if element is not None:
            log10_number_densities: Array = jnp.asarray(
                [
                    value.log10_number_density(element)
                    for value in self.data.values()
                    if element in value.species.composition()
                ]
            )
        else:
            log10_number_densities = jnp.asarray(
                [value.log10_number_density() for value in self.data.values()]
            )

        return logsumexp_base10(log10_number_densities)

    def number_density(self, element: str | None = None) -> Array:
        """Number density of all species or an individual element

        Args:
            element: Element to compute the number density for, or None to compute for all species.
                Defaults to None.

        Returns:
            Number density for all species or `element` if not None.
        """
        return jnp.power(10, self.log10_number_density(element))

    def element_number_density(self) -> Array:
        """Number density of all elements"""
        return jnp.sum(
            jnp.asarray([value.element_number_density() for value in self.data.values()])
        )

    def mass(self, element: str | None = None) -> Array:
        """Total mass of the species or an individual element

        Args:
            element: Element to compute the mass for, or None to compute for the species. Defaults
                to None.

        Returns:
            Mass in kg for the species or `element` if not None.
        """
        return jnp.sum(jnp.asarray([value.mass(element) for value in self.data.values()]))

    def molecules(self, element: str | None = None) -> Array:
        """Total number of molecules of the species or an individual element

        Args:
            element: Element to compute the number of molecules for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of molecules for the species or number of `element` if not None.
        """
        return jnp.sum(jnp.asarray([value.molecules(element) for value in self.data.values()]))

    def elements(self) -> Array:
        """Total number of elements"""
        return jnp.sum(jnp.asarray([value.elements() for value in self.data.values()]))

    def moles(self, element: str | None = None) -> Array:
        """Total number of moles of the species or an individual element

        Args:
            element: Element to compute the number of moles for, or None to compute for the
                species. Defaults to None.

        Returns:
            Number of moles for the species or `element` if not None.
        """
        return jnp.sum(jnp.asarray([value.moles(element) for value in self.data.values()]))

    def element_moles(self) -> Array:
        """Total number of moles of elements"""
        return jnp.sum(jnp.asarray([value.element_moles() for value in self.data.values()]))

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


class Atmosphere(
    ImmutableDict[GasSpecies, GasNumberDensitySpeciesSetter],
    CollectionMixin[GasSpecies, GasNumberDensitySpeciesSetter],
):
    """Bulk properties of the atmosphere"""

    @override
    def __init__(self, data: dict[GasSpecies, GasNumberDensitySpeciesSetter], planet: Planet):
        super().__init__(data)
        self._planet: Planet = planet
        self.atmosphere = self

    @override
    def output_dict(self) -> dict[str, float]:
        """Output dictionary

        Returns:
            Output dictionary
        """
        output: dict[str, float] = super().output_dict()
        output[f"{self.output_prefix}pressure"] = self.pressure().item()
        output[f"{self.output_prefix}temperature"] = self.temperature()  # type: ignore
        output[f"{self.output_prefix}volume"] = self.volume().item()

        return output

    def fugacities_by_hill_formula(self) -> dict[str, Array]:
        """Fugacities by Hill formula

        Returns:
            Fugacities by Hill formula
        """
        fugacities: dict[str, Array] = {}
        for gas_species, value in self.data.items():
            fugacities[gas_species.hill_formula] = value.fugacity()

        return fugacities

    def log10_molar_mass(self) -> Array:
        """Log10 molar mass"""
        molar_masses: Array = jnp.array([value.species.molar_mass for value in self.data.values()])
        log10_number_densities: Array = jnp.asarray([value.value for value in self.data.values()])
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
            jnp.asarray([value.log10_pressure() for value in self.data.values()])
        )
        return jnp.power(10, log10_pressure)

    def temperature(self) -> ArrayLike:
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
            + jnp.log10(self.planet.surface_area)
            - jnp.log10(self.planet.surface_gravity)
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
        self.atmosphere: Atmosphere = Atmosphere(
            {species: collection.abundance for species, collection in self.data.items()}, planet
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
        for index, container in enumerate(self.data.values()):
            container.value = value[index]

    def output_raw_solution(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for species, container in self.data.items():
            output[species.name] = container.abundance.value.item()

        return output

    def output_solution(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for species, container in self.data.items():
            output[f"{species.name}"] = container.abundance.pressure().item()

        return output


class CondensedCollection(
    ImmutableDict[CondensedSpecies, CondensedSpeciesContainer],
    CollectionMixin[CondensedSpecies, CondensedSpeciesContainer],
):
    """A collection of condensed species"""

    @override
    def __init__(
        self,
        data: dict[CondensedSpecies, CondensedSpeciesContainer],
        planet: Planet,
        atmosphere: Atmosphere,
    ):
        super().__init__(data)
        self._planet: Planet = planet
        self.atmosphere: Atmosphere = atmosphere

    @classmethod
    def create(cls, species: Species, planet: Planet, atmosphere: Atmosphere) -> Self:
        """Creates a condensed collection from species.

        Args:
            species: Species
            planet: Planet
            atmosphere: Atmosphere
        """
        init_dict: dict[CondensedSpecies, CondensedSpeciesContainer] = {
            condensed_species: CondensedSpeciesContainer(condensed_species)
            for condensed_species in species.condensed_species().values()
        }

        return cls(init_dict, planet, atmosphere)

    @property
    def value(self) -> Array:
        """The solution as an array for the solver"""
        value_list: list[Array] = []
        for container in self.data.values():
            value_list.append(container.activity.value)
            value_list.append(container.abundance.value)
            value_list.append(container.stability.value)

        return jnp.asarray(value_list)

    @value.setter
    def value(self, value: Array) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        for index, container in enumerate(self.data.values()):
            start_index: int = index * container.NUMBER
            container.activity.value = value[start_index]
            container.abundance.value = value[start_index + 1]
            container.stability.value = value[start_index + 2]

    def output_raw_solution(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for species, container in self.data.items():
            output[f"{ACTIVITY_PREFIX}{species.name}"] = container.activity.value.item()
            output[species.name] = container.abundance.value.item()
            output[f"{STABILITY_PREFIX}{species.name}"] = container.stability.value.item()

        return output

    def output_solution(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for species, container in self.data.items():
            output[f"{ACTIVITY_PREFIX}{species.name}"] = jnp.power(
                10, container.activity.value
            ).item()
            output[f"mass_{species.name}"] = container.abundance.mass().item()

        return output


SomeContainer = GasSpeciesContainer | CondensedSpeciesContainer


class Solution(CollectionMixin[ChemicalSpecies, SomeContainer]):
    """The solution

    Args:
        gas_collection: Gas collection
        condensed_collection: Condensed collection
        planet: Planet

    Attributes:
        gas: Gas collection
        condensed: Condensed collection
    """

    @override
    def __init__(
        self,
        gas_collection: GasCollection,
        condensed_collection: CondensedCollection,
        planet: Planet,
    ):
        self.gas: GasCollection = gas_collection
        self.condensed: CondensedCollection = condensed_collection
        self._planet: Planet = planet
        self.atmosphere = self.gas.atmosphere

    @property
    def data(self) -> Mapping[ChemicalSpecies, SomeContainer]:
        # Linter gets confused due to invariance of ChainMap type parameters
        return ChainMap(self.gas.data, self.condensed.data)  # type: ignore

    @property
    def value(self) -> Array:
        """The solution as an array for the solver"""
        return jnp.concatenate((self.gas.value, self.condensed.value))

    @value.setter
    def value(self, value: Array) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        number_of_gas_species: int = len(self.gas)
        self.gas.value = value[:number_of_gas_species]
        self.condensed.value = value[number_of_gas_species:]

    @classmethod
    def create(cls, species: Species, planet: Planet) -> Self:
        gas_collection: GasCollection = GasCollection.create(species, planet)
        condensed_collection: CondensedCollection = CondensedCollection.create(
            species, planet, gas_collection.atmosphere
        )

        return cls(gas_collection, condensed_collection, planet)

    @property
    def number(self) -> int:
        """Total number of solution quantities"""
        return sum(collection.NUMBER for collection in self.data.values())

    def unique_elements(self) -> tuple[str, ...]:
        """Unique elements in the species

        Returns:
            A list of unique elements
        """
        unique_elements: list[str] = []
        for species in self:
            unique_elements.extend(species.elements)

        return tuple(set(unique_elements))

    def get_reaction_array(self) -> Array:
        """Gets the reaction array

        Returns:
            The reaction array
        """
        reaction_list: list = []
        for collection in self.gas.values():
            reaction_list.append(collection.abundance.value)
        for collection in self.condensed.values():
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
        for element in self.unique_elements():
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
            for collection in self.data.values():
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
        for species, collection in self.data.items():
            output[f"{SPECIES_PREFIX}{species.name}"] = collection.output_dict()
        output |= self._output_elements()
        output["atmosphere"] = self.atmosphere.output_dict()
        output["planet"] = self.planet.asdict()
        output["raw_solution"] = self.output_raw_solution()
        output["solution"] = self.output_solution()

        return output

    def output_raw_solution(self) -> dict[str, float]:
        return self.gas.output_raw_solution() | self.condensed.output_raw_solution()

    def output_solution(self) -> dict[str, float]:
        return self.gas.output_solution() | self.condensed.output_solution()

    def __getitem__(self, key) -> SomeContainer:
        return self.data[key]

    def __iter__(self) -> Iterator[ChemicalSpecies]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":

    # Quick testing
    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    H2O_l = LiquidSpecies("H2O")
    species_: Species = Species([H2O_g, H2_g, O2_g, H2O_l])
    planet_: Planet = Planet()

    # Must create and then set a value
    # out_test = GasCollection.create(species_, planet_)
    # out_test.value = jnp.array([26, 28, 24])

    # print(out_test.atmosphere)
    # print(out_test.atmosphere.moles())
    # print(out_test.atmosphere.number_density())
    # out_test.value = jnp.array([1, 2, 3])
    # print(out_test.atmosphere.number_density(element="H"))

    print("Trying solution")

    solution: Solution = Solution.create(species_, planet_)
    solution.value = jnp.array([6, 7, 8, 9])
    print(solution.data)
    print(solution[H2O_g].moles("O"))

    print(solution.output_full())
    # print(solution.output_dict())
    # print(solution.atmosphere.output_dict())
    # print(solution.atmosphere.elements())
