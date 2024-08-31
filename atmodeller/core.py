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
from typing import Mapping, NamedTuple

import jax.numpy as jnp

from atmodeller import GRAVITATIONAL_CONSTANT
from atmodeller.eos.interfaces import IdealGas, RealGasProtocol
from atmodeller.interfaces import ChemicalSpecies, CondensedSpecies, ImmutableList
from atmodeller.solubility.compositions import composition_solubilities
from atmodeller.solubility.interfaces import NoSolubility, SolubilityProtocol
from atmodeller.utilities import filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


logger: logging.Logger = logging.getLogger(__name__)


class Planet(NamedTuple):
    """The properties of a planet

    Defaults values are for a fully molten Earth.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to Earth.
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        surface_radius: Radius of the planetary surface in m. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
    """

    planet_mass: float = 5.972e24
    """Mass of the planet in kg"""
    core_mass_fraction: float = 0.295334691460966
    """Mass fraction of the core relative to the planetary mass in kg/kg"""
    mantle_melt_fraction: float = 1.0
    """Mass fraction of the molten mantle in kg/kg"""
    surface_radius: float = 6371000.0
    """Radius of the surface in m"""
    surface_temperature: float = 2000.0
    """Temperature of the surface in K"""

    @property
    def mantle_mass(self) -> float:
        """Mantle mass"""
        return self.planet_mass * (1.0 - self.core_mass_fraction)

    @property
    def mantle_melt_mass(self) -> float:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_solid_mass(self) -> float:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def surface_area(self) -> float:
        """Surface area"""
        return 4.0 * jnp.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> float:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2

    def to_dict(self) -> dict:
        """Gets a dictionary of the values"""
        base_dict: dict = self._asdict()
        base_dict["mantle_mass"] = self.mantle_mass
        base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        return base_dict


class GasSpecies(ChemicalSpecies):
    """A gas species

    Args:
        formula: Chemical formula (e.g. CO2, C, CH4, etc.)
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        solid_melt_distribution_coefficient: Distribution coefficient between the gas trapped in
            solids and melt. Defaults to 0.
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
        """Distribution coefficient between the gas trapped in solids and melt"""
        return self._solid_melt_distribution_coefficient

    # def __repr__(self) -> str:
    #     base_repr: str = super().__repr__().rstrip(")")
    #     return f"{base_repr}, " f"solubility={self.solubility!r}, " f"eos={self._eos!r})"


class SolidSpecies(CondensedSpecies):
    """A solid species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
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
    """

    @override
    def __init__(self, formula: str, **kwargs):
        super().__init__(formula, "l", **kwargs)


class Species(ImmutableList[ChemicalSpecies]):
    """A list of species

    Args:
        initlist: Initial list of species. Defaults to None.
    """

    def __init__(self, *args, melt_composition: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.conform_solubilities_to_composition(melt_composition)

    @property
    def names(self) -> tuple[str, ...]:
        """Unique names of the species"""
        return tuple([species.name for species in self.data])

    @property
    def number(self) -> int:
        """Number of species"""
        return len(self.data)

    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species"""
        return filter_by_type(self, GasSpecies)

    def condensed_species(self) -> dict[int, CondensedSpecies]:
        """Condensed species"""
        return filter_by_type(self, CondensedSpecies)

    def elements(self) -> tuple[str, ...]:
        """Unique elements in the species

        Returns:
            Unique elements in the species
        """
        elements: list[str] = []
        for species in self.data:
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        return tuple(sorted_elements)

    def species_index(self, find_species: ChemicalSpecies) -> int:
        """Gets the index of a species

        Args:
            find_species: Species to find

        Returns:
            Index of the species

        Raises:
            ValueError: The species is not in the species list
        """
        for index, species in enumerate(self.data):
            if species is find_species:
                return index

        raise ValueError(f"{find_species.name} is not in the species list")

    def get_species(self, find_species: ChemicalSpecies) -> ChemicalSpecies:
        """Gets a species

        Args:
            find_species: Species to find

        Returns:
            The species

        Raises:
            ValueError: The species is not in the species list
        """
        index: int = self.species_index(find_species)

        return self.data[index]

    def get_species_from_name(self, species_name: str) -> ChemicalSpecies:
        """Gets a species from its name

        Args:
            species_name: Unique name of the species

        Returns:
            The species

        Raises:
            ValueError: The species is not in the species list
        """
        for species in self.data:
            if species.name == species_name:
                return species

        raise ValueError(f"{species_name} is not in the species list")

    def conform_solubilities_to_composition(self, melt_composition: str | None) -> None:
        """Conforms the solubilities of the gas species to the planet composition.

        Args:
            melt_composition: Composition of the melt. Defaults to None.

        Raises:
            ValueError if the melt composition does not exist.
        """
        if melt_composition is not None:
            logger.info("Setting solubilities for %s melt composition", melt_composition)
            try:
                solubilities: Mapping[str, SolubilityProtocol] = composition_solubilities[
                    melt_composition.casefold()
                ]
            except KeyError as exc:
                raise ValueError(f"Cannot find solubilities for {melt_composition}") from exc

            for gas_species in self.gas_species().values():
                try:
                    gas_species.solubility = solubilities[gas_species.hill_formula]
                    logger.info(
                        "Found solubility law for %s: %s",
                        gas_species.hill_formula,
                        gas_species.solubility.__class__.__name__,
                    )
                except KeyError:
                    logger.info("No solubility law for %s", gas_species.hill_formula)
                    gas_species.solubility = NoSolubility()
