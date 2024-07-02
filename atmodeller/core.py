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
from typing import Generic, Mapping, TypeVar

import numpy as np
import numpy.typing as npt

from atmodeller import GRAVITATIONAL_CONSTANT
from atmodeller.eos.interfaces import IdealGas, RealGasProtocol
from atmodeller.interfaces import ChemicalSpecies, CondensedSpecies
from atmodeller.solubility.compositions import composition_solubilities
from atmodeller.solubility.interfaces import NoSolubility, SolubilityProtocol
from atmodeller.utilities import dataclass_to_logger, filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

T = TypeVar("T")

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet

    Defines the properties of a planet that are relevant for interior modeling. It provides default
    values suitable for modelling a fully molten Earth-like planet.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to Earth.
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        surface_radius: Radius of the planetary surface in m. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
        melt_composition: Melt composition of the planet. Default is None.
    """

    planet_mass: float = 5.972e24
    """Mass of the planet in kg"""
    core_mass_fraction: float = 0.295334691460966
    """Mass fraction of the core relative to the planetary mass (kg/kg)"""
    mantle_melt_fraction: float = 1.0
    """Mass fraction of the molten mantle"""
    surface_radius: float = 6371000.0
    """Radius of the surface in m"""
    surface_temperature: float = 2000.0
    """Temperature of the surface in K"""
    melt_composition: str | None = None
    """Melt composition"""
    mantle_mass: float = field(init=False)
    """Mass of the mantle"""
    mantle_melt_mass: float = field(init=False)
    """Mass of the molten mantle"""
    mantle_solid_mass: float = field(init=False)
    """Mass of the solid mantle"""
    surface_area: float = field(init=False)
    """Surface area"""
    surface_gravity: float = field(init=False)
    """Surface gravity"""

    def __post_init__(self):
        self.mantle_mass = self.planet_mass * (1 - self.core_mass_fraction)
        self.mantle_melt_mass = self.mantle_mass * self.mantle_melt_fraction
        self.mantle_solid_mass = self.mantle_mass * (1 - self.mantle_melt_fraction)
        self.surface_area = 4.0 * np.pi * self.surface_radius**2
        self.surface_gravity = GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2
        logger.info("Creating a planet")
        dataclass_to_logger(self, logger)


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


class Species(UserList[ChemicalSpecies]):
    """A list of species

    Args:
        initlist: Initial list of species. Defaults to None.
    """

    @property
    def names(self) -> list[str]:
        """Unique names of the species"""
        return [species.name for species in self.data]

    @property
    def number(self) -> int:
        """Number of species"""
        return self.number_gas_species + self.number_condensed_species

    @property
    def gas_species(self) -> list[GasSpecies]:
        """Gas species"""
        return list(filter_by_type(self, GasSpecies).values())

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return len(self.gas_species)

    @property
    def condensed_species(self) -> list[CondensedSpecies]:
        """Condensed species"""
        return list(filter_by_type(self, CondensedSpecies).values())

    @property
    def number_condensed_species(self) -> int:
        """Number of condensed species"""
        return len(self.condensed_species)

    def elements(self) -> list[str]:
        """Unique elements in the species

        Returns:
            A list of unique elements
        """
        elements: list[str] = []
        for species in self.data:
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))

        return unique_elements

    def species_index(self, find_species: ChemicalSpecies) -> int:
        """Gets the index of a species

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

    def conform_solubilities_to_composition(self, melt_composition: str | None = None) -> None:
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


class SolutionComponent(Generic[T]):
    """A component of the solution

    Args:
        species: Species relevant to this solution component, which is usually a sublist derived
            from :class:`Species`.

    Attributes:
        data: Dictionary of the solution values, which are usually log10 of the physical values.
    """

    def __init__(self, species: list[T]):
        self._species: list[T] = species
        self.data: dict[T, float] = {}

    @property
    def number(self) -> int:
        """Number of solution quantities"""
        return len(self._species)

    @property
    def log10(self) -> dict[T, float]:
        """Log10 values"""
        return self.data

    @property
    def physical(self) -> dict[T, float]:
        """Physical values"""
        return {key: 10**value for key, value in self.data.items()}

    def clip_values(
        self, minimum_value: float | None = None, maximum_value: float | None = None
    ) -> None:
        """clips the values.

        Args:
            minimum_value: Minimum value. Defaults to None, meaning do not clip.
            maximum_value: Maximum value. Defaults to None, meaning do not clip.
        """
        for key, value in self.data.items():
            self.data[key] = np.clip(value, minimum_value, maximum_value)

    def perturb_values(self, perturb: float = 0) -> None:
        """Perturbs the values.

        Args:
            perturb: Maximum log10 value to perturb the values. Defaults to 0.
        """
        for key in self.data:
            self.data[key] += perturb * (2 * np.random.rand() - 1)


class GasSolution(SolutionComponent[GasSpecies]):
    """The gas solution"""

    @property
    def total_pressure(self) -> float:
        """Total pressure"""
        return sum(self.physical.values())

    @property
    def mean_molar_mass(self) -> float:
        """Mean molar mass"""
        mass: float = 0
        for species, pressure in self.physical.items():
            mass += species.molar_mass * pressure
        mass /= self.total_pressure

        return mass

    @property
    def volume_mixing_ratios(self) -> dict[GasSpecies, float]:
        """Volume mixing ratios"""
        vmr: dict[GasSpecies, float] = {}
        for species, pressure in self.physical.items():
            vmr[species] = pressure / self.total_pressure

        return vmr

    def log10_fugacity_coefficients(self, temperature: float) -> dict[GasSpecies, float]:
        """Log10 fugacity coefficients

        Args:
            temperature: Temperature in K

        Returns:
            Log10 fugacity coefficients
        """
        return {
            key: np.log10(value) for key, value in self.fugacity_coefficients(temperature).items()
        }

    def fugacity_coefficients(self, temperature: float) -> dict[GasSpecies, float]:
        """Fugacity coefficients

        Args:
            temperature: Temperature in K

        Returns:
            Fugacity coefficients
        """
        fugacity_coefficients: dict[GasSpecies, float] = {}
        for species in self.data:
            fugacity_coefficients[species] = species.eos.fugacity_coefficient(
                temperature, self.total_pressure
            )

        return fugacity_coefficients

    def log10_fugacities(self, temperature: float) -> dict[GasSpecies, float]:
        """Log10 fugacities

        Args:
            temperature: Temperature in K

        Returns:
            Log10 gas fugacities
        """
        log10_fugacities: dict[GasSpecies, float] = {}
        for species, log10_pressure in self.data.items():
            log10_fugacities[species] = (
                log10_pressure + self.log10_fugacity_coefficients(temperature)[species]
            )

        return log10_fugacities

    def fugacities(self, temperature: float) -> dict[GasSpecies, float]:
        """Fugacities

        Args:
            temperature: Temperature in K

        Returns:
            Gas fugacities
        """
        return {key: 10**value for key, value in self.log10_fugacities(temperature).items()}

    def fugacities_by_hill_formula(self, temperature: float) -> dict[str, float]:
        """Gas fugacities by hill formula

        Args:
            temperature: Temperature in K

        Returns:
            Gas fugacities by hill formula
        """
        return {key.hill_formula: value for key, value in self.fugacities(temperature).items()}


class Solution:
    """The solution

    Stores and updates the solution and assembles the appropriate vectors to solve the coupled
    reaction network and mass balance system. The solution is separated into four components: the
    gas pressures, the activities of condensed species, the mass of condensed species, and the
    stability criteria. All solution quantities are log10.

    Args:
        species: Species
    """

    def __init__(self, species: Species):
        self._species: Species = species
        self._gas: GasSolution = GasSolution(species.gas_species)
        self._activity: SolutionComponent[CondensedSpecies] = SolutionComponent[CondensedSpecies](
            species.condensed_species
        )
        self._mass: SolutionComponent[CondensedSpecies] = SolutionComponent[CondensedSpecies](
            species.condensed_species
        )
        self._stability: SolutionComponent[CondensedSpecies] = SolutionComponent[CondensedSpecies](
            species.condensed_species
        )

    @property
    def number(self) -> int:
        """Total number of solution quantities"""
        return self.gas.number + self.activity.number + self.mass.number + self.stability.number

    @property
    def gas(self) -> GasSolution:
        """Gas solution"""
        return self._gas

    @property
    def activity(self) -> SolutionComponent[CondensedSpecies]:
        """Activity solution"""
        return self._activity

    @property
    def mass(self) -> SolutionComponent[CondensedSpecies]:
        """Mass solution"""
        return self._mass

    @property
    def stability(self) -> SolutionComponent[CondensedSpecies]:
        """Stability solution"""
        return self._stability

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """The solution as an array for the solver"""
        data: npt.NDArray[np.float_] = np.zeros(self.number, dtype=np.float_)
        index: int = 0
        for species in self._species.gas_species:
            index = self._species.species_index(species)
            data[index] = self.gas.data[species]
        for counter, species in enumerate(self._species.condensed_species):
            index = self._species.species_index(species)
            data[index] = self.activity.data[species]
            index = self._species.number + counter
            data[index] = self.mass.data[species]
            index += self._species.number_condensed_species
            data[index] = self.stability.data[species]

        return data

    @data.setter
    def data(self, value: npt.NDArray[np.float_]) -> None:
        """Sets the solution from an array.

        Args:
            value: An array, which is usually passed by the solver.
        """
        index: int = 0
        for species in self._species.gas_species:
            index = self._species.species_index(species)
            self.gas.data[species] = value[index]
        for counter, species in enumerate(self._species.condensed_species):
            index = self._species.species_index(species)
            self.activity.data[species] = value[index]
            index = self._species.number + counter
            self.mass.data[species] = value[index]
            index += self._species.number_condensed_species
            self.stability.data[species] = value[index]

    def stability_array(self) -> npt.NDArray:
        """The condensate stability array"""
        stability_array: npt.NDArray = np.zeros(self._species.number, dtype=np.float_)
        for species, value in self.stability.data.items():
            index: int = self._species.species_index(species)
            stability_array[index] = value

        return stability_array

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        output: dict[str, float] = {}
        for species, value in zip(self._species.data, self.data[: self._species.number]):
            output[species.name] = value
        for species, value in zip(self._species.condensed_species, self.mass.data.values()):
            output[f"mass_{species.name}"] = value
        for species, value in zip(self._species.condensed_species, self.stability.data.values()):
            output[f"stability_{species.name}"] = value

        return output

    def solution_dict(self) -> dict[str, float]:
        """Solution in a dictionary"""
        output: dict[str, float] = {}
        for species, pressure in self.gas.physical.items():
            output[species.name] = pressure
        for species, activity in self.activity.physical.items():
            output[species.name] = activity
        for species, condensed_mass in self.mass.physical.items():
            output[f"mass_{species.name}"] = condensed_mass

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
