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

if TYPE_CHECKING:
    from atmodeller.constraints import SystemConstraints


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
        logger.info("Creating a new planet")
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

    @staticmethod
    def formula_matrix(
        elements: list[str], species: list[ChemicalSpecies]
    ) -> npt.NDArray[np.int_]:
        """Creates a formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Args:
            elements: A list of elements
            species: A list of species

        Returns:
            The formula matrix
        """
        matrix: npt.NDArray[np.int_] = np.zeros((len(elements), len(species)), dtype=np.int_)
        for element_index, element in enumerate(elements):
            for species_index, single_species in enumerate(species):
                try:
                    count: int = single_species.composition()[element].count
                except KeyError:
                    count = 0
                matrix[element_index, species_index] = count

        return matrix


@dataclass
class Solution:
    """The solution

    Stores and updates the solution and assembles the appropriate vectors to solve the coupled
    reaction network and mass balance system. Importantly, the solution quantities depend on the
    applied constraints. All quantities must be positive so log10 is used. The ordering of the
    solution vector must be maintained for consistency and is organised as follows:

    1. Species fugacities and activities, ordered according to the `species` argument.

    2. Mass of the condensed species

    3. Stability factors for condensed species

    Args:
        _species: Species
    """

    _species: Species
    # These are all log10
    _species_solution: dict[ChemicalSpecies, float] = field(init=False, default_factory=dict)
    _mass_solution: dict[CondensedSpecies, float] = field(init=False, default_factory=dict)
    _stability_solution: dict[CondensedSpecies, float] = field(init=False, default_factory=dict)

    @property
    def species_solution(self) -> dict[ChemicalSpecies, float]:
        return self._species_solution

    @property
    def mass_solution(self) -> dict[CondensedSpecies, float]:
        return self._mass_solution

    @property
    def stability_solution(self) -> dict[CondensedSpecies, float]:
        return self._stability_solution

    @property
    def number(self) -> int:
        """Number of solution quantities

        The factor of two is because each (possibly stable) condensate has a stability factor.
        """
        return self._species.number_species() + 2 * self._species.number_condensed_species

    @property
    def data(self) -> npt.NDArray[np.float_]:
        data: npt.NDArray[np.float_] = np.zeros(self.number, dtype=np.float_)
        index = 0
        # Fill data with species solutions
        for species in self._species:
            data[index] = self._species_solution[species]
            index += 1
        # Fill data with mass solutions
        for species in self._species.condensed_species:
            data[index] = self._mass_solution[species]
            index += 1
        # Fill data with stability solutions
        for species in self._species.condensed_species:
            data[index] = self._stability_solution[species]
            index += 1

        return data

    @data.setter
    def data(self, value: npt.NDArray) -> None:
        """Sets the solution dictionaries

        Args:
            value: A vector, which is usually passed by the solver.
        """
        index = 0

        for species in self._species:
            self._species_solution[species] = value[index]
            index += 1
        for species in self._species.condensed_species:
            self._mass_solution[species] = value[index]
            index += 1
        for species in self._species.condensed_species:
            self._stability_solution[species] = value[index]
            index += 1

    @property
    def species_values(self) -> npt.NDArray:
        return np.array(list(self._species_solution.values()))

    @property
    def stability_array(self) -> npt.NDArray:
        stability_array: npt.NDArray = np.zeros(self._species.number_species(), dtype=float)
        for species in self._species.condensed_species:
            index: int = self._species.species_index(species)
            stability_array[index] = self._stability_solution[species]

        return stability_array

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
    def gas_mean_molar_mass(self) -> float:
        """Mean molar mass of the gas"""
        mass: float = 0
        for species in self._species.gas_species:
            mass += species.molar_mass * self.gas_pressures[species]
        mass /= self.total_pressure

        return mass

    @property
    def volume_mixing_ratios(self) -> dict[GasSpecies, float]:
        """Volume mixing ratios"""
        vmr: dict[GasSpecies, float] = {}
        for species in self._species.gas_species:
            vmr[species] = self.gas_pressures[species] / self.total_pressure

        return vmr

    @property
    def log10_activities(self) -> dict[CondensedSpecies, float]:
        """Log10 activities"""
        activities: dict[CondensedSpecies, float] = {}
        for species in self._species.condensed_species:
            activities[species] = self._species_solution[species]

        return activities

    @property
    def activities(self) -> dict[CondensedSpecies, float]:
        """Activities"""
        return {species: 10**value for species, value in self.log10_activities.items()}

    @property
    def condensed_masses(self) -> dict[CondensedSpecies, float]:
        """Masses of condensed species"""
        return {
            condensed_species: 10 ** self._mass_solution[condensed_species]
            for condensed_species in self._species.condensed_species
        }

    def log10_fugacity_coefficients(self, temperature: float) -> dict[GasSpecies, float]:
        """Log10 fugacity coefficients

        Args:
            temperature: Temperature in K

        Returns:
            Log10 fugacity coefficients
        """
        log10_coefficients: dict[GasSpecies, float] = {}
        for species in self._species.gas_species:
            log10_coefficients[species] = np.log10(
                species.eos.fugacity_coefficient(temperature, self.total_pressure)
            )

        return log10_coefficients

    def fugacity_coefficients(self, temperature: float) -> dict[GasSpecies, float]:
        """Fugacity coefficients

        Args:
            temperature: Temperature in K

        Returns:
            Fugacity coefficients
        """
        return {
            key: 10**value for key, value in self.log10_fugacity_coefficients(temperature).items()
        }

    def log10_gas_fugacities(self, temperature: float) -> dict[GasSpecies, float]:
        """Log10 gas fugacities

        Args:
            temperature: Temperature in K

        Returns:
            Log10 gas fugacities
        """
        log10_fugacities: dict[GasSpecies, float] = {}
        for species in self._species.gas_species:
            log10_fugacities[species] = (
                self.log10_gas_pressures[species]
                + self.log10_fugacity_coefficients(temperature)[species]
            )

        return log10_fugacities

    def gas_fugacities(self, temperature: float) -> dict[GasSpecies, float]:
        """Gas fugacities

        Args:
            temperature: Temperature in K

        Returns:
            Gas fugacities
        """
        return {key: 10**value for key, value in self.log10_gas_fugacities(temperature).items()}

    def gas_fugacities_by_hill_formula(self, temperature: float) -> dict[str, float]:
        """Gas fugacities by hill formula

        Args:
            temperature: Temperature in K

        Returns:
            Gas fugacities by hill formula
        """
        return {key.hill_formula: value for key, value in self.gas_fugacities(temperature).items()}

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        output: dict[str, float] = {}
        for species, value in zip(self._species.data, self.species_solution.values()):
            output[species.name] = value
        for species, value in zip(self._species.condensed_species, self.mass_solution.values()):
            output[f"mass_{species.name}"] = value
        for species, value in zip(
            self._species.condensed_species, self.stability_solution.values()
        ):
            output[f"stability_{species.name}"] = value

        return output

    def raw_solution_dict(self) -> dict[str, float]:
        """Raw solution in a dictionary"""
        output: dict[str, float] = {}
        for species, value in zip(self._species.data, self.species_solution.values()):
            output[species.name] = value
        for species, value in zip(self._species.condensed_species, self.mass_solution.values()):
            output[f"mass_{species.name}"] = value
        for species, value in zip(
            self._species.condensed_species, self.stability_solution.values()
        ):
            output[f"stability_{species.name}"] = value

        return output

    def solution_dict(self) -> dict[str, float]:
        """Solution in a dictionary"""
        output: dict[str, float] = {}
        for species, pressure in self.gas_pressures.items():
            output[species.name] = pressure
        for species, activity in self.activities.items():
            output[species.name] = activity
        for species, condensed_mass in self.condensed_masses.items():
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
