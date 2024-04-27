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
from functools import wraps
from typing import TYPE_CHECKING, Callable, Mapping, Optional

import numpy as np
from molmass import Composition, Formula

from atmodeller import NOBLE_GASES
from atmodeller.constraints import ActivityConstant, Constraint
from atmodeller.eos.interfaces import IdealGas, RealGasProtocol
from atmodeller.solubility.compositions import composition_solubilities
from atmodeller.solubility.interfaces import NoSolubility, SolubilityProtocol
from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDatasetABC,
)
from atmodeller.utilities import UnitConversion, filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet


class ThermodynamicDataset(ThermodynamicDatasetABC):
    """Combines thermodynamic data from multiple datasets.

    Args:
        datasets: A list of thermodynamic data to use. Defaults to Holland and Powell, and JANAF.
    """

    _DATA_SOURCE: str = "Combined"

    def __init__(
        self,
        datasets: list[ThermodynamicDatasetABC] | None = None,
    ):
        if datasets is None:
            self.datasets: list[ThermodynamicDatasetABC] = []
            self.add_dataset(ThermodynamicDatasetHollandAndPowell())
            self.add_dataset(ThermodynamicDatasetJANAF())
        else:
            self.datasets = datasets

    def add_dataset(self, dataset: ThermodynamicDatasetABC) -> None:
        """Adds a thermodynamic dataset

        Args:
            dataset: A thermodynamic dataset
        """
        if len(self.datasets) >= 1:
            logger.warning("Combining different thermodynamic data may result in inconsistencies")
        logger.info("Adding thermodynamic data: %s", dataset.data_source)
        self.datasets.append(dataset)

    @override
    def get_species_data(
        self, species: ChemicalComponent, **kwargs
    ) -> ThermodynamicDataForSpeciesProtocol | None:
        """See base class."""
        for dataset in self.datasets:
            if dataset is not None:
                return dataset.get_species_data(species, **kwargs)

        raise KeyError(f"Thermodynamic data for {species.formula} is not available in any dataset")


class ChemicalComponent:
    """A chemical component and its properties

    Args:
        formula: Chemical formula (e.g., CO2, C, CH4, etc.)
        phase: cr, g, and l for (crystalline) solid, gas, and liquid, respectively
        thermodynamic_dataset: The thermodynamic dataset. Defaults to JANAF.
        name: Name in the thermodynamic dataset. Defaults to None.
        filename: Filename in the thermodynamic dataset. Defaults to None.

    Attributes:
        formula: Chemical formula
        phase: Phase (cr, g, or l)
        thermodynamic_data: The thermodynamic data for the species
        atoms: Number of atoms
        composition: Composition
        hill_formula: Hill formula
        is_homonuclear_diatomic: True if homonuclear diatomic, otherwise False
        is_noble: True if a noble gas, otherwise False
        molar_mass: Molar mass
    """

    def __init__(
        self,
        formula: str,
        phase: str,
        *,
        thermodynamic_dataset: ThermodynamicDatasetABC | None = None,
        name: str | None = None,
        filename: str | None = None,
    ):
        self._formula: Formula = Formula(formula)
        self.phase: str = phase
        if thermodynamic_dataset is None:
            thermodynamic_dataset_ = ThermodynamicDatasetJANAF()
        else:
            thermodynamic_dataset_: ThermodynamicDatasetABC = thermodynamic_dataset
        self.thermodynamic_data: ThermodynamicDataForSpeciesProtocol | None = (
            thermodynamic_dataset_.get_species_data(self, name=name, filename=filename)
        )
        assert self.thermodynamic_data is not None
        logger.info(
            "Creating %s %s using thermodynamic data in %s",
            self.__class__.__name__,
            self.formula,
            thermodynamic_dataset_.data_source,
        )

    @property
    def atoms(self) -> int:
        """Number of atoms"""
        return self._formula.atoms

    def composition(self) -> Composition:
        """Composition"""
        return self._formula.composition()

    @property
    def elements(self) -> list[str]:
        return list(self.composition().keys())

    @property
    def formula(self) -> str:
        return str(self._formula)

    @property
    def hill_formula(self) -> str:
        """Hill formula"""
        return self._formula.formula

    @property
    def is_homonuclear_diatomic(self) -> bool:
        """True if homonuclear diatomic, otherwise False."""
        composition = self.composition()
        if len(list(composition.keys())) == 1 and list(composition.values())[0].count == 2:
            return True
        else:
            return False

    @property
    def is_noble(self) -> bool:
        """True if a noble gas, otherwise False."""
        if self.formula in NOBLE_GASES:
            return True
        else:
            return False

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol"""
        return UnitConversion.g_to_kg(self._formula.mass)

    @property
    def name(self) -> str:
        """Unique name, combining formula and phase"""
        return f"{self.formula}_{self.phase}"


def _mass_decorator(func) -> Callable:
    """Returns the reservoir masses of either the gas species or one of its elements."""

    @wraps(func)
    def mass_wrapper(
        self: GasSpecies,
        system: InteriorAtmosphereSystem,
        *,
        element: Optional[str] = None,
    ) -> dict[str, float]:
        """Wrapper to return the reservoir masses of either the gas species or one of its elements.

        Args:
            element: Returns the reservoir masses of this element. Defaults to None to return the
                species masses.

        Returns:
            Reservoir masses of either the gas species or one of its elements.
        """
        mass: dict[str, float] = func(self, system)
        if element is not None:
            try:
                mass_scale_factor: float = (
                    UnitConversion.g_to_kg(self.composition()[element].mass) / self.molar_mass
                )
            except KeyError:  # Element not in formula so mass is zero.
                mass_scale_factor = 0
            for key in mass:
                mass[key] *= mass_scale_factor

        return mass

    return mass_wrapper


class GasSpecies(ChemicalComponent):
    """A gas species

    Args:
        formula: Chemical formula (e.g. CO2, C, CH4, etc.)
        phase: Phase. Defaults to g for gas.
        thermodynamic_dataset: The thermodynamic dataset. Defaults to JANAF
        name: Name in the thermodynamic dataset. Defaults to None.
        filename: Filename in the thermodynamic dataset. Defaults to None.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0.
        solubility: Solubility model. Defaults to no solubility
        eos: A gas equation of state. Defaults to an ideal gas.

    Attributes:
        formula: Chemical formula
        phase: g for gas
        thermodynamic_data: The thermodynamic data for the species
        atoms: Number of atoms
        composition: Composition
        hill_formula: Hill formula
        is_homonuclear_diatomic: True if homonuclear diatomic, otherwise False
        is_noble: True if a noble gas, otherwise False
        molar_mass: Molar mass
        solubility: Solubility model
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt
        eos: A gas equation of state
    """

    def __init__(
        self,
        formula: str,
        phase="g",
        *,
        thermodynamic_dataset: ThermodynamicDatasetABC | None = None,
        name: str | None = None,
        filename: str | None = None,
        solid_melt_distribution_coefficient: float = 0,
        solubility: SolubilityProtocol | None = None,
        eos: RealGasProtocol | None = None,
    ):
        super().__init__(
            formula,
            phase,
            thermodynamic_dataset=thermodynamic_dataset,
            name=name,
            filename=filename,
        )
        self.solid_melt_distribution_coefficient: float = solid_melt_distribution_coefficient
        self.solubility: SolubilityProtocol = NoSolubility() if solubility is None else solubility
        self.eos: RealGasProtocol = IdealGas() if eos is None else eos

    @_mass_decorator
    def mass(
        self,
        system: InteriorAtmosphereSystem,
        *,
        element: Optional[str] = None,
    ) -> dict[str, float]:
        """Calculates the total mass of the species or element in each reservoir

        Args:
            system: Interior atmosphere system
            element: Returns the mass for an element. Defaults to None to return the species mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total reservoir masses of the species (element=None) or element (element=element)
        """
        # Only used by the decorator.
        del element

        planet: Planet = system.planet
        pressure: float = system.solution_dict()[self.name]
        fugacity: float = system.fugacities_dict[f"f{self.formula}"]

        # Atmosphere
        mass_in_atmosphere: float = UnitConversion.bar_to_Pa(pressure) / planet.surface_gravity
        mass_in_atmosphere *= (
            planet.surface_area * self.molar_mass / system.atmospheric_mean_molar_mass
        )

        # Melt
        ppmw_in_melt: float = self.solubility.concentration(
            fugacity=fugacity,
            temperature=planet.surface_temperature,
            pressure=system.total_pressure,
            **system.fugacities_dict,
        )
        mass_in_melt: float = (
            system.planet.mantle_melt_mass * ppmw_in_melt * UnitConversion.ppm_to_fraction()
        )

        # Solid
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass_in_solid: float = (
            system.planet.mantle_solid_mass * ppmw_in_solid * UnitConversion.ppm_to_fraction()
        )

        output: dict[str, float] = {
            "atmosphere": mass_in_atmosphere,
            "melt": mass_in_melt,
            "solid": mass_in_solid,
        }

        return output


class CondensedSpecies(ChemicalComponent):
    """A condensed species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        phase: Phase
        thermodynamic_dataset: The thermodynamic dataset. Defaults to JANAF
        name: Name in the thermodynamic dataset. Defaults to None.
        filename: Filename in the thermodynamic dataset. Defaults to None.

    Attributes:
        formula: Chemical formula
        phase: Phase
        thermodynamic_data: The thermodynamic data for the species
        atoms: Number of atoms
        composition: Composition
        hill_formula: Hill formula
        is_homonuclear_diatomic: True if homonuclear diatomic, otherwise False
        is_noble: True if a noble gas, otherwise False
        molar_mass: Molar mass
        activity: Activity, which is always ideal
    """

    def __init__(
        self,
        formula: str,
        phase: str,
        *,
        thermodynamic_dataset: ThermodynamicDatasetABC | None,
        name: str | None,
        filename: str | None,
    ):
        super().__init__(
            formula,
            phase,
            thermodynamic_dataset=thermodynamic_dataset,
            name=name,
            filename=filename,
        )
        self.activity: Constraint = ActivityConstant(species=str(self.formula))


class SolidSpecies(CondensedSpecies):
    """Solid species"""

    @override
    def __init__(
        self,
        formula: str,
        phase: str = "cr",
        *,
        thermodynamic_dataset: ThermodynamicDatasetABC | None = None,
        name: str | None = None,
        filename: str | None = None,
    ):
        super().__init__(
            formula,
            phase,
            thermodynamic_dataset=thermodynamic_dataset,
            name=name,
            filename=filename,
        )


class LiquidSpecies(CondensedSpecies):
    """Liquid species"""

    def __init__(
        self,
        formula: str,
        phase: str = "l",
        *,
        thermodynamic_dataset: ThermodynamicDatasetABC | None = None,
        name: str | None = None,
        filename: str | None = None,
    ):
        super().__init__(
            formula,
            phase,
            thermodynamic_dataset=thermodynamic_dataset,
            name=name,
            filename=filename,
        )


class Species(UserList):
    """A list of species

    Args:
        initlist: Initial list of species. Defaults to None.

    Attributes:
        data: List of species contained in the system
    """

    def __init__(self, initlist: list[ChemicalComponent] | None = None):
        self.data: list[ChemicalComponent]  # For typing
        super().__init__(initlist)

    @property
    def elements(self) -> list[str]:
        elements: list[str] = []
        for species in self.data:
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))

        return unique_elements

    @property
    def number(self) -> int:
        """Number of species"""
        return len(self.data)

    @property
    def number_elements(self) -> int:
        """Number of elements"""
        return len(self.elements)

    @property
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species"""
        return filter_by_type(self, GasSpecies)

    @property
    def gas_species_by_formula(self) -> dict[str, GasSpecies]:
        """Gas species by name"""
        return {value.formula: value for value in self.gas_species.values()}

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return len(self.gas_species)

    @property
    def condensed_species(self) -> dict[int, CondensedSpecies]:
        """Condensed species"""
        return filter_by_type(self, CondensedSpecies)

    @property
    def condensed_elements(self) -> list[str]:
        """Elements in condensed species"""
        elements: list[str] = []
        for species in self.condensed_species.values():
            elements.extend(species.elements)
        unique_elements: list[str] = list(set(elements))

        return unique_elements

    @property
    def number_condensed_species(self) -> int:
        """Number of condensed species"""
        return len(self.condensed_species)

    @property
    def indices(self) -> dict[str, int]:
        """Indices of the species"""
        return {formula: index for index, formula in enumerate(self.formulas)}

    @property
    def formulas(self) -> list[str]:
        """Chemical formulas of the species"""
        return [species.formula for species in self.data]

    @property
    def names(self) -> list[str]:
        """Unique names of the species"""
        return [species.name for species in self.data]

    def conform_solubilities_to_composition(self, melt_composition: str | None = None) -> None:
        """Conforms the solubilities of the species to the planet composition.

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

            for species in self.gas_species.values():
                try:
                    species.solubility = solubilities[species.formula]
                    logger.info(
                        "Found solubility law for %s: %s",
                        species.formula,
                        species.solubility.__class__.__name__,
                    )
                except KeyError:
                    logger.info("No solubility law for %s", species.formula)
                    species.solubility = NoSolubility()

    def composition_matrix(self) -> np.ndarray:
        """Creates a matrix where species (rows) are split into their element counts (columns).

        Returns:
            For example, self.species = ['CO2', 'H2O'] would return:
                [[0, 1, 2],
                 [2, 0, 1]]
            if the columns represent the elements H, C, and O, respectively.
        """
        matrix: np.ndarray = np.zeros((self.number, self.number_elements), dtype=int)
        for species_index, species in enumerate(self.data):
            for element_index, element in enumerate(self.elements):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                matrix[species_index, element_index] = count

        return matrix
