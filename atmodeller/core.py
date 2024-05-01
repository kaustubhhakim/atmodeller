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
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np
from molmass import Formula

from atmodeller.activity.interfaces import ActivityProtocol, ConstantActivity
from atmodeller.eos.interfaces import IdealGas, RealGasProtocol
from atmodeller.solubility.compositions import composition_solubilities
from atmodeller.solubility.interfaces import NoSolubility, SolubilityProtocol
from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDataset,
)
from atmodeller.thermodata.janaf import ThermodynamicDatasetJANAF
from atmodeller.utilities import UnitConversion, filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet


class _ChemicalSpecies:
    """A chemical species and its properties

    Args:
        formula: Chemical formula (e.g., CO2, C, CH4, etc.)
        phase: cr, g, and l for (crystalline) solid, gas, and liquid, respectively
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF.
        thermodata_name: Name of the component in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
    """

    def __init__(
        self,
        formula: str,
        phase: str,
        *,
        thermodata_dataset: ThermodynamicDataset = ThermodynamicDatasetJANAF(),
        thermodata_name: str | None = None,
        thermodata_filename: str | None = None,
    ):
        self._formula: Formula = Formula(formula)
        self._phase: str = phase
        thermodata: ThermodynamicDataForSpeciesProtocol | None = (
            thermodata_dataset.get_species_data(
                self, name=thermodata_name, filename=thermodata_filename
            )
        )
        assert thermodata is not None
        self._thermodata: ThermodynamicDataForSpeciesProtocol = thermodata
        logger.info(
            "Creating %s %s (hill formula=%s) using thermodynamic data in %s",
            self.__class__.__name__,
            self.formula,
            self.hill_formula,
            self.thermodata.data_source,
        )

    @property
    def elements(self) -> list[str]:
        """Elements in species"""
        return list(self.formula.composition().keys())

    @property
    def formula(self) -> Formula:
        """Formula object"""
        return self._formula

    @property
    def hill_formula(self) -> str:
        """Hill formula"""
        return self.formula.formula

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol"""
        return UnitConversion.g_to_kg(self.formula.mass)

    @property
    def name(self) -> str:
        """Unique name by combining formula and phase"""
        return f"{self.formula}_{self.phase}"

    @property
    def phase(self) -> str:
        """Phase"""
        return self._phase

    @property
    def thermodata(self) -> ThermodynamicDataForSpeciesProtocol:
        """Thermodynamic data for the species"""
        return self._thermodata


class GasSpecies(_ChemicalSpecies):
    """A gas species

    Args:
        formula: Chemical formula (e.g. CO2, C, CH4, etc.)
        phase: Phase. Defaults to g for gas.
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0.
        solubility: Solubility model. Defaults to no solubility
        eos: A gas equation of state. Defaults to an ideal gas.

    Attributes:
        solubility: Solubility model
    """

    @override
    def __init__(
        self,
        formula: str,
        phase="g",
        *,
        solid_melt_distribution_coefficient: float = 0,
        solubility: SolubilityProtocol = NoSolubility(),
        eos: RealGasProtocol = IdealGas(),
        **kwargs,
    ):
        super().__init__(formula, phase, **kwargs)
        self._solid_melt_distribution_coefficient: float = solid_melt_distribution_coefficient
        self.solubility: SolubilityProtocol = solubility
        self._eos: RealGasProtocol = eos

    @property
    def eos(self) -> RealGasProtocol:
        """A gas equation of state"""
        return self._eos

    @property
    def solid_melt_distribution_coefficient(self) -> float:
        """Distribution coefficient between solid and melt"""
        return self._solid_melt_distribution_coefficient

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

        Returns:
            Total reservoir masses of the species or element
        """
        planet: Planet = system.planet
        pressure: float = system.solution_dict()[self.name]
        # TODO: Use Hill for convention
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

        if element is not None:
            try:
                mass_scale_factor: float = (
                    UnitConversion.g_to_kg(self.formula.composition()[element].mass)
                    / self.molar_mass
                )
            except KeyError:  # Element not in formula so mass is zero.
                mass_scale_factor = 0
            for key in output:
                output[key] *= mass_scale_factor

        return output


class _CondensedSpecies(_ChemicalSpecies):
    """A condensed species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        phase: Phase
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        activity: Activity model. Defaults to unity for a pure component.

    Attributes:
        activity: Activity model
    """

    @override
    def __init__(
        self,
        formula: str,
        phase: str,
        *,
        activity: ActivityProtocol = ConstantActivity(),
        **kwargs,
    ):
        super().__init__(formula, phase, **kwargs)
        self._activity: ActivityProtocol = activity

    @property
    def activity(self) -> ActivityProtocol:
        """An activity model"""
        return self._activity


class SolidSpecies(_CondensedSpecies):
    """A solid species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        phase: Phase. Defaults to cr for solid.
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        activity: Activity model. Defaults to unity for a pure component.

    Attributes:
        activity: Activity model
    """

    @override
    def __init__(self, formula: str, phase: str = "cr", **kwargs):
        super().__init__(formula, phase, **kwargs)


class LiquidSpecies(_CondensedSpecies):
    """A liquid species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        phase: Phase. Defaults to l for liquid.
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
        activity: Activity model. Defaults to unity for a pure component.

    Attributes:
        activity: Activity model
    """

    @override
    def __init__(self, formula: str, phase: str = "l", **kwargs):
        super().__init__(formula, phase, **kwargs)


class Species(UserList):
    """A list of species

    Args:
        initlist: Initial list of species. Defaults to None.

    Attributes:
        data: List of species contained in the system
    """

    def __init__(self, initlist: list[_ChemicalSpecies] | None = None):
        self.data: list[_ChemicalSpecies]  # For typing
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
        return {str(value.formula): value for value in self.gas_species.values()}

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return len(self.gas_species)

    @property
    def condensed_species(self) -> dict[int, _CondensedSpecies]:
        """Condensed species"""
        return filter_by_type(self, _CondensedSpecies)

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
        return [str(species.formula) for species in self.data]

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
                    species.solubility = solubilities[str(species.formula)]
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
            A matrix of element counts
        """
        matrix: np.ndarray = np.zeros((self.number, self.number_elements), dtype=int)
        for species_index, species in enumerate(self.data):
            for element_index, element in enumerate(self.elements):
                try:
                    count: int = species.formula.composition()[element].count
                except KeyError:
                    count = 0
                matrix[species_index, element_index] = count

        return matrix
