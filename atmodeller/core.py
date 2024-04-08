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

import importlib.resources
import logging
import sys
from abc import ABC, abstractmethod
from collections import UserList
from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
import pandas as pd
from molmass import Composition, Formula
from scipy.constants import kilo
from thermochem import janaf

from atmodeller import DATA_DIRECTORY, NOBLE_GASES
from atmodeller.constraints import ActivityConstant, Constraint
from atmodeller.eos.interfaces import IdealGas, RealGas
from atmodeller.solubilities import NoSolubility, Solubility, composition_solubilities
from atmodeller.utilities import UnitConversion, filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet


class ThermodynamicDataForSpeciesABC(ABC):
    """Thermodynamic data for a species to compute the Gibbs energy of formation.

    Args:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation
        *args: Arbitrary positional arguments used by child classes
        **kwargs: Arbitrary keyword arguments used by child classes

    Attributes:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation
    """

    def __init__(self, species: ChemicalComponent, data_source: str, data: Any, *args, **kwargs):
        del args
        del kwargs
        self.species: ChemicalComponent = species
        self.data_source: str = data_source
        self.data: Any = data

    @abstractmethod
    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol.

        Args:
            temperature: Temperature in kelvin
            pressure: Total pressure in bar

        Returns:
            The standard Gibbs free energy of formation in J/mol
        """


class ThermodynamicDatasetABC(ABC):
    """Thermodynamic dataset base class"""

    _DATA_SOURCE: str
    # JANAF standards below. May be overwritten by child classes.
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @abstractmethod
    def get_species_data(
        self, species: ChemicalComponent, **kwargs
    ) -> ThermodynamicDataForSpeciesABC | None:
        """Gets the thermodynamic data for a species or None if not available

        Args:
            species: Species
            **kwargs: Arbitrary keyword arguments

        Returns:
            Thermodynamic data for the species or None if not available
        """

    @property
    def data_source(self) -> str:
        """The source of the data."""
        return self._DATA_SOURCE

    @property
    def enthalpy_reference_temperature(self) -> float:
        """Enthalpy reference temperature in kelvin"""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def standard_state_pressure(self) -> float:
        """Standard state pressure in bar"""
        return self._STANDARD_STATE_PRESSURE


class ThermodynamicDatasetJANAF(ThermodynamicDatasetABC):
    """JANAF thermodynamic dataset"""

    _DATA_SOURCE: str = "JANAF"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @staticmethod
    def get_modified_hill_formula(species: ChemicalComponent) -> str:
        """Gets the modified Hill formula.

        JANAF uses the modified Hill formula to index its data tables. In short, H, if present,
        should appear after C (if C is present), otherwise it must be the first element.

        Args:
            species: Species

        Returns:
            The species represented in the JANAF format
        """
        elements: dict[str, int] = {
            element: properties.count for element, properties in species.composition().items()
        }

        if "C" in elements:
            ordered_elements: list[str] = ["C"]
        else:
            ordered_elements = []

        if "H" in elements:
            ordered_elements.append("H")

        ordered_elements.extend(sorted(elements.keys() - {"C", "H"}))

        formula_string: str = "".join(
            [
                element + (str(elements[element]) if elements[element] > 1 else "")
                for element in ordered_elements
            ]
        )
        logger.debug("Modified Hill formula = %s", formula_string)

        return formula_string

    @override
    def get_species_data(
        self,
        species: ChemicalComponent,
        *,
        name: str | None = None,
        filename: str | None = None,
        **kwargs,
    ) -> ThermodynamicDataForSpeciesABC | None:
        """See base class."""
        del kwargs

        db: janaf.Janafdb = janaf.Janafdb()

        # Defined by JANAF convention
        janaf_formula: str = self.get_modified_hill_formula(species)

        def get_phase_data(phases: list[str]) -> janaf.JanafPhase | None:
            """Gets the phase data for a list of phases in order of priority.

            Args:
                phases: Phases to search for in the JANAF database.

            Returns:
                Phase data if it exists in JANAF, otherwise None
            """
            if filename is not None:
                phase_data: janaf.JanafPhase | None = db.getphasedata(filename=filename)
            else:
                try:
                    phase_data = db.getphasedata(formula=janaf_formula, name=name, phase=phases[0])
                except ValueError:
                    # Cannot find the phase, so keep iterating through the list of options
                    phase_data = get_phase_data(phases[1:])
                except IndexError:
                    # Reached the end of the phases to try meaning no phase data was found
                    phase_data = None

            return phase_data

        if isinstance(species, GasSpecies):
            if species.is_homonuclear_diatomic or species.is_noble:
                phase_data = get_phase_data(["ref", "g"])
            else:
                phase_data = get_phase_data(["g"])

        elif isinstance(species, SolidSpecies):
            phase_data = get_phase_data(["cr", "ref"])  # ref included for C (graphite)

        elif isinstance(species, LiquidSpecies):
            phase_data = get_phase_data(["l", "l,g"])  # l,g included for Water at 1, 10, 100 bar

        else:
            logger.error("Thermodynamic data is unknown for %s", species.__class__.__name__)
            msg: str = f"{self.__class__.__name__} does not support {species.__class__.__name__} "
            msg += " because it has no phase information"
            raise ValueError(msg)

        if phase_data is None:
            logger.warning(
                "Thermodynamic data for %s (%s) not found in %s",
                species.formula,
                janaf_formula,
                self.data_source,
            )

            return None
        else:
            logger.info(
                "Thermodynamic data for %s (%s) found in %s",
                species.formula,
                janaf_formula,
                self.data_source,
            )
            logger.info("Phase data = %s", phase_data)

            return self.ThermodynamicDataForSpecies(species, self.data_source, phase_data)

    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesABC):
        """JANAF thermodynamic data for a species"""

        @override
        def __init__(self, species: ChemicalComponent, data_source: str, data: janaf.JanafPhase):
            """See base class."""
            super().__init__(species, data_source, data)

        @override
        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            """See base class."""
            del pressure
            # thermochem v0.8.2 returns J not kJ. Main branch now returns kJ hence kilo conversion.
            # https://github.com/adelq/thermochem/pull/25
            gibbs: float = self.data.DeltaG(temperature) * kilo

            return gibbs


class ThermodynamicDatasetHollandAndPowell(ThermodynamicDatasetABC):
    """Thermodynamic dataset from :cite:t:`HP91,HP98`.

    The book 'Equilibrium thermodynamics in petrology: an introduction' by R. Powell also has
    a useful appendix A with equations.
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath("Mindata161127.csv")
        )
        with data as data_path:
            self.data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        self.data["name of phase component"] = self.data["name of phase component"].str.strip()
        self.data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        self.data.drop(columns="Abbreviation", inplace=True)
        self.data.set_index("name of phase component", inplace=True)
        self.data = self.data.loc[:, :"Vmax"]
        self.data = self.data.astype(float)

    @override
    def get_species_data(
        self, species: ChemicalComponent, name: str | None = None, **kwargs
    ) -> ThermodynamicDataForSpeciesABC | None:
        """See base class."""
        del kwargs

        try:
            phase_data: pd.Series | None = self.data.loc[name]
            logger.debug(
                "Thermodynamic data for %s (%s) found in %s",
                species.formula,
                name,
                self.data_source,
            )

            return self.ThermodynamicDataForSpecies(
                species, self.data_source, phase_data, self.enthalpy_reference_temperature
            )

        except KeyError:
            phase_data = None
            logger.warning(
                "Thermodynamic data for %s (%s) not found in %s",
                species.formula,
                name,
                self.data_source,
            )

            return None

    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesABC):
        """Thermodynamic data for a species

        Args:
            species: Species
            data_source: Source of the thermodynamic data
            data: Data used to compute the Gibbs energy of formation
            enthalpy_reference_temperature: Enthalpy reference temperature

        Attributes:
            species: Species
            data_source: Source of the thermodynamic data
            data: Data used to compute the Gibbs energy of formation
            enthalpy_reference_temperature: Enthalpy reference temperature
            dkdp: Derivative of bulk modulus (K) with respect to pressure. Set to 4.
            dkdt_factor: Factor for computing the temperature-dependence of K. Set to 1.5e-4.
        """

        @override
        def __init__(
            self,
            species: ChemicalComponent,
            data_source: str,
            data: pd.Series,
            enthalpy_reference_temperature: float,
        ):
            super().__init__(species, data_source, data)
            self.enthalpy_reference_temperature: float = enthalpy_reference_temperature
            self.dkdp: float = 4.0
            self.dkdt_factor: float = -1.5e-4

        @override
        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            """See base class"""
            gibbs: float = self._get_enthalpy(temperature) - temperature * self._get_entropy(
                temperature
            )

            if isinstance(self.species, CondensedSpecies):
                gibbs += self._get_volume_pressure_integral(temperature, pressure)

            logger.debug(
                "Species = %s, standard Gibbs energy of formation = %f",
                self.species.formula,
                gibbs,
            )

            return gibbs

        def _get_enthalpy(self, temperature: float) -> float:
            """Calculates the enthalpy at temperature.

            Args:
                temperature: Temperature in kelvin

            Returns:
                Enthalpy in J
            """
            enthalpy0 = self.data["Hf"]  # J
            # Coefficients for calculating the heat capacity
            a = self.data["a"]  # J/K
            b = self.data["b"]  # J/K^2
            c = self.data["c"]  # J K
            d = self.data["d"]  # J K^(-1/2)

            enthalpy_integral: float = (
                enthalpy0
                + a * (temperature - self.enthalpy_reference_temperature)
                + b / 2 * (temperature**2 - self.enthalpy_reference_temperature**2)
                - c * (1 / temperature - 1 / self.enthalpy_reference_temperature)
                + 2 * d * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return enthalpy_integral

        def _get_entropy(self, temperature: float) -> float:
            """Calculates the entropy at temperature.

            Args:
                temperature: Temperature in kelvin

            Returns:
                Entropy in J/K
            """
            entropy0 = self.data["S"]  # J/K
            # Coefficients for calculating the heat capacity
            a = self.data["a"]  # J/K
            b = self.data["b"]  # J/K^2
            c = self.data["c"]  # J K
            d = self.data["d"]  # J K^(-1/2)

            entropy_integral: float = (
                entropy0
                + a * np.log(temperature / self.enthalpy_reference_temperature)
                + b * (temperature - self.enthalpy_reference_temperature)
                - c / 2 * (1 / temperature**2 - 1 / self.enthalpy_reference_temperature**2)
                - 2 * d * (1 / temperature**0.5 - 1 / self.enthalpy_reference_temperature**0.5)
            )
            return entropy_integral

        def _get_volume_at_temperature(self, temperature: float) -> float:
            """Calculates the volume at temperature.

            The exponential arises from the strict derivation, but often an expansion is performed
            where exp(x) = 1+x as in Holland and Powell (1998). Below the exp term is retained, but
            the equation in Holland and Powell (1998) p311 is expanded.

            Args:
                temperature: Temperature in kelvin

            Returns:
                Volume in J/bar
            """
            volume0 = self.data["V"]  # J/bar
            alpha0 = self.data["a0"]  # K^(-1), thermal expansivity

            volume: float = volume0 * np.exp(
                alpha0 * (temperature - self.enthalpy_reference_temperature)
                - 2 * 10.0 * alpha0 * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return volume

        def _get_bulk_modulus_at_temperature(self, temperature: float) -> float:
            """Calculates the bulk modulus at temperature.

            Holland and Powell (1998), p312 in the text

            Args:
                temperature: Temperature in kelvin

            Returns:
                Bulk modulus in bar
            """
            bulk_modulus0 = self.data["K"]  # Bulk modulus in bar
            bulk_modulus: float = bulk_modulus0 * (
                1 + self.dkdt_factor * (temperature - self.enthalpy_reference_temperature)
            )
            return bulk_modulus

        def _get_volume_pressure_integral(self, temperature: float, pressure: float) -> float:
            """Computes the volume-pressure integral.

            Holland and Powell (1998), p312.

            Args:
                temperature: Temperature in kelvin
                pressure: Pressure in bar

            Returns:
                The volume-pressure integral
            """
            volume: float = self._get_volume_at_temperature(temperature)
            bulk_modulus: float = self._get_bulk_modulus_at_temperature(temperature)
            integral_vp: float = (
                volume
                * bulk_modulus
                / (self.dkdp - 1)
                * (
                    (1 + self.dkdp * (pressure - 1.0) / bulk_modulus) ** (1.0 - 1.0 / self.dkdp)
                    - 1
                )
            )  # J, use P-1.0 instead of P.
            return integral_vp


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
    ) -> ThermodynamicDataForSpeciesABC | None:
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
        self.thermodynamic_data: ThermodynamicDataForSpeciesABC | None = (
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
        solubility: Solubility | None = None,
        eos: RealGas | None = None,
    ):
        super().__init__(
            formula,
            phase,
            thermodynamic_dataset=thermodynamic_dataset,
            name=name,
            filename=filename,
        )
        self.solid_melt_distribution_coefficient: float = solid_melt_distribution_coefficient
        self.solubility: Solubility = NoSolubility() if solubility is None else solubility
        self.eos: RealGas = IdealGas() if eos is None else eos

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
                solubilities: dict[str, Solubility] = composition_solubilities[
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
