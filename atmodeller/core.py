"""Core

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
from collections import UserList
from dataclasses import KW_ONLY, dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
from molmass import Composition, Formula
from thermochem import janaf

from atmodeller import DATA_ROOT_PATH, GAS_CONSTANT_BAR, NOBLE_GASES
from atmodeller.interfaces import (
    ConstraintABC,
    RealGasABC,
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDatasetABC,
)
from atmodeller.solubilities import NoSolubility, Solubility, composition_solubilities
from atmodeller.utilities import UnitConversion, filter_by_type

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet


@dataclass(kw_only=True)
class IdealGas(RealGasABC):
    """An ideal gas, PV=RT"""

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        return self.ideal_volume(temperature, pressure)

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        volume_integral: float = GAS_CONSTANT_BAR * temperature * np.log(pressure)
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral


@dataclass(kw_only=True, frozen=True)
class ConstantConstraint(ConstraintABC):
    """A constraint of a constant value

    Args:
        name: The name of the constraint, which should be one of: 'activity', 'fugacity',
            'pressure', or 'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.
        value: The constant value, which is usually in kg for masses and bar for pressures or
            fugacities.

    Attributes:
        name: The name of the constraint
        species: The species to constrain
        value: The constant value
    """

    value: float

    def get_value(self, **kwargs) -> float:
        """Returns the constant value. See base class."""
        del kwargs
        return self.value


@dataclass(kw_only=True, frozen=True)
class ActivityConstant(ConstantConstraint):
    """A constant activity

    Args:
        species: The species to constrain
        value: The constant value. Defaults to unity for ideal behaviour.

    Attributes:
        species: The species to constrain
        value: The constant value
    """

    name: str = field(init=False, default="activity")
    value: float = 1.0


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

    def get_data(self, species: ChemicalComponent) -> ThermodynamicDataForSpeciesProtocol | None:
        """See base class"""

        db: janaf.Janafdb = janaf.Janafdb()

        # Regardless of the user input for name_in_dataset, this is defined by JANAF convention.
        species.name_in_dataset = self.get_modified_hill_formula(species)

        def get_phase_data(phases: list[str]) -> janaf.JanafPhase | None:
            """Gets the phase data for a list of phases in order of priority.

            Args:
                phases: Phases to search for in the JANAF database.

            Returns:
                Phase data if it exists in JANAF, otherwise None
            """
            try:
                phase_data: janaf.JanafPhase | None = db.getphasedata(
                    formula=species.name_in_dataset, phase=phases[0]
                )
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
            phase_data = get_phase_data(["l"])

        else:
            msg: str = "Thermodynamic data is unknown for %s" % species.__class__.__name__
            logger.error(msg)
            msg = "%s does not support %s because it has no phase information" % (
                self.__class__.__name__,
                species.__class__.__name__,
            )
            logger.error(msg)
            raise ValueError(msg)

        if phase_data is None:
            msg = "Thermodynamic data for %s is not available in %s (%s name = %s)" % (
                species.formula,
                self.DATA_SOURCE,
                self.DATA_SOURCE,
                species.name_in_dataset,
            )
            logger.warning(msg)

            return None
        else:
            msg = "Thermodynamic data for %s found in %s (%s name = %s)" % (
                species.formula,
                self.DATA_SOURCE,
                self.DATA_SOURCE,
                species.name_in_dataset,
            )
            logger.debug(msg)

            return self.ThermodynamicDataForSpecies(species, self.DATA_SOURCE, phase_data)

    @dataclass(frozen=True)
    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesProtocol):
        """JANAF thermodynamic data for a species

        See base class.
        """

        species: ChemicalComponent
        data_source: str
        data: janaf.JanafPhase

        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            """Gets the standard Gibbs free energy of formation in J/mol.

            See base class.
            """
            del pressure
            gibbs: float = self.data.DeltaG(temperature)

            return gibbs


class ThermodynamicDatasetHollandAndPowell(ThermodynamicDatasetABC):
    """Holland and Powell thermodynamic dataset

    https://ui.adsabs.harvard.edu/abs/1998JMetG..16..309H

    The book 'Equilibrium thermodynamics in petrology: an introduction' by R. Powell also has
    a useful appendix A with equations.
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    def __init__(self):
        data_path: Path = DATA_ROOT_PATH / Path("Mindata161127.csv")  # type: ignore
        self.data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        self.data["name of phase component"] = self.data["name of phase component"].str.strip()
        self.data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        self.data.drop(columns="Abbreviation", inplace=True)
        self.data.set_index("name of phase component", inplace=True)
        self.data = self.data.loc[:, :"Vmax"]
        self.data = self.data.astype(float)

    def get_data(self, species: ChemicalComponent) -> ThermodynamicDataForSpeciesProtocol | None:
        try:
            phase_data: pd.Series | None = self.data.loc[species.name_in_dataset]
            msg = "Thermodynamic data for %s found in %s (%s name = %s)" % (
                species.formula,
                self.DATA_SOURCE,
                self.DATA_SOURCE,
                species.name_in_dataset,
            )
            logger.debug(msg)

            return self.ThermodynamicDataForSpecies(
                species, self.DATA_SOURCE, phase_data, self._ENTHALPY_REFERENCE_TEMPERATURE
            )

        except KeyError:
            phase_data = None
            msg = "Thermodynamic data for %s is not available in %s (%s name = %s)" % (
                species.formula,
                self.DATA_SOURCE,
                self.DATA_SOURCE,
                species.name_in_dataset,
            )
            logger.warning(msg)

            return None

    @dataclass(frozen=True)
    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesProtocol):
        """Holland and Powell thermodynamic data for a species

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
            dKdP: Derivative of bulk modulus (K) with respect to pressure. Set to 4.
            dKdT_factor: Factor for computing the temperature-dependence of K. Set to 1.5e-4.
        """

        species: ChemicalComponent
        data_source: str
        data: pd.Series
        enthalpy_reference_temperature: float
        dKdP: float = field(init=False, default=4.0)
        dKdT_factor: float = field(init=False, default=-1.5e-4)

        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            """Gets the standard Gibbs free energy of formation in units of J/mol.

            Args:
                temperature: Temperature in kelvin
                pressure: Pressure (total) in bar

            Returns:
                The standard Gibbs free energy of formation in J/mol
            """
            gibbs: float = self._get_enthalpy(temperature) - temperature * self._get_entropy(
                temperature
            )

            if isinstance(self.species, CondensedSpecies):
                gibbs += self._get_volume_pressure_integral(temperature, pressure)

            logger.debug(
                "Species = %s, standard Gibbs energy of formation = %f",
                self.species.name_in_dataset,
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
            H = self.data["Hf"]  # J
            # Coefficients for calculating the heat capacity
            a = self.data["a"]  # J/K
            b = self.data["b"]  # J/K^2
            c = self.data["c"]  # J K
            d = self.data["d"]  # J K^(-1/2)

            integral_H: float = (
                H
                + a * (temperature - self.enthalpy_reference_temperature)
                + b / 2 * (temperature**2 - self.enthalpy_reference_temperature**2)
                - c * (1 / temperature - 1 / self.enthalpy_reference_temperature)
                + 2 * d * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return integral_H

        def _get_entropy(self, temperature: float) -> float:
            """Calculates the entropy at temperature.

            Args:
                temperature: Temperature in kelvin

            Returns:
                Entropy in J/K
            """
            S = self.data["S"]  # J/K
            # Coefficients for calculating the heat capacity
            a = self.data["a"]  # J/K
            b = self.data["b"]  # J/K^2
            c = self.data["c"]  # J K
            d = self.data["d"]  # J K^(-1/2)

            integral_S: float = (
                S
                + a * np.log(temperature / self.enthalpy_reference_temperature)
                + b * (temperature - self.enthalpy_reference_temperature)
                - c / 2 * (1 / temperature**2 - 1 / self.enthalpy_reference_temperature**2)
                - 2 * d * (1 / temperature**0.5 - 1 / self.enthalpy_reference_temperature**0.5)
            )
            return integral_S

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
            V = self.data["V"]  # J/bar
            alpha0 = self.data["a0"]  # K^(-1), thermal expansivity

            volume_T: float = V * np.exp(
                alpha0 * (temperature - self.enthalpy_reference_temperature)
                - 2 * 10.0 * alpha0 * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return volume_T

        def _get_bulk_modulus_at_temperature(self, temperature: float) -> float:
            """Calculates the bulk modulus at temperature.

            Holland and Powell (1998), p312 in the text

            Args:
                temperature: Temperature in kelvin

            Returns:
                Bulk modulus in bar
            """
            K = self.data["K"]  # Bulk modulus in bar
            bulk_modulus_T: float = K * (
                1 + self.dKdT_factor * (temperature - self.enthalpy_reference_temperature)
            )
            return bulk_modulus_T

        def _get_volume_pressure_integral(self, temperature: float, pressure: float) -> float:
            """Computes the volume-pressure integral.

            Holland and Powell (1998), p312.

            Args:
                temperature: Temperature in kelvin
                pressure: Pressure in bar

            Returns:
                The volume-pressure integral
            """
            V_T: float = self._get_volume_at_temperature(temperature)
            K_T: float = self._get_bulk_modulus_at_temperature(temperature)
            integral_VP: float = (
                V_T
                * K_T
                / (self.dKdP - 1)
                * ((1 + self.dKdP * (pressure - 1.0) / K_T) ** (1.0 - 1.0 / self.dKdP) - 1)
            )  # J, use P-1.0 instead of P.
            return integral_VP


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
        logger.info("Adding thermodynamic data: %s", dataset.DATA_SOURCE)
        self.datasets.append(dataset)

    def get_data(self, species: ChemicalComponent) -> ThermodynamicDataForSpeciesProtocol | None:
        """See base class."""
        for dataset in self.datasets:
            if dataset is not None:
                return dataset.get_data(species)

        msg: str = "Thermodynamic data for %s is not available in any dataset" % (species.formula)
        logger.error(msg)
        raise KeyError(msg)


@dataclass
class ChemicalComponent:
    """A chemical component and its properties

    Args:
        formula: Chemical formula (e.g., CO2, C, CH4, etc.)
        thermodynamic_dataset: The thermodynamic dataset. Defaults to JANAF
        name_in_dataset: Name for locating Gibbs data in the thermodynamic dataset. Defaults to an
            empty string which means the `formula` should be used.

    Attributes:
        formula: Chemical formula
        thermodynamic_dataset: The thermodynamic dataset
        name_in_dataset: Name for locating Gibbs data in the thermodynamic dataset
        atoms: Number of atoms
        composition: Composition
        hill_formula: Hill formula
        is_homonuclear_diatomic: True if homonuclear diatomic, otherwise False
        is_noble: True if a noble gas, otherwise False
        molar_mass: Molar mass
    """

    formula: str
    _: KW_ONLY
    thermodynamic_dataset: ThermodynamicDatasetABC = field(
        default_factory=ThermodynamicDatasetJANAF
    )
    name_in_dataset: str = ""  # Empty string to maintain type for type checking
    _formula: Formula = field(init=False)
    _thermodynamic_data: ThermodynamicDataForSpeciesProtocol | None = field(init=False)

    def __post_init__(self):
        if not self.name_in_dataset:  # Empty string
            self.name_in_dataset = self.formula
        self._formula = Formula(self.formula)
        self._thermodynamic_data = self.thermodynamic_dataset.get_data(self)
        assert self._thermodynamic_data is not None
        logger.info(
            "Creating %s %s using thermodynamic data in %s (%s name = %s)",
            self.__class__.__name__,
            self.formula,
            self._thermodynamic_data.data_source,
            self._thermodynamic_data.data_source,
            self.name_in_dataset,
        )

    @property
    def atoms(self) -> int:
        """Number of atoms"""
        return self._formula.atoms

    def composition(self) -> Composition:
        """Composition"""
        return self._formula.composition()

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


@dataclass
class GasSpecies(ChemicalComponent):
    """A gas species

    Args:
        formula: Chemical formula (e.g. CO2, C, CH4, etc.)
        thermodynamic_dataset: The thermodynamic dataset. Defaults to JANAF
        name_in_dataset: Name for locating Gibbs data in the thermodynamic dataset. Defaults to an
            empty string which means the `formula` should be used.
        solubility: Solubility model. Defaults to no solubility
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0
        eos: A gas equation of state. Defaults to an ideal gas.

    Attributes:
        formula: Chemical formula
        thermodynamic_dataset: The thermodynamic dataset
        name_in_dataset: Name for locating Gibbs data in the thermodynamic dataset
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

    _: KW_ONLY
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0
    eos: RealGasABC = field(default_factory=IdealGas)

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
        pressure: float = system.solution_dict[self.formula]
        fugacity: float = system.fugacities_dict[self.formula]

        # Atmosphere
        mass_in_atmosphere: float = UnitConversion.bar_to_Pa(pressure) / planet.surface_gravity
        mass_in_atmosphere *= (
            planet.surface_area * self.molar_mass / system.atmospheric_mean_molar_mass
        )

        # Melt
        ppmw_in_melt: float = self.solubility.concentration(
            fugacity=fugacity,
            temperature=planet.surface_temperature,
            log10_fugacities_dict=system.log10_fugacities_dict,
            pressure=system.total_pressure,
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


@dataclass
class CondensedSpecies(ChemicalComponent):
    """A condensed species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        thermodynamic_dataset: The thermodynamic dataset. Defaults to JANAF
        name_in_dataset: Name for locating Gibbs data in the thermodynamic dataset

    Attributes:
        formula: Chemical formula
        thermodynamic_dataset: The thermodynamic dataset
        name_in_dataset: Name for locating Gibbs data in the thermodynamic dataset
        atoms: Number of atoms
        composition: Composition
        hill_formula: Hill formula
        is_homonuclear_diatomic: True if homonuclear diatomic, otherwise False
        is_noble: True if a noble gas, otherwise False
        molar_mass: Molar mass
        activity: Activity, which is always ideal
    """

    _: KW_ONLY
    activity: ConstraintABC = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.activity = ActivityConstant(species=self.formula)


@dataclass(kw_only=True)
class SolidSpecies(CondensedSpecies):
    """Solid species"""


@dataclass(kw_only=True)
class LiquidSpecies(CondensedSpecies):
    """Liquid species"""


class Species(UserList):
    """Collections of species for an interior-atmosphere system

    A collection of species. It provides methods to filter species based on their phases (gas,
    condensed).

    Args:
        initlist: Initial list of species. Defaults to None.

    Attributes:
        data: List of species contained in the system.
    """

    def __init__(self, initlist: list[ChemicalComponent] | None = None):
        self.data: list[ChemicalComponent]  # For typing.
        super().__init__(initlist)

    @property
    def number(self) -> int:
        """Number of species"""
        return len(self.data)

    @property
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species"""
        return filter_by_type(self, GasSpecies)

    @property
    def number_gas_species(self) -> int:
        """Number of gas species"""
        return len(self.gas_species)

    @property
    def condensed_species(self) -> dict[int, CondensedSpecies]:
        """Condensed species."""
        return filter_by_type(self, CondensedSpecies)

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

    def conform_solubilities_to_planet_composition(self, planet: Planet) -> None:
        """Ensure that the solubilities of the species are consistent with the planet composition.

        Args:
            planet: A planet.
        """
        if planet.melt_composition is not None:
            msg: str = (
                # pylint: disable=consider-using-f-string
                "Setting solubilities to be consistent with the melt composition (%s)"
                % planet.melt_composition
            )
            logger.info(msg)
            try:
                solubilities: dict[str, Solubility] = composition_solubilities[
                    planet.melt_composition.casefold()
                ]
            except KeyError:
                logger.error("Cannot find solubilities for %s", planet.melt_composition)
                raise

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

    def _species_sorter(self, species: ChemicalComponent) -> tuple[int, str]:
        """Sorter for the species

        Sorts first by species complexity and second by species name.

        Args:
            species: Species

        Returns:
            A tuple to sort first by number of elements and second by species name.
        """
        return (species.atoms, species.formula)
