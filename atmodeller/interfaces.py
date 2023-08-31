"""Interfaces.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Type, Union

import numpy as np
import pandas as pd
from molmass import Formula
from thermochem import janaf

from atmodeller import DATA_ROOT_PATH
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.core import InteriorAtmosphereSystem, Planet


class SystemConstraint(Protocol):
    """A constraint to apply to an interior-atmosphere system."""

    name: str
    species: str

    def get_value(self, *args, **kwargs) -> float:
        """Computes the value of the constraint for given input arguments.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The evaluation of the constraint according to *args and **kwargs.
        """
        ...


@dataclass(kw_only=True)
class ConstantSystemConstraint(ABC):
    """A constant value constraint.

    Args:
        name: Constraint name, which must be one of: fugacity, pressure, or mass.
        species: The species to constrain. Usually a species for a pressure or fugacity constraint
            or an element for a mass constraint.
        value: The constant value. Imposed value in kg for masses and bar for pressures or
            fugacities.

    Attributes:
        name: Constraint name.
        species: The species to constrain.
        value: The constant value.
    """

    name: str
    species: str
    value: float

    def get_value(self, *args, **kwargs) -> float:
        """Returns the constant value. See base class."""
        del args
        del kwargs
        return self.value


@dataclass(kw_only=True)
class IdealityConstant(ConstantSystemConstraint):
    """A constant activity or fugacity coefficient.

    Note:
        name and species are usually updated after instantiation to ensure they are consistent with
        the species that instantiated this class.

    Args:
        value: The constant value. Defaults to 1 (i.e. ideal behaviour).

    Attributes:
        name: Constraint name.
        species: The species to constrain.
        value: The constant value.
    """

    name: str = field(init=False, default="")
    species: str = field(init=False, default="")
    value: float = 1.0


class Solubility(ABC):
    """Solubility base class."""

    def power_law(self, fugacity: float, constant: float, exponent: float) -> float:
        """Power law. Fugacity in bar and returns ppmw."""
        return constant * fugacity**exponent

    @abstractmethod
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Dissolved volatile concentration in ppmw in the melt.

        Args:
            fugacity: Fugacity of the species.
            temperature: Temperature.
            fugacities_dict: Fugacities of all species in the system.

        Returns:
            ppmw of the species in the melt.
        """
        raise NotImplementedError

    def __call__(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Dissolved volatile concentration in ppmw in the melt.

        See self._solubility.
        """
        solubility: float = self._solubility(fugacity, temperature, fugacities_dict)
        logger.debug(
            "%s, f = %f, T = %f, ppmw = %f",
            self.__class__.__name__,
            fugacity,
            temperature,
            solubility,
        )
        return solubility


class NoSolubility(Solubility):
    """No solubility."""

    def _solubility(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return 0.0


class ThermodynamicDataBase(ABC):
    """Thermodynamic data base class."""

    _DATA_SOURCE: str
    _ENTHALPY_REFERENCE_TEMPERATURE: float  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @abstractmethod
    def __init__(self, species: ChemicalComponent):
        self.species: ChemicalComponent = species

    @property
    def DATA_SOURCE(self) -> str:
        """Identifies the source of the data."""
        return self._DATA_SOURCE

    @property
    def ENTHALPY_REFERENCE_TEMPERATURE(self) -> float:
        """Enthalpy reference temperature."""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def STANDARD_STATE_PRESSURE(self) -> float:
        """Standard state pressure."""
        return self._STANDARD_STATE_PRESSURE

    @abstractmethod
    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            temperature: Temperature.
            pressure: Pressure (total).

        Returns:
            The standard Gibbs free energy of formation (J/mol).
        """
        ...


@dataclass(kw_only=True)
class GasSpeciesOutput:
    """Output for a gas species."""

    mass_in_atmosphere: float  # kg
    mass_in_solid: float  # kg
    mass_in_melt: float  # kg
    moles_in_atmosphere: float  # moles
    moles_in_melt: float  # moles
    moles_in_solid: float  # moles
    ppmw_in_solid: float  # ppm by weight
    ppmw_in_melt: float  # ppm by weight
    fugacity: float  # bar
    fugacity_coefficient: float  # dimensionless
    pressure_in_atmosphere: float  # bar
    volume_mixing_ratio: float  # dimensionless
    mass_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the gas species or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: GasSpecies, element: Optional[str] = None, **kwargs) -> float:
        """Wrapper to return the mass of either the gas species or one of its elements.

        Args:
            element: Returns the mass of this element. Defaults to None to return the species mass.
            **kwargs: Catches keyword arguments to forward to func.

        Returns:
            Mass of either the gas species or element.
        """
        mass: float = func(self, **kwargs)
        if element is not None:
            try:
                mass *= (
                    UnitConversion.g_to_kg(self.formula.composition()[element].mass)
                    / self.molar_mass
                )
            except KeyError:  # Element not in formula so mass is zero.
                mass = 0

        return mass

    return mass_wrapper


class ThermodynamicDataJANAF(ThermodynamicDataBase):
    """Thermodynamic data from the JANAF tables.

    Args:
        species: Chemical component.
    """

    _DATA_SOURCE: str = "JANAF"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    def __init__(self, species: ChemicalComponent):
        """Init.

        Quoting from the JANAF documentation: In the JANAF tables, we have generally chosen the
        ideal diatomic gas for the reference state of permanent gases such as O2, N2, Cl2 etc.
        """

        super().__init__(species)
        self.data: janaf.JanafPhase = self._get_phase_data()

    def _get_phase_data(self) -> janaf.JanafPhase:
        """Gets the relevant phase data for the species.

        Returns:
            A JanafPhase instance.
        """

        db = janaf.Janafdb()

        def get_phase_data(phase):
            try:
                phase_data = db.getphasedata(
                    formula=self.species.modified_hill_formula, phase=phase
                )
            except ValueError:
                return None
            return phase_data

        if isinstance(self.species, GasSpecies):
            if self.species.is_homonuclear_diatomic:
                phase_data_ref = get_phase_data("ref")
                phase_data_g = get_phase_data("g")
                if phase_data_ref is None and phase_data_g is None:
                    msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                        self.species.name_in_thermodynamic_data,
                        self.species.modified_hill_formula,
                        self.DATA_SOURCE,
                    )
                    logger.warning(msg)
                    raise KeyError(msg)
                phase = phase_data_ref or phase_data_g
            else:
                phase = get_phase_data("g")
                if phase is None:
                    msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                        self.species.name_in_thermodynamic_data,
                        self.species.modified_hill_formula,
                        self.DATA_SOURCE,
                    )
                    logger.warning(msg)
                    raise KeyError(msg)
        else:
            phase = get_phase_data("ref")

        assert phase is not None
        logger.debug(
            "Thermodynamic data for %s (%s) = %s",
            self.species.name_in_thermodynamic_data,
            self.species.modified_hill_formula,
            phase,
        )

        return phase

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            temperature: Temperature.
            pressure: Pressure (total).

        Returns:
            The standard Gibbs free energy of formation (J/mol).
        """
        del pressure
        gibbs: float = self.data.DeltaG(temperature)

        logger.debug(
            "Species = %s, standard Gibbs energy of formation = %f",
            self.species.name_in_thermodynamic_data,
            gibbs,
        )

        return gibbs


class ThermodynamicDataHollandAndPowell(ThermodynamicDataBase):
    """Thermodynamic data from Holland and Powell (1998).

    https://ui.adsabs.harvard.edu/abs/1998JMetG..16..309H

    The book 'Equilibrium thermodynamics in petrology: an introduction' by R. Powell also has
    a useful appendix A with equations.

    See the comments in the data file that is parsed by __init__

    Args:
        species: Chemical component.

    Raises:
        KeyError: Thermodynamic data is not available for the species.
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar
    dKdP: float = 4.0  # Derivative of bulk modulus (K) with respect to pressure.
    dKdT_factor: float = -1.5e-4  # Factor for computing the temperature-dependence of K.

    def __init__(self, species: ChemicalComponent):
        super().__init__(species)
        self.data: pd.Series = self._get_phase_data()

    def _get_phase_data(self) -> pd.Series:
        """Gets the relevant phase data for the species.

        Returns:
            A pandas series.
        """

        data_path: Path = DATA_ROOT_PATH / Path("Mindata161127.csv")  # type: ignore
        data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        data["name of phase component"] = data["name of phase component"].str.strip()
        data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        data.drop(columns="Abbreviation", inplace=True)
        data.set_index("name of phase component", inplace=True)
        data = data.loc[:, :"Vmax"]
        data = data.astype(float)

        try:
            return data.loc[self.species.name_in_thermodynamic_data]
        except KeyError as exc:
            msg: str = "Thermodynamic data for %s (%s) is not available in %s" % (
                self.species.name_in_thermodynamic_data,
                self.species.hill_formula,
                self.DATA_SOURCE,
            )
            logger.error(msg)
            raise KeyError from exc

    def get_enthalpy(self, temperature: float) -> float:
        """Calculates the enthalpy at temperature.

        Args:
            temperature: Temperature.

        Returns:
            Enthalpy at temperature.
        """
        H = self.data["Hf"]  # J
        a = self.data["a"]  # J/K           Coeff for calc heat capacity.
        b = self.data["b"]  # J/K^2         Coeff for calc heat capacity.
        c = self.data["c"]  # J K           Coeff for calc heat capacity.
        d = self.data["d"]  # J K^(-1/2)    Coeff for calc heat capacity.

        integral_H: float = (
            H
            + a * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
            + b / 2 * (temperature**2 - self.ENTHALPY_REFERENCE_TEMPERATURE**2)
            - c * (1 / temperature - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE)
            + 2 * d * (temperature**0.5 - self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
        )
        return integral_H

    def get_entropy(self, temperature: float) -> float:
        """Calculates the entropy at temperature.

        Args:
            temperature: Temperature.

        Returns:
            Entropy at temperature.
        """
        S = self.data["S"]  # J/K
        a = self.data["a"]  # J/K           Coeff for calc heat capacity.
        b = self.data["b"]  # J/K^2         Coeff for calc heat capacity.
        c = self.data["c"]  # J K           Coeff for calc heat capacity.
        d = self.data["d"]  # J K^(-1/2)    Coeff for calc heat capacity.

        integral_S: float = (
            S
            + a * np.log(temperature / self.ENTHALPY_REFERENCE_TEMPERATURE)
            + b * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
            - c / 2 * (1 / temperature**2 - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE**2)
            - 2 * d * (1 / temperature**0.5 - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
        )
        return integral_S

    def get_volume_at_temperature(self, temperature: float) -> float:
        """Calculates the volume at temperature.

        The exponential arises from the strict derivation, but often an expansion is performed
        where exp(x) = 1+x as in Holland and Powell (1998). Below the exp term is retained, but
        the equation in Holland and Powell (1998) p311 is expanded.

        Args:
            temperature: Temperature.

        Returns:
            Volume at temperature.
        """
        V = self.data["V"]  # J/bar
        alpha0 = self.data["a0"]  # K^(-1), thermal expansivity

        volume_T: float = V * np.exp(
            alpha0 * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
            - 2 * 10.0 * alpha0 * (temperature**0.5 - self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
        )
        return volume_T

    def get_bulk_modulus_at_temperature(self, temperature: float) -> float:
        """Calculates the bulk modulus at temperature.

        Holland and Powell (1998), p312 in the text.

        Args:
            temperature: Temperature.

        Returns:
            Bulk modulus at temperature.
        """
        K = self.data["K"]  # bar, bulk modulus
        bulk_modulus_T: float = K * (
            1 + self.dKdT_factor * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)
        )
        return bulk_modulus_T

    def get_volume_pressure_integral(self, temperature: float, pressure: float) -> float:
        """Computes the volume-pressure integral.

        Holland and Powell (1998), p312.

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            The volume-pressure integral.
        """
        V_T: float = self.get_volume_at_temperature(temperature)
        K_T: float = self.get_bulk_modulus_at_temperature(temperature)
        integral_VP: float = (
            V_T
            * K_T
            / (self.dKdP - 1)
            * ((1 + self.dKdP * (pressure - 1.0) / K_T) ** (1.0 - 1.0 / self.dKdP) - 1)
        )  # J, use P-1.0 instead of P.
        return integral_VP

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            temperature: Temperature.
            pressure: Pressure (total).

        Returns:
            The standard Gibbs free energy of formation (J/mol).
        """

        gibbs: float = self.get_enthalpy(temperature) - temperature * self.get_entropy(temperature)

        if isinstance(self.species, SolidSpecies):
            gibbs += self.get_volume_pressure_integral(temperature, pressure)

        logger.debug(
            "Species = %s, standard Gibbs energy of formation = %f",
            self.species.name_in_thermodynamic_data,
            gibbs,
        )

        return gibbs


class ThermodynamicData(ThermodynamicDataBase):
    """Combines thermodynamic data from multiple datasets.

    Args:
        species: Chemical component.
        datasets: A list of thermodynamic data to use. Defaults to Holland and Powell, and JANAF.
    """

    _DATA_SOURCE: str = "Combined"
    _STANDARD_STATE_PRESSURE: float = 1  # bar
    # We assume the JANAF reference temperature, which is close enough to the reference temperature
    # of Holland and Powell of 298 K (which could in fact be the same, if they simply decided to
    # drop the decimal points when reporting the reference temperature?).
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K

    def __init__(
        self,
        species: ChemicalComponent,
        datasets: Union[list[ThermodynamicDataBase], None] = None,
    ):
        super().__init__(species)
        if datasets is None:
            self.datasets: list[ThermodynamicDataBase] = []
            self.add_dataset(ThermodynamicDataHollandAndPowell(species))
            self.add_dataset(ThermodynamicDataJANAF(species))
        else:
            self.datasets = datasets

    def add_dataset(self, dataset: ThermodynamicDataBase) -> None:
        """Adds a thermodynamic dataset.

        Args:
            dataset: A thermodynamic dataset.
        """
        if len(self.datasets) >= 1:
            logger.warning("Combining different thermodynamic data may result in inconsistencies")
        logger.info("Adding thermodynamic data: %s", dataset.DATA_SOURCE)
        self.datasets.append(dataset)

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            temperature: Temperature.
            pressure: Pressure (total).

        Returns:
            The standard Gibbs free energy of formation (J/mol).
        """
        for dataset in self.datasets:
            try:
                gibbs: float = dataset.get_formation_gibbs(
                    temperature=temperature, pressure=pressure
                )
                return gibbs
            except KeyError:
                continue

        msg: str = "Thermodynamic data for %s (%s) is not available in any dataset" % (
            self.species.name_in_thermodynamic_data,
            self.species.hill_formula,
        )
        logger.error(msg)
        raise KeyError(msg)


@dataclass(kw_only=True)
class ChemicalComponent(ABC):
    """Abstract base class representing a chemical component and its properties.

    Args:
        chemical_formula: Chemical formula (e.g., CO2, C, CH4, etc.).
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        ideality: Ideality object for thermodynamic calculations. See subclasses for specific use.
            Defaults to Ideal.
        thermodynamic_class: Class for thermodynamic data. Defaults to JANAF.

    Attributes:
        chemical_formula: Chemical formula.
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        formula: Formula object derived from the chemical formula.
        ideality: Ideality object for thermodynamic calculations. See subclasses for specific use.
        thermodynamic_class: Class for thermodynamic data.
        thermodynamic_data: Instance of thermodynamic_class for this chemical component.
    """

    chemical_formula: str
    name_in_thermodynamic_data: str
    ideality: SystemConstraint = field(default_factory=IdealityConstant)
    thermodynamic_class: Type[ThermodynamicDataBase] = ThermodynamicDataJANAF
    formula: Formula = field(init=False)
    thermodynamic_data: ThermodynamicDataBase = field(init=False)
    output: Any = field(init=False, default=None)

    def __post_init__(self):
        logger.info(
            "Creating a %s: %s (%s)",
            self.__class__.__name__,
            self.name_in_thermodynamic_data,
            self.chemical_formula,
        )
        self.formula = Formula(self.chemical_formula)
        self.ideality.species = self.chemical_formula
        self.thermodynamic_data = self.thermodynamic_class(self)

    @property
    def molar_mass(self) -> float:
        """Molar mass in kg/mol."""
        return UnitConversion.g_to_kg(self.formula.mass)

    @property
    def hill_formula(self) -> str:
        """Hill formula."""
        return self.formula.formula

    @property
    def is_homonuclear_diatomic(self) -> bool:
        """True if the species is homonuclear diatomic otherwise False."""

        composition = self.formula.composition()

        if len(list(composition.keys())) == 1 and list(composition.values())[0].count == 2:
            return True
        else:
            return False

    @cached_property
    def modified_hill_formula(self) -> str:
        """Modified Hill formula.

        JANAF uses the modified Hill formula to index its data tables. In short, H, if present,
        should appear after C (if C is present), otherwise it must be the first element.
        """
        elements: dict[str, int] = {
            element: properties.count for element, properties in self.formula.composition().items()
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


@dataclass(kw_only=True)
class GasSpecies(ChemicalComponent):
    """A gas species.

    For a gas species, 'self.ideality' refers to its fugacity coefficient, where the fugacity is
    equal to the fugacity coefficient multiplied by the species' partial pressure.

    Args:
        chemical_formula: Chemical formula (e.g. CO2, C, CH4, etc.).
        ideality: Ideality object representing the fugacity coefficient for thermodynamic
            calculations. Defaults to Ideal (i.e., unity).
        thermodynamic_class: Class for thermodynamic data. Defaults to JANAF.
        solubility: Solubility model. Defaults to no solubility.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0.

    Attributes:
        chemical_formula: Chemical formula.
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        formula: Formula object derived from the chemical formula.
        ideality: Ideality object representing the fugacity coefficient for thermodynamic
            calculations.
        thermodynamic_class: Class for thermodynamic data.
        thermodynamic_data: Instance of thermodynamic_class for this chemical component.
        solubility: Solubility model.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
        output: Stores calculated values for output.
    """

    name_in_thermodynamic_data: str = field(init=False)
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0
    output: Union[GasSpeciesOutput, None] = field(init=False, default=None)

    def __post_init__(self):
        self.name_in_thermodynamic_data = self.chemical_formula
        super().__post_init__()
        self.ideality.name = "fugacity_coefficient"

    @property
    def fugacity_coefficient(self) -> SystemConstraint:
        """Fugacity coefficient."""
        return self.ideality

    @_mass_decorator
    def mass(
        self,
        *,
        planet: Planet,
        system: InteriorAtmosphereSystem,
        element: Optional[str] = None,
    ) -> float:
        """Calculates the total mass of the species or element.

        Args:
            planet: Planet properties.
            system: Interior atmosphere system.
            element: Returns the mass for an element. Defaults to None to return the species mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the species (element=None) or element (element=element).
        """

        del element

        pressure: float = system.solution_dict[self.chemical_formula]
        fugacity: float = system.fugacities_dict[self.chemical_formula]
        fugacity_coefficient: float = system.fugacity_coefficients_dict[self.chemical_formula]

        # Atmosphere.
        mass_in_atmosphere: float = UnitConversion.bar_to_Pa(pressure) / planet.surface_gravity
        mass_in_atmosphere *= (
            planet.surface_area * self.molar_mass / system.atmospheric_mean_molar_mass
        )
        volume_mixing_ratio: float = pressure / system.total_pressure
        moles_in_atmosphere: float = mass_in_atmosphere / self.molar_mass

        # Melt.
        prefactor: float = planet.mantle_mass * planet.mantle_melt_fraction
        ppmw_in_melt: float = self.solubility(
            fugacity,
            planet.surface_temperature,
            system.fugacities_dict,
        )
        mass_in_melt: float = prefactor * ppmw_in_melt * UnitConversion.ppm_to_fraction()
        moles_in_melt: float = mass_in_melt / self.molar_mass

        # Solid.
        prefactor: float = planet.mantle_mass * (1 - planet.mantle_melt_fraction)
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass_in_solid: float = prefactor * ppmw_in_solid * UnitConversion.ppm_to_fraction()
        moles_in_solid: float = mass_in_solid / self.molar_mass

        self.output = GasSpeciesOutput(
            mass_in_atmosphere=mass_in_atmosphere,
            mass_in_solid=mass_in_solid,
            mass_in_melt=mass_in_melt,
            moles_in_atmosphere=moles_in_atmosphere,
            moles_in_melt=moles_in_melt,
            moles_in_solid=moles_in_solid,
            ppmw_in_solid=ppmw_in_solid,
            ppmw_in_melt=ppmw_in_melt,
            fugacity=fugacity,
            fugacity_coefficient=fugacity_coefficient,
            pressure_in_atmosphere=pressure,
            volume_mixing_ratio=volume_mixing_ratio,
        )

        return self.output.mass_in_total


@dataclass(kw_only=True)
class SolidSpeciesOutput:
    """Output for a solid species."""

    activity: float


@dataclass(kw_only=True)
class SolidSpecies(ChemicalComponent):
    """A solid species.

    For a solid species, 'self.ideality' refers to its activity, where the activity is equal to the
    activity coefficient multiplied by the species' volume mixing ratio.

    Args:
        chemical_formula: Chemical formula (e.g., CO2, C, CH4, etc.).
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        thermodynamic_class: Class for thermodynamic data. Defaults to JANAF.
        ideality: Ideality object representing activity for thermodynamic calculations. Defaults to
            Ideal (i.e., unity).

    Attributes:
        chemical_formula: Chemical formula.
        name_in_thermodynamic_data: Name for locating Gibbs data in the thermodynamic data.
        formula: Formula object derived from the chemical formula.
        ideality: Ideality object representing activity for thermodynamic calculations.
        thermodynamic_class: Class for thermodynamic data.
        thermodynamic_data: Instance of thermodynamic_class for this chemical component.
        output: Stores calculated values for output.
    """

    output: Union[SolidSpeciesOutput, None] = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        self.ideality.name = "activity"

    @property
    def activity(self) -> SystemConstraint:
        """Activity."""
        return self.ideality
