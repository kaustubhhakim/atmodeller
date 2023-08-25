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
from functools import cached_property
from typing import Protocol

from molmass import Formula

from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


class Ideality(ABC):
    """Abstract base class for calculating ideality based on temperature and pressure.

    Ideality can refer to either ideal gas behaviour or an ideal solution.

    This class defines a method to calculate ideality based on temperature and pressure.
    Subclasses must implement the '_ideality' method to provide the specific calculation.

    Attributes:
        None
    """

    @abstractmethod
    def _ideality(self, *, temperature: float, pressure: float) -> float:
        """Calculates the ideality of a substance.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in bar.

        Returns:
            The calculated ideality value.
        """
        raise NotImplementedError

    def __call__(self, *, temperature: float, pressure: float) -> float:
        """Calculates the ideality of a substance at a given temperature and pressure.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in bar.

        Returns:
            The calculated ideality value.
        """
        return self._ideality(temperature=temperature, pressure=pressure)


class NonIdealConstant(Ideality):
    """A non-ideal constant activity or fugacity coefficient.

    Useful for testing or representing non-ideal behavior.

    This class provides the _ideality method that returns a constant value, allowing you to
    simulate non-ideal behavior with a specific coefficient.

    Args:
        constant: The constant activity or fugacity coefficient.

    Attributes:
        constant: The constant activity or fugacity coefficient.
    """

    def __init__(self, constant: float):
        self.constant: float = constant

    def _ideality(self, *args, **kwargs) -> float:
        """Calculates the ideality value for a constant activity or fugacity coefficient.

        Args:
            *args: Positional arguments (not used).
            **kwargs: Keyword arguments (not used).

        Returns:
            The constant activity or fugacity coefficient.
        """
        del args
        del kwargs
        return self.constant


class Ideal(NonIdealConstant):
    """An ideal solution or an ideal gas.

    An ideal solution with an activity coefficient of unity or an ideal gas with a fugacity
    coefficient of unity.

    This class provides the _ideality method that returns a value of 1.0, indicating that the
    solution or gas is ideal with no deviations from ideality.
    """

    def __init__(self, constant: float = 1.0):
        super().__init__(constant)


class BufferedFugacity(ABC):
    """Abstract base class for calculating buffered fugacity based on temperature and pressure.

    This class defines a method to calculate the log10(fugacity) of a buffer substance in terms of
    temperature. Subclasses must implement the '_fugacity' method to provide the specific
    calculation.

    Attributes:
        None
    """

    @abstractmethod
    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """Calculates the log10(fugacity) of the buffer in terms of temperature.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in bar. Defaults to 1 bar.

        Returns:
            Log10 of the fugacity.
        """
        raise NotImplementedError

    def __call__(
        self, *, temperature: float, pressure: float = 1, log10_shift: float = 0
    ) -> float:
        """Calculates the log10(fugacity) of the buffer plus an optional shift.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in bar. Defaults to 1 bar.
            log10_shift: Log10 shift. Defaults to 0.

        Returns:
            Log10 of the fugacity including the shift.
        """
        return self._fugacity(temperature=temperature, pressure=pressure) + log10_shift


@dataclass(kw_only=True)
class ChemicalComponent(ABC):
    """Abstract base class representing a chemical component and its properties.

    Args:
        chemical_formula: Chemical formula (e.g., CO2, C, CH4, etc.).
        common_name: Common name for locating Gibbs data in the thermodynamic database.
        ideality: Ideality object for thermodynamic calculations. See subclasses for specific use.
            Defaults to Ideal.

    Attributes:
        chemical_formula: Chemical formula.
        common_name: Common name in the thermodynamic database.
        formula: Formula object derived from the chemical formula.
        ideality: Ideality object for thermodynamic calculations. See subclasses for specific use.
    """

    chemical_formula: str
    common_name: str
    ideality: Ideality = field(default_factory=Ideal)
    formula: Formula = field(init=False)
    # TODO: select source of thermodynamic data for this species.  Do all the reading in/caching
    # to set up the interpolation functions

    def __post_init__(self):
        logger.info(
            "Creating a %s: %s (%s)",
            self.__class__.__name__,
            self.common_name,
            self.chemical_formula,
        )
        self.formula = Formula(self.chemical_formula)

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


class StandardGibbsFreeEnergyOfFormationProtocol(Protocol):
    """Standard Gibbs free energy of formation protocol."""

    _DATA_SOURCE: str
    _ENTHALPY_REFERENCE_TEMPERATURE: float  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

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

    def get(self, species: ChemicalComponent, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            species: A species.
            temperature: Temperature.
            pressure: Pressure (total) is relevant for condensed phases, but not for ideal gases.

        Returns:
            The standard Gibbs free energy of formation (J/mol).

        Raises:
            KeyError: Thermodynamic data is not available for the species.
        """
        ...


class SystemConstraint(Protocol):
    """A value constraint to apply to an interior-atmosphere system.

    Args:
        species: The species to constrain. Usually a species for a pressure or fugacity constraint
            or an element for a mass constraint.
        value: Imposed value in kg for masses and bar for pressures or fugacities.

    Attributes:
        species: The species to constrain. Usually a species for a pressure or fugacity constraint
            or an element for a mass constraint.
        value: Imposed value in kg for masses and bar for pressures or fugacities.
    """

    @property
    def species(self) -> str:
        ...

    def get_value(self, **kwargs) -> float:
        ...
