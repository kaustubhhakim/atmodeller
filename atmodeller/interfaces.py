"""Interfaces

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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Protocol

import numpy as np

from atmodeller import GAS_CONSTANT, GAS_CONSTANT_BAR
from atmodeller.utilities import UnitConversion

if TYPE_CHECKING:
    from atmodeller.core import ChemicalComponent

logger: logging.Logger = logging.getLogger(__name__)

# Solubility limiter is applied universally
MAXIMUM_PPMW: float = UnitConversion.weight_percent_to_ppmw(10)  # 10% by weight


class GetValueABC(ABC):
    """An object with a get_value method."""

    @abstractmethod
    def get_value(self, *args, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            An evaluation based on the provided arguments
        """
        ...

    def get_log10_value(self, *args, **kwargs) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            *args: Positional arguments only
            **kwargs: Keyword arguments only

        Returns:
            An evaluation of the log10 value based on the provided arguments
        """
        return np.log10(self.get_value(*args, **kwargs))


@dataclass(kw_only=True, frozen=True)
class ConstraintABC(GetValueABC):
    """A constraint to apply to an interior-atmosphere system.

    Args:
        name: The name of the constraint, which should be one of: 'activity', 'fugacity',
            'pressure', or 'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.

    Attributes:
        name: The name of the constraint
        species: The species to constrain
    """

    name: str
    species: str

    @property
    def full_name(self) -> str:
        """Combines the species name and constraint name to give a unique descriptive name."""
        if self.species:
            full_name: str = f"{self.species}_"
        else:
            full_name = ""
        full_name += self.name

        return full_name


@dataclass(kw_only=True)
class RealGasABC(GetValueABC):
    """A real gas equation of state (EOS)

    This base class requires a specification for the volume and volume integral. Then the
    fugacity and related quantities can be computed using the standard relation:

    RTlnf = integral(VdP)

    If critical_temperature and critical_pressure are set to their default value of unity, then
    these quantities are effectively not used, and the model coefficients should be in terms of
    the real temperature and pressure. But for corresponding state models, which are formulated in
    terms of a reduced temperature and a reduced pressure, the critical_temperature and
    critical_pressure must be set to appropriate values for the species under consideration.

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
    """

    critical_temperature: float = 1  # Default of one is equivalent to not used
    critical_pressure: float = 1  # Default of one is equivalent to not used
    standard_state_pressure: float = field(init=False, default=1)  # 1 bar

    def scaled_pressure(self, pressure: float) -> float:
        """Scaled pressure, i.e. a reduced pressure when critical pressure is not unity

        Args:
            pressure: Pressure in bar

        Returns:
            The scaled (reduced) pressure, which is dimensionless
        """
        scaled_pressure: float = pressure / self.critical_pressure

        return scaled_pressure

    def scaled_temperature(self, temperature: float) -> float:
        """Scaled temperature, i.e. a reduced temperature when critical temperature is not unity

        Args:
            temperature: Temperature in kelvin

        Returns:
            The scaled (reduced) temperature, which is dimensionless
        """
        scaled_temperature: float = temperature / self.critical_temperature

        return scaled_temperature

    def compressibility_parameter(self, temperature: float, pressure: float, **kwargs) -> float:
        """Compressibility parameter at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar
            **kwargs: Catches unused keyword arguments. Used for overrides in subclasses.

        Returns:
            The compressibility parameter, Z, which is dimensionless
        """
        del kwargs
        volume: float = self.volume(temperature, pressure)
        volume_ideal: float = self.ideal_volume(temperature, pressure)
        Z: float = volume / volume_ideal

        return Z

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Fugacity coefficient evaluated at temperature and pressure, which is dimensionaless.
        """
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)

        return fugacity_coefficient

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Natural log of the fugacity
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            fugacity coefficient, which is non-dimensional
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    def ideal_volume(self, temperature: float, pressure: float) -> float:
        """Ideal volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            ideal volume in m^3 mol^(-1)
        """
        volume_ideal: float = GAS_CONSTANT_BAR * temperature / pressure

        return volume_ideal

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP)

        Be careful with units. If this function uses the same constants (and GAS_CONSTANT_BAR) as
        volume() then the units will be m^3 mol^(-1) bar. But this method requires that the units
        returned are J mol^(-1). Hence the following conversion is often necessary:

            1 J = 10^(-5) m^(3) bar

        There are functions to do this conversion in utilities.py.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        ...


def limit_solubility(bound: float = MAXIMUM_PPMW) -> Callable:
    """A decorator to limit the solubility in ppmw.

    Args:
        bound: The maximum limit of the solubility in ppmw. Defaults to MAXIMUM_PPMW.

    Returns:
        The decorator.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: SolubilityABC, *args, **kwargs):
            result: float = func(self, *args, **kwargs)
            if result > bound:
                msg: str = "%s solubility (%d ppmw) will be limited to %d ppmw" % (
                    self.__class__.__name__,
                    result,
                    bound,
                )
                logger.warning(msg)

            return np.clip(result, 0, bound)  # Limit the result between 0 and 'bound'

        return wrapper

    return decorator


class SolubilityABC(GetValueABC):
    """A solubility law for a species."""

    def power_law(self, fugacity: float, constant: float, exponent: float) -> float:
        """Computes solubility from a power law.

        Args:
            fugacity: Fugacity of the species in bar
            constant: Constant for the power law
            exponent: Exponent for the power law

        Returns:
            Dissolved volatile concentration in the melt in ppmw.
        """
        return constant * fugacity**exponent

    @abstractmethod
    def _solubility(
        self,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Dissolved volatile concentration in the melt in ppmw.

        Args:
            fugacity: Fugacity of the species in bar
            temperature: Temperature in kelvin
            log10_fugacities_dict: Log10 fugacities of all species in the system
            pressure: Total pressure

        Returns:
            Dissolved volatile concentration in the melt in ppmw.
        """
        raise NotImplementedError

    @limit_solubility()  # Note this limiter is always applied.
    def get_value(
        self,
        *,
        fugacity: float,
        temperature: float,
        log10_fugacities_dict: dict[str, float],
        pressure: float,
    ) -> float:
        """Dissolved volatile concentration in the melt in ppmw.

        See self._solubility.
        """
        solubility: float = self._solubility(
            fugacity, temperature, log10_fugacities_dict, pressure
        )
        # logger.debug(
        #     "%s, f = %f, T = %f, ppmw = %f",
        #     self.__class__.__name__,
        #     fugacity,
        #     temperature,
        #     solubility,
        # )
        return solubility


@dataclass(frozen=True)
class ThermodynamicDataForSpeciesProtocol(Protocol):
    """Protocol for a class with a method that returns the Gibbs energy of formation for a species.

    Args:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation

    Attributes:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation
    """

    species: ChemicalComponent
    data_source: str
    data: Any

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol.

        Args:
            temperature: Temperature in kelvin
            pressure: Total pressure in bar

        Returns:
            The standard Gibbs free energy of formation in J/mol
        """
        ...


class ThermodynamicDatasetABC(ABC):
    """Thermodynamic dataset base class"""

    _DATA_SOURCE: str
    # JANAF standards below. May be overwritten by child classes.
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @abstractmethod
    def get_data(self, species: ChemicalComponent) -> ThermodynamicDataForSpeciesProtocol | None:
        """Gets the thermodynamic data for a species, otherwise None if not available

        Args:
            species: Species

        Returns:
            Thermodynamic data for the species, otherwise None is not available
        """
        ...

    @property
    def DATA_SOURCE(self) -> str:
        """Identifies the source of the data."""
        return self._DATA_SOURCE

    @property
    def ENTHALPY_REFERENCE_TEMPERATURE(self) -> float:
        """Enthalpy reference temperature in kelvin"""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def STANDARD_STATE_PRESSURE(self) -> float:
        """Standard state pressure in bar"""
        return self._STANDARD_STATE_PRESSURE
