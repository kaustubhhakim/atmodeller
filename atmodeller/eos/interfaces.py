"""Interfaces for real gas equations of state.

See the LICENSE file for licensing information.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Type

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import GetValueABC


@dataclass(kw_only=True)
class FugacityModelABC(GetValueABC):
    """A fugacity model.

    This base class requires a specification for the volume and volume integral, since then the
    fugacity and related quantities can be computed using the standard relation:

    RTlnf = integral(VdP).

    Args:
        scaling: Scaling depending on the units of the coefficients that are used in this model.
            For pressure in bar this is unity and for pressure in kbar this is kilo. Defaults to
            unity.

    Attributes:
        scaling: Scaling depending on the units of pressure.
        GAS_CONSTANT: Gas constant with the appropriate units.
        standard_state_pressure: Standard state pressure with the appropriate units. Set to 1 bar.
    """

    scaling: float = 1
    GAS_CONSTANT: float = field(init=False, default=GAS_CONSTANT)
    standard_state_pressure: float = field(init=False, default=1)  # 1 bar

    def __post_init__(self):
        """Scales pressure quantities to ensure they use the internal pressure units."""
        self.GAS_CONSTANT /= self.scaling
        self.standard_state_pressure /= self.scaling

    def compressibility_parameter(self, temperature: float, pressure: float, **kwargs) -> float:
        """Compressibility parameter at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure
            **kwargs: Catches unused keyword arguments, but used for overrides in subclasses.

        Returns:
            The compressibility parameter, Z
        """
        del kwargs
        volume: float = self.volume(temperature, pressure)
        volume_ideal: float = self.ideal_volume(temperature, pressure)
        Z: float = volume / volume_ideal

        return Z

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Note that the input 'pressure' must ALWAYS be in bar, so it is scaled here using
        'self.scaling' since self.fugacity_coefficient requires the internal units of pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            Fugacity coefficient evaluated at temperature and pressure, which is dimensionaless.
        """
        pressure /= self.scaling
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)

        return fugacity_coefficient

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity.

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity.
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            self.GAS_CONSTANT * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in the same units as the input pressure.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity.
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))  # bar
        fugacity /= self.scaling  # to units of input pressure for consistency.

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity coefficient, which is non-dimensional.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    def ideal_volume(self, temperature: float, pressure: float) -> float:
        """Ideal volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure.

        Returns:
            ideal volume
        """
        volume_ideal: float = self.GAS_CONSTANT * temperature / pressure

        return volume_ideal

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        ...


@dataclass(kw_only=True)
class ReducedFugacityModelABC(FugacityModelABC):
    """A fugacity model that is formulated in terms of reduced temperature and pressure.

    See base class.

    Args:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure
        scaling: Scaling depending on the units of the coefficients that are used in this model.
            For pressure in bar this is unity and for pressure in kbar this is kilo. Defaults to
            unity.

    Attributes:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure
        scaling: Scaling depending on the units of pressure
        GAS_CONSTANT: Gas constant with the appropriate units
        standard_state_pressure: Standard state pressure with the appropriate units. Set to 1 bar.
    """

    Tc: float
    Pc: float

    def reduced_pressure(self, pressure: float) -> float:
        """Reduced pressure.

        Args:
            pressure: Pressure.

        Returns:
            The reduced pressure, which is dimensionless.
        """
        Pr: float = pressure / self.Pc

        return Pr

    def reduced_temperature(self, temperature: float) -> float:
        """Reduced temperature.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            The reduced temperature, which is dimensionless.
        """
        Tr: float = temperature / self.Tc

        return Tr

    @property
    def reduced_standard_state_pressure(self) -> float:
        """Reduced standard state pressure.

        Args:
            pressure: Pressure.

        Returns:
            The reduced standard state pressure, which is dimensionless.
        """
        Pr0: float = self.standard_state_pressure / self.Pc

        return Pr0


@dataclass(kw_only=True)
class CombinedReducedFugacityModel(ReducedFugacityModelABC):
    """Combines multiple fugacity models for different pressure ranges into a single model.

    Args:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        classes: Reduced fugacity classes with coefficients specified and ordered by increasing
            pressure.
        upper_pressure_bounds: The upper pressure bound relevant to the fugacity class by position.
        scaling: Scaling depending on the units of the coefficients that are used in this model.
            For pressure in bar this is unity and for pressure in kbar this is kilo. Defaults to
            unity.

    Attributes:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        classes: Reduced fugacity classes with coefficients specified and ordered by increasing
            pressure.
        upper_pressure_bounds: The upper pressure bound relevant to the fugacity class by position.
        models: Instantiated fugacity classes.
        scaling: Scaling depending on the units of pressure
        GAS_CONSTANT: Gas constant with the appropriate units
        standard_state_pressure: Standard state pressure with the appropriate units. Set to 1 bar.
    """

    classes: tuple[Type[ReducedFugacityModelABC], ...]
    upper_pressure_bounds: tuple[float, ...]
    models: list[FugacityModelABC] = field(init=False, default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        for fugacity_class in self.classes:
            self.models.append(fugacity_class(Tc=self.Tc, Pc=self.Pc))

    def _get_index(self, pressure: float) -> int:
        """Gets the index of the appropriate fugacity model using the pressure.

        Args:
            pressure: Pressure.

        Returns:
            Index of the relevant fugacity model.
        """
        for index, pressure_high in enumerate(self.upper_pressure_bounds):
            if pressure < pressure_high:
                return index
        # If the pressure is higher than all specified pressure ranges, use the last model.
        return len(self.models) - 1

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        index: int = self._get_index(pressure)
        volume: float = self.models[index].volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        index: int = self._get_index(pressure)
        volume: float = self.models[index].volume_integral(temperature, pressure)

        return volume


@dataclass(frozen=True)
class critical_data:
    """Critical temperature and pressure of a gas species.

    Args:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure in bar

    Attributes:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure in bar
    """

    Tc: float
    Pc: float


# Critical temperature and pressure data for a corresponding states model, based on Table 2 in
# Shi and Saxena (1992) with some additions. For simplicity, we just compile one set of critical
# data, even though they vary a little between studies which could result in subtle differences.
critical_data_dictionary: dict[str, critical_data] = {
    "H2O": critical_data(647.25, 221.1925),
    "CO2": critical_data(304.15, 73.8659),
    "CH4": critical_data(191.05, 46.4069),
    "CO": critical_data(133.15, 34.9571),
    "O2": critical_data(154.75, 50.7638),
    "H2": critical_data(33.25, 12.9696),
    "S2": critical_data(208.15, 72.954),
    "SO2": critical_data(430.95, 78.7295),
    "COS": critical_data(377.55, 65.8612),
    "H2S": critical_data(373.55, 90.0779),
    "N2": critical_data(126.2, 33.9),  # Saxena and Fei (1987)
    "Ar": critical_data(151, 48.6),  # Saxena and Fei (1987)
}
