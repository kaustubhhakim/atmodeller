"""Base classes for the fugacity models from Shi and Saxena (1992) and Saxena and Fei (1988).

See the LICENSE file for licensing information.
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from atmodeller.eos.interfaces import ReducedFugacityModelABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SaxenaABC(ReducedFugacityModelABC):
    """Shi and Saxena (1992) fugacity model.

    The model presented in Shi and Saxena (1992) is a general form that can be adapted to the
    previous work of Saxena and Fei (1988).

    Shi and Saxena (1992), Thermodynamic modeling of the C-H-O-S fluid system, American
    Mineralogist, Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid
    phases.

    http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

    Args:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        a_coefficients: a coefficients (see paper). Defaults to empty.
        b_coefficients: b coefficients (see paper). Defaults to empty.
        c_coefficients: c coefficients (see paper). Defaults to empty.
        d_coefficients: d coefficients (see paper). Defaults to empty.

    Attributes:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        a_coefficients: a coefficients.
        b_coefficients: b coefficients.
        c_coefficients: c coefficients.
        d_coefficients: d coefficients.
        scaling: See base class.
        GAS_CONSTANT: See base class.
        standard_state_pressure: Standard state pressure. Set to 1 bar.
    """

    a_coefficients: tuple[float, ...] = field(default_factory=tuple)
    b_coefficients: tuple[float, ...] = field(default_factory=tuple)
    c_coefficients: tuple[float, ...] = field(default_factory=tuple)
    d_coefficients: tuple[float, ...] = field(default_factory=tuple)

    @abstractmethod
    def _get_compressibility_coefficient(
        self, temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation.

        Shi and Saxena (1992), Equation 1.

        Args:
            temperature: Temperature in kelvin.
            coefficients: Tuple of the coefficients a, b, c, or d.

        Returns
            The relevant coefficient.
        """
        ...

    def a(self, temperature: float) -> float:
        """a parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            a parameter.
        """
        a: float = self._get_compressibility_coefficient(temperature, self.a_coefficients)

        return a

    def b(self, temperature: float) -> float:
        """b parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            b parameter.
        """
        b: float = self._get_compressibility_coefficient(temperature, self.b_coefficients)

        return b

    def c(self, temperature: float) -> float:
        """c parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            c parameter.
        """
        c: float = self._get_compressibility_coefficient(temperature, self.c_coefficients)

        return c

    def d(self, temperature: float) -> float:
        """d parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            d parameter.
        """
        d: float = self._get_compressibility_coefficient(temperature, self.d_coefficients)

        return d

    def compressibility_parameter(self, temperature: float, pressure: float) -> float:
        """Compressibility parameter at temperature and pressure.

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Shi and Saxena (1992), Equation 2.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            The compressibility parameter, Z.
        """
        Pr: float = self.reduced_pressure(pressure)
        Z: float = (
            self.a(temperature)
            + self.b(temperature) * Pr
            + self.c(temperature) * Pr**2
            + self.d(temperature) * Pr**3
        )

        return Z

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Shi and Saxena (1992), Equation 1.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        Z: float = self.compressibility_parameter(temperature, pressure)
        volume: float = Z * self.ideal_volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP).

        Shi and Saxena (1992), Equation 11.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        Pr: float = self.reduced_pressure(pressure)
        P0r: float = self.reduced_standard_state_pressure
        volume_integral: float = (
            (
                self.a(temperature) * np.log(Pr / P0r)
                + self.b(temperature) * (Pr - P0r)
                + (1.0 / 2) * self.c(temperature) * (Pr**2 - P0r**2)
                + (1.0 / 3) * self.d(temperature) * (Pr**3 - P0r**3)
            )
            * self.GAS_CONSTANT
            * temperature
        )

        return volume_integral


@dataclass(kw_only=True)
class SaxenaFiveCoefficients(SaxenaABC):
    """Fugacity model with five coefficients, which is generally used for low pressures.

    See base class.
    """

    def _get_compressibility_coefficient(
        self, temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation.

        Shi and Saxena (1992), Equation 3b.

        Args:
            temperature: Temperature in kelvin.
            coefficients: Tuple of the coefficients a, b, c, or d.

        Returns
            The relevant coefficient.
        """
        Tr: float = self.reduced_temperature(temperature)
        coefficient: float = (
            coefficients[0]
            + coefficients[1] / Tr
            + coefficients[2] / Tr ** (3 / 2)
            + coefficients[3] / Tr**3
            + coefficients[4] / Tr**4
        )

        return coefficient


@dataclass(kw_only=True)
class SaxenaEightCoefficients(SaxenaABC):
    """Fugacity model with eight coefficients, which is generally used for high pressures.

    See base class.
    """

    def _get_compressibility_coefficient(
        self, temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation.

        Shi and Saxena (1992), Equation 3a.

        Args:
            temperature: Temperature in kelvin.
            coefficients: Tuple of the coefficients a, b, c, or d.

        Returns
            The relevant coefficient.
        """
        Tr: float = self.reduced_temperature(temperature)
        coefficient: float = (
            coefficients[0]
            + coefficients[1] * Tr
            + coefficients[2] / Tr
            + coefficients[3] * Tr**2
            + coefficients[4] / Tr**2
            + coefficients[5] * Tr**3
            + coefficients[6] / Tr**3
            + coefficients[7] * np.log(Tr)
        )

        return coefficient
