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
"""Redox buffers"""

# Use physical symbol conventions so pylint: disable=C0103

import logging
import sys
from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import ConstraintProtocol
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class _RedoxBuffer(ABC):
    """A redox buffer

    Args:
        species: Species
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.
        evaluation_pressure: Optional constant pressure in bar to always evaluate the redox buffer.
            Defaults to None, meaning that the input pressure argument is used instead.

    Attributes:
        species: Species
        log10_shift: Log10 shift relative to the buffer.
        evaluation_pressure: Optional constant pressure in bar to always use to evaluate the redox
            buffer. Defaults to None, meaning that the input pressure argument is used instead.
        full_name: Combines the species and fugacity to give a unique constraint name.
    """

    def __init__(
        self, species: str, *, log10_shift: float = 0, evaluation_pressure: float | None = None
    ):
        self._species: str = species
        self.log10_shift: float = log10_shift
        self.evaluation_pressure: float | None = evaluation_pressure

    @property
    def full_name(self) -> str:
        return f"{self.species}_{self.name}"

    @property
    def name(self) -> str:
        return "fugacity"

    @property
    def species(self) -> str:
        return self._species

    @abstractmethod
    def _get_buffer_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        """Log10 value at the buffer

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            log10 of the fugacity at the buffer
        """

    def get_log10_value(self, temperature: float, pressure: float, **kwargs) -> float:
        """Log10 value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar. This is ignored if :attr:`pressure` is not None.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """
        if self.evaluation_pressure is not None:
            pressure = self.evaluation_pressure
            logger.debug(
                "Evaluate %s at constant pressure = %f", self.__class__.__name__, pressure
            )
        log10_value: float = self._get_buffer_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        log10_value += self.log10_shift

        return log10_value

    def get_value(self, temperature: float, pressure: float, **kwargs) -> float:
        """Value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar. This is ignored if :attr:`pressure` is not None.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Fugacity including any shift
        """
        log10_value: float = self.get_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        value: float = 10**log10_value

        return value


class _OxygenFugacityBuffer(_RedoxBuffer):
    """A redox buffer that constrains oxygen fugacity as a function of temperature."""

    @override
    def __init__(self, species: str = "O2", **kwargs):
        super().__init__(species, **kwargs)


class IronWustiteBufferHirschmann(_OxygenFugacityBuffer, ConstraintProtocol):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`"""

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float = 1, **kwargs) -> float:
        del kwargs
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )

        return fugacity


class IronWustiteBufferONeill(_OxygenFugacityBuffer, ConstraintProtocol):
    """Iron-wustite buffer :cite:p:`OE02`

    Gibbs energy of reaction is at 1 bar :cite:p:`OE02{Table 6}`.
    """

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float = 1, **kwargs) -> float:
        del pressure
        del kwargs
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )

        return fugacity


class IronWustiteBufferBallhaus(_OxygenFugacityBuffer, ConstraintProtocol):
    """Iron-wustite buffer :cite:p:`BBG91`"""

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float = 1, **kwargs) -> float:
        del kwargs
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )

        return fugacity


class IronWustiteBufferFischer(_OxygenFugacityBuffer, ConstraintProtocol):
    """Iron-wustite buffer :cite:p:`F11`

    See :cite:t:`F11{Table S2}` in supplementary materials.
    """

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: float = 1, **kwargs) -> float:
        del kwargs
        pressure_GPa: float = UnitConversion.bar_to_GPa(pressure)
        a_coeff: float = 6.44059 + 0.00463099 * pressure_GPa
        b_coeff: float = (
            -28.1808
            + 0.556272 * pressure_GPa
            - 0.00143757 * pressure_GPa**2
            + 4.0256e-6 * pressure_GPa**3
            - 5.4861e-9 * pressure_GPa**4  # Note typo in Table S2. Must be pressure**4.
        )
        b_coeff *= 1000 / temperature
        fugacity: float = a_coeff + b_coeff

        return fugacity


IronWustiteBuffer: Type[_OxygenFugacityBuffer] = IronWustiteBufferHirschmann
