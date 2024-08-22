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

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from atmodeller import GAS_CONSTANT
from atmodeller.thermodata.interfaces import RedoxBufferProtocol
from atmodeller.utilities import ExperimentalCalibration, unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class _RedoxBuffer(ABC, RedoxBufferProtocol):
    """A redox buffer

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.
        calibration: Calibration temperature and pressure range. Defaults to empty.

    Attributes:
        log10_shift: Log10 shift relative to the buffer
        calibration: Calibration temperature and pressure range
    """

    @override
    def __init__(
        self,
        log10_shift: float = 0,
        *,
        calibration: ExperimentalCalibration = ExperimentalCalibration()
    ):
        self.log10_shift: float = log10_shift
        self.calibration: ExperimentalCalibration = calibration
        logger.debug("Setting experimental calibration = %s", calibration)

    @abstractmethod
    def _get_buffer_log10_value(
        self, temperature: float, pressure: ArrayLike, **kwargs
    ) -> ArrayLike:
        """Log10 value at the buffer

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            log10 of the fugacity at the buffer
        """

    # TODO: Not sure how a penalty will behave with JAX,
    @override
    def get_log10_value(
        self, temperature: float, pressure: ArrayLike, penalty: bool = False, **kwargs
    ) -> ArrayLike:
        """Log10 value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            penalty: Apply penalty function. Defaults to True.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """
        log10_value: ArrayLike = self._get_buffer_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        log10_value = log10_value + self.log10_shift

        if penalty:
            log10_value = log10_value + self.calibration.get_penalty(temperature, pressure)

        return log10_value

    @override
    def get_value(
        self, temperature: float, pressure: ArrayLike, penalty: bool = True, **kwargs
    ) -> ArrayLike:
        """Value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            penalty: Apply penalty function. Defaults to True.
            **kwargs: Arbitrary keyword arguments

        Returns:
            Fugacity including any shift
        """
        log10_value: ArrayLike = self.get_log10_value(
            temperature=temperature, pressure=pressure, penalty=penalty, **kwargs
        )
        value: ArrayLike = jnp.power(10, log10_value)

        return value


# 27.5 GPa is given in the abstract of :cite:t:`HGD08`
IronWustiteBufferHirschmann08Calibration: ExperimentalCalibration = ExperimentalCalibration(
    pressure_max=27.5 * unit_conversion.GPa_to_bar
)


class IronWustiteBufferHirschmann08(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`"""

    @override
    def __init__(
        self,
        log10_shift: float = 0,
        *,
        calibration: ExperimentalCalibration = IronWustiteBufferHirschmann08Calibration
    ):
        super().__init__(log10_shift, calibration=calibration)

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: ArrayLike, **kwargs) -> Array:
        del kwargs
        fugacity: Array = (
            -0.8853 * jnp.log(temperature)
            - 28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
        )

        return fugacity


# From :cite:t:`H21`: "It extrapolates smoothly to higher temperature, though not calibrated above
# 3000 K. Extrapolation to lower temperatures (<1000 K) or higher pressures (>100 GPa) is not
# recommended."
IronWustiteBufferHirschmann21Calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=1000, pressure_max=100 * unit_conversion.GPa_to_bar
)


class IronWustiteBufferHirschmann21(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`H21`"""

    @override
    def __init__(
        self,
        log10_shift: float = 0,
        *,
        calibration: ExperimentalCalibration = IronWustiteBufferHirschmann21Calibration
    ):
        super().__init__(log10_shift, calibration=calibration)
        self.a: Array = jnp.array([6.844864, 1.175691e-1, 1.143873e-3, 0, 0])
        self.b: Array = jnp.array([5.791364e-4, -2.891434e-4, -2.737171e-7, 0, 0])
        self.c: Array = jnp.array([-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7])
        self.d: Array = jnp.array([-2.769002e4, 5.285977e2, -2.919275, 0, 0])
        self.e: Array = jnp.array([8.463095, -3.000307e-3, 7.213445e-5, 0, 0])
        self.f: Array = jnp.array([1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0])
        self.g: Array = jnp.array([-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4])
        self.h: Array = jnp.array([-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0])

    def _evaluate_m(self, pressure: ArrayLike, coefficients: Array) -> Array:
        """Evaluates an m parameter

        Args:
            pressure: Pressure in GPa
            coefficients: Coefficients

        Return:
            m parameter
        """
        m: Array = (
            coefficients[0]
            + coefficients[1] * pressure
            + coefficients[2] * pressure**2
            + coefficients[3] * pressure**3
            + coefficients[4] * pressure ** (1 / 2)
        )

        return m

    def _evaluate_fO2(
        self, temperature: float, pressure: ArrayLike, coefficients: tuple[Array, ...]
    ) -> Array:
        """Evaluates the fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa
            coefficients: Coefficients

        Returns:
            log10fO2
        """
        log10fO2: Array = (
            self._evaluate_m(pressure, coefficients[0])
            + self._evaluate_m(pressure, coefficients[1]) * temperature
            + self._evaluate_m(pressure, coefficients[2]) * temperature * jnp.log(temperature)
            + self._evaluate_m(pressure, coefficients[3]) / temperature
        )

        return log10fO2

    def _fcc_bcc_iron(self, temperature: float, pressure: ArrayLike) -> Array:
        """log10fO2 for fcc and bcc iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for fcc and bcc iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self.a, self.b, self.c, self.d)
        )

        return log10fO2

    def _hcp_iron(self, temperature: float, pressure: ArrayLike) -> Array:
        """log10fO2 for hcp iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for hcp iron
        """
        log10fO2: ArrayLike = self._evaluate_fO2(
            temperature, pressure, (self.e, self.f, self.g, self.h)
        )

        return log10fO2

    def _use_hcp(self, temperature: float, pressure: ArrayLike) -> Array:
        """Check to use hcp iron formulation for fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa
        """
        x: Array = jnp.array([-18.64, 0.04359, -5.069e-6])
        threshold: Array = x[0] + x[1] * temperature + x[2] * temperature**2

        return jnp.array(pressure) > threshold

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: ArrayLike, **kwargs) -> Array:
        del kwargs
        pressure_GPa: ArrayLike = pressure * unit_conversion.bar_to_GPa

        def hcp_case() -> Array:
            return self._hcp_iron(temperature, pressure_GPa)

        def fcc_bcc_case() -> Array:
            return self._fcc_bcc_iron(temperature, pressure_GPa)

        return lax.cond(self._use_hcp(temperature, pressure_GPa), hcp_case, fcc_bcc_case)


class IronWustiteBufferHirschmann(RedoxBufferProtocol):
    """Composite iron-wustite buffer using :cite:t:`OP93,HGD08` and :cite:t:`H21`"""

    @override
    def __init__(
        self,
        log10_shift: float = 0,
    ):
        self.low_temperature_buffer: _RedoxBuffer = IronWustiteBufferHirschmann08(log10_shift)
        self.high_temperature_buffer: _RedoxBuffer = IronWustiteBufferHirschmann21(log10_shift)

    @override
    def get_log10_value(self, temperature: float, pressure: ArrayLike, **kwargs) -> ArrayLike:
        """Log10 value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """
        try:
            assert self.high_temperature_buffer.calibration.temperature_min is not None
        except AssertionError as exc:
            raise ValueError(
                "temperature_min is not defined for the high temperature buffer"
            ) from exc

        if temperature < self.high_temperature_buffer.calibration.temperature_min:
            return self.low_temperature_buffer.get_log10_value(temperature, pressure, **kwargs)
        else:
            return self.high_temperature_buffer.get_log10_value(temperature, pressure, **kwargs)

    @override
    def get_value(self, temperature: float, pressure: ArrayLike, **kwargs) -> Array:
        """Value including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Arbitrary keyword arguments

        Returns:
            Fugacity including any shift
        """
        log10_value: ArrayLike = self.get_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        value: Array = jnp.power(10, log10_value)

        return value


class IronWustiteBufferONeill(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`OE02`

    Gibbs energy of reaction is at 1 bar :cite:p:`OE02{Table 6}`.
    """

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: ArrayLike, **kwargs) -> Array:
        del pressure
        del kwargs
        fugacity: Array = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * jnp.log(temperature))
            / (jnp.log(10) * GAS_CONSTANT * temperature)
        )

        return fugacity


class IronWustiteBufferBallhaus(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`BBG91`"""

    @override
    def _get_buffer_log10_value(self, temperature: float, pressure: ArrayLike, **kwargs) -> Array:
        del kwargs
        fugacity: Array = (
            14.07
            - 28784 / temperature
            - 2.04 * jnp.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )

        return fugacity


class IronWustiteBufferFischer(_RedoxBuffer):
    """Iron-wustite buffer :cite:p:`F11`

    See :cite:t:`F11{Table S2}` in supplementary materials.
    """

    @override
    def _get_buffer_log10_value(
        self, temperature: float, pressure: ArrayLike, **kwargs
    ) -> ArrayLike:
        del kwargs
        pressure_GPa: ArrayLike = pressure * unit_conversion.bar_to_GPa
        a_coeff: ArrayLike = 6.44059 + 0.00463099 * pressure_GPa
        b_coeff: ArrayLike = (
            -28.1808
            + 0.556272 * pressure_GPa
            - 0.00143757 * pressure_GPa**2
            + 4.0256e-6 * pressure_GPa**3
            - 5.4861e-9 * pressure_GPa**4  # Note typo in Table S2. Must be pressure**4.
        )
        b_coeff = b_coeff * 1000 / temperature
        fugacity: ArrayLike = a_coeff + b_coeff

        return fugacity


IronWustiteBuffer: Type[RedoxBufferProtocol] = IronWustiteBufferHirschmann
