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

import sys
from abc import ABC, abstractmethod
from typing import Protocol

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.utilities import ExperimentalCalibrationNew, unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


class RedoxBufferProtocol(FugacityConstraintProtocol, Protocol):
    """Redox buffer protocol"""

    log10_shift: ArrayLike
    evaluation_pressure: ArrayLike | None
    _calibration: ExperimentalCalibrationNew

    @property
    def calibration(self) -> ExperimentalCalibrationNew: ...

    @property
    def value(self) -> ArrayLike:
        return self.log10_shift

    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...


class RedoxBuffer(ABC, RedoxBufferProtocol):
    """Redox buffer

    Child classes must set self._calibration and self._evaluation_pressure_scaling

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.
        evaluation_pressure: Pressure to evaluate the buffer at. Defaults to None, meaning use
            the total pressure which is passed in as an argument.

    Attributes:
        log10_shift: Log10 shift relative to the buffer
        evaluation_pressure: Pressure to evaluate the buffer at or None
    """

    def __init__(
        self,
        log10_shift: ArrayLike = 0,
        evaluation_pressure: ArrayLike | None = None,
    ):
        self.log10_shift: ArrayLike = log10_shift
        self.evaluation_pressure: ArrayLike | None = evaluation_pressure

    @property
    def calibration(self) -> ExperimentalCalibrationNew:
        """Experimental calibration"""
        return self._calibration

    @abstractmethod
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike: ...

    def get_scaled_pressure(self, pressure: ArrayLike) -> ArrayLike:
        if self.evaluation_pressure is not None:
            return self.convert_pressure_units(self.evaluation_pressure)
        else:
            return self.convert_pressure_units(pressure)

    @abstractmethod
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity at the buffer

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity at the buffer
        """

    @override
    @jit
    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity
        """
        return self.log10_fugacity_buffer(temperature, pressure) + self.log10_shift

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        return self.log10_fugacity(temperature, pressure) * jnp.log(10)

    def tree_flatten(self) -> tuple[tuple[ArrayLike, ArrayLike | None], None]:
        children = (self.log10_shift, self.evaluation_pressure)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del aux_data
        return cls(*children)


@register_pytree_node_class
class IronWustiteBufferHirschmann08(RedoxBuffer):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`

    Experimental calibration values are provided in the abstract of :cite:t:`HGD08`.

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
        evaluation_pressure: Pressure to evaluate the buffer at. Defaults to None, meaning use
            the total pressure which is passed in as an argument.

    Attributes:
        log10_shift: Log10 shift relative to the buffer.
        evaluation_pressure: Pressure to evaluate the buffer at.
    """

    @override
    def __init__(self, log10_shift: ArrayLike = 0, evaluation_pressure: ArrayLike | None = None):
        self._calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
            pressure_max=27.5 * unit_conversion.GPa_to_bar
        )
        super().__init__(log10_shift, evaluation_pressure)

    @override
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Units are bar"""
        return pressure

    @override
    @jit
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity
        """
        scaled_pressure: ArrayLike = self.get_scaled_pressure(pressure)
        log10_fugacity_buffer: Array = (
            -0.8853 * jnp.log(temperature)
            - 28776.8 / temperature
            + 14.057
            + 0.055 * (scaled_pressure - 1) / temperature
        )

        return log10_fugacity_buffer


@register_pytree_node_class
class IronWustiteBufferHirschmann21(RedoxBuffer):
    """Iron-wustite buffer :cite:p:`H21`

    Regarding the calibration, :cite:t:`H21` states that: 'It extrapolates smoothly to higher
    temperature, though not calibrated above 3000 K. Extrapolation to lower temperatures (<1000 K)
    or higher pressures (>100 GPa) is not recommended.'

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
        evaluation_pressure: Pressure to evaluate the buffer at. Defaults to None, meaning use
            the total pressure which is passed in as an argument.

    Attributes:
        log10_shift: Log10 shift relative to the buffer
        evaluation_pressure: Pressure to evaluate the buffer at
    """

    @override
    def __init__(self, log10_shift: ArrayLike = 0, evaluation_pressure: ArrayLike | None = None):
        super().__init__(log10_shift, evaluation_pressure)
        self._calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
            temperature_min=1000, pressure_max=100 * unit_conversion.GPa_to_bar
        )
        self._a: tuple[float, ...] = (6.844864, 1.175691e-1, 1.143873e-3, 0, 0)
        self._b: tuple[float, ...] = (5.791364e-4, -2.891434e-4, -2.737171e-7, 0.0, 0.0)
        self._c: tuple[float, ...] = (-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7)
        self._d: tuple[float, ...] = (-2.769002e4, 5.285977e2, -2.919275, 0, 0)
        self._e: tuple[float, ...] = (8.463095, -3.000307e-3, 7.213445e-5, 0, 0)
        self._f: tuple[float, ...] = (1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0)
        self._g: tuple[float, ...] = (-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4)
        self._h: tuple[float, ...] = (-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0)

    @override
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Units are GPa"""
        return pressure * unit_conversion.bar_to_GPa

    @jit
    def _evaluate_m(self, pressure: ArrayLike, coefficients: tuple[float, ...]) -> Array:
        """Evaluates an m parameter

        Args:
            pressure: Pressure in GPa
            coefficients: Coefficients

        Return:
            m parameter
        """
        m: Array = (
            coefficients[4] * jnp.power(pressure, 0.5)
            + coefficients[0]
            + coefficients[1] * pressure
            + coefficients[2] * jnp.power(pressure, 2)
            + coefficients[3] * jnp.power(pressure, 3)
        )

        return m

    @jit
    def _evaluate_fO2(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        coefficients: tuple[tuple[float, ...], ...],
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

    @jit
    def _fcc_bcc_iron(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """log10fO2 for fcc and bcc iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for fcc and bcc iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self._a, self._b, self._c, self._d)
        )

        return log10fO2

    @jit
    def _hcp_iron(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """log10fO2 for hcp iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for hcp iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self._e, self._f, self._g, self._h)
        )

        return log10fO2

    @jit
    def _use_hcp(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Check to use hcp iron formulation for fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Returns:
            True/False whether to use the hcp iron formulation
        """
        x: tuple[float, ...] = (-18.64, 0.04359, -5.069e-6)
        threshold: Array = x[2] * jnp.power(temperature, 2) + x[1] * temperature + x[0]

        return jnp.array(pressure) > threshold

    @override
    @jit
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity
        """
        scaled_pressure: ArrayLike = self.get_scaled_pressure(pressure)

        def hcp_case() -> Array:
            return self._hcp_iron(temperature, scaled_pressure)

        def fcc_bcc_case() -> Array:
            return self._fcc_bcc_iron(temperature, scaled_pressure)

        buffer_value: Array = jnp.where(
            self._use_hcp(temperature, scaled_pressure), hcp_case(), fcc_bcc_case()
        )

        return buffer_value


@register_pytree_node_class
class IronWustiteBufferHirschmann(RedoxBuffer):
    """Composite iron-wustite buffer using :cite:t:`OP93,HGD08` and :cite:t:`H21`"""

    @override
    def __init__(
        self,
        log10_shift: ArrayLike = 0,
        evaluation_pressure: ArrayLike | None = None,
    ):
        super().__init__(log10_shift, evaluation_pressure)
        self._low_temperature_buffer: RedoxBufferProtocol = IronWustiteBufferHirschmann08(
            log10_shift, evaluation_pressure
        )
        self._high_temperature_buffer: RedoxBufferProtocol = IronWustiteBufferHirschmann21(
            log10_shift, evaluation_pressure
        )
        self._calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
            pressure_max=100 * unit_conversion.GPa_to_bar
        )

    # Not used for a composite redox buffer but required by the interface
    @override
    def convert_pressure_units(self, pressure: ArrayLike) -> ArrayLike:
        """Units are bar"""
        return pressure

    @jit
    def _use_low_temperature(self, temperature: ArrayLike) -> Array:
        """Check to use the low temperature buffer for fO2

        Args:
            temperature: Temperature

        Returns:
            True/False whether to use the low temperature formulation
        """
        return jnp.asarray(temperature) < self._high_temperature_buffer.calibration.temperature_min

    @override
    @jit
    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity at the buffer

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity at the buffer
        """

        def low_temperature_case() -> ArrayLike:
            return self._low_temperature_buffer.log10_fugacity_buffer(temperature, pressure)

        def high_temperature_case() -> ArrayLike:
            return self._high_temperature_buffer.log10_fugacity_buffer(temperature, pressure)

        # Apply the low or high temperature case functions element-wise using jnp.where
        buffer_value: Array = jnp.where(
            self._use_low_temperature(temperature),
            low_temperature_case(),
            high_temperature_case(),
        )

        return buffer_value


IronWustiteBuffer = IronWustiteBufferHirschmann
