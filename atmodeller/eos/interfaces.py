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
"""Interfaces for real gas equations of state"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
from jax import Array, lax
from jax.typing import ArrayLike

from atmodeller import GAS_CONSTANT, GAS_CONSTANT_BAR
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentalCalibration:
    """Experimental calibration range

    Args:
        temperature_min: Minimum temperature in K. Defaults to None (i.e. not specified).
        temperature_max: Maximum temperature in K. Defaults to None (i.e. not specified).
        pressure_min: Minimum pressure in bar. Defaults to None (i.e. not specified).
        pressure_max: Maximum pressure in bar. Defaults to None (i.e. not specified).
        temperature_penalty: Penalty coefficients for temperature. Defaults to 1000.
        pressure_penalty: Penalty coefficient for pressure. Defaults to 1000.
    """

    temperature_min: float | None = None
    """Minimum temperature in K"""
    temperature_max: float | None = None
    """Maximum temperature in K"""
    pressure_min: float | None = None
    """Minimum pressure in bar"""
    pressure_max: float | None = None
    """Maximum pressure in bar"""
    temperature_penalty: float = 1e3
    """Temperature penalty"""
    pressure_penalty: float = 1e3
    """Pressure penalty"""
    _clips_to_apply: list[Callable] = field(init=False, default_factory=list, repr=False)
    """Clips to apply"""

    def __post_init__(self):
        if self.temperature_min is not None:
            logger.info(
                "Set minimum evaluation temperature (temperature > %f)", self.temperature_min
            )
            self._clips_to_apply.append(self._clip_temperature_min)
        if self.temperature_max is not None:
            logger.info(
                "Set maximum evaluation temperature (temperature < %f)", self.temperature_max
            )
            self._clips_to_apply.append(self._clip_temperature_max)
        if self.pressure_min is not None:
            logger.info("Set minimum evaluation pressure (pressure > %f)", self.pressure_min)
            self._clips_to_apply.append(self._clip_pressure_min)
        if self.pressure_max is not None:
            logger.info("Set maximum evaluation pressure (pressure < %f)", self.pressure_max)
            self._clips_to_apply.append(self._clip_pressure_max)

    def _clip_pressure_max(self, temperature: float, pressure: ArrayLike) -> tuple[float, Array]:
        """Clips maximum pressure

        Args:
            temperature: Temperature in K
            pressure: pressure in bar

        Returns:
            Temperature, and clipped pressure
        """
        assert self.pressure_max is not None

        return temperature, jnp.minimum(pressure, jnp.array(self.pressure_max))

    def _clip_pressure_min(self, temperature: float, pressure: ArrayLike) -> tuple[float, Array]:
        """Clips minimum pressure

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Temperature, and clipped pressure
        """
        assert self.pressure_min is not None

        return temperature, jnp.maximum(pressure, jnp.array(self.pressure_min))

    def _clip_temperature_max(
        self, temperature: float, pressure: ArrayLike
    ) -> tuple[float, Array]:
        """Clips maximum temperature

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Clipped temperature, and pressure
        """
        assert self.temperature_max is not None

        return min(temperature, self.temperature_max), jnp.array(pressure)

    def _clip_temperature_min(
        self, temperature: float, pressure: ArrayLike
    ) -> tuple[float, Array]:
        """Clips minimum temperature

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Clipped temperature, and pressure
        """
        assert self.temperature_min is not None

        return max(temperature, self.temperature_min), jnp.array(pressure)

    def get_within_range(self, temperature: float, pressure: ArrayLike) -> tuple[float, ArrayLike]:
        """Gets temperature and pressure conditions within the calibration range.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            temperature in K, pressure in bar, according to prescribed clips
        """
        for clip_func in self._clips_to_apply:
            temperature, pressure = clip_func(temperature, pressure)

        return temperature, pressure

    def get_penalty(self, temperature: float, pressure: ArrayLike) -> Array:
        """Gets a penalty value if temperature and pressure are outside the calibration range

        This is based on the quadratic penalty method.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            A penalty value
        """
        pressure_: Array = jnp.asarray(pressure)
        temperature_clip, pressure_clip = self.get_within_range(temperature, pressure)
        penalty = self.pressure_penalty * jnp.power(
            pressure_clip - pressure_, 2
        ) + self.temperature_penalty * jnp.power(temperature_clip - temperature, 2)

        return penalty


@dataclass(kw_only=True)
class RealGas(ABC):
    r"""A real gas equation of state (EOS)

    Fugacity is computed using the standard relation:

    .. math::

        R T \ln f = \int V dP

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`f` is fugacity, :math:`V`
    is volume, and :math:`P` is pressure.

    Args:
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    calibration: ExperimentalCalibration = field(
        default_factory=ExperimentalCalibration, repr=False
    )
    """Calibration range of the temperature and pressure"""

    def compressibility_factor(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
        """Compressibility parameter

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility parameter, which is dimensionless
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        volume_ideal: ArrayLike = self.ideal_volume(temperature, pressure)
        compressibility_factor: ArrayLike = volume / volume_ideal

        return compressibility_factor

    def ln_fugacity(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
        """Natural log of the fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Natural log of the fugacity
        """
        ln_fugacity: ArrayLike = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: ArrayLike) -> Array:
        """Fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: Array = jnp.exp(self.ln_fugacity(temperature, pressure))
        logger.debug("fugacity = %s", fugacity)

        return fugacity

    def ln_fugacity_coefficient(self, temperature: float, pressure: ArrayLike) -> Array:
        """Natural log of the fugacity coefficient

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Natural log of the fugacity coefficient
        """
        return -jnp.log(pressure) + self.ln_fugacity(temperature, pressure)

    def fugacity_coefficient(self, temperature: float, pressure: ArrayLike) -> Array:
        """Fugacity coefficient

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            fugacity coefficient, which is non-dimensional
        """
        return jnp.exp(self.ln_fugacity_coefficient(temperature, pressure))

    def ideal_volume(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
        r"""Ideal volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Ideal volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        volume_ideal: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure

        return volume_ideal

    @abstractmethod
    def volume(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume integral

        .. math::

            \int V dP

        This method requires that the units returned are :math:`\mathrm{J}\mathrm{mol}^{-1}`. Hence
        the following conversion is often necessary:

        .. math::

            1\ \mathrm{J} = 10^{-5}\ \mathrm{m}^3\mathrm{bar}

        There are functions to do this conversion in :class:`atmodeller.utilities.UnitConversion`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


@dataclass(kw_only=True)
class CorrespondingStatesMixin(ABC):
    """A corresponding states model

    Args:
        critical_temperature: Critical temperature in K. Defaults to 1, effectively meaning that
            the scaled temperature is numerically the same as the actual temperature, albeit
            without units.
        critical_pressure: Critical pressure in bar. Defaults to 1, effectively meaning that the
            scale pressure is numerically the same as the actual pressure, albeit without units.
    """

    critical_temperature: float = 1
    """Critical temperature in K"""
    critical_pressure: float = 1
    """Critical pressure in bar"""

    # Required for Saxena but not for Holland and Powell
    def scaled_pressure(self, pressure: ArrayLike) -> ArrayLike:
        """Scaled pressure

        This is a reduced pressure when :attr:`critical_pressure` is not unity.

        Args:
            pressure: Pressure in bar

        Returns:
            The scaled (reduced) pressure, which is dimensionless
        """
        scaled_pressure: ArrayLike = pressure / self.critical_pressure

        return scaled_pressure

    # Required for Saxena but not for Holland and Powell
    def scaled_temperature(self, temperature: float) -> float:
        """Scaled temperature

        This is a reduced temperature when :attr:`critical_temperature` is not unity.

        Args:
            temperature: Temperature in K

        Returns:
            The scaled (reduced) temperature, which is dimensionless
        """
        scaled_temperature: float = temperature / self.critical_temperature

        return scaled_temperature


@dataclass(kw_only=True)
class VirialCompensation(CorrespondingStatesMixin, RealGas):
    r"""A virial compensation term for the increasing deviation of the MRK volumes with pressure

    General form of the equation :cite:t:`HP98` and also see :cite:t:`HP91{Equations 4 and 9}`:

    .. math::

        V_\mathrm{virial} = a(P-P0) + b(P-P0)^\frac{1}{2} + c(P-P0)^\frac{1}{4}

    This form also works for the virial compensation term from :cite:t:`HP91`, in which
    case :math:`c=0`. :attr:`critical_pressure` and :attr:`critical_temperature` are required for
    gases which are known to obey approximately the principle of corresponding states.

    Although this looks similar to an EOS, it only calculates an additional perturbation to the
    volume and the volume integral of an MRK EOS, and hence it does not return a meaningful volume
    or volume integral by itself.

    Args:
        a_coefficients: Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`, where
            :math:`a_0` and :math:`a_1` may be scaled (internally) by critical parameters for
            corresponding states.
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly and may be determined from experimental data.
        critical_temperature: Critical temperature in K. Defaults to unity meaning not a
            corresponding states model.
        critical_pressure: Critical pressure in bar. Defaults to unity meaning not a corresponding
            states model.
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    a_coefficients: Array
    r"""Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`, where :math:`a_0` and
    :math:`a_1` may be additionally (internally) scaled by critical parameters
    (:attr:`critical_temperature` and :attr:`critical_pressure`) for corresponding states."""
    b_coefficients: Array
    """Coefficients for the `b` parameter. See :attr:`a_coefficients` documentation."""
    c_coefficients: Array
    """Coefficients for the `c` parameter. See :attr:`a_coefficients` documentation."""
    P0: ArrayLike
    """Pressure at which the MRK equation begins to overestimate the molar volume significantly 
    and may be determined from experimental data."""

    def a(self, temperature: float) -> Array:
        r"""`a` parameter :cite:p:`HP98`

        This is also the `d` parameter in :cite:t:`HP91`.

        Args:
            temperature: Temperature in K

        Returns:
            `a` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}\mathrm{bar}^{-1}`
        """
        a: Array = (
            self.a_coefficients[0] * self.critical_temperature
            + self.a_coefficients[1] * temperature
        )
        a = a / self.critical_pressure**2

        return a

    def b(self, temperature: float) -> Array:
        r"""`b` parameter :cite:p:`HP98`

        This is also the `c` parameter in :cite:t:`HP91`.

        Args:
            temperature: Temperature in K

        Returns:
            `b` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}\mathrm{bar}^\frac{-1}{2}`
        """
        b: Array = (
            self.b_coefficients[0] * self.critical_temperature
            + self.b_coefficients[1] * temperature
        )
        b = b / self.critical_pressure ** (3 / 2)

        return b

    def c(self, temperature: float) -> Array:
        r"""`c` parameter :cite:p:`HP98`

        Args:
            temperature: Temperature in K

        Returns:
            `c` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}\mathrm{bar}^\frac{-1}{4}`
        """
        c: Array = (
            self.c_coefficients[0] * self.critical_temperature
            + self.c_coefficients[1] * temperature
        )
        c = c / self.critical_pressure ** (5 / 4)

        return c

    def delta_pressure(self, pressure: ArrayLike) -> Array:
        """Pressure difference

        Args:
            pressure: Pressure in bar

        Returns:
            Pressure difference relative to :attr:`P0`
        """
        # TODO: Careful. Maybe jnp.where is better for array-based operations
        delta_pressure: Array = lax.cond(
            jnp.asarray(pressure) > jnp.asarray(self.P0),
            lambda pressure: jnp.asarray(pressure - self.P0, dtype=jnp.float_),
            lambda pressure: jnp.array(0, dtype=jnp.float_),
            pressure,
        )

        return delta_pressure

    def volume(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume contribution

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume contribution in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        delta_pressure: Array = self.delta_pressure(pressure)
        volume: Array = (
            self.a(temperature) * delta_pressure
            + self.b(temperature) * delta_pressure**0.5
            + self.c(temperature) * delta_pressure**0.25
        )

        return volume

    def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume integral :math:`V dP` contribution

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral contribution in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        delta_pressure: Array = self.delta_pressure(pressure)
        volume_integral: Array = (
            self.a(temperature) / 2.0 * delta_pressure**2
            + 2.0 / 3.0 * self.b(temperature) * delta_pressure ** (3.0 / 2.0)
            + 4.0 / 5.0 * self.c(temperature) * delta_pressure ** (5.0 / 4.0)
        )
        volume_integral = volume_integral * unit_conversion.m3_bar_to_J

        return volume_integral


@dataclass(kw_only=True)
class CORK(CorrespondingStatesMixin, RealGas):
    """A Compensated-Redlich-Kwong (CORK) EOS :cite:p:`HP91`

    Args:
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly and may be determined from experimental data.
        mrk: MRK model for computing the MRK contribution
        a_virial: `a` coefficients for the virial compensation. Defaults to zero.
        b_virial: `b` coefficients for the virial compensation. Defaults to zero.
        c_virial: `c` coefficients for the virial compensation. Defaults to zero.
        critical_temperature: Critical temperature in K. Defaults to unity meaning not a
            corresponding states model.
        critical_pressure: Critical pressure in bar. Defaults to unity meaning not a corresponding
            states model.
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    P0: ArrayLike
    """Pressure at which the MRK equation begins to overestimate the molar volume significantly 
    and may be determined from experimental data."""
    mrk: RealGas
    """MRK model for computing the MRK contribution"""
    a_virial: Array = jnp.array((0, 0))
    """`a` coefficients for the virial compensation"""
    b_virial: Array = jnp.array((0, 0))
    """`b` coefficients for the virial compensation"""
    c_virial: Array = jnp.array((0, 0))
    """`c` coefficients for the virial compensation"""
    virial: VirialCompensation = field(init=False)
    """The virial compensation model"""

    def __post_init__(self):
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            P0=self.P0,
            critical_temperature=self.critical_temperature,
            critical_pressure=self.critical_pressure,
        )

    @override
    def volume(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume :cite:p:`HP91{Equation 7a}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        return self.virial.volume(temperature, pressure) + self.mrk.volume(temperature, pressure)

    @override
    def volume_integral(self, temperature: float, pressure: float) -> Array:
        r"""Volume integral :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        return self.virial.volume_integral(temperature, pressure) + self.mrk.volume_integral(
            temperature, pressure
        )


@dataclass(kw_only=True)
class CombinedEOSModel(RealGas):
    """Combines multiple EOS models for different pressure ranges into a single EOS model.

    Args:
        models: EOS models ordered by increasing pressure from lowest to highest
        upper_pressure_bounds: Upper pressure bound in bar relevant to the EOS by position
    """

    models: tuple[RealGas, ...]
    """EOS models ordered by increasing pressure from lowest to highest"""
    upper_pressure_bounds: Array
    """Upper pressure bound in bar relevant to the EOS by position"""

    def _get_index(self, pressure: ArrayLike) -> int:
        """Gets the index of the appropriate EOS model using the upper pressure bound

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """

        def body_fun(i: int, carry: int) -> int:
            pressure_high: Array = self.upper_pressure_bounds[i]
            condition = pressure >= pressure_high
            # TODO: Careful. Maybe jnp.where is better for array-based operations
            new_index: int = lax.cond(condition, lambda _: i + 1, lambda _: carry, None)

            return new_index

        init_carry: int = 0  # Initial carry value
        index = lax.fori_loop(0, len(self.upper_pressure_bounds), body_fun, init_carry)

        return index

    @override
    def volume(self, temperature: float, pressure: ArrayLike) -> Array:
        index = self._get_index(pressure)
        volume: Array = lax.switch(
            index, [model.volume for model in self.models], temperature, pressure
        )
        return volume

    @override
    def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)

        def compute_integral(i: int, pressure_high: ArrayLike, pressure_low: ArrayLike) -> Array:
            """Compute pressure integral."""
            return self.models[i].volume_integral(temperature, pressure_high) - self.models[
                i
            ].volume_integral(temperature, pressure_low)

        def case_0() -> Array:
            return self.models[0].volume_integral(temperature, pressure)

        def case_1() -> Array:
            volume0 = self.models[0].volume_integral(temperature, self.upper_pressure_bounds[0])
            dvolume = compute_integral(1, pressure, self.upper_pressure_bounds[0])
            return volume0 + dvolume

        def case_2() -> Array:
            volume0 = self.models[0].volume_integral(temperature, self.upper_pressure_bounds[0])
            dvolume0 = compute_integral(
                1, self.upper_pressure_bounds[1], self.upper_pressure_bounds[0]
            )
            dvolume1 = compute_integral(2, pressure, self.upper_pressure_bounds[1])
            return volume0 + dvolume0 + dvolume1

        return lax.switch(index, [case_0, case_1, case_2])
