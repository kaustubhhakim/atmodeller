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

# Use symbols from the relevant papers for consistency so pylint: disable=C0103

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, lax
from jax.typing import ArrayLike
from numpy.polynomial.polynomial import Polynomial

from atmodeller import GAS_CONSTANT, GAS_CONSTANT_BAR
from atmodeller.utilities import ExperimentalCalibration, unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class RealGasProtocol(Protocol):

    def fugacity_coefficient(self, temperature: float, pressure: ArrayLike) -> ArrayLike: ...


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

    def compressibility_parameter(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
        """Compressibility parameter

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility parameter, which is dimensionless
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        volume_ideal: ArrayLike = self.ideal_volume(temperature, pressure)
        compressibility_parameter: ArrayLike = volume / volume_ideal

        return compressibility_parameter

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
class IdealGas(RealGas):
    r"""An ideal gas equation of state:

    .. math::

        R T = P V

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`P` is pressure, and
    :math:`V` is volume.

    Args:
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    @override
    def fugacity_coefficient(self, *args, **kwargs) -> Array:
        """Fugacity coefficient

        Since we know that the fugacity coefficient of an ideal gas is unity by definition we
        simply impose it here to pass a constant to the solver. This seems to avoid some NaN
        crashes with the Optimistix solver.
        """
        del args
        del kwargs
        return jnp.array(1)

    @override
    def volume(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
        return self.ideal_volume(temperature, pressure)

    @override
    def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
        volume_integral: Array = GAS_CONSTANT_BAR * temperature * jnp.log(pressure)
        volume_integral = volume_integral * unit_conversion.m3_bar_to_J

        return volume_integral


@dataclass(kw_only=True)
class ModifiedRedlichKwongABC(RealGas):
    r"""A Modified Redlich Kwong (MRK) equation of state :cite:p:`{e.g.}HP91{Equation 3}`:

    .. math::

        P = \frac{RT}{V-b} - \frac{a(T)}{V(V+b)\sqrt{T}}

    where :math:`P` is pressure, :math:`T` is temperature, :math:`V` is the molar volume, :math:`R`
    the gas constant, :math:`a(T)` is the Redlich-Kwong function of :math:`T`, and :math:`b` is the
    Redlich-Kwong constant.

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) `a` parameter
        b0: The Redlich-Kwong constant `b`
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    a_coefficients: Array
    """Coefficients for the Modified Redlich Kwong (MRK) `a` parameter"""
    b0: float
    """The Redlich-Kwong constant `b`"""

    @abstractmethod
    def a(self, temperature: float) -> Array:
        r"""MRK `a` parameter computed from :attr:`a_coefficients`.

        Args:
            temperature: Temperature in K

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self) -> float:
        r"""MRK `b` parameter computed from :attr:`b0`

        Units are :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class MRKExplicitABC(CorrespondingStatesMixin, ModifiedRedlichKwongABC):
    """A Modified Redlich Kwong (MRK) EOS in explicit form"""

    @override
    def a(self, temperature: float) -> Array:
        r"""MRK `a` parameter from :attr:`a_coefficients` :cite:p:`HP91{Equation 9}`

        Args:
            temperature: Temperature in K

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        a: Array = (
            self.a_coefficients[0] * self.critical_temperature ** (5.0 / 2)
            + self.a_coefficients[1] * self.critical_temperature ** (3.0 / 2) * temperature
            + self.a_coefficients[2] * self.critical_temperature ** (1.0 / 2) * temperature**2
        )
        a = a / self.critical_pressure

        return a

    @property
    def b(self) -> float:
        r"""MRK `b` parameter computed from :attr:`b0` :cite:p:`HP91{Equation 9}`.

        Units are :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        b: float = self.b0 * self.critical_temperature / self.critical_pressure

        return b

    @override
    def volume(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume-explicit equation :cite:p:`HP91{Equation 7}`

        Without complications of critical phenomena the MRK equation can be simplified using the
        approximation:

        .. math::

            V \sim \frac{RT}{P} + b

        where :math:`V` is volume, :math:`R` is the gas constant, :math:`T` is temperature,
        :math:`P` is pressure, and :math:`b` is :attr:`b`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            MRK volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        volume: Array = (
            jnp.sqrt(temperature)
            * -self.a(temperature)
            * GAS_CONSTANT_BAR
            / (GAS_CONSTANT_BAR * temperature + self.b * pressure)
            / (GAS_CONSTANT_BAR * temperature + 2.0 * self.b * pressure)
            + GAS_CONSTANT_BAR * temperature / pressure
            + self.b
        )

        return volume

    @override
    def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume-explicit integral :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        volume_integral: Array = (
            GAS_CONSTANT_BAR * temperature * jnp.log(pressure)
            + self.b * pressure
            + self.a(temperature)
            / self.b
            / jnp.sqrt(temperature)
            * (
                jnp.log(GAS_CONSTANT_BAR * temperature + self.b * pressure)
                - jnp.log(GAS_CONSTANT_BAR * temperature + 2.0 * self.b * pressure)
            )
        )
        volume_integral = volume_integral * unit_conversion.m3_bar_to_J

        return volume_integral


# TODO: Update to support JAX
@dataclass(kw_only=True)
class MRKImplicitABC(ModifiedRedlichKwongABC):
    """A Modified Redlich Kwong (MRK) EOS in implicit form

    Args:
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    @override
    def a(self, temperature: float) -> Array:
        r"""MRK `a` parameter from :attr:`a_coefficients` :cite:p:`HP91{Equation 6}`

        Args:
            temperature: Temperature in K

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        a: Array = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * self.delta_temperature_for_a(temperature)
            + self.a_coefficients[2] * self.delta_temperature_for_a(temperature) ** 2
            + self.a_coefficients[3] * self.delta_temperature_for_a(temperature) ** 3
        )

        return a

    @abstractmethod
    def delta_temperature_for_a(self, temperature: float) -> float:
        """Temperature difference for the calculation of the `a` parameter

        Args:
            temperature: Temperature in K

        Returns:
            A temperature difference
        """

    @property
    def b(self) -> float:
        """MRK `b` parameter computed from :attr:`b0`.

        :class:`~MRKImplicitABC` is not used for corresponding states models so :attr:`b0` is the
        `b` coefficient.
        """
        return self.b0

    def A_factor(self, temperature: float, pressure: ArrayLike) -> Array:
        """`A` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `A` factor, which is non-dimensional
        """
        del pressure
        A_factor: Array = self.a(temperature) / (self.b * GAS_CONSTANT_BAR * temperature**1.5)

        return A_factor

    def B_factor(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
        """`B` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `B` factor, which is non-dimensional
        """
        B_factor: ArrayLike = self.b * pressure / (GAS_CONSTANT_BAR * temperature)

        return B_factor

    @override
    def volume_integral(
        self,
        temperature: float,
        pressure: ArrayLike,
    ) -> Array:
        r"""Volume integral :cite:p:`HP91{Equation A.2}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        z: ArrayLike = self.compressibility_parameter(temperature, pressure)
        A: Array = self.A_factor(temperature, pressure)
        B: ArrayLike = self.B_factor(temperature, pressure)
        # The base class requires a specification of the volume_integral, but the equations are in
        # terms of the fugacity coefficient.
        ln_fugacity_coefficient: Array = -jnp.log(z - B) - A * jnp.log(1 + B / z) + z - 1  # type: ignore
        ln_fugacity: Array = jnp.log(pressure) + ln_fugacity_coefficient
        volume_integral: Array = GAS_CONSTANT_BAR * temperature * ln_fugacity
        volume_integral = volume_integral * unit_conversion.m3_bar_to_J

        return volume_integral

    # TODO: Needs updating to support JAX. Can use optimistix root finder, but then need to tailor
    # the initial guesses to find the correct root

    def volume_roots(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Real and (potentially) physically meaningful volume solutions of the MRK equation

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume solutions of the MRK equation in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        coefficients: list[float] = []
        coefficients.append(-self.a(temperature) * self.b / np.sqrt(temperature))
        coefficients.append(
            -self.b * GAS_CONSTANT_BAR * temperature
            - self.b**2 * pressure
            + self.a(temperature) / np.sqrt(temperature)
        )
        coefficients.append(-GAS_CONSTANT_BAR * temperature)
        coefficients.append(pressure)

        polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="V")
        logger.debug("MRK equation = %s", polynomial)
        volume_roots: npt.NDArray = polynomial.roots()
        # Numerical solution could result in a small imaginery component, even though the root is
        # real.
        real_roots: npt.NDArray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
        # Physically meaningful volumes must be positive.
        positive_roots: npt.NDArray = real_roots[real_roots > 0]
        # In general, several roots could be returned, and subclasses will need to determine which
        # is the correct volume to use depending on the phase (liquid, gas, etc.)
        logger.debug("V = %s", positive_roots)

        return positive_roots


# TODO: Update to support JAX
@dataclass(kw_only=True)
class MRKCriticalBehaviour(RealGas):
    r"""A MRK equation of state that accommodates critical behaviour :cite:p:`HP91{Appendix A}`

    Args:
        mrk_fluid: MRK EOS for the supercritical fluid
        mrk_gas: MRK EOS for the subcritical gas
        mrk_liquid: MRK EOS for the subcritical liquid
        Ta: Temperature at which :math:`a_{\mathrm gas} = a` by constrained fitting :cite:p:`HP91`.
        calibration: Calibration temperature and pressure range. Defaults to empty.
    """

    mrk_fluid: MRKImplicitABC
    """MRK EOS for the supercritical fluid"""
    mrk_gas: MRKImplicitABC
    """MRK EOS for the subcritical gas"""
    mrk_liquid: MRKImplicitABC
    """MRK EOS for the subcritical liquid"""
    Ta: float
    r"""Temperature at which :math:`a_{\mathrm gas} = a` by constrained fitting"""

    @abstractmethod
    def Psat(self, temperature: float) -> float:
        """Saturation curve :cite:p:`{e.g.}HP91{Equation 5}`

        Args:
            temperature: Temperature in K

        Returns:
            Saturation curve pressure in bar
        """

    @override
    def volume(self, temperature: float, pressure: float) -> float:
        Psat: float = self.Psat(temperature)

        if temperature <= self.Ta and pressure <= Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Gas phase")
            volume = self.mrk_gas.volume(temperature, pressure)

        elif temperature <= self.Ta and pressure > Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Liquid phase")
            volume = self.mrk_liquid.volume(temperature, pressure)

        else:
            logger.debug("Fluid phase")
            volume = self.mrk_fluid.volume(temperature, pressure)

        return volume

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        Psat: float = self.Psat(temperature)

        if temperature <= self.Ta and pressure <= Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            # logger.debug("Gas phase")
            volume_integral = self.mrk_gas.volume_integral(temperature, pressure)

        elif temperature <= self.Ta and pressure > Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            # logger.debug("Performing pressure integration")
            volume_integral = self.mrk_gas.volume_integral(temperature, Psat)
            volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
            volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)

        else:
            # logger.debug("Fluid phase")
            volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        return volume_integral


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


@dataclass(frozen=True)
class CriticalData:
    """Critical temperature and pressure of a gas species.

    Args:
        temperature: Critical temperature in K
        pressure: Critical pressure in bar
    """

    temperature: float
    """Critical temperature in K"""
    pressure: float
    """Critical pressure in bar"""


critical_parameters_H2O: CriticalData = CriticalData(647.25, 221.1925)
"""Critical parameters for H2O :cite:p:`SS92{Table 2}`"""
critical_parameters_CO2: CriticalData = CriticalData(304.15, 73.8659)
"""Critical parameters for CO2 :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 304.2 K and 73.8 bar
"""
critical_parameters_CH4: CriticalData = CriticalData(191.05, 46.4069)
"""Critical parameters for CH4 :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 190.6 K and 46 bar
"""
critical_parameters_CO: CriticalData = CriticalData(133.15, 34.9571)
"""Critical parameters for CO :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 132.9 K and 35 bar
"""
critical_parameters_O2: CriticalData = CriticalData(154.75, 50.7638)
"""Critical parameters for O2 :cite:p:`SS92{Table 2}`"""
critical_parameters_H2: CriticalData = CriticalData(33.25, 12.9696)
"""Critical parameters for H2 :cite:p:`SS92{Table 2}`"""
critical_parameters_H2_holland: CriticalData = CriticalData(41.2, 21.1)
"""Critical parameters for H2 :cite:p:`HP91`"""
critical_parameters_S2: CriticalData = CriticalData(208.15, 72.954)
"""Critical parameters for S2 :cite:p:`SS92{Table 2}`

http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

:cite:p:`HP11` state that the critical parameters are from :cite:t:`RPS77`. However, in the fifth
edition of this book (:cite:t:`PPO00`) S2 is not given (only S is).
"""
critical_parameters_SO2: CriticalData = CriticalData(430.95, 78.7295)
"""Critical parameters for SO2 :cite:p:`SS92{Table 2}`"""
critical_parameters_COS: CriticalData = CriticalData(377.55, 65.8612)
"""Critical parameters for COS :cite:p:`SS92{Table 2}`"""
critical_parameters_H2S: CriticalData = CriticalData(373.55, 90.0779)
"""Critical parameters for H2S :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 373.4 K and 0.08963 bar
"""
critical_parameters_N2: CriticalData = CriticalData(126.2, 33.9)
"""Critical parameters for N2 :cite:p:`SF87{Table 1}`"""
critical_parameters_Ar: CriticalData = CriticalData(151, 48.6)
"""Critical parameters for Ar :cite:p:`SF87{Table 1}`"""

critical_parameters: dict[str, CriticalData] = {
    "Ar": critical_parameters_Ar,
    "CH4": critical_parameters_CH4,
    "CO": critical_parameters_CO,
    "CO2": critical_parameters_CO2,
    "COS": critical_parameters_COS,
    "H2": critical_parameters_H2,
    "H2_Holland": critical_parameters_H2_holland,
    "H2O": critical_parameters_H2O,
    "H2S": critical_parameters_H2S,
    "N2": critical_parameters_N2,
    "O2": critical_parameters_O2,
    "S2": critical_parameters_S2,
    "SO2": critical_parameters_SO2,
}
"""Critical parameters for gases

These critical data could be extended to more species using :cite:t:`PPO00{Appendix A.19}`
"""
