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
"""Core classes and functions for real gas equations of state

Units for temperature and pressure are K and bar, respectively.
"""

# Use symbols from the relevant papers for consistency so pylint: disable=C0103

import logging
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array, grad, jit, lax
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from numpy.polynomial.polynomial import Polynomial

from atmodeller import GAS_CONSTANT, GAS_CONSTANT_BAR, PRESSURE_REFERENCE
from atmodeller.interfaces import ActivityProtocol, RealGasProtocol
from atmodeller.utilities import (
    ExperimentalCalibrationNew,
    PyTreeNoData,
    safe_exp,
    unit_conversion,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


class RealGas(ABC, RealGasProtocol):
    r"""A real gas equation of state (EOS)

    Fugacity is computed using the standard relation:

    .. math::

        R T \ln f = \int V dP

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`f` is fugacity, :math:`V`
    is volume, and :math:`P` is pressure.
    """

    @abstractmethod
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity in bar
        """

    @abstractmethod
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """

    @jit
    def dvolume_dpressure(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Derivative of volume with respect to pressure

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Derivative of volume with respect to pressure
        """
        dvolume_dpressure_fn: Callable = grad(self.volume, argnums=1)

        return dvolume_dpressure_fn(temperature, pressure)

    @jit
    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Log activity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log activity
        """
        # The standard state is defined at 1 bar (see PRESSURE_REFERENCE), so we do not need to
        # perform a division (by unity) to get activity, which is non-dimensional.

        return self.log_fugacity(temperature, pressure)

    @jit
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Compressibility factor

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

    @jit
    def fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: Array = safe_exp(self.log_fugacity(temperature, pressure))

        return fugacity

    @jit
    def log_fugacity_coefficient(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log of the fugacity coefficient

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log of the fugacity coefficient, which is dimensionless
        """
        return self.log_fugacity(temperature, pressure) - self.ideal_log_fugacity(
            temperature, pressure
        )

    @jit
    def fugacity_coefficient(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Fugacity coefficient

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            fugacity coefficient, which is non-dimensional
        """
        return safe_exp(self.log_fugacity_coefficient(temperature, pressure))

    @jit
    def ideal_log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity of an ideal gas

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity of an ideal gas
        """
        del temperature
        ideal_log_fugacity: ArrayLike = jnp.log(pressure)

        return ideal_log_fugacity

    @jit
    def ideal_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Volume of an ideal gas

        This is required to compute the compressibility parameter.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Ideal volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure

        return ideal_volume


# TODO: Eventually this will probably utilise a class that assembles different equations of state
# for given pressure ranges.
@register_pytree_node_class
class RealGasBounded(RealGas):
    """A real gas equation of state that is bounded

    Args:
        real_gas: Real gas equation of state to bound
        calibration: Calibration that encodes the bounds
    """

    def __init__(
        self,
        real_gas: RealGas,
        calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(),
    ):
        super().__init__()
        self._real_gas = real_gas
        self._calibration: ExperimentalCalibrationNew = calibration
        self._pressure_min: Array = jnp.array(calibration.pressure_min)
        self._pressure_max: Array = jnp.array(calibration.pressure_max)

    @property
    def calibration(self) -> ExperimentalCalibrationNew:
        """Experimental calibration"""
        return self._calibration

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity that is bounded

        Outside of the experimental calibration there is no guarantee that the real gas fugacity
        is sensible and it is difficult a priori to predict how reasonable the extrapolation will
        be. Therefore, below the minimum calibration pressure we assume an ideal gas, and above the
        calibration pressure we assume a fixed fugacity coefficient determined at the maximum
        pressure. This maintains relatively smooth behaviour of the function beyond the calibrated
        values, which is often required to guide the solver.

        This method could also implement a bound on temperature, if eventually required or desired.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        pressure_clipped: Array = jnp.clip(pressure, self._pressure_min, self._pressure_max)
        # jax.debug.print("pressure_clipped = {out}", out=pressure_clipped)

        # Calculate log fugacity in different regions
        log_fugacity_below: ArrayLike = self.ideal_log_fugacity(temperature, pressure_clipped)
        # jax.debug.print("log_fugacity_below = {out}", out=log_fugacity_below)
        # Evaluating the function within bounds helps to ensure a more stable numerical solution
        log_fugacity_in_range: ArrayLike = self._real_gas.log_fugacity(
            temperature, pressure_clipped
        )
        # jax.debug.print("log_fugacity_in_range = {out}", out=log_fugacity_in_range)
        log_fugacity_at_Pmax: ArrayLike = self._real_gas.log_fugacity(
            temperature, self._pressure_max
        )
        # jax.debug.print("log_fugacity_at_Pmax = {out}", out=log_fugacity_at_Pmax)

        # Compute the difference in volume relative to ideal at the maximum calibration pressure
        dvolume: ArrayLike = self._real_gas.volume(
            temperature, self._pressure_max
        ) - self.ideal_volume(temperature, self._pressure_max)

        # VdP taking account of the extrapolated volume change above the calibration pressure.
        log_fugacity_above: ArrayLike = (
            log_fugacity_at_Pmax
            + jnp.log(pressure / self._pressure_max)
            + dvolume * (pressure - self._pressure_max) / (GAS_CONSTANT_BAR * temperature)
        )

        log_fugacity = lax.select(
            pressure > self._pressure_min, log_fugacity_in_range, log_fugacity_below
        )
        # FIXME: Commented out for testing
        # log_fugacity = lax.select(pressure < self._pressure_max, log_fugacity, log_fugacity_above)

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        pressure_clipped: Array = jnp.clip(pressure, self._pressure_min, self._pressure_max)

        # Calculate volume in different regions
        ideal_volume: ArrayLike = self.ideal_volume(temperature, pressure)
        # jax.debug.print("volume_below = {out}", out=volume_below)
        # Evaluate the function within bounds to help to ensure a more stable numerical solution
        volume_in_range: ArrayLike = self._real_gas.volume(temperature, pressure_clipped)

        # Compute the difference in volume relative to ideal at the maximum calibration pressure
        dvolume: ArrayLike = self._real_gas.volume(
            temperature, self._pressure_max
        ) - self.ideal_volume(temperature, self._pressure_max)

        # Determine the appropriate volume based on pressure ranges
        volume = lax.select(pressure > self._pressure_min, volume_in_range, ideal_volume)
        volume = lax.select(pressure < self._pressure_max, volume, ideal_volume + dvolume)

        return volume

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {"real_gas": self._real_gas, "calibration": self._calibration}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


# @dataclass(kw_only=True)
# class CorrespondingStatesMixin(ABC):
#     """A corresponding states model

#     Args:
#         critical_temperature: Critical temperature in K. Defaults to 1, effectively meaning that
#             the scaled temperature is numerically the same as the actual temperature, albeit
#             without units.
#         critical_pressure: Critical pressure in bar. Defaults to 1, effectively meaning that the
#             scale pressure is numerically the same as the actual pressure, albeit without units.
#     """

#     critical_temperature: float = 1
#     """Critical temperature in K"""
#     critical_pressure: float = 1
#     """Critical pressure in bar"""

#     def scaled_pressure(self, pressure: ArrayLike) -> ArrayLike:
#         """Scaled pressure

#         This is a reduced pressure when :attr:`critical_pressure` is not unity.

#         Args:
#             pressure: Pressure in bar

#         Returns:
#             The scaled (reduced) pressure, which is dimensionless
#         """
#         scaled_pressure: ArrayLike = pressure / self.critical_pressure

#         return scaled_pressure

#     def scaled_temperature(self, temperature: float) -> float:
#         """Scaled temperature

#         This is a reduced temperature when :attr:`critical_temperature` is not unity.

#         Args:
#             temperature: Temperature in K

#         Returns:
#             The scaled (reduced) temperature, which is dimensionless
#         """
#         scaled_temperature: float = temperature / self.critical_temperature

#         return scaled_temperature


# @dataclass(kw_only=True)
# class ModifiedRedlichKwongABC(RealGas):
#     r"""A Modified Redlich Kwong (MRK) equation of state :cite:p:`{e.g.}HP91{Equation 3}`:

#     .. math::

#         P = \frac{RT}{V-b} - \frac{a(T)}{V(V+b)\sqrt{T}}

#     where :math:`P` is pressure, :math:`T` is temperature, :math:`V` is the molar volume, :math:`R`
#     the gas constant, :math:`a(T)` is the Redlich-Kwong function of :math:`T`, and :math:`b` is the
#     Redlich-Kwong constant.

#     Args:
#         a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) `a` parameter
#         b0: The Redlich-Kwong constant `b`
#         calibration: Calibration temperature and pressure range. Defaults to empty.
#     """

#     a_coefficients: Array
#     """Coefficients for the Modified Redlich Kwong (MRK) `a` parameter"""
#     b0: float
#     """The Redlich-Kwong constant `b`"""

#     @abstractmethod
#     def a(self, temperature: float) -> Array:
#         r"""MRK `a` parameter computed from :attr:`a_coefficients`.

#         Args:
#             temperature: Temperature in K

#         Returns:
#             MRK `a` parameter in
#             :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
#         """
#         raise NotImplementedError

#     @property
#     @abstractmethod
#     def b(self) -> float:
#         r"""MRK `b` parameter computed from :attr:`b0`

#         Units are :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
#         """
#         raise NotImplementedError


# @dataclass(kw_only=True)
# class MRKExplicitABC(CorrespondingStatesMixin, ModifiedRedlichKwongABC):
#     """A Modified Redlich Kwong (MRK) EOS in explicit form"""

#     @override
#     def a(self, temperature: float) -> Array:
#         r"""MRK `a` parameter from :attr:`a_coefficients` :cite:p:`HP91{Equation 9}`

#         Args:
#             temperature: Temperature in K

#         Returns:
#             MRK `a` parameter in
#             :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
#         """
#         a: Array = (
#             self.a_coefficients[0] * self.critical_temperature ** (5.0 / 2)
#             + self.a_coefficients[1] * self.critical_temperature ** (3.0 / 2) * temperature
#             + self.a_coefficients[2] * self.critical_temperature ** (1.0 / 2) * temperature**2
#         )
#         a = a / self.critical_pressure

#         return a

#     @property
#     def b(self) -> float:
#         r"""MRK `b` parameter computed from :attr:`b0` :cite:p:`HP91{Equation 9}`.

#         Units are :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
#         """
#         b: float = self.b0 * self.critical_temperature / self.critical_pressure

#         return b

#     @override
#     def volume(self, temperature: float, pressure: ArrayLike) -> Array:
#         r"""Volume-explicit equation :cite:p:`HP91{Equation 7}`

#         Without complications of critical phenomena the MRK equation can be simplified using the
#         approximation:

#         .. math::

#             V \sim \frac{RT}{P} + b

#         where :math:`V` is volume, :math:`R` is the gas constant, :math:`T` is temperature,
#         :math:`P` is pressure, and :math:`b` is :attr:`b`.

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             MRK volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
#         """
#         volume: Array = (
#             jnp.sqrt(temperature)
#             * -self.a(temperature)
#             * GAS_CONSTANT_BAR
#             / (GAS_CONSTANT_BAR * temperature + self.b * pressure)
#             / (GAS_CONSTANT_BAR * temperature + 2.0 * self.b * pressure)
#             + GAS_CONSTANT_BAR * temperature / pressure
#             + self.b
#         )

#         return volume

#     @override
#     def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
#         r"""Volume-explicit integral :cite:p:`HP91{Equation 8}`

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
#         """
#         volume_integral: Array = (
#             GAS_CONSTANT_BAR * temperature * jnp.log(pressure)
#             + self.b * pressure
#             + self.a(temperature)
#             / self.b
#             / jnp.sqrt(temperature)
#             * (
#                 jnp.log(GAS_CONSTANT_BAR * temperature + self.b * pressure)
#                 - jnp.log(GAS_CONSTANT_BAR * temperature + 2.0 * self.b * pressure)
#             )
#         )
#         volume_integral = volume_integral * unit_conversion.m3_bar_to_J

#         return volume_integral


# # TODO: Update to support JAX
# @dataclass(kw_only=True)
# class MRKImplicitABC(ModifiedRedlichKwongABC):
#     """A Modified Redlich Kwong (MRK) EOS in implicit form

#     Args:
#         calibration: Calibration temperature and pressure range. Defaults to empty.
#     """

#     @override
#     def a(self, temperature: float) -> Array:
#         r"""MRK `a` parameter from :attr:`a_coefficients` :cite:p:`HP91{Equation 6}`

#         Args:
#             temperature: Temperature in K

#         Returns:
#             MRK `a` parameter in
#             :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
#         """
#         a: Array = (
#             self.a_coefficients[0]
#             + self.a_coefficients[1] * self.delta_temperature_for_a(temperature)
#             + self.a_coefficients[2] * self.delta_temperature_for_a(temperature) ** 2
#             + self.a_coefficients[3] * self.delta_temperature_for_a(temperature) ** 3
#         )

#         return a

#     @abstractmethod
#     def delta_temperature_for_a(self, temperature: float) -> float:
#         """Temperature difference for the calculation of the `a` parameter

#         Args:
#             temperature: Temperature in K

#         Returns:
#             A temperature difference
#         """

#     @property
#     def b(self) -> float:
#         """MRK `b` parameter computed from :attr:`b0`.

#         :class:`~MRKImplicitABC` is not used for corresponding states models so :attr:`b0` is the
#         `b` coefficient.
#         """
#         return self.b0

#     def A_factor(self, temperature: float, pressure: ArrayLike) -> Array:
#         """`A` factor :cite:p:`HP91{Appendix A}`

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             `A` factor, which is non-dimensional
#         """
#         del pressure
#         A_factor: Array = self.a(temperature) / (self.b * GAS_CONSTANT_BAR * temperature**1.5)

#         return A_factor

#     def B_factor(self, temperature: float, pressure: ArrayLike) -> ArrayLike:
#         """`B` factor :cite:p:`HP91{Appendix A}`

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             `B` factor, which is non-dimensional
#         """
#         B_factor: ArrayLike = self.b * pressure / (GAS_CONSTANT_BAR * temperature)

#         return B_factor

#     @override
#     def volume_integral(
#         self,
#         temperature: float,
#         pressure: ArrayLike,
#     ) -> Array:
#         r"""Volume integral :cite:p:`HP91{Equation A.2}`

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
#         """
#         z: ArrayLike = self.compressibility_factor(temperature, pressure)
#         A: Array = self.A_factor(temperature, pressure)
#         B: ArrayLike = self.B_factor(temperature, pressure)
#         # The base class requires a specification of the volume_integral, but the equations are in
#         # terms of the fugacity coefficient.
#         ln_fugacity_coefficient: Array = -jnp.log(z - B) - A * jnp.log(1 + B / z) + z - 1  # type: ignore
#         ln_fugacity: Array = jnp.log(pressure) + ln_fugacity_coefficient
#         volume_integral: Array = GAS_CONSTANT_BAR * temperature * ln_fugacity
#         volume_integral = volume_integral * unit_conversion.m3_bar_to_J

#         return volume_integral

#     # TODO: Needs updating to support JAX. Can use optimistix root finder, but then need to tailor
#     # the initial guesses to find the correct root

#     def volume_roots(self, temperature: float, pressure: ArrayLike) -> Array:
#         r"""Real and (potentially) physically meaningful volume solutions of the MRK equation

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume solutions of the MRK equation in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
#         """
#         coefficients: list[float] = []
#         coefficients.append(-self.a(temperature) * self.b / np.sqrt(temperature))
#         coefficients.append(
#             -self.b * GAS_CONSTANT_BAR * temperature
#             - self.b**2 * pressure
#             + self.a(temperature) / np.sqrt(temperature)
#         )
#         coefficients.append(-GAS_CONSTANT_BAR * temperature)
#         coefficients.append(pressure)

#         polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="V")
#         logger.debug("MRK equation = %s", polynomial)
#         volume_roots: npt.NDArray = polynomial.roots()
#         # Numerical solution could result in a small imaginery component, even though the root is
#         # real.
#         real_roots: npt.NDArray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
#         # Physically meaningful volumes must be positive.
#         positive_roots: npt.NDArray = real_roots[real_roots > 0]
#         # In general, several roots could be returned, and subclasses will need to determine which
#         # is the correct volume to use depending on the phase (liquid, gas, etc.)
#         logger.debug("V = %s", positive_roots)

#         return positive_roots


# # TODO: Update to support JAX
# @dataclass(kw_only=True)
# class MRKCriticalBehaviour(RealGas):
#     r"""A MRK equation of state that accommodates critical behaviour :cite:p:`HP91{Appendix A}`

#     Args:
#         mrk_fluid: MRK EOS for the supercritical fluid
#         mrk_gas: MRK EOS for the subcritical gas
#         mrk_liquid: MRK EOS for the subcritical liquid
#         Ta: Temperature at which :math:`a_{\mathrm gas} = a` by constrained fitting :cite:p:`HP91`.
#         calibration: Calibration temperature and pressure range. Defaults to empty.
#     """

#     mrk_fluid: MRKImplicitABC
#     """MRK EOS for the supercritical fluid"""
#     mrk_gas: MRKImplicitABC
#     """MRK EOS for the subcritical gas"""
#     mrk_liquid: MRKImplicitABC
#     """MRK EOS for the subcritical liquid"""
#     Ta: float
#     r"""Temperature at which :math:`a_{\mathrm gas} = a` by constrained fitting"""

#     @abstractmethod
#     def Psat(self, temperature: float) -> float:
#         """Saturation curve :cite:p:`{e.g.}HP91{Equation 5}`

#         Args:
#             temperature: Temperature in K

#         Returns:
#             Saturation curve pressure in bar
#         """

#     @override
#     def volume(self, temperature: float, pressure: float) -> float:
#         Psat: float = self.Psat(temperature)

#         if temperature <= self.Ta and pressure <= Psat:
#             logger.debug(
#                 "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
#                 temperature,
#                 self.Ta,
#                 pressure,
#                 Psat,
#             )
#             logger.debug("Gas phase")
#             volume = self.mrk_gas.volume(temperature, pressure)

#         elif temperature <= self.Ta and pressure > Psat:
#             logger.debug(
#                 "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
#                 temperature,
#                 self.Ta,
#                 pressure,
#                 Psat,
#             )
#             logger.debug("Liquid phase")
#             volume = self.mrk_liquid.volume(temperature, pressure)

#         else:
#             logger.debug("Fluid phase")
#             volume = self.mrk_fluid.volume(temperature, pressure)

#         return volume

#     @override
#     def volume_integral(self, temperature: float, pressure: float) -> float:
#         Psat: float = self.Psat(temperature)

#         if temperature <= self.Ta and pressure <= Psat:
#             logger.debug(
#                 "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
#                 temperature,
#                 self.Ta,
#                 pressure,
#                 Psat,
#             )
#             # logger.debug("Gas phase")
#             volume_integral = self.mrk_gas.volume_integral(temperature, pressure)

#         elif temperature <= self.Ta and pressure > Psat:
#             logger.debug(
#                 "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
#                 temperature,
#                 self.Ta,
#                 pressure,
#                 Psat,
#             )
#             # logger.debug("Performing pressure integration")
#             volume_integral = self.mrk_gas.volume_integral(temperature, Psat)
#             volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
#             volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)

#         else:
#             # logger.debug("Fluid phase")
#             volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

#         return volume_integral


# @dataclass(kw_only=True)
# class VirialCompensation(CorrespondingStatesMixin, RealGas):
#     r"""A virial compensation term for the increasing deviation of the MRK volumes with pressure

#     General form of the equation :cite:t:`HP98` and also see :cite:t:`HP91{Equations 4 and 9}`:

#     .. math::

#         V_\mathrm{virial} = a(P-P0) + b(P-P0)^\frac{1}{2} + c(P-P0)^\frac{1}{4}

#     This form also works for the virial compensation term from :cite:t:`HP91`, in which
#     case :math:`c=0`. :attr:`critical_pressure` and :attr:`critical_temperature` are required for
#     gases which are known to obey approximately the principle of corresponding states.

#     Although this looks similar to an EOS, it only calculates an additional perturbation to the
#     volume and the volume integral of an MRK EOS, and hence it does not return a meaningful volume
#     or volume integral by itself.

#     Args:
#         a_coefficients: Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`, where
#             :math:`a_0` and :math:`a_1` may be scaled (internally) by critical parameters for
#             corresponding states.
#         b_coefficients: As above for the b coefficients
#         c_coefficients: As above for the c coefficients
#         P0: Pressure at which the MRK equation begins to overestimate the molar volume
#             significantly and may be determined from experimental data.
#         critical_temperature: Critical temperature in K. Defaults to unity meaning not a
#             corresponding states model.
#         critical_pressure: Critical pressure in bar. Defaults to unity meaning not a corresponding
#             states model.
#         calibration: Calibration temperature and pressure range. Defaults to empty.
#     """

#     a_coefficients: Array
#     r"""Coefficients for a polynomial of the form :math:`a=a_0+a_1 T`, where :math:`a_0` and
#     :math:`a_1` may be additionally (internally) scaled by critical parameters
#     (:attr:`critical_temperature` and :attr:`critical_pressure`) for corresponding states."""
#     b_coefficients: Array
#     """Coefficients for the `b` parameter. See :attr:`a_coefficients` documentation."""
#     c_coefficients: Array
#     """Coefficients for the `c` parameter. See :attr:`a_coefficients` documentation."""
#     P0: ArrayLike
#     """Pressure at which the MRK equation begins to overestimate the molar volume significantly
#     and may be determined from experimental data."""

#     def a(self, temperature: float) -> Array:
#         r"""`a` parameter :cite:p:`HP98`

#         This is also the `d` parameter in :cite:t:`HP91`.

#         Args:
#             temperature: Temperature in K

#         Returns:
#             `a` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}\mathrm{bar}^{-1}`
#         """
#         a: Array = (
#             self.a_coefficients[0] * self.critical_temperature
#             + self.a_coefficients[1] * temperature
#         )
#         a = a / self.critical_pressure**2

#         return a

#     def b(self, temperature: float) -> Array:
#         r"""`b` parameter :cite:p:`HP98`

#         This is also the `c` parameter in :cite:t:`HP91`.

#         Args:
#             temperature: Temperature in K

#         Returns:
#             `b` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}\mathrm{bar}^\frac{-1}{2}`
#         """
#         b: Array = (
#             self.b_coefficients[0] * self.critical_temperature
#             + self.b_coefficients[1] * temperature
#         )
#         b = b / self.critical_pressure ** (3 / 2)

#         return b

#     def c(self, temperature: float) -> Array:
#         r"""`c` parameter :cite:p:`HP98`

#         Args:
#             temperature: Temperature in K

#         Returns:
#             `c` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}\mathrm{bar}^\frac{-1}{4}`
#         """
#         c: Array = (
#             self.c_coefficients[0] * self.critical_temperature
#             + self.c_coefficients[1] * temperature
#         )
#         c = c / self.critical_pressure ** (5 / 4)

#         return c

#     def delta_pressure(self, pressure: ArrayLike) -> Array:
#         """Pressure difference

#         Args:
#             pressure: Pressure in bar

#         Returns:
#             Pressure difference relative to :attr:`P0`
#         """
# TODO: Careful. Maybe jnp.where is better for array-based operations
#         delta_pressure: Array = lax.cond(
#             jnp.asarray(pressure) > jnp.asarray(self.P0),
#             lambda pressure: jnp.asarray(pressure - self.P0, dtype=jnp.float_),
#             lambda pressure: jnp.array(0, dtype=jnp.float_),
#             pressure,
#         )

#         return delta_pressure

#     def volume(self, temperature: float, pressure: ArrayLike) -> Array:
#         r"""Volume contribution

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume contribution in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
#         """
#         delta_pressure: Array = self.delta_pressure(pressure)
#         volume: Array = (
#             self.a(temperature) * delta_pressure
#             + self.b(temperature) * delta_pressure**0.5
#             + self.c(temperature) * delta_pressure**0.25
#         )

#         return volume

#     def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
#         r"""Volume integral :math:`V dP` contribution

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume integral contribution in :math:`\mathrm{J}\mathrm{mol}^{-1}`
#         """
#         delta_pressure: Array = self.delta_pressure(pressure)
#         volume_integral: Array = (
#             self.a(temperature) / 2.0 * delta_pressure**2
#             + 2.0 / 3.0 * self.b(temperature) * delta_pressure ** (3.0 / 2.0)
#             + 4.0 / 5.0 * self.c(temperature) * delta_pressure ** (5.0 / 4.0)
#         )
#         volume_integral = volume_integral * unit_conversion.m3_bar_to_J

#         return volume_integral


# @dataclass(kw_only=True)
# class CORK(CorrespondingStatesMixin, RealGas):
#     """A Compensated-Redlich-Kwong (CORK) EOS :cite:p:`HP91`

#     Args:
#         P0: Pressure at which the MRK equation begins to overestimate the molar volume
#             significantly and may be determined from experimental data.
#         mrk: MRK model for computing the MRK contribution
#         a_virial: `a` coefficients for the virial compensation. Defaults to zero.
#         b_virial: `b` coefficients for the virial compensation. Defaults to zero.
#         c_virial: `c` coefficients for the virial compensation. Defaults to zero.
#         critical_temperature: Critical temperature in K. Defaults to unity meaning not a
#             corresponding states model.
#         critical_pressure: Critical pressure in bar. Defaults to unity meaning not a corresponding
#             states model.
#         calibration: Calibration temperature and pressure range. Defaults to empty.
#     """

#     P0: ArrayLike
#     """Pressure at which the MRK equation begins to overestimate the molar volume significantly
#     and may be determined from experimental data."""
#     mrk: RealGas
#     """MRK model for computing the MRK contribution"""
#     a_virial: Array = jnp.array((0, 0))
#     """`a` coefficients for the virial compensation"""
#     b_virial: Array = jnp.array((0, 0))
#     """`b` coefficients for the virial compensation"""
#     c_virial: Array = jnp.array((0, 0))
#     """`c` coefficients for the virial compensation"""
#     virial: VirialCompensation = field(init=False)
#     """The virial compensation model"""

#     def __post_init__(self):
#         self.virial = VirialCompensation(
#             a_coefficients=self.a_virial,
#             b_coefficients=self.b_virial,
#             c_coefficients=self.c_virial,
#             P0=self.P0,
#             critical_temperature=self.critical_temperature,
#             critical_pressure=self.critical_pressure,
#         )

#     @override
#     def volume(self, temperature: float, pressure: ArrayLike) -> Array:
#         r"""Volume :cite:p:`HP91{Equation 7a}`

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
#         """
#         return self.virial.volume(temperature, pressure) + self.mrk.volume(temperature, pressure)

#     @override
#     def volume_integral(self, temperature: float, pressure: float) -> Array:
#         r"""Volume integral :cite:p:`HP91{Equation 8}`

#         Args:
#             temperature: Temperature in K
#             pressure: Pressure in bar

#         Returns:
#             Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
#         """
#         return self.virial.volume_integral(temperature, pressure) + self.mrk.volume_integral(
#             temperature, pressure
#         )


# @dataclass(kw_only=True)
# class CombinedEOSModel(RealGas):
#     """Combines multiple EOS models for different pressure ranges into a single EOS model.

#     Args:
#         models: EOS models ordered by increasing pressure from lowest to highest
#         upper_pressure_bounds: Upper pressure bound in bar relevant to the EOS by position
#     """

#     models: tuple[RealGas, ...]
#     """EOS models ordered by increasing pressure from lowest to highest"""
#     upper_pressure_bounds: Array
#     """Upper pressure bound in bar relevant to the EOS by position"""

#     def _get_index(self, pressure: ArrayLike) -> int:
#         """Gets the index of the appropriate EOS model using the upper pressure bound

#         Args:
#             pressure: Pressure in bar

#         Returns:
#             Index of the relevant EOS model
#         """

#         def body_fun(i: int, carry: int) -> int:
#             pressure_high: Array = self.upper_pressure_bounds[i]
#             condition = pressure >= pressure_high
# TODO: Careful. Maybe jnp.where is better for array-based operations
#             new_index: int = lax.cond(condition, lambda _: i + 1, lambda _: carry, None)

#             return new_index

#         init_carry: int = 0  # Initial carry value
#         index = lax.fori_loop(0, len(self.upper_pressure_bounds), body_fun, init_carry)

#         return index

#     @override
#     def volume(self, temperature: float, pressure: ArrayLike) -> Array:
#         index = self._get_index(pressure)
#         volume: Array = lax.switch(
#             index, [model.volume for model in self.models], temperature, pressure
#         )
#         return volume

#     @override
#     def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
#         index: int = self._get_index(pressure)

#         def compute_integral(i: int, pressure_high: ArrayLike, pressure_low: ArrayLike) -> Array:
#             """Compute pressure integral."""
#             return self.models[i].volume_integral(temperature, pressure_high) - self.models[
#                 i
#             ].volume_integral(temperature, pressure_low)

#         def case_0() -> Array:
#             return self.models[0].volume_integral(temperature, pressure)

#         def case_1() -> Array:
#             volume0 = self.models[0].volume_integral(temperature, self.upper_pressure_bounds[0])
#             dvolume = compute_integral(1, pressure, self.upper_pressure_bounds[0])
#             return volume0 + dvolume

#         def case_2() -> Array:
#             volume0 = self.models[0].volume_integral(temperature, self.upper_pressure_bounds[0])
#             dvolume0 = compute_integral(
#                 1, self.upper_pressure_bounds[1], self.upper_pressure_bounds[0]
#             )
#             dvolume1 = compute_integral(2, pressure, self.upper_pressure_bounds[1])
#             return volume0 + dvolume0 + dvolume1

#         return lax.switch(index, [case_0, case_1, case_2])
