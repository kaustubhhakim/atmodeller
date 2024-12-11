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
"""Classes that aggregate EOS

Units for temperature and pressure are K and bar, respectively.
"""

import logging
import sys
from abc import ABC
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos.core import RealGas
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibrationNew

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


class CombinedRealGasABC(RealGas, ABC):
    """Base class for combining real gas EOS

    Args:
        real_gases: Real gases to combine
        calibrations: Experimental calibrations (or None) that correspond by position to the
            entries in `real_gases`.
    """

    def __init__(
        self,
        real_gases: list[RealGas],
        calibrations: list[ExperimentalCalibrationNew | None],
    ):
        self._real_gases: list[RealGas] = real_gases
        self._calibrations: list[ExperimentalCalibrationNew | None] = calibrations
        self._upper_pressure_bounds: Array = self._get_upper_pressure_bounds()

    @property
    def volume_functions(self) -> list[Callable]:
        return [eos.volume for eos in self._real_gases]

    @property
    def volume_integral_functions(self) -> list[Callable]:
        return [eos.volume_integral for eos in self._real_gases]

    # Do not jit. Causes problems with initialization.
    def _get_upper_pressure_bounds(self) -> Array:
        """Gets the upper pressure bounds based on each experimental calibration.

        Returns:
            Upper pressure bounds
        """
        upper_pressure_bounds: list[float] = []

        for nn, calibration in enumerate(self._calibrations):
            if calibration is None:
                try:
                    calibration_next = self._calibrations[nn + 1]
                except IndexError:
                    logger.debug("Last entry has no calibration")
                    continue
                try:
                    assert isinstance(calibration_next, ExperimentalCalibrationNew)
                except AssertionError:
                    msg = "'None' entries must be bracketed by experimental calibrations"
                    raise ValueError(msg)

                # Minimum pressure of the next entry used as the maximum pressure of this entry.
                pressure_bound: float = calibration_next.pressure_min
            else:
                try:
                    assert calibration.pressure_max is not None
                except AssertionError:
                    msg = "Maximum pressure cannot be None"
                    raise ValueError(msg)

                pressure_bound = calibration.pressure_max

            upper_pressure_bounds.append(pressure_bound)

        logger.debug("upper_pressure_bounds = %s", upper_pressure_bounds)

        return jnp.array(upper_pressure_bounds)

    @jit
    def _get_index(self, pressure: ArrayLike) -> int:
        """Gets the index of the appropriate EOS model based on `pressure`.

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """

        def body_fun(i: int, carry: int) -> int:
            pressure_high: Array = lax.dynamic_slice(self._upper_pressure_bounds, (i,), (1,))
            pressure_high = jnp.squeeze(pressure_high)
            condition = pressure >= pressure_high
            new_index: int = lax.cond(condition, lambda _: i + 1, lambda _: carry, None)

            return new_index

        initial_carry: int = 0
        index: int = lax.fori_loop(0, len(self._upper_pressure_bounds), body_fun, initial_carry)

        return index

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)
        volume: Array = lax.switch(index, self.volume_functions, temperature, pressure)

        return volume

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "real_gases": self._real_gases,
            "calibrations": self._calibrations,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class CombinedRealGas(CombinedRealGasABC):
    """Combined real gas EOS

    This class selects the volume integral to use based on an index.
    """

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)
        volume_integral: Array = lax.switch(
            index, self.volume_integral_functions, temperature, pressure
        )

        return volume_integral


@register_pytree_node_class
class CombinedRealGasRemoveSteps(CombinedRealGasABC):
    """Combined real gas EOS with separate integrations for each EOS

    This class computes the contribution to the volume integral separately for each EOS based on
    the range covered by its P-T calibration, and then combines them.
    """

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: int = self._get_index(pressure)

        def compute_integral(ii: int, pressure_high: ArrayLike, pressure_low: ArrayLike) -> Array:
            volume_integral_high: Array = lax.switch(
                ii, self.volume_integral_functions, temperature, pressure_high
            )
            # jax.debug.print("volume_integral_high = {out}", out=volume_integral_high)
            volume_integral_low: Array = lax.switch(
                ii, self.volume_integral_functions, temperature, pressure_low
            )
            # jax.debug.print("volume_integral_low = {out}", out=volume_integral_low)
            integral_contribution = volume_integral_high - volume_integral_low

            return integral_contribution

        def body_fun(ii: int, carry: Array) -> Array:
            # jax.debug.print("index = {out}", out=index)
            pressure_high: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, ii, keepdims=False
            )
            # jax.debug.print("pressure_high = {out}", out=pressure_high)
            pressure_low: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, ii - 1, keepdims=False
            )
            # jax.debug.print("pressure_low = {out}", out=pressure_low)
            carry = carry + compute_integral(ii, pressure_high, pressure_low)

            return carry

        # Initialize. Must be 0.0 to ensure float array.
        total_integral: Array = jnp.array(0.0)

        def add_only_first_integral(total_integral: Array) -> Array:
            integral: Array = lax.switch(0, self.volume_integral_functions, temperature, pressure)

            return total_integral + integral

        total_integral: Array = lax.cond(
            index == 0, add_only_first_integral, lambda x: x, total_integral
        )

        # Loop over and accumulate the vdP integrations over the EOS. This will only do something
        # if index >= 2, so will effectively be ignored for index = 1, as desired
        total_integral: Array = lax.fori_loop(1, index, body_fun, total_integral)
        # jax.debug.print("total_integral after lax.fori_loop = {out}", out=total_integral)

        def add_final_integral(total_integral: Array) -> Array:
            # Account for the first integral, which thus far has not been included for index > 0.
            pressure_high = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, 0, keepdims=False
            )
            # Do not evaluate a difference because log(0) is not defined.
            # TODO: Check. This evaluation for P<1 will be negative for an ideal gas.
            first_integral: Array = lax.switch(
                0, self.volume_integral_functions, temperature, pressure_high
            )

            pressure_low: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, index - 1, keepdims=False
            )
            final_integral: Array = compute_integral(index, pressure, pressure_low)

            return first_integral + final_integral + total_integral

        total_integral = lax.cond(index > 0, add_final_integral, lambda x: x, total_integral)
        # jax.debug.print("total_integral = {out}", out=total_integral)

        return total_integral


@register_pytree_node_class
class RealGasBounded(RealGas):
    """A real gas equation of state that is bounded

    Args:
        real_gas: Real gas equation of state to bound
        calibration: Calibration that encodes the bounds
    """

    def __init__(
        self,
        real_gas: RealGasProtocol,
        calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(),
    ):
        super().__init__()
        self._real_gas: RealGasProtocol = real_gas
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
            temperature: Temperature in K
            pressure: Pressure in bar

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
        log_fugacity = lax.select(pressure < self._pressure_max, log_fugacity, log_fugacity_above)

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

    @override
    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array: ...

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {"real_gas": self._real_gas, "calibration": self._calibration}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)
