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
from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike, Float

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos.core import IdealGas, RealGas
from atmodeller.utilities import ExperimentalCalibration, as_j64

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier

logger: logging.Logger = logging.getLogger(__name__)


class CombinedRealGas(RealGas):
    """Combined real gas EOS with separate volume integrations for each EOS

    This class computes the contribution to the volume integral separately for each EOS based on
    the range covered by its P-T calibration, and then combines them.

    Args:
        real_gases: Real gases to combine
        calibrations: Experimental calibrations that correspond to `real_gases`
    """

    real_gases: tuple[RealGas, ...]
    """Real gases to combine"""
    calibrations: tuple[ExperimentalCalibration, ...]
    """Experimental calibrations"""
    _upper_pressure_bounds: tuple[Array, ...] = eqx.field(init=False)

    def __post_init__(self):
        self._upper_pressure_bounds = self._get_upper_pressure_bounds()

    @classmethod
    def create(
        cls,
        real_gases: Sequence[RealGas],
        calibrations: Sequence[ExperimentalCalibration],
        extrapolate: bool = True,
    ) -> RealGas:
        """Create an instance with the given real gases and calibrations

        Reasonable extrapolation behaviour is required to ensure that the function is bounded to
        avoid throwing NaNs or infs which will crash the solver. Physically, it is reasonable to
        extend the lower bound using the ideal gas law and the upper bound assuming a linear
        pressure dependence of the compressibility factor.

        There is no bounding for temperature; hence it is assumed that the extrapolation
        behaviour of temperature is reasonable. This is practically useful because the calibrations
        are often restricted to a lower temperature range than the high temperatures that are
        typically of interest for hot rocks and magma ocean planets.

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to `real_gases`
            extrapolate: Extrapolate the EOS to have reasonable behaviour below the minimum and
                above the maximum calibration pressure if required. Defaults to True.
        """
        real_gases_list: list[RealGas] = list(real_gases)
        calibrations_list: list[ExperimentalCalibration] = list(calibrations)

        if extrapolate:
            if not jnp.isnan(calibrations_list[0].pressure_min):
                cls._append_lower_bound(real_gases_list, calibrations_list)
            if not jnp.isnan(calibrations_list[-1].pressure_max):
                cls._append_upper_bound(real_gases_list, calibrations_list)

        return cls(tuple(real_gases_list), tuple(calibrations_list))

    @classmethod
    def _append_lower_bound(
        cls,
        real_gases: list[RealGas],
        calibrations: list[ExperimentalCalibration],
    ) -> None:
        """Appends the lower bound, which gives ideal gas behaviour

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to `real_gases`
        """
        real_gases.insert(0, IdealGas())
        pressure_max: Float[Array, ""] = calibrations[0].pressure_min
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_max=pressure_max)
        calibrations.insert(0, calibration)

    @classmethod
    def _append_upper_bound(
        cls,
        real_gases: list[RealGas],
        calibrations: list[ExperimentalCalibration],
    ) -> None:
        """Appends the upper bound

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to `real_gases`
        """
        pressure_min: Float[Array, ""] = calibrations[-1].pressure_max
        real_gas: RealGas = UpperBoundRealGas(real_gases[-1], pressure_min)
        real_gases.append(real_gas)
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=pressure_min)
        calibrations.append(calibration)

    @property
    def volume_functions(self) -> tuple[Callable, ...]:
        """Volume functions"""
        return tuple(eos.volume for eos in self.real_gases)

    @property
    def upper_pressure_bounds(self) -> Array:
        return jnp.array(self._upper_pressure_bounds)

    @property
    def volume_integral_functions(self) -> tuple[Callable, ...]:
        """Volume integral functions"""
        return tuple(eos.volume_integral for eos in self.real_gases)

    def _get_upper_pressure_bounds(self) -> tuple[Array, ...]:
        """Gets the upper pressure bounds based on each experimental calibration.

        Returns:
            Upper pressure bounds
        """
        upper_pressure_bounds: list[Array] = []

        for ii, calibration in enumerate(self.calibrations):
            try:
                assert not jnp.isnan(calibration.pressure_max)
            except AssertionError:
                if ii == len(self.calibrations) - 1:
                    continue
                else:
                    msg: str = "Maximum pressure cannot be None"
                    raise ValueError(msg)

            pressure_bound = calibration.pressure_max
            upper_pressure_bounds.append(pressure_bound)

        return tuple(upper_pressure_bounds)

    @eqx.filter_jit
    def _get_index(self, pressure: ArrayLike) -> Array:
        """Gets the index of the appropriate EOS model based on `pressure`.

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """
        index: Array = jnp.searchsorted(self.upper_pressure_bounds, pressure, side="right")
        # jax.debug.print("pressure = {pressure}, index = {index}", pressure=pressure, index=index)

        return index

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        temperature, pressure = jnp.broadcast_arrays(temperature, pressure)
        original_shape: tuple[int, ...] = temperature.shape
        # jax.debug.print("temperature = {out}", out=temperature)
        # jax.debug.print("pressure = {out}", out=pressure)

        # Flatten for vmap
        temperature = temperature.ravel()
        pressure = pressure.ravel()

        indices: Array = self._get_index(pressure)
        # jax.debug.print("indices = {out}", out=indices)

        def apply_volume(index: ArrayLike, temperature, pressure) -> Array:
            return lax.switch(index, self.volume_functions, temperature, pressure)

        vmap_volume: Callable = eqx.filter_vmap(apply_volume, in_axes=(0, 0, 0))
        volume: Array = vmap_volume(indices, temperature, pressure)
        # jax.debug.print("volume = {out}", out=volume)

        return jnp.reshape(volume, original_shape)

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: Array = self._get_index(pressure)

        def compute_integral(i: Array, pressure_high: ArrayLike, pressure_low: ArrayLike) -> Array:
            """Computes the volume integral for the given pressure range.

            Args:
                i: Index of the EOS model
                pressure_high: Upper pressure bound
                pressure_low: Lower pressure bound

            Returns:
                Volume integral"""
            volume_integral_high: Array = lax.switch(
                i, self.volume_integral_functions, temperature, pressure_high
            )
            # jax.debug.print(
            #    "compute_integral: volume_integral_high = {out}", out=volume_integral_high
            # )
            volume_integral_low: Array = lax.switch(
                i, self.volume_integral_functions, temperature, pressure_low
            )
            # jax.debug.print(
            #    "compute_integral: volume_integral_low = {out}", out=volume_integral_low
            # )
            integral: Array = volume_integral_high - volume_integral_low

            return integral

        def scan_fn(carry: Array, i: Array) -> tuple:
            """Scan function for accumulating the volume integral

            Args:
                carry: Accumulated volume integral
                i: Index of the EOS model

            Returns:
                Updated carry and None (unused)
            """
            # jax.debug.print("scan_fn: i = {out}", out=i)
            pressure_low = lax.dynamic_index_in_dim(
                self.upper_pressure_bounds, i - 1, keepdims=False
            )
            # jax.debug.print("pressure_low = {out}", out=pressure_low)

            # Middle integrals
            mask_middle: Array = i < index
            # jax.debug.print("mask_middle = {out}", out=mask_middle)
            pressure_high = lax.dynamic_index_in_dim(self.upper_pressure_bounds, i, keepdims=False)
            # jax.debug.print("pressure_high = {out}", out=pressure_high)
            contrib_middle: Array = compute_integral(i, pressure_high, pressure_low)
            carry = carry + jnp.where(mask_middle, contrib_middle, 0.0)

            # Final integral
            mask_final: Array = i == index
            # jax.debug.print("mask_final = {out}", out=mask_final)
            contrib_final: Array = compute_integral(i, pressure, pressure_low)
            carry = carry + jnp.where(mask_final, contrib_final, 0.0)
            # jax.debug.print("carry = {out}", out=carry)

            return carry, None

        def add_first_integral(total_integral: Array) -> Array:
            """Adds the contribution of the first integral to the total integral.

            This is necessary because the first integral is not included in the loop over the
            EOS models.

            Args:
                total_integral: Total integral so far

            Returns:
                Total integral with the first integral contribution added
            """
            # If the index is 0, then the first integral is the only one that is added.
            integral: Array = lax.switch(0, self.volume_integral_functions, temperature, pressure)
            # Otherwise, the first integral is added to the total integral.
            pressure_max: Array = lax.dynamic_index_in_dim(
                self.upper_pressure_bounds, 0, keepdims=False
            )
            integral2: Array = lax.switch(
                0, self.volume_integral_functions, temperature, pressure_max
            )

            # jax.debug.print("add_only_first_integral: integral = {out}", out=integral)
            return jnp.where(index == 0, total_integral + integral, total_integral + integral2)

        # Initialize. Must be 0.0 to ensure float array.
        total_integral: Array = jnp.array(0.0)
        total_integral = add_first_integral(total_integral)

        # Scan over the indices of the EOS models.
        loop_indices: Array = jnp.arange(1, self.upper_pressure_bounds.shape[0] + 1)
        # jax.debug.print("loop_indices = {out}", out=loop_indices)
        total_integral, _ = lax.scan(scan_fn, total_integral, loop_indices)

        return total_integral


class UpperBoundRealGas(RealGas):
    """An upper bound for an EOS

    This is used to extrapolate an EOS assuming that the compressibility factor is a linear
    function of pressure. Importantly, this class is not intended to be used directly, but rather
    as a component of `CombinedRealGas`.

    `p_eval` is used to evaluate the compressibility factor and its gradient, and therefore to
    avoid recompilation of JAX functions it is converted to a JAX array of dtype float64, which
    is the expected type for the pressure argument of the EOS functions.

    Args:
        real_gas: Real gas to evaluate the compressibility factor at `p_eval`.
        p_eval: Evaluation pressure in bar. This is usually the maximum calibration pressure
            of `real_gas`. Defaults to 1 bar.
    """

    real_gas: RealGas
    """Real gas to evaluate the compressibility factor at `p_eval`"""
    p_eval: Float[Array, ""] = eqx.field(converter=as_j64, default=1.0, static=True)
    """Evaluation pressure in bar. Defaults to 1 bar."""

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def _z0(self, temperature: ArrayLike) -> ArrayLike:
        """Compressibility factor of the previous EOS to blend smoothly with.

        Args:
            temperature: Temperature in K
        """
        return self.real_gas.compressibility_factor(temperature, self.p_eval)

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def _dzdp0(self, temperature: ArrayLike) -> ArrayLike:
        """Gradient of the compressibility factor of the previous EOS to blend smoothly with.

        Args:
            temperature: Temperature in K
        """
        return self.real_gas.dzdp(temperature, self.p_eval)

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity cannot be computed.

        This method should not be used because the volume integral is only defined above `p_eval`,
        meaning that the log fugacity cannot be calculated.
        """
        del temperature
        del pressure

        raise NotImplementedError("This method should not be used")

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        compressibility_factor: ArrayLike = self._z0(temperature) + self._dzdp0(temperature) * (
            pressure - self.p_eval  # type: ignore
        )

        return compressibility_factor

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        volume: ArrayLike = (
            self.compressibility_factor(temperature, pressure)
            * GAS_CONSTANT_BAR
            * temperature
            / pressure
        )

        return volume

    @override
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        volume_integral: Array = (
            (
                jnp.log(pressure / self.p_eval)
                * (self._z0(temperature) - self._dzdp0(temperature) * self.p_eval)
                + self._dzdp0(temperature) * (pressure - self.p_eval)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )

        return volume_integral
