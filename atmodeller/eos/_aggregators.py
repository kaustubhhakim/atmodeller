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
from dataclasses import field
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array, jit, lax
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos.core import IdealGas, RealGas
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibration

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier

logger: logging.Logger = logging.getLogger(__name__)


class CombinedRealGasABC(RealGas):
    """Base class for combining real gas EOS

    Args:
        real_gases: Real gases to combine
        calibrations: Experimental calibrations that correspond to `real_gases`
    """

    real_gases: list[RealGasProtocol]
    """Real gases to combine"""
    calibrations: list[ExperimentalCalibration]
    """Experimental calibrations"""
    _upper_pressure_bounds: Array = field(init=False)

    def __post_init__(self):
        self._upper_pressure_bounds: Array = self._get_upper_pressure_bounds()

    @property
    def volume_functions(self) -> list[Callable]:
        """Volume functions"""
        return [eos.volume for eos in self.real_gases]

    @property
    def volume_integral_functions(self) -> list[Callable]:
        """Volume integral functions"""
        return [eos.volume_integral for eos in self.real_gases]

    # Jitting might cause problem with initialization?
    def _get_upper_pressure_bounds(self) -> Array:
        """Gets the upper pressure bounds based on each experimental calibration.

        Returns:
            Upper pressure bounds
        """
        upper_pressure_bounds: list[float] = []

        for ii, calibration in enumerate(self.calibrations):
            try:
                assert calibration.pressure_max is not None
            except AssertionError:
                if ii == len(self.calibrations) - 1:
                    continue
                else:
                    msg: str = "Maximum pressure cannot be None"
                    raise ValueError(msg)

            pressure_bound = calibration.pressure_max
            upper_pressure_bounds.append(pressure_bound)

        return jnp.array(upper_pressure_bounds)

    @eqx.filter_jit
    def _get_index(self, pressure: ArrayLike) -> Array:
        """Gets the index of the appropriate EOS model based on `pressure`.

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
        """
        index: Array = jnp.searchsorted(self._upper_pressure_bounds, pressure, side="right")
        # jax.debug.print("pressure = {pressure}, index = {index}", pressure=pressure, index=index)

        return index

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        index: Array = self._get_index(pressure)
        volume: Array = lax.switch(index, self.volume_functions, temperature, pressure)

        return volume

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity


class UpperBoundRealGas(RealGas):
    """An upper bound for an EOS

    This is used to extrapolate an EOS assuming that the compressibility factor is a linear
    function of pressure. Importantly, this class is not intended to be used directly, but rather
    as a component of `CombinedRealGas`.

    Args:
        real_gas: Real gas to evaluate the compressibility factor at `p_eval`.
        p_eval: Evaluation pressure in bar. This is usually the maximum calibration pressure
            of `real_gas`. Defaults to 1 bar.
        dzdp: Gradient of the compressibility factor. Defaults to 0.
    """

    real_gas: RealGasProtocol
    """Real gas to evaluate the compressibility factor at `p_eval`"""
    p_eval: float = 1
    """Evaluation pressure in bar"""
    dzdp: float = 0
    """Gradient of the compressibility factor"""

    @eqx.filter_jit
    def _z0(self, temperature: ArrayLike) -> ArrayLike:
        """Compressibility factor of the previous EOS to blend smoothly with.

        Args:
            temperature: Temperature in K
        """
        return self.real_gas.compressibility_factor(temperature, self.p_eval)

    @override
    @eqx.filter_jit
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
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        compressibility_factor: ArrayLike = self._z0(temperature) + self.dzdp * (
            pressure - self.p_eval
        )

        return compressibility_factor

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        return self.compressibility_factor(temperature, pressure) * self.ideal_volume(
            temperature, pressure
        )

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        volume_integral: Array = (
            (
                jnp.log(pressure / self.p_eval) * (self._z0(temperature) - self.dzdp * self.p_eval)
                + self.dzdp * (pressure - self.p_eval)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )

        return volume_integral


class CombinedRealGasSwitch(CombinedRealGasABC):
    """Combined real gas EOS

    This class selects the EOS to use based on an index. This is only meaningful if subsequent EOS
    contain the contribution of previous EOS, otherwise there will be a mismatch across the joining
    boundary and the resulting EOS will not be continuous.
    """

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: Array = self._get_index(pressure)
        volume_integral: Array = lax.switch(
            index, self.volume_integral_functions, temperature, pressure
        )

        return volume_integral


class CombinedRealGas(CombinedRealGasABC):
    """Combined real gas EOS with separate volume integrations for each EOS

    This class computes the contribution to the volume integral separately for each EOS based on
    the range covered by its P-T calibration, and then combines them.
    """

    @classmethod
    def create(
        cls,
        real_gases: list[RealGasProtocol],
        calibrations: list[ExperimentalCalibration],
        dzdp: float = 0,
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
            dzdp: Constant compressibility (pressure) gradient for the upper bound extrapolation
                (if relevant). Defaults to 0.
            extrapolate: Extrapolate the EOS to have reasonable behaviour below the minimum and
                above the maximum calibration pressure if required. Defaults to True.
        """
        if extrapolate:
            if calibrations[0].pressure_min is not None:
                cls._append_lower_bound(real_gases, calibrations)
            if calibrations[-1].pressure_max is not None:
                cls._append_upper_bound(real_gases, calibrations, dzdp)

        return cls(real_gases, calibrations)

    @classmethod
    def _append_lower_bound(
        cls,
        real_gases: list[RealGasProtocol],
        calibrations: list[ExperimentalCalibration],
    ) -> None:
        """Appends the lower bound, which gives ideal gas behaviour

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to `real_gases`
        """
        real_gases.insert(0, IdealGas())
        pressure_max: float = calibrations[0].pressure_min  # type: ignore check done before
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_max=pressure_max)
        calibrations.insert(0, calibration)

    @classmethod
    def _append_upper_bound(
        cls,
        real_gases: list[RealGasProtocol],
        calibrations: list[ExperimentalCalibration],
        dzdp: float,
    ) -> None:
        """Appends the upper bound

        Args:
            real_gases: Real gases to combine
            calibrations: Experimental calibrations that correspond to `real_gases`
            dzdp: Constant compressibility (pressure) gradient
        """
        pressure_min: float = calibrations[-1].pressure_max  # type: ignore check done before
        real_gas: RealGasProtocol = UpperBoundRealGas(real_gases[-1], pressure_min, dzdp)
        real_gases.append(real_gas)
        calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=pressure_min)
        calibrations.append(calibration)

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        index: Array = self._get_index(pressure)

        def compute_integral(
            ii: Array, pressure_high: ArrayLike, pressure_low: ArrayLike
        ) -> Array:
            volume_integral_high: Array = lax.switch(
                ii, self.volume_integral_functions, temperature, pressure_high
            )
            # jax.debug.print("compute_integral: index = {ii}", ii=ii)
            # jax.debug.print(
            #    "compute_integral: volume_integral_high = {out}", out=volume_integral_high
            # )
            volume_integral_low: Array = lax.switch(
                ii, self.volume_integral_functions, temperature, pressure_low
            )
            # jax.debug.print(
            #    "compute_integral: volume_integral_low = {out}", out=volume_integral_low
            # )
            integral_contribution = volume_integral_high - volume_integral_low

            return integral_contribution

        def body_fun(ii: Array, carry: Array) -> Array:
            # jax.debug.print("body_fun: index = {out}", out=index)
            pressure_high: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, ii, keepdims=False
            )
            # jax.debug.print("body_fun: pressure_high = {out}", out=pressure_high)
            pressure_low: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, ii - 1, keepdims=False
            )
            # jax.debug.print("body_fun: pressure_low = {out}", out=pressure_low)
            carry = carry + compute_integral(ii, pressure_high, pressure_low)
            # jax.debug.print("body_fun: carry = {out}", out=carry)

            return carry

        # Initialize. Must be 0.0 to ensure float array.
        total_integral: Array = jnp.array(0.0)

        def add_only_first_integral(total_integral: Array) -> Array:
            integral: Array = lax.switch(0, self.volume_integral_functions, temperature, pressure)
            # jax.debug.print("add_only_first_integral: integral = {out}", out=integral)

            return total_integral + integral

        total_integral: Array = lax.cond(
            index == 0, add_only_first_integral, lambda x: x, total_integral
        )

        # TODO: Use lax scan for reverse differentiation compatibility
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
            # jax.debug.print("first_integral = {out}", out=first_integral)

            pressure_low: Array = lax.dynamic_index_in_dim(
                self._upper_pressure_bounds, index - 1, keepdims=False
            )
            final_integral: Array = compute_integral(index, pressure, pressure_low)

            return first_integral + final_integral + total_integral

        total_integral = lax.cond(index > 0, add_final_integral, lambda x: x, total_integral)
        # jax.debug.print("total_integral = {out}", out=total_integral)

        return total_integral
