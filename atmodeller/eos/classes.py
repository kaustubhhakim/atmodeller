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
"""Concrete classes for real gas equations of state"""

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE
from atmodeller.eos.core import RealGas
from atmodeller.utilities import OptxSolver

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier


class BeattieBridgeman(RealGas):
    r"""Beattie-Bridgeman equation :cite:p:`HWZ58{Equation 1}`

    .. math::

        PV^2 = RT\left(1-\frac{c}{VT^3}\right)\left(V+B_0-\frac{bB_0}{V}\right)
        - A_0\left(1-\frac{a}{V}\right)

    Args:
        A0: A0 empirical constant
        a: a empirical constant
        B0: B0 empirical constant
        b: b empirical constant
        c: c empirical constant
    """

    A0: float
    """A0 empirical constant"""
    a: float
    """a empirical constant"""
    B0: float
    """B0 empirical constant"""
    b: float
    """b empirical constant"""
    c: float
    """c empirical constant"""

    @eqx.filter_jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume :cite:p:`HWZ58{Equation 2}`

        .. math::

            PV^4 - RTV^3 - \left(RTB_0 - \frac{Rc}{T^2}-A_0\right)V^2
            +\left(RTbB_0+\frac{RcB_0}{T^2}-aA_0\right)V - \frac{RcbB_0}{T^2}=0

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

        # jax.debug.print("volume = {out}", out=volume)
        # jax.debug.print("temperature = {out}", out=temperature)
        # jax.debug.print("pressure = {out}", out=pressure)

        coeff0: Array = 1 / jnp.square(temperature) * -GAS_CONSTANT_BAR * self.c * self.b * self.B0
        coeff1: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self.c * self.B0
            + GAS_CONSTANT_BAR * temperature * self.b * self.B0
            - self.a * self.A0
        )
        coeff2: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self.c
            - GAS_CONSTANT_BAR * temperature * self.B0
            + self.A0
        )
        coeff3: ArrayLike = -GAS_CONSTANT_BAR * temperature

        residual: Array = (
            coeff0
            + coeff1 * volume
            + coeff2 * jnp.power(volume, 2)
            + coeff3 * jnp.power(volume, 3)
            + pressure * jnp.power(volume, 4)
        )

        return residual

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity :cite:p:`HWZ58{Equation 11}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        log_fugacity: Array = (
            jnp.log(GAS_CONSTANT_BAR * temperature / volume)
            + (
                self.B0
                - self.c / jnp.power(temperature, 3)
                - self.A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 2
            / volume
            - (
                self.b * self.B0
                + self.c * self.B0 / jnp.power(temperature, 3)
                - self.a * self.A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 3
            / (2 * jnp.square(volume))
            + (self.c * self.b * self.B0 / jnp.power(temperature, 3))
            * 4
            / (3 * jnp.power(volume, 3))
        )

        return log_fugacity

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Solves the BB equation numerically to compute the volume.

        :cite:t:`HWZ58` doesn't say which root to take, but one real root is very small and the
        maximum real root gives a volume that agrees with the tabulated compressibility factor
        for all species.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        initial_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(
            self._objective_function, solver, initial_volume, args=kwargs, throw=False
        )
        volume = sol.value
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        return self.log_fugacity(temperature, pressure) * GAS_CONSTANT_BAR * temperature
