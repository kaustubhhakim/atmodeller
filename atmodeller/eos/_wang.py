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
"""Real gas Virial EOS

The papers state a volume integration from :math:`P_0` to :math:`P`, where :math:`f(P_0=1)=1`.
Hence for bounded EOS a minimum pressure of 1 bar is assumed.
"""

import logging

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller import override
from atmodeller.constants import GAS_CONSTANT_BAR, PRESSURE_REFERENCE
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.utilities import ExperimentalCalibration, to_native_floats

logger: logging.Logger = logging.getLogger(__name__)


class VirialQuadratic(RealGas):
    """A real gas EOS that implements the virial formulation

    Args:
        a_coefficients: `a` coefficients
        b_coefficients: `b` coefficients
        c_coefficients: `c` coefficients
        critical_data: Critical data. Defaults to empty.
    """

    a_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """`a` coefficients"""
    b_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """`b` coefficients"""
    c_coefficients: tuple[float, ...] = eqx.field(converter=to_native_floats)
    """`c` coefficients"""

    @override
    @eqx.filter_jit
    def _get_compressibility_coefficient(
        self, temperature: ArrayLike, coefficients: tuple[float, ...]
    ) -> Array:
        """General form of the coefficients for the compressibility calculation

        Args:
            temperature: Temperature in K
            coefficients: Tuple of the coefficients `a`, `b`, `c`

        Returns
            The relevant coefficient
        """
        coefficient: Array = (
            jnp.asarray(coefficients[0])
            + coefficients[1] * temperature
            + coefficients[2] * jnp.square(temperature)
        )

        return coefficient

    @eqx.filter_jit
    def a(self, temperature: ArrayLike) -> Array:
        """`a` parameter

        Args:
            temperature: Temperature in K

        Returns:
            a parameter
        """
        a: Array = self._get_compressibility_coefficient(temperature, self.a_coefficients)

        return a

    @eqx.filter_jit
    def b(self, temperature: ArrayLike) -> Array:
        """`b` parameter

        Args:
            temperature: Temperature in K

        Returns:
            b parameter
        """
        b: Array = self._get_compressibility_coefficient(temperature, self.b_coefficients)

        return b

    @eqx.filter_jit
    def c(self, temperature: ArrayLike) -> Array:
        """`c` parameter

        Args:
            temperature: Temperature in K

        Returns:
            c parameter
        """
        c: Array = self._get_compressibility_coefficient(temperature, self.c_coefficients)

        return c

    @override
    @eqx.filter_jit
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Compressibility factor

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility factor, which is dimensionless
        """
        Z: Array = (
            self.a(temperature)
            + self.b(temperature) * pressure
            + self.c(temperature) * jnp.square(pressure)
        )

        return Z

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume :cite:p:`SS92{Equation 1}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        Z: Array = self.compressibility_factor(temperature, pressure)
        volume_ideal: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        volume: Array = Z * volume_ideal

        return volume

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        volume_integral: Array = (
            (
                self.a(temperature) * jnp.log(pressure / PRESSURE_REFERENCE)
                + self.b(temperature) * (pressure - PRESSURE_REFERENCE)
                + (1.0 / 2) * self.c(temperature) * (jnp.square(pressure) - PRESSURE_REFERENCE**2)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )

        return volume_integral


experimental_calibration_wang18: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=1200,
    temperature_max=4100,
    pressure_min=1,
    pressure_max=1387e3,
)
"""Experimental calibration for :cite:`WLL18` models"""

# Virial EOS for H4Si from :cite:t:`WLL18`
# The coefficients are for the quadratic virial equation of state

H4Si_wang18: RealGas = VirialQuadratic(
    a_coefficients=(1.0, 0, 0),
    b_coefficients=(3.8552e-4, -1.822e-7, 2.54e-11),
    c_coefficients=(-1.942e-10, 1.088e-13, -1.62e-17),
)

H4Si_wang18_bounded: RealGas = CombinedRealGas.create(
    [H4Si_wang18], [experimental_calibration_wang18]
)
"""OSi MRK corresponding states bounded :cite:p:`C16`"""


def get_wang_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of virial EOS

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["H4Si_wang18"] = H4Si_wang18_bounded

    return eos_models
