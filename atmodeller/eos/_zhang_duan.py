#!/usr/bin/env python3
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
"""Real gas EOS from :cite:t:`ZD09`"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos.core import RealGas
from atmodeller.interfaces import RealGasProtocol

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@register_pytree_node_class
class ZhangDuan(RealGas):
    """Real gas EOS from :cite:t:`ZD09`

    TODO: LaTeX math

    Args:
        epsilon: Lenard-Jones parameter (epsilon/kB) in K
        sigma: Lenard-Jones parameter (10^-10) m
    """

    def __init__(self, epsilon: float, sigma: float):
        self._epsilon: float = epsilon
        self._sigma: float = sigma
        self._coefficients: tuple[float, ...] = (
            2.95177298930e-2,
            -6.33756452413e3,
            -2.75265428882e5,
            1.29128089283e-3,
            -1.45797416153e2,
            7.65938947237e4,
            2.58661493537e-6,
            0.52126532146,
            -1.39839523753e2,
            -2.36335007175e-8,
            5.35026383543e-3,
            -0.27110649951,
            2.50387836486e4,
            0.73226726041,
            1.54833359970e-2,
        )

    @jit
    def Pm(self, pressure: ArrayLike) -> ArrayLike:
        """Scaled pressure

        Args:
            pressure: Pressure in bar

        Returns:
            The scaled pressure
        """
        # TODO: Check units
        scaled_pressure: ArrayLike = 3.0636 * jnp.power(self._sigma, 3) * pressure / self._epsilon

        return scaled_pressure

    @jit
    def Tm(self, temperature: ArrayLike) -> ArrayLike:
        """Scaled temperature

        Args:
            temperature: Temperature in K

        Returns:
            The scaled temperature
        """
        # TODO: Check units
        scaled_temperature: ArrayLike = 154.0 * temperature / self._epsilon

        return scaled_temperature

    @jit
    def Vm(self, volume: ArrayLike) -> ArrayLike:
        """Scaled volume

        Args:
            volume: Volume

        Returns:
            The scaled volume
        """
        # TODO: Check units
        sigma_term: ArrayLike = jnp.power(self._sigma / 3.691, 3)
        scaled_volume: ArrayLike = volume / 1000.0 / sigma_term

        return scaled_volume

    @jit
    def _get_parameter(
        self, temperature: ArrayLike, coefficients: tuple[float, float, float]
    ) -> ArrayLike:
        Tm: ArrayLike = self.Tm(temperature)

        return (
            coefficients[0] + coefficients[1] / jnp.square(Tm) + coefficients[2] / jnp.power(Tm, 3)
        )

    # @override
    # @jit
    # def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
    #     r"""Volume :cite:p:`SS92{Equation 1}`

    #     Args:
    #         temperature: Temperature in K
    #         pressure: Pressure in bar

    #     Returns:
    #         Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
    #     """
    #     Z: Array = self.compressibility_factor(temperature, pressure)
    #     volume: Array = Z * self.ideal_volume(temperature, pressure)

    #     return volume

    @jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

        pressure = pressure / 10  # to MPa

        Pm: ArrayLike = self.Pm(pressure)
        Tm: ArrayLike = self.Tm(temperature)
        Vm: ArrayLike = self.Vm(volume)

        # ptr: ArrayLike = Pm / (GAS_CONSTANT_BAR * Tm)
        ptr: ArrayLike = Pm / (8.314472 * Tm)

        a: ArrayLike = self._get_parameter(temperature, self._coefficients[0:3])
        b: ArrayLike = self._get_parameter(temperature, self._coefficients[3:6])
        c: ArrayLike = self._get_parameter(temperature, self._coefficients[6:9])
        d: ArrayLike = self._get_parameter(temperature, self._coefficients[9:12])

        term1: Array = (
            1 / jnp.asarray(Vm)
            + a / jnp.square(Vm)
            + b / jnp.power(Vm, 3)
            + c / jnp.power(Vm, 5)
            + d / jnp.power(Vm, 6)
        )

        a13: float = self._coefficients[12]
        a14: float = self._coefficients[13]
        a15: float = self._coefficients[14]
        term2: Array = a13 / jnp.power(Tm, 3) / jnp.power(Vm, 3)
        term2 = term2 * (a14 + a15 / jnp.square(Vm))
        term2 = term2 * jnp.exp(-a15 / jnp.square(Vm))

        residual: Array = term1 + term2 - ptr

        return residual

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Solves the RK equation numerically to compute the volume.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        # FIXME: Solve for Vm then convert back
        initial_volume: ArrayLike = 22  # e-5  # self.initial_volume(temperature, pressure)
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        # atol reduced since typical volumes are around 1e-5 to 1e-6
        solver = optx.Newton(rtol=1.0e-6, atol=1.0e-12)
        sol = optx.root_find(
            self._objective_function,
            solver,
            initial_volume,
            args=kwargs,
            throw=True,  # TODO: Update throw to False when working
        )
        volume: ArrayLike = sol.value
        # jax.debug.print("volume = {out}", out=volume)

        jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

        return volume

    @jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume integral :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        # FIXME
        raise NotImplementedError()

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        log_fugacity: Array = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_BAR * temperature
        )

        return log_fugacity

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {
            "epsilon": self._epsilon,
            "sigma": self._sigma,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


# Converted from Perple_X
def zd09_pure_species(i, p, t, r, pr, nopt_51=1e-6, max_iter=100):
    """
    Python implementation of the Zhang & Duan (2009) EoS for pure species.

    Parameters:
    - i: species index (1=H2O, 2=CO2, 3=CO, 4=CH4, 5=H2, 7=O2, 16=C2H6)
    - p: pressure in MPa
    - t: temperature in K
    - r: gas constant in J/mol/K
    - pr: reference pressure in MPa
    - nopt_51: convergence threshold
    - max_iter: max iteration for Newton-Raphson

    Returns:
    - vol: molar volume in cm^3/mol
    - lnfug: log fugacity
    """

    # Species-specific constants from the Fortran data blocks
    eps = np.zeros(17)
    sig3 = np.zeros(17)
    eps[[1, 2, 3, 4, 5, 7, 16]] = [510, 235, 105.6, 154, 31.2, 0, 124.5]
    sig3[[1, 2, 3, 4, 5, 7, 16]] = [
        23.887872,
        54.439939,
        49.027896,
        50.28426837,
        25.153757,
        0,
        37.933056,
    ]

    if i not in [1, 2, 3, 4, 5, 7, 16]:
        raise ValueError("Unsupported species index")

    # Initial guess for volume (in cm³/mol)
    vol = 50.0

    prt = p / (10.0 * r * t)
    gamm = 6.123507682 * sig3[i] ** 2
    et = eps[i] / t
    et2 = et**2

    b = (0.5870171892 + (-5.314333643 - 1.498847241 * et) * et2) * sig3[i]
    c = (0.5106889412 + (-2.431331151 + 8.294070444 * et) * et2) * sig3[i] ** 2
    d = (0.4045789083 + (3.437865241 - 5.988792021 * et) * et2) * sig3[i] ** 4
    e = (-0.07351354702 + (0.7017349038 - 0.2308963611 * et) * et2) * sig3[i] ** 5
    f = 1.985438372 * et2 * et * sig3[i] ** 2
    ge = 16.60301885 * et2 * et * sig3[i] ** 4

    for it in range(max_iter):
        vi = 1.0 / vol
        expg = np.exp(-gamm * vi * vi)

        veq = -vi - b * vi**2 + (-f * expg - c) * vi**3 + (-ge * expg - d) * vi**5 - e * vi**6

        dveq = (
            -veq * vi
            + b * vi**3
            + 2.0 * (f * expg + c) * vi**4
            + (-2.0 * f * expg * gamm + 4.0 * ge * expg + 4.0 * d) * vi**6
            + 5.0 * e * vi**7
            - 2.0 * ge * expg * gamm * vi**8
        )

        dv = -(prt + veq) / dveq

        if dv < 0.0 and (vol + dv < 0.0):
            vol *= 0.8
        else:
            vol += dv

        if abs(dv / vol) < nopt_51:
            expg = np.exp(gamm / (vol * vol))
            lnfug = (
                np.log(r * t / vol / pr * 1e1)
                + 0.5 * (f + ge / gamm) * (1.0 - 1.0 / expg) / gamm
                + (
                    2.0 * b
                    + (
                        1.5 * c
                        + (f - 0.5 * ge / gamm) / expg
                        + (1.25 * d + ge / expg + 1.2 * e / vol) / vol**2
                    )
                    / vol
                )
                / vol
            )
            vol *= 10.0  # Convert from J/bar to cm³/mol
            return vol, lnfug

    # If it doesn't converge, return None
    return None, None


CH4_zhang09: RealGasProtocol = ZhangDuan(154.0, 3.691)
H2O_zhang09: RealGasProtocol = ZhangDuan(510.0, 2.88)
CO2_zhang09: RealGasProtocol = ZhangDuan(235.0, 3.79)
H2_zhang09: RealGasProtocol = ZhangDuan(31.2, 2.93)
CO_zhang09: RealGasProtocol = ZhangDuan(105.6, 3.66)
O2_zhang09: RealGasProtocol = ZhangDuan(124.5, 3.36)
C2H6_zhang09: RealGasProtocol = ZhangDuan(246.1, 4.35)


def get_zhang_eos_models() -> dict[str, RealGasProtocol]:
    """Gets a dictionary of Zhang and Duan EOS models that are bounded

    TODO: Make bounded?

    The naming convention is as follows:
        [species]_[eos model]_[citation]

    'cs' refers to corresponding states.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGasProtocol] = {}
    eos_models["CH4_zhang09"] = CH4_zhang09
    eos_models["H2O_zhang09"] = H2O_zhang09
    eos_models["CO2_zhang09"] = CO2_zhang09
    eos_models["H2_zhang09"] = H2_zhang09
    eos_models["CO_zhang09"] = CO_zhang09
    eos_models["O2_zhang09"] = O2_zhang09
    eos_models["C2H6_zhang09"] = C2H6_zhang09

    return eos_models


def test():
    # Table 6 comparisons
    temperature_low: float = 1203.15
    pressure_low: float = 9500
    temperature_high: float = 1873.15
    pressure_high: float = 25000

    volume_low_ZD = H2O_zhang09.volume(temperature_low, pressure_low)
    print("volume_low (Zhang and Duan) = ", volume_low_ZD, ", target = 2.22e-05")
    volume_high_ZD = H2O_zhang09.volume(temperature_high, pressure_high)
    print("volume_high (Zhang and Duan) = ", volume_high_ZD, ", target = 1.941e-05")

    from atmodeller.eos._holland_powell import H2O_cork_holland98 as H2O_cork_holland

    print("")
    # Agrees with the data in Table 6.
    volume_low_HP = H2O_cork_holland.volume(temperature_low, pressure_low)
    print("volume_low (Holland and Powell) = ", volume_low_HP, ", target = 2.160e-05")
    volume_high_HP = H2O_cork_holland.volume(temperature_high, pressure_high)
    print("volume_high (Holland and Powell) = ", volume_high_HP, ", target = 1.837e-05")


if __name__ == "__main__":
    test()
