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

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.typing import ArrayLike

from atmodeller.eos.core import RealGas

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override


if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class ZhangDuan(RealGas):
    """Real gas EOS from :cite:t:`ZD09`

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

    @override
    @jit
    def compressibility_factor(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Compressibility factor :cite:p:`ZD09{Equation 8}`

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility factor, which is dimensionless
        """
        Tr: ArrayLike = self.scaled_temperature(temperature)
        Pr: ArrayLike = self.scaled_pressure(pressure)
        Z: Array = (
            self._a(Tr)
            + self._b(Tr) * Pr
            + self._c(Tr) * jnp.square(Pr)
            + self._d(Tr) * jnp.power(Pr, 3)
        )

        return Z

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Volume :cite:p:`SS92{Equation 1}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        Z: Array = self.compressibility_factor(temperature, pressure)
        volume: Array = Z * self.ideal_volume(temperature, pressure)

        return volume

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
