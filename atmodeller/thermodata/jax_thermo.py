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
"""JAX thermochemical data"""

# Convenient to use chemical formulas so pylint: disable=C0103

from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike
from scipy.constants import gas_constant

Href: float = 298.15
"""Enthalpy reference temperature in K"""
Pref: float = 1.0
"""Standard state pressure in bar"""


class ThermoData(NamedTuple):
    """Thermochemical data

    Args:
        b1: Enthalpy integration constant
        b2: Entropy integration constant
        cp_coefficients: Coefficients for the heat capacity
        T_min: Minimum temperature for data fit in K
        T_max: Maximum temperature for data fit in K
    """

    b1: ArrayLike
    b2: ArrayLike
    cp_coefficients: tuple[float, ...]
    T_min: float
    T_max: float

    def cp_over_R(self, temperature: ArrayLike) -> Array:
        """Heat capacity relative to the gas constant (R)

        Args:
            temperature: Temperature

        Returns:
            Heat capacity (J/K/mol) relative to R
        """
        return _cp_over_R(self.cp_coefficients, temperature)

    def G_over_RT(self, temperature: ArrayLike) -> Array:
        """Gibbs

        Args:
            temperature: Temperature

        Returns:
            Gibbs energy (J/mol) relative to RT
        """
        return _G_over_RT(self.cp_coefficients, self.b1, self.b2, temperature)

    def S_over_R(self, temperature: ArrayLike) -> Array:
        """Entropy relative to R

        Args:
            temperature: Temperature

        Returns:
            Entropy (J/K/mol) relative to R
        """
        return _S_over_R(self.cp_coefficients, self.b2, temperature)

    def H_over_RT(self, temperature: ArrayLike) -> Array:
        """Enthalpy relative to RT

        Args:
            temperature: Temperature

        Returns:
            Enthalpy (J/mol) relative to RT
        """
        return _H_over_RT(self.cp_coefficients, self.b1, temperature)


@partial(jit, static_argnames=["cp_coefficients"])
def _cp_over_R(cp_coefficients: tuple[float, ...], temperature: ArrayLike) -> Array:
    """Heat capacity relative to the gas constant (R)

    Args:
        cp_coefficients: Coefficients for the heat capacity
        temperature: Temperature in K

    Returns:
        Heat capacity (J/K/mol) relative to R
    """
    cp: Array = (
        cp_coefficients[0] * jnp.power(temperature, -2)
        + cp_coefficients[1] * jnp.power(temperature, -1)
        + cp_coefficients[2]
        + cp_coefficients[3] * temperature
        + cp_coefficients[4] * jnp.power(temperature, 2)
        + cp_coefficients[5] * jnp.power(temperature, 3)
        + cp_coefficients[6] * jnp.power(temperature, 4)
    )

    return cp


@partial(jit, static_argnames=["cp_coefficients", "b2"])
def _S_over_R(cp_coefficients: tuple[float, ...], b2: float, temperature: ArrayLike) -> Array:
    """Entropy relative to the gas constant (R)

    Args:
        cp_coefficients: Coefficients for the heat capacity
        b2: Entropy integration constant
        temperature: Temperature in K

    Returns:
        Entropy (J/K/mol) relative to the gas constant
    """
    S: Array = (
        -cp_coefficients[0] * jnp.power(temperature, -2) / 2
        - cp_coefficients[1] * jnp.power(temperature, -1)
        + cp_coefficients[2] * jnp.log(temperature)
        + cp_coefficients[3] * temperature
        + cp_coefficients[4] * jnp.power(temperature, 2) / 2
        + cp_coefficients[5] * jnp.power(temperature, 3) / 3
        + cp_coefficients[6] * jnp.power(temperature, 4) / 4
        + b2
    )

    return S


@partial(jit, static_argnames=["cp_coefficients", "b1"])
def _H_over_RT(cp_coefficients: tuple[float, ...], b1: float, temperature: ArrayLike) -> Array:
    """Enthalpy relative to RT

    Args:
        cp_coefficients: Coefficients for the heat capacity
        b1: Enthalpy integration constant
        temperature: Temperature in K

    Returns:
        Enthalpy (J/mol) relative to RT
    """
    H: Array = (
        -cp_coefficients[0] * jnp.power(temperature, -2)
        + cp_coefficients[1] * jnp.log(temperature) / temperature
        + cp_coefficients[2]
        + cp_coefficients[3] * temperature / 2
        + cp_coefficients[4] * jnp.power(temperature, 2) / 3
        + cp_coefficients[5] * jnp.power(temperature, 3) / 4
        + cp_coefficients[6] * jnp.power(temperature, 4) / 5
        + b1 / temperature
    )

    return H


@partial(jit, static_argnames=["cp_coefficients", "b1", "b2"])
def _G_over_RT(
    cp_coefficients: tuple[float, ...], b1: float, b2: float, temperature: ArrayLike
) -> Array:
    """Gibbs

    Args:
        cp_coefficients: Coefficients for the heat capacity
        b1: Enthalpy integration constant
        b2: Entropy integration constant
        temperature: Temperature in K

    Returns:
        Gibbs energy
    """
    H_over_RT: Array = _H_over_RT(cp_coefficients, b1, temperature)
    S_over_R: Array = _S_over_R(cp_coefficients, b2, temperature)

    G: Array = H_over_RT - S_over_R

    return G


CO_200_1000 = ThermoData(
    -1.303131878e4,
    -7.859241350,
    (
        1.489045326e4,
        -2.922285939e2,
        5.724527170,
        -8.176235030e-3,
        1.456903469e-5,
        -1.087746302e-8,
        3.027941827e-12,
    ),
    200.0,
    1000.0,
)
"""CO for 200 to 1000 K :cite:p:`MZG02{Page 84}`"""

CO_1000_2000 = ThermoData(
    -2.466261084e3,
    -1.387413108e1,
    (
        4.619197250e5,
        -1.944704863e3,
        5.916714180,
        -5.664282830e-4,
        1.398814540e-7,
        -1.787680361e-11,
        9.620935570e-16,
    ),
    1000.0,
    6000.0,
)
"""CO for 1000 to 6000 K :cite:p:`MZG02{Page 84}`"""

CO_6000_20000 = ThermoData(
    5.701421130e6,
    -2.060704786e3,
    (
        8.868662960e8,
        -7.500377840e5,
        2.495474979e2,
        -3.956351100e-2,
        3.297772080e-6,
        -1.318409933e-10,
        1.998937948e-15,
    ),
    6000.0,
    20000.0,
)
"""CO for 6000 to 20000 K :cite:p:`MZG02{Page 84}`"""

CO = CO_200_1000
# CO = CO_1000_2000
# CO = CO_6000_20000

if __name__ == "__main__":

    temperature_in = 1000.0
    print(CO)

    # Agrees with online JANAF tables
    cp = CO.cp_over_R(temperature_in) * gas_constant
    # Agrees with online JANAF tables
    S = CO.S_over_R(temperature_in) * gas_constant
    H = CO.H_over_RT(temperature_in) * gas_constant * temperature_in
    G = CO.G_over_RT(temperature_in)

    print(cp, S, H, G)

    G_scaled = G * gas_constant * temperature_in

    print(G_scaled)
