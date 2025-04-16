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

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, lax
from jax.typing import ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibration, OptxSolver, safe_exp, unit_conversion

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier


class ZhangDuan(RealGas):
    r"""Real gas EOS :cite:p:`ZD09`

    Args:
        epsilon: Lenard-Jones parameter (epsilon/kB) in K
        sigma: Lenard-Jones parameter in :math:`10^{-10}` m
        initial_volume: Initial volume guess in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`. The choice
            of the initial volume is very important to avoid a jump to a different solution branch
            that is not the correct physical one. Ideally we want an initial volume that will
            converge across the entire range of pressures and temperatures, at least up to the
            maximum calibration pressure of 10 GPa. The initial volume may be different for each
            species.
    """

    coefficients: ClassVar[tuple[float, ...]] = (
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
    """Coefficients"""

    epsilon: float
    """Lenard-Jones parameter (epsilon/kB) in K"""
    sigma: float
    r"""Lenard-Jones parameter in :math:`10^{-10}` m"""
    initial_volume: float = 1.0e-5
    r"""Initial volume guess in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`"""

    @eqx.filter_jit
    def _Pm(self, pressure: ArrayLike) -> Array:
        """Scaled pressure

        Args:
            pressure: Pressure in bar

        Returns:
            Scaled pressure
        """
        pressure_MPa: ArrayLike = pressure * unit_conversion.bar_to_MPa
        scaled_pressure: Array = 3.0636 * jnp.power(self.sigma, 3) * pressure_MPa / self.epsilon
        # jax.debug.print("scaled_pressure = {out}", out=scaled_pressure)

        return scaled_pressure

    @eqx.filter_jit
    def _Tm(self, temperature: ArrayLike) -> ArrayLike:
        """Scaled temperature

        Args:
            temperature: Temperature in K

        Returns:
            Scaled temperature
        """
        scaled_temperature: ArrayLike = 154.0 * temperature / self.epsilon
        # jax.debug.print("scaled_temperature = {out}", out=scaled_temperature)

        return scaled_temperature

    @eqx.filter_jit
    def _Vm(self, volume: ArrayLike) -> Array:
        r"""Scaled volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`

        Returns:
            Scaled volume
        """
        volume_cm3: ArrayLike = volume * unit_conversion.m3_to_cm3
        sigma_term: Array = jnp.power(self.sigma / 3.691, 3)
        scaled_volume: Array = volume_cm3 / 1000.0 / sigma_term  # type:ignore
        # jax.debug.print("scaled_volume = {out}", out=scaled_volume)

        return scaled_volume

    @eqx.filter_jit
    def _get_parameter(self, Tm: ArrayLike, coefficients: tuple[float, ...]) -> Array:
        """Gets the parameter (coefficient) for polynomials involving Tm terms

        Args:
            Tm: Scaled temperature
            coefficients: Coefficients for this term

        Returns:
            Parameter (coefficient)
        """
        return (
            coefficients[0] + coefficients[1] / jnp.square(Tm) + coefficients[2] / jnp.power(Tm, 3)
        )

    @eqx.filter_jit
    def _S1(self, Tm: ArrayLike, Vm: ArrayLike) -> Array:
        """S1 term :cite:p:`ZD09{Equation 15}`

        Args:
            Tm: Scaled temperature
            Vm: Scaled volume

        Returns:
            S1 term
        """
        b: Array = self._get_parameter(Tm, self.coefficients[0:3])
        c: Array = self._get_parameter(Tm, self.coefficients[3:6])
        d: Array = self._get_parameter(Tm, self.coefficients[6:9])
        e: Array = self._get_parameter(Tm, self.coefficients[9:12])
        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]

        S1: Array = (
            b / Vm
            + c / (2 * jnp.square(Vm))
            + d / (4 * jnp.power(Vm, 4))
            + e / (5 * jnp.power(Vm, 5))
        )
        S1 = S1 + (
            a13
            / (2 * a15 * jnp.power(Tm, 3))
            * (a14 + 1 - (a14 + 1 + a15 / jnp.square(Vm)) * safe_exp(-a15 / jnp.square(Vm)))
        )
        # jax.debug.print("S1 = {out}", out=S1)

        return S1

    @eqx.filter_jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume :cite:p:`ZD09{Equation 8}`.

        Note that the left-hand side of :cite:t:`ZD09{Equation 8}` is the compressibility factor
        so should be expressed in terms of P, V, R, and T, and not the scaled equivalents.

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]
        # jax.debug.print("temperature = {temperature}", temperature=temperature)
        # jax.debug.print("pressure = {pressure}", pressure=pressure)

        Tm: ArrayLike = self._Tm(temperature)
        # jax.debug.print("Tm = {Tm}", Tm=Tm)
        Vm: ArrayLike = self._Vm(volume)
        # jax.debug.print("Vm = {Vm}", Vm=Vm)
        # Zhang and Duan (2009) use scaled quantities, according to the paper, but this is
        # presumably a typo and in fact the actual P and T should be used. This agrees with the
        # Perple_X implementation and the reported volumes in Table 6.
        ptr: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        # jax.debug.print("ptr = {ptr}", ptr=ptr)

        b: ArrayLike = self._get_parameter(Tm, self.coefficients[0:3])
        # jax.debug.print("b = {b}", b=b)
        c: ArrayLike = self._get_parameter(Tm, self.coefficients[3:6])
        # jax.debug.print("c = {c}", c=c)
        d: ArrayLike = self._get_parameter(Tm, self.coefficients[6:9])
        # jax.debug.print("d = {d}", d=d)
        e: ArrayLike = self._get_parameter(Tm, self.coefficients[9:12])
        # jax.debug.print("e = {e}", e=e)

        term1: Array = (
            jnp.asarray(1)
            + b / jnp.asarray(Vm)
            + c / jnp.power(Vm, 2)
            + d / jnp.power(Vm, 4)
            + e / jnp.power(Vm, 5)
        )
        # jax.debug.print("term1 = {term1}", term1=term1)

        a13: float = self.coefficients[12]
        a14: float = self.coefficients[13]
        a15: float = self.coefficients[14]
        term2: Array = (
            a13
            / jnp.power(Tm, 3)
            / jnp.power(Vm, 2)
            * (a14 + a15 / jnp.square(Vm))
            * safe_exp(-a15 / jnp.square(Vm))
        )
        # jax.debug.print("term2 = {term2}", term2=term2)

        residual: Array = term1 + term2 - ptr
        # jax.debug.print("residual = {residual}", residual=residual)

        return residual

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        # Find the broadcasted shape of temperature and pressure
        temperature = jnp.asarray(temperature)
        pressure = jnp.asarray(pressure)
        out_shape: tuple[int, ...] = lax.broadcast_shapes(temperature.shape, pressure.shape)

        # Broadcast the initial volume to match
        initial: Array = jnp.broadcast_to(self.initial_volume, out_shape)

        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(self._objective_function, solver, initial, args=kwargs, throw=THROW)
        volume: ArrayLike = sol.value
        # jax.debug.print("volume = {out}", out=volume)
        # jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])

        return volume

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity :cite:p:`ZD09{Equation 14}`

        This is for a pure species and does not include the terms to enable end member mixing.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        Vm: Array = self._Vm(volume)
        Tm: ArrayLike = self._Tm(temperature)
        # Compute the compressibility factor directly to avoid another solve for volume
        Z: ArrayLike = pressure * volume / (GAS_CONSTANT_BAR * temperature)
        log_fugacity_coefficient: Array = -jnp.log(Z) + self._S1(Tm, Vm) + Z - 1
        log_fugacity: Array = log_fugacity_coefficient + jnp.log(pressure)
        # jax.debug.print("log_fugacity_coefficient = {out}", out=log_fugacity_coefficient)

        return log_fugacity

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
        log_fugacity: Array = self.log_fugacity(temperature, pressure)
        volume_integral: Array = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral


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
    eps = np.array([510, 235, 105.6, 154, 31.2, 124.5, 246.1])
    sig = np.array([2.88, 3.79, 3.66, 3.691, 2.93, 3.36, 4.35])

    sig3 = sig**3

    # Initial guess for volume (in cm³/mol)
    vol = 50.0

    # P is in bar, divide by ten converts to MPa
    # r is 8.314, t is in K
    # Note that these appear to be the actual P and T, not scaled variables.
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
        # vi is inverse of the actual volume, not scaled
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
            # TODO: Perple_X has this term, but the value seems to be a factor of ten larger
            # than the volume in cm^3. Unsure why.
            vol *= 10.0  # Convert from J/bar to cm³/mol
            return vol, lnfug

    # If it doesn't converge, return None
    return None, None


# The choice of the initial volume guess is really important and the value should ideally allow the
# model to converge to the correct physical route across the entire range of pressures and
# temperatures, at least up to the maximum calibration pressure of 10 GPa.
CH4_zhang09: RealGasProtocol = ZhangDuan(154.0, 3.691)

H2O_zhang09: RealGasProtocol = ZhangDuan(510.0, 2.88)

CO2_zhang09: RealGasProtocol = ZhangDuan(235.0, 3.79)

# Tested boundedness (not the same as physical correctness) for 500<T<10000 K and 0<P<10 GPa
H2_zhang09: RealGasProtocol = ZhangDuan(31.2, 2.93, 10 * unit_conversion.cm3_to_m3)
"""H2 :cite:p:`ZD09`"""
H2_experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=250,
    temperature_max=423,
    pressure_min=2 * unit_conversion.MPa_to_bar,
    pressure_max=700 * unit_conversion.MPa_to_bar,
)
r"""Experimental calibration for H2 :cite:p:`ZD09{Table 5}`"""
H2_zhang09_bounded: RealGasProtocol = CombinedRealGas.create(
    [H2_zhang09], [H2_experimental_calibration]
)
"""H2 bounded to experimental data :cite:p:`ZD09{Table 5}`"""

CO_zhang09: RealGasProtocol = ZhangDuan(105.6, 3.66)
O2_zhang09: RealGasProtocol = ZhangDuan(124.5, 3.36)
C2H6_zhang09: RealGasProtocol = ZhangDuan(246.1, 4.35)


def get_zhang_eos_models() -> dict[str, RealGasProtocol]:
    """Gets a dictionary of Zhang and Duan EOS models

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGasProtocol] = {}
    eos_models["CH4_zhang09"] = CH4_zhang09
    eos_models["H2O_zhang09"] = H2O_zhang09
    eos_models["CO2_zhang09"] = CO2_zhang09
    eos_models["H2_zhang09"] = H2_zhang09  # _bounded
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


def perple_X():
    """The volume agrees with the data in Table 6 if the volume is divided by 10."""

    temperature_low: float = 1203.15
    pressure_low: float = 9500
    temperature_high: float = 1873.15
    pressure_high: float = 25000

    GAS_CONSTANT = GAS_CONSTANT_BAR * 1e5

    volume_low_perplex, lnfug_low_perplex = zd09_pure_species(
        0, pressure_low, temperature_low, GAS_CONSTANT, 1
    )
    print("volume_low_perplex = ", volume_low_perplex, ", target = 2.22e-05")
    print("lnfug_low_perplex = ", lnfug_low_perplex)
    print("fug_low_perplex = ", np.exp(lnfug_low_perplex))  # type: ignore

    volume_low_ZD = H2O_zhang09.volume(temperature_low, pressure_low)
    print("volume_low = ", volume_low_ZD, ", target = 2.22e-05")
    log_fugacity_coefficient_ZD = H2O_zhang09.log_fugacity_coefficient(
        temperature_low, pressure_low
    )
    print("log_fugacity_coefficient_ZD = ", log_fugacity_coefficient_ZD)
    fugacity_ZD = np.exp(log_fugacity_coefficient_ZD) * pressure_low
    print("fugacity_ZD = ", fugacity_ZD)

    print()

    volume_high_perplex, lnfug_high_perplex = zd09_pure_species(
        0, pressure_high, temperature_high, GAS_CONSTANT, 1
    )
    print("volume_high_perplex = ", volume_high_perplex, ", target = 1.941e-05")
    print("lnfug_high_perplex = ", lnfug_high_perplex)
    print("fug_high_perplex = ", np.exp(lnfug_high_perplex))  # type: ignore

    volume_high_ZD = H2O_zhang09.volume(temperature_high, pressure_high)
    print("volume_high = ", volume_high_ZD, ", target = 1.941e-05")
    log_fugacity_coefficient_ZD = H2O_zhang09.log_fugacity_coefficient(
        temperature_high, pressure_high
    )
    print("log_fugacity_coefficient_ZD = ", log_fugacity_coefficient_ZD)
    fugacity_ZD = np.exp(log_fugacity_coefficient_ZD) * pressure_high
    print("fugacity_ZD = ", fugacity_ZD)


if __name__ == "__main__":
    # print("test below")
    # test()
    # print()
    print("perple_X below")
    perple_X()
