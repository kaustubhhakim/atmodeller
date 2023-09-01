"""Fugacity coefficients and non-ideal effects.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import fsolve

from atmodeller import GAS_CONSTANT

logger: logging.Logger = logging.getLogger(__name__)


# calculate pure gas fugacity coefficient and Modified Redlich-Kwang volume
def Calc_lambda(P, T, a, b, V_init):
    """
    Because HP98 dataset uses pressure in kbar, for easier implementation,
    pressure in this function is in kbar, temperature in K
    """

    R = 8.314472e-3  # Note unit in kJ requires P in kbar
    EOS = (
        lambda V: P * V**3
        - R * T * V**2
        - (b * R * T + b**2 * P - a / np.sqrt(T)) * V
        - a * b / np.sqrt(T)
    )
    Jacob = lambda V: 3 * P * V**2 - 2 * R * T * V - (b * R * T + b**2 * P - a / np.sqrt(T))
    V_mrk = fsolve(EOS, V_init, fprime=Jacob, xtol=1e-6)

    # compressibility factor
    Z = P * V_mrk / (R * T)
    B = b * P / (R * T)
    A = a / (b * R * T**1.5)
    lnlambda = Z - 1.0 - np.log(Z - B) - A * np.log(1.0 + B / Z)

    return lnlambda, V_mrk


# calculate pure gas fugacity using the CORK equation from HP98
# return value is RTlnf in Joules, rather than f
def Calc_V_f(P, T, name):
    R = 8.314472e-3  # Note unit in kJ requires P in kbar
    if (name == "H2O") | (name == "CO2"):
        if name == "H2O":
            a0 = 1113.4
            a1 = -0.88517
            a2 = 4.53e-3
            a3 = -1.3183e-5
            a4 = -0.22291
            a5 = -3.8022e-4
            a6 = 1.7791e-7
            a7 = 5.8487
            a8 = -2.1370e-2
            a9 = 6.8133e-5
            b = 1.465
            c = 1.9853e-3
            d = -8.9090e-2
            e = 8.0331e-2
            Tc = 673.0
            P0 = 2.0
            Psat = -13.627e-3 + 7.29395e-7 * T**2 - 2.34622e-9 * T**3 + 4.83607e-15 * T**5

            if T < Tc:
                a = a0 + a1 * (Tc - T) + a2 * (Tc - T) ** 2 + a3 * (Tc - T) ** 3
                a_gas = a0 + a7 * (Tc - T) + a8 * (Tc - T) ** 2 + a9 * (Tc - T) ** 3

            else:
                a = a0 + a4 * (T - Tc) + a5 * (T - Tc) ** 2 + a6 * (T - Tc) ** 3

        else:
            a0 = 741.2
            a1 = -0.10891
            a2 = -3.903e-4
            Tc = 304.2
            P0 = 5.0
            a = a0 + a1 * T + a2 * T**2
            b = 3.057
            c = 5.40776e-3 - 1.59046e-6 * T
            d = -1.78198e-1 + 2.45317e-5 * T
            e = 0
            Psat = 0

        if T >= Tc:
            V_init = R * T / P + b
            lnlambda_mrk, V_mrk = Calc_lambda(P, T, a, b, V_init)

        else:
            if P <= Psat:
                V_init = R * T / P + 10.0 * b
                lnlambda_mrk, V_mrk = Calc_lambda(P, T, a_gas, b, V_init)

            else:
                V_init = R * T / P + 10.0 * b
                lnlambda1, V_mrk = Calc_lambda(Psat, T, a_gas, b, V_init)

                V_init = b / 2.0
                lnlambda2, V_mrk = Calc_lambda(Psat, T, a, b, V_init)

                V_init = R * T / P + b
                lnlambda3, V_mrk = Calc_lambda(P, T, a, b, V_init)

                lnlambda_mrk = lnlambda1 - lnlambda2 + lnlambda3

        # virial contribution to MRK (modified Redlich-Kwang)
        if P >= P0:
            lnlambda_vir = (
                1.0
                / (R * T)
                * (
                    c / 2.0 * (P - P0) ** 2
                    + 2.0 / 3.0 * d * (P - P0) ** (3.0 / 2.0)
                    + 4.0 / 5.0 * e * (P - P0) ** (5.0 / 4.0)
                )
            )
            V_vir = c * (P - P0) + d * (P - P0) ** 0.5 + e * (P - P0) ** 0.25

        else:
            lnlambda_vir = 0.0
            V_vir = 0.0

        lnlambda = lnlambda_mrk + lnlambda_vir
        V = V_mrk + V_vir
        RTlnf = R * T * lnlambda + R * T * np.log(P / 1e-3)
        RTlnf = 1e3 * RTlnf  # convert to J

        return V, RTlnf

    # TODO: remove. Dan has now created class-based approach below.
    # else:
    #     if name == "CO":
    #         Tc = 132.9
    #         Pc = 0.0350

    #     elif name == "CH4":
    #         Tc = 190.6
    #         Pc = 0.0460

    #     elif name == "H2":
    #         Tc = 41.2
    #         Pc = 0.0211

    #     a0 = 5.45963e-5
    #     a1 = -8.63920e-6
    #     b0 = 9.18301e-4
    #     c0 = -3.30558e-5
    #     c1 = 2.30524e-6
    #     d0 = 6.93054e-7
    #     d1 = -8.38293e-8

    #     a = a0 * Tc ** (5.0 / 2.0) / Pc + a1 * Tc ** (3.0 / 2.0) / Pc * T
    #     b = b0 * Tc / Pc
    #     c = c0 * Tc / Pc ** (3.0 / 2.0) + c1 / Pc ** (3.0 / 2.0) * T
    #     d = d0 * Tc / Pc**2 + d1 / Pc**2 * T

    #     V = (
    #         R * T / P
    #         + b
    #         - a * R * np.sqrt(T) / (R * T + b * P) / (R * T + 2.0 * b * P)
    #         + c * np.sqrt(P)
    #         + d * P
    #     )
    #     RTlnf = (
    #         R * T * np.log(1000 * P)
    #         + b * P
    #         + a / b / np.sqrt(T) * (np.log(R * T + b * P) - np.log(R * T + 2.0 * b * P))
    #         + 2 / 3 * c * P * np.sqrt(P)
    #         + d / 2 * P**2
    #     )

    #     return V, 1e3 * RTlnf


@dataclass(kw_only=True, frozen=True)
class CorkFull(ABC):
    """Full Cork equation from Holland and Powell (1991)."""

    a0: float
    a1: float
    a2: float
    # a3-a9 only required for H2O.
    b: float
    c0: float
    c1: float
    d0: float
    d1: float
    Tc: float
    P0: float

    # TODO: Might not be possible for H2O EOS.
    # @abstractmethod
    # def a(self, temperature: float) -> float:
    #     raise NotImplementedError

    def c(self, temperature: float) -> float:
        """Virial-type coefficient c. Equation 4, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient c.
        """
        c: float = self.c0 + self.c1 * temperature
        return c

    def d(self, temperature: float) -> float:
        """Virial-type coefficient d. Equation 4, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient d.
        """
        d: float = self.d0 + self.d1 * temperature
        return d

    def volume(self, *, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume (units TODO).
        """
        volume: float = self.volume_MRK(
            temperature=temperature, pressure=pressure
        ) + self.volume_virial(temperature=temperature, pressure=pressure)

        return volume

    def _objective_function_volume_MRK(
        self, volume: float, temperature: float, pressure: float, a: Callable[[float], float]
    ) -> float:
        """Equation A.1, Holland and Powell (1991)."""

        R: float = GAS_CONSTANT * 1.0e-3  # Note unit in kJ requires pressure in kbar.
        residual: float = (
            pressure * volume**3
            - R * temperature * volume**2
            - (
                self.b * R * temperature
                + self.b**2 * pressure
                - a(temperature) / np.sqrt(temperature)
            )
            * volume
            - a(temperature) * self.b / np.sqrt(temperature)
        )

        return residual

    def _volume_MRK_jacobian(
        self, volume: float, temperature: float, pressure: float, a: Callable[[float], float]
    ):
        """Jacobian of Equation A.1, Holland and Powell (1991)."""

        R: float = GAS_CONSTANT * 1.0e-3  # Note unit in kJ requires pressure in kbar.
        jacobian: float = (
            3 * pressure * volume**2
            - 2 * R * temperature * volume
            - (
                self.b * R * temperature
                + self.b**2 * pressure
                - a(temperature) / np.sqrt(temperature)
            )
        )
        return jacobian

    def volume_virial(self, *, temperature: float, pressure: float) -> float:
        """Virial-type volume term.

        In the Appendix of Holland and Powell (1991) it states that the virial component of volume
        is added to the MRK component if the pressure is above P0.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            Virial-type volume term.
        """
        if pressure > self.P0:
            volume_virial: float = self.c(temperature) * (pressure - self.P0) ** 0.5 + self.d(
                temperature
            ) * (pressure - self.P0)
        else:
            volume_virial = 0
        return volume_virial


@dataclass(kw_only=True, frozen=True)
class CorkFullCO2(CorkFull):
    """Full Cork equation for CO2 from Holland and Powell (1991)."""

    a0: float = field(init=False, default=741.2)
    a1: float = field(init=False, default=-0.10891)
    a2: float = field(init=False, default=-3.903e-4)
    b: float = field(init=False, default=3.057)
    c0: float = field(init=False, default=-2.26924e-1)
    c1: float = field(init=False, default=7.73793e-5)
    d0: float = field(init=False, default=1.33790e-2)
    d1: float = field(init=False, default=-1.01740e-5)
    Tc: float = field(init=False, default=304.2)
    P0: float = field(init=False, default=5.0)

    def a(self, temperature: float) -> float:
        """Coefficient a. Holland and Powell (1991), p270.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient a.
        """
        a: float = self.a0 + self.a1 * temperature + self.a2 * temperature**2
        return a


@dataclass(kw_only=True, frozen=True)
class CorkFullH2O(CorkFull):
    """Full Cork equation for H2O from Holland and Powell (1991)."""

    a0: float = field(init=False, default=1113.4)
    a1: float = field(init=False, default=-0.88517)
    a2: float = field(init=False, default=4.53e-3)
    a3: float = field(init=False, default=-1.3183e-5)
    a4: float = field(init=False, default=-0.22291)
    a5: float = field(init=False, default=-3.8022e-4)
    a6: float = field(init=False, default=1.7791e-7)
    a7: float = field(init=False, default=5.8487)
    a8: float = field(init=False, default=-2.1370e-2)
    a9: float = field(init=False, default=6.8133e-5)
    b: float = field(init=False, default=1.465)
    c0: float = field(init=False, default=-3.025650e-2)
    c1: float = field(init=False, default=-5.343144e-6)
    d0: float = field(init=False, default=-3.2297554e-3)
    d1: float = field(init=False, default=2.2215221e-6)
    Tc: float = field(init=False, default=673.0)
    P0: float = field(init=False, default=2.0)

    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991).

        Args:
            temperature: Temperature.

        Returns:
            Saturation curve pressure.
        """
        Psat: float = (
            -13.627e-3
            + 7.29395e-7 * temperature**2
            - 2.34622e-9 * temperature**3
            + 4.83607e-15 * temperature**5
        )
        return Psat

    def a_gas(self, temperature: float) -> float:
        """Coefficient a for gaseous H2O. Equation 6a, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient a for gaseous H2O.
        """
        a: float = (
            self.a0
            + self.a7 * (self.Tc - temperature)
            + self.a8 * (self.Tc - temperature) ** 2
            + self.a9 * (self.Tc - temperature) ** 3
        )
        return a

    def a(self, temperature: float) -> float:
        """Coefficient a for liquid H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient a for liquid H2O.
        """
        if temperature < self.Tc:
            a: float = (
                self.a0
                + self.a1 * (self.Tc - temperature)
                + self.a2 * (self.Tc - temperature) ** 2
                + self.a3 * (self.Tc - temperature) ** 3
            )
        else:
            a = (
                self.a0
                + self.a4 * (temperature - self.Tc)
                + self.a5 * (temperature - self.Tc) ** 2
                + self.a6 * (temperature - self.Tc) ** 3
            )
        return a

    def volume_MRK(self, *, temperature: float, pressure: float) -> float:
        """MRK volume.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            MRK volume (units TODO).
        """
        R: float = GAS_CONSTANT * 1.0e-3  # Note unit in kJ requires pressure in kbar.

        if temperature >= self.Tc:
            # Only one real root.
            # This would always be triggered for CO2, but trigger warning/exception if not.
            volume_init: float = R * temperature / pressure + self.b
            a: Callable[[float], float] = self.a
        else:
            if pressure <= self.Psat(temperature):
                volume_init: float = R * temperature / pressure + 10.0 * self.b
                a = self.a_gas
            else:
                # TODO: do stuff
                ...

        volume_MRK = fsolve(
            self._objective_function_volume_MRK,
            volume_init,
            args=(temperature, pressure, a),
            fprime=self._volume_MRK_jacobian,
        )


@dataclass(kw_only=True, frozen=True)
class CorkSimple:
    """Simplified Cork equation from Holland and Powell (1991).

    Although originally fit to CO2 data, this predicts the volumes and fugacities for several other
    gases which are known to obey approximately the principle of corresponding states.

    Corresponding states parameters from Table 2 in Holland and Powell (1991).

    Args:
        Tc: TODO.
        Pc: TODO.

    Attributes:
        TODO.
    """

    # FIXME: Note units.
    Tc: float
    Pc: float
    a0: float = field(init=False, default=5.45963e-5)
    a1: float = field(init=False, default=-8.63920e-6)
    b0: float = field(init=False, default=9.18301e-4)
    c0: float = field(init=False, default=-3.30558e-5)
    c1: float = field(init=False, default=2.30524e-6)
    d0: float = field(init=False, default=6.93054e-7)
    d1: float = field(init=False, default=-8.38293e-8)

    def a(self, temperature: float) -> float:
        """Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient a.
        """
        a: float = (
            self.a0 * self.Tc ** (5.0 / 2.0) / self.Pc
            + self.a1 * self.Tc ** (3.0 / 2.0) / self.Pc * temperature
        )
        return a

    def b(self, temperature: float) -> float:
        """Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient b.
        """
        del temperature
        b: float = self.b0 * self.Tc / self.Pc
        return b

    def c(self, temperature: float) -> float:
        """Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient c.
        """
        c: float = (
            self.c0 * self.Tc / self.Pc ** (3.0 / 2.0)
            + self.c1 / self.Pc ** (3.0 / 2.0) * temperature
        )
        return c

    def d(self, temperature: float) -> float:
        """Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            Coefficient d.
        """
        d: float = self.d0 * self.Tc / self.Pc**2 + self.d1 / self.Pc**2 * temperature
        return d

    def RTlnf(self, *, temperature: float, pressure: float) -> float:
        """Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            RTlnf in Joules. TODO: Must be per mol?
        """
        R = GAS_CONSTANT * 1.0e-3  # Note unit in kJ requires pressure in kbar.
        RTlnf: float = (
            R * temperature * np.log(1000 * pressure)
            + self.b(temperature) * pressure
            + self.a(temperature)
            / self.b(temperature)
            / np.sqrt(temperature)
            * (
                np.log(R * temperature + self.b(temperature) * pressure)
                - np.log(R * temperature + 2.0 * self.b(temperature) * pressure)
            )
            + 2 / 3 * self.c(temperature) * pressure * np.sqrt(pressure)
            + self.d(temperature) / 2 * pressure**2
        )
        RTlnf *= 1e3  # kJ to J. Must actually be J/mol?
        return RTlnf

    def fugacity(self, *, temperature: float, pressure: float) -> float:
        """Fugacity.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity in bar.
        """
        fugacity: float = np.exp(
            self.RTlnf(temperature=temperature, pressure=pressure) / (GAS_CONSTANT * temperature)
        )
        return fugacity

    def fugacity_coefficient(self, *, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity coefficient.
        """
        # TODO: Clean up. Clunky for fugacity to require kbar yet return bar.
        fugacity_coefficient: float = self.fugacity(temperature=temperature, pressure=pressure) / (
            pressure * 1e3
        )
        return fugacity_coefficient

    def volume(self, temperature: float, pressure: float) -> float:
        """Equation 7a, Holland and Powell (1991).

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume (units TODO).
        """
        R = GAS_CONSTANT * 1.0e-3  # Note unit in kJ requires P in kbar.
        volume: float = (
            R * temperature / pressure
            + self.b(temperature)
            - self.a(temperature)
            * R
            * np.sqrt(temperature)
            / (R * temperature + self.b(temperature) * pressure)
            / (R * temperature + 2.0 * self.b(temperature) * pressure)
            + self.c(temperature) * np.sqrt(pressure)
            + self.d(temperature) * pressure
        )
        return volume


@dataclass(kw_only=True, frozen=True)
class CorkSimpleCO2(CorkSimple):
    """Critical constants for CO2.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=304.2)
    Pc: float = field(init=False, default=0.0738)


@dataclass(kw_only=True, frozen=True)
class CorkCH4(CorkSimple):
    """Critical constants for CH4.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=190.6)
    Pc: float = field(init=False, default=0.0460)


@dataclass(kw_only=True, frozen=True)
class CorkH2(CorkSimple):
    """Critical constants for H2.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=41.2)
    Pc: float = field(init=False, default=0.0211)


@dataclass(kw_only=True, frozen=True)
class CorkCO(CorkSimple):
    """Critical constants for CO.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=132.9)
    Pc: float = field(init=False, default=0.0350)
