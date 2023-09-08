#!/usr/bin/env python3

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
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.constants import kilo
from scipy.optimize import fsolve

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import GetValueABC
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)

GAS_CONSTANT_KJ: float = GAS_CONSTANT / kilo


# calculate pure gas fugacity coefficient and Modified Redlich-Kwang volume
def Calc_lambda(P, T, a, b, V_init):
    """
    Because HP98 dataset uses pressure in kbar, for easier implementation,
    pressure in this function is in kbar, temperature in K
    """

    EOS = (
        lambda V: P * V**3
        - GAS_CONSTANT_KJ * T * V**2
        - (b * GAS_CONSTANT_KJ * T + b**2 * P - a / np.sqrt(T)) * V
        - a * b / np.sqrt(T)
    )
    Jacob = (
        lambda V: 3 * P * V**2
        - 2 * GAS_CONSTANT_KJ * T * V
        - (b * GAS_CONSTANT_KJ * T + b**2 * P - a / np.sqrt(T))
    )
    V_mrk = fsolve(EOS, V_init, fprime=Jacob, xtol=1e-6)

    # compressibility factor
    Z = P * V_mrk / (GAS_CONSTANT_KJ * T)
    B = b * P / (GAS_CONSTANT_KJ * T)
    A = a / (b * GAS_CONSTANT_KJ * T**1.5)
    lnlambda = Z - 1.0 - np.log(Z - B) - A * np.log(1.0 + B / Z)

    return lnlambda, V_mrk


# calculate pure gas fugacity using the CORK equation from HP98
# return value is RTlnf in Joules, rather than f
def Calc_V_f(P, T, name):
    R = GAS_CONSTANT * 1e-3  # Note unit in kJ requires P in kbar
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
                lnlambda_mrk, V_mrk = Calc_lambda(P, T, a_gas, b, V_init)  # type: ignore

            else:
                V_init = R * T / P + 10.0 * b
                lnlambda1, V_mrk = Calc_lambda(Psat, T, a_gas, b, V_init)  # type: ignore

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
        V = V_mrk + V_vir  # type: ignore
        RTlnf = R * T * lnlambda + R * T * np.log(P / 1e-3)
        RTlnf = 1e3 * RTlnf  # convert to J

        return V, RTlnf

    else:
        if name == "CO":
            Tc = 132.9
            Pc = 0.0350

        elif name == "CH4":
            Tc = 190.6
            Pc = 0.0460

        elif name == "H2":
            Tc = 41.2
            Pc = 0.0211

        a0 = 5.45963e-5
        a1 = -8.63920e-6
        b0 = 9.18301e-4
        c0 = -3.30558e-5
        c1 = 2.30524e-6
        d0 = 6.93054e-7
        d1 = -8.38293e-8

        a = a0 * Tc ** (5.0 / 2.0) / Pc + a1 * Tc ** (3.0 / 2.0) / Pc * T  # type: ignore
        b = b0 * Tc / Pc  # type: ignore
        c = c0 * Tc / Pc ** (3.0 / 2.0) + c1 / Pc ** (3.0 / 2.0) * T  # type: ignore
        d = d0 * Tc / Pc**2 + d1 / Pc**2 * T  # type: ignore

        V = (
            R * T / P
            + b
            - a * R * np.sqrt(T) / (R * T + b * P) / (R * T + 2.0 * b * P)
            + c * np.sqrt(P)
            + d * P
        )
        RTlnf = (
            R * T * np.log(1000 * P)
            + b * P
            + a / b / np.sqrt(T) * (np.log(R * T + b * P) - np.log(R * T + 2.0 * b * P))
            + 2 / 3 * c * P * np.sqrt(P)
            + d / 2 * P**2
        )

        return V, 1e3 * RTlnf


@dataclass(kw_only=True, frozen=True)
class VirialCompensation:
    """A compensation term for the increasing deviation of the MRK volumes with pressure.

    General form of the equation from Holland and Powell (1998):

        V_virial = a(P-P0) + b(P-P0)**0.5 + c(P-P0)**0.25

    This form also works for the virial compensation term from Holland and Powell (1991), in which
    case c=0. Pc and Tc are required for gases which are known to obey approximately the principle
    of corresponding states.

    Args:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally scaled by Tc and Pc in the case of corresponding states.
        b_coefficients: As above for the b coefficients.
        c_coefficients: As above for the c coefficients.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar. Defaults
            to zero for the corresponding states case.
        Tc: Critical temperature in kelvin. Defaults to 1 (not used for full CORK).
        Pc: Critical pressure in kbar. Defaults to 1 (not used for full CORK).

    Attributes:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally scaled by Tc and Pc in the case of corresponding states.
        b_coefficients: As above for the b coefficients.
        c_coefficients: As above for the c coefficients.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar.
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure in kbar.
    """

    a_coefficients: tuple[float, float]
    b_coefficients: tuple[float, float]
    c_coefficients: tuple[float, float]
    P0: float = 0  # For corresponding states this should default to zero.
    Pc: float = 1  # Only non unity for corresponding states.
    Tc: float = 1  # Only non unity for corresponding states.

    def a(self, temperature: float) -> float:
        """a parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            a parameter.
        """
        a: float = self.a_coefficients[0] * self.Tc + self.a_coefficients[1] * temperature
        a /= self.Pc**2

        return a

    def b(self, temperature: float) -> float:
        """b parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            b parameter.
        """
        b: float = self.b_coefficients[0] * self.Tc + self.b_coefficients[1] * temperature
        b /= self.Pc ** (3 / 2)

        return b

    def c(self, temperature: float) -> float:
        """c parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            c parameter.
        """
        c: float = self.c_coefficients[0] * self.Tc + self.c_coefficients[1] * temperature
        c /= self.Pc ** (5 / 4)

        return c

    def ln_fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Natural log of the virial contribution to the fugacity coefficient.

        Equation A.2., Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            ln(fugacity_coefficient).
        """

        ln_fugacity_coefficient: float = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_KJ * temperature
        )

        return ln_fugacity_coefficient

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient of the virial contribution.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity_coefficient.
        """
        fugacity_coefficient: float = np.exp(self.ln_fugacity_coefficient(temperature, pressure))

        return fugacity_coefficient

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume contribution.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume term.
        """
        volume: float = (
            self.a(temperature) * (pressure - self.P0)
            + self.b(temperature) * (pressure - self.P0) ** 0.5
            + self.c(temperature) * (pressure - self.P0) ** 0.25
        )

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP) contribution.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume integral in kJ mol^(-1).
        """
        volume_integral: float = (
            self.a(temperature) / 2.0 * (pressure - self.P0) ** 2
            + 2.0 / 3.0 * self.b(temperature) * (pressure - self.P0) ** (3.0 / 2.0)
            + 4.0 / 5.0 * self.c(temperature) * (pressure - self.P0) ** (5.0 / 4.0)
        )

        return volume_integral


# TODO: Move comment elsewhere.
#    In the Appendix of Holland and Powell (1991) it states that the virial component of
#    fugacity is added to the MRK component if the pressure is above P0.


@dataclass(kw_only=True)
class CorkFull(GetValueABC):
    """A Full Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    The units in Holland and Powell (1991) are K and kbar, so note unit conversions where relevant.

    Constants correspond to Table 1 in Holland and Powell (1991).

    Args:
        Tc: Critical temperature in kelvin.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar.
        a0: Constant (Table 1, Holland and Powell, 1991).
        a1: Constant (Table 1, Holland and Powell, 1991).
        a2: Constant (Table 1, Holland and Powell, 1991).
        b: Constant (Table 1, Holland and Powell, 1991).

    Attributes:
        Tc: Critical temperature in kelvin.
        p0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar.
        a0: Constant (Table 1, Holland and Powell, 1991).
        a1: Constant (Table 1, Holland and Powell, 1991).
        a2: Constant (Table 1, Holland and Powell, 1991).
        b: Constant (Table 1, Holland and Powell, 1991).
        virial: Virial contribution object.
    """

    Tc: float  # kelvin.
    P0: float  # kbar.
    # Constants from Table 1, Holland and Powell (1991).
    a0: float
    a1: float
    a2: float
    # a3-a9 only required for H2O so they are not included in this base class.
    b: float
    a_virial: tuple[float, float]
    b_virial: tuple[float, float]
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            P0=self.P0,
        )

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            Fugacity coefficient evaluated at temperature and pressure.
        """
        pressure_kbar: float = pressure / kilo
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure_kbar)

        return fugacity_coefficient

    @abstractmethod
    def a(self, temperature: float) -> float:
        """Parameter a in Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a in kJ^2 kbar^(-1) K^(1/2) mol^(-2).
        """
        raise NotImplementedError

    @staticmethod
    def compressibility_factor(temperature: float, pressure: float, volume: float) -> float:
        """Compressibility factor.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.
            volume: Volume in kJ kbar^(-1) mol^(-1).

        Returns:
            Compressibility factor, which is non-dimensional.
        """
        # TODO: Meng computes the compressibility factor using only the MRK volume. See line 48
        # in his outgassing-2 Jupyter notebook. And equation A.2. in HP91 seems to suggest a virial
        # contribution as the last term, so maybe it should be V MRK?
        compressibility: float = pressure * volume / (GAS_CONSTANT_KJ * temperature)

        return compressibility

    def A_factor(self, temperature: float, pressure: float) -> float:
        """A factor in Appendix A of Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            A factor, which is non-dimensional.
        """
        del pressure
        A: float = self.a(temperature) / (self.b * GAS_CONSTANT_KJ * temperature**1.5)

        return A

    def B_factor(self, temperature: float, pressure: float) -> float:
        """B factor in Appendix A of Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            B factor, which is non-dimensional.
        """
        B: float = self.b * pressure / (GAS_CONSTANT_KJ * temperature)

        return B

    def ln_fugacity_coefficient_MRK(self, temperature: float, pressure: float) -> float:
        """Natural log of the MRK contribution to the fugacity coefficient.

        Equation A.2., Holland and Powell (1991).

        These equations can be applied directly in the absence of critical phenomena in the range
        of temperature and pressure under consideration. Hence these are appropriate for CO2.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            ln(fugacity_cofficient_MRK).
        """

        A: float = self.A_factor(temperature, pressure)
        B: float = self.B_factor(temperature, pressure)
        volume: float = self.volume_MRK_above_Tc(temperature, pressure)
        z: float = self.compressibility_factor(temperature, pressure, volume)

        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)

        return ln_fugacity_coefficient

    def ln_fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity coefficient including both MRK and virial contributions.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            ln(fugacity_coefficient).
        """

        ln_fugacity: float = self.ln_fugacity_coefficient_MRK(temperature, pressure)
        if pressure >= self.P0:
            ln_fugacity += self.virial.ln_fugacity_coefficient(temperature, pressure)

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in kbar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity in kbar.
        """
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)
        fugacity: float = fugacity_coefficient * pressure  # kbar
        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity coefficient.
        """
        fugacity_coefficient: float = np.exp(self.ln_fugacity_coefficient(temperature, pressure))
        return fugacity_coefficient

    def _objective_function_volume_MRK(
        self, volume: float, temperature: float, pressure: float, a: Callable[[float], float]
    ) -> float:
        """Residual function for the MRK volume from Equation A.1, Holland and Powell (1991).

        Args:
            volume: Volume.
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.
            a: A callable to compute function a.

        Returns:
            Residual of the MRK volume.
        """

        residual: float = (
            pressure * volume**3
            - GAS_CONSTANT_KJ * temperature * volume**2
            - (
                self.b * GAS_CONSTANT_KJ * temperature
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
        """Jacobian of Equation A.1, Holland and Powell (1991).

        Args:
            volume: Volume.
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.
            a: A callable to compute function a.

        Returns:
            Jacobian of the MRK volume.
        """

        jacobian: float = (
            3 * pressure * volume**2
            - 2 * GAS_CONSTANT_KJ * temperature * volume
            - (
                self.b * GAS_CONSTANT_KJ * temperature
                + self.b**2 * pressure
                - a(temperature) / np.sqrt(temperature)
            )
        )
        return jacobian

    def volume_MRK_above_Tc(self, temperature: float, pressure: float) -> float:
        """MRK volume above the critical temperature.

        There is only one real root.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume in kJ kbar^(-1) mol^(-1).
        """

        # TODO: This cannot work, otherwise the evaluation in Appendix A of Holland and Powell
        # (1991) cannot work. To check with Meng.
        # assert temperature >= self.Tc

        volume_init: float = GAS_CONSTANT_KJ * temperature / pressure + self.b

        volume_MRK: np.ndarray = fsolve(
            self._objective_function_volume_MRK,
            volume_init,
            args=(temperature, pressure, self.a),
            fprime=self._volume_MRK_jacobian,
        )  # type: ignore

        return volume_MRK[0]


@dataclass(kw_only=True)
class CorkFullCO2(CorkFull):
    """Full Cork equation for CO2 from Holland and Powell (1991)."""

    a0: float = field(init=False, default=741.2)
    a1: float = field(init=False, default=-0.10891)
    a2: float = field(init=False, default=-3.903e-4)
    b: float = field(init=False, default=3.057)
    a_virial: tuple[float, float] = field(init=False, default=(1.33790e-2, -1.01740e-5))
    b_virial: tuple[float, float] = field(init=False, default=(-2.26924e-1, 7.73793e-5))
    Tc: float = field(init=False, default=304.2)
    P0: float = field(init=False, default=5.0)

    def a(self, temperature: float) -> float:
        """Function a. Holland and Powell (1991), p270.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Function a.
        """
        a: float = self.a0 + self.a1 * temperature + self.a2 * temperature**2
        return a


@dataclass(kw_only=True)
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
    a_virial: tuple[float, float] = field(init=False, default=(-3.2297554e-3, 2.2215221e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-3.025650e-2, -5.343144e-6))
    Tc: float = field(init=False, default=673.0)  # FIXME: Paper says should be 695 K.
    P0: float = field(init=False, default=2.0)

    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991).

        Args:
            temperature: Temperature.

        Returns:
            Saturation curve pressure in kbar.
        """
        Psat: float = (
            -13.627e-3
            + 7.29395e-7 * temperature**2
            - 2.34622e-9 * temperature**3
            + 4.83607e-15 * temperature**5
        )
        return Psat

    def a_gas(self, temperature: float) -> float:
        """Parameter a for gaseous H2O. Equation 6a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a for gaseous H2O.
        """
        a: float = (
            self.a0
            + self.a7 * (self.Tc - temperature)
            + self.a8 * (self.Tc - temperature) ** 2
            + self.a9 * (self.Tc - temperature) ** 3
        )
        return a

    def a(self, temperature: float) -> float:
        """Parameter a for liquid and supercritical H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a for liquid and supercritical H2O.
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

    def volume_MRK_dense_phase_below_Tc(self, temperature: float, pressure: float) -> float:
        """MRK volume for the dense phase below the critical temperature.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume units TODO.
        """

        assert temperature < self.Tc

        volume_init: float = self.b / 2

        volume_MRK: np.ndarray = fsolve(
            self._objective_function_volume_MRK,
            volume_init,
            args=(temperature, pressure, self.a),
            fprime=self._volume_MRK_jacobian,
        )  # type: ignore

        return volume_MRK[0]

    def volume_MRK_gaseous_phase_below_Tc(self, temperature: float, pressure: float) -> float:
        """MRK volume for the gaseous phase below the critical temperature.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume units TODO.
        """

        assert temperature < self.Tc

        volume_init: float = GAS_CONSTANT_KJ * temperature / pressure + 10.0 * self.b

        volume_MRK: np.ndarray = fsolve(
            self._objective_function_volume_MRK,
            volume_init,
            args=(temperature, pressure, self.a_gas),
            fprime=self._volume_MRK_jacobian,
        )  # type: ignore

        return volume_MRK[0]

    # TODO: FIXME: Needs refreshing to work for critical behaviour.
    def lnf_MRK(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity. Appendix A, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            ln(fugacity) (units TODO).
        """

        # TODO: Check is this fugacity or fugacity coefficient?
        if temperature >= self.Tc:
            ln_fugacity: float = super().ln_fugacity_coefficient_MRK(temperature, pressure)

        elif pressure <= self.Psat(temperature):
            A: float = self.A_factor(temperature, pressure)
            B: float = self.B_factor(temperature, pressure)
            volume: float = self.volume_MRK_gaseous_phase_below_Tc(temperature, pressure)
            z: float = self.compressibility_factor(temperature, pressure, volume)
            ln_fugacity: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)

        else:  # pressure > self.Psat(temperature):
            # Step (1), Appendix A, Holland and Powell (1991).
            A: float = self.A_factor(temperature, pressure)
            B: float = self.B_factor(temperature, pressure)
            volume = self.volume_MRK_gaseous_phase_below_Tc(temperature, self.Psat(temperature))
            z: float = self.compressibility_factor(temperature, pressure, volume)
            ln_fugacity1: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)

            # Step (2), Appendix A, Holland and Powell (1991).
            volume = self.volume_MRK_dense_phase_below_Tc(temperature, self.Psat(temperature))
            z: float = self.compressibility_factor(temperature, pressure, volume)
            ln_fugacity2: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)

            # Step (3), Appendix A, Holland and Powell (1991).
            ln_fugacity3: float = super().ln_fugacity_coefficient_MRK(temperature, pressure)

            # Step (4), Appendix A, Holland and Powell (1991).
            ln_fugacity = ln_fugacity1 - ln_fugacity2 + ln_fugacity3

        return ln_fugacity


@dataclass(kw_only=True)
class CorkSimple(GetValueABC):
    """A Simplified Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    Although originally fit to CO2 data, this predicts the volumes and fugacities for several other
    gases which are known to obey approximately the principle of corresponding states. The
    corresponding states parameters are from Table 2 in Holland and Powell (1991). Note also in
    this case it appears P0 is always zero, even though for the full CORK equations it determines
    whether or not the virial contribution is added. It assumes there are no complications of
    critical behaviour in the P-T range considered.

    The units in Holland and Powell (1991) are K and kbar, so note unit conversions where relevant.

    Args:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure in kbar.

    Attributes:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure in kbar.
        a0: Universal constant (Table 2, Holland and Powell, 1991).
        a1: Universal constant (Table 2, Holland and Powell, 1991).
        b0: Universal constant (Table 2, Holland and Powell, 1991).
        a_virial: Constants for virial contribution (d0 and d1 in Table 2).
        b_virial: Constants for virial contribution (c0 and c1 in Table 2).
        c_virial: Constants for vitial contribution (not used).
        virial: Virial contribution object.
    """

    Tc: float  # kelvin
    Pc: float  # kbar
    # Universal constants from Table 2, Holland and Powell (1991).
    a0: float = field(init=False, default=5.45963e-5)
    a1: float = field(init=False, default=-8.63920e-6)
    b0: float = field(init=False, default=9.18301e-4)
    a_virial: tuple[float, float] = field(init=False, default=(6.93054e-7, -8.38293e-8))
    b_virial: tuple[float, float] = field(init=False, default=(-3.30558e-5, 2.30524e-6))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            Pc=self.Pc,
            Tc=self.Tc,
        )

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            Fugacity coefficient evaluated at temperature and pressure.
        """
        pressure_kbar: float = pressure / kilo
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure_kbar)

        return fugacity_coefficient

    def a(self, temperature: float) -> float:
        """Parameter a in Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a in kJ^2 kbar^(-1) K^(1/2) mol^(-2).
        """
        a: float = (
            self.a0 * self.Tc ** (5.0 / 2.0) / self.Pc
            + self.a1 * self.Tc ** (3.0 / 2.0) / self.Pc * temperature
        )
        return a

    @property
    def b(self) -> float:
        """Parameter b in Equation 9, Holland and Powell (1991).

        Returns:
            Parameter b in kJ kbar^(-1) mol^(-1).
        """
        b: float = self.b0 * self.Tc / self.Pc
        return b

    def compressibility_factor(self, temperature: float, pressure: float) -> float:
        """Compressibility factor.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Compressibility factor, which is non-dimensional.
        """
        compressibility: float = (
            pressure * self.volume(temperature, pressure) / (GAS_CONSTANT_KJ * temperature)
        )

        return compressibility

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity in kbar.
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT_KJ * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in kbar.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity in kbar.
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))  # bar
        fugacity /= kilo  # kbar

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity coefficient.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    def volume_MRK(self, temperature: float, pressure: float) -> float:
        """Volume from MRK contribution. Equation 7, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume in kJ kbar^(-1) mol^(-1).
        """
        volume: float = (
            GAS_CONSTANT_KJ * temperature / pressure
            + self.b
            - self.a(temperature)
            * GAS_CONSTANT_KJ
            * np.sqrt(temperature)
            / (GAS_CONSTANT_KJ * temperature + self.b * pressure)
            / (GAS_CONSTANT_KJ * temperature + 2.0 * self.b * pressure)
        )

        return volume

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume. Equation 7a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume in kJ kbar^(-1) mol^(-1).
        """
        volume: float = self.volume_MRK(temperature, pressure) + self.virial.volume(
            temperature, pressure
        )

        return volume

    def volume_cm3(self, temperature: float, pressure: float) -> float:
        """Volume in cm^3 mol^(-1). Equation 7a, Holland and Powell (1991).

        This is useful for comparing with Figs 8a-c in Holland and Powell (1991), which shows the
        volume-temperature plots for CO, CH4, and H2.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume in cm^3 mol^(-1).
        """
        volume_kJ_per_kbar: float = self.volume(temperature, pressure)
        volume_cm3: float = volume_kJ_per_kbar * UnitConversion.kJ_per_kbar_to_cm3()

        return volume_cm3

    def volume_integral_MRK(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP) from MRK contribution. Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume integral in kJ mol^(-1).
        """
        volume_integral: float = (
            GAS_CONSTANT_KJ * temperature * np.log(1000 * pressure)
            + self.b * pressure
            + self.a(temperature)
            / self.b
            / np.sqrt(temperature)
            * (
                np.log(GAS_CONSTANT_KJ * temperature + self.b * pressure)
                - np.log(GAS_CONSTANT_KJ * temperature + 2.0 * self.b * pressure)
            )
        )

        return volume_integral

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume integral in kJ mol^(-1).
        """
        volume_integral: float = self.volume_integral_MRK(
            temperature, pressure
        ) + self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CorkSimpleCO2(CorkSimple):
    """Critical constants for CO2.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=304.2)
    Pc: float = field(init=False, default=0.0738)


@dataclass(kw_only=True)
class CorkCH4(CorkSimple):
    """Critical constants for CH4.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=190.6)
    Pc: float = field(init=False, default=0.0460)


@dataclass(kw_only=True)
class CorkH2(CorkSimple):
    """Critical constants for H2.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=41.2)
    Pc: float = field(init=False, default=0.0211)


@dataclass(kw_only=True)
class CorkCO(CorkSimple):
    """Critical constants for CO.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=132.9)
    Pc: float = field(init=False, default=0.0350)


# Holland and Powell (2011) use this model for additional species below, using critical constants
# from the following reference:
# Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
# McGraw-Hill, New York.


@dataclass(kw_only=True)
class CorkS2(CorkSimple):
    """Critical constants for S2."""

    # Data not in The Properties of Gases and Liquids.  Use S instead?

    Tc: float  # TODO = field(init=False, default=132.9)
    Pc: float  # TODO = field(init=False, default=0.0350)


@dataclass(kw_only=True)
class CorkH2S(CorkSimple):
    """Critical constants for H2S.

    Appendix A.19 in The Properties of Gases and Liquids (2001), 5th edition.
    """

    Tc: float = field(init=False, default=373.4)
    Pc: float = field(init=False, default=0.08963)


def main():
    """For testing."""

    # 1 bar = 10^5 Pa
    # 1 kbar = 10^8 Pa
    # 10 kbar = 1 GPa

    # Comparison with Kite's H2 fugacity coefficient is not great. But around >30kbar the fugacity
    # coefficient for H2 maxes out and then decreases again.

    pressure: float = 4  # 4  # 1.8200066513507675  # 10
    temperature: float = 2000  # 1500  # 2000

    # These tests are for CO, CH4, and H2. The results agree with Meng.
    # test_simple_cork(temperature, pressure)

    test_full_cork(temperature, pressure)


def test_simple_cork(temperature, pressure):
    # Meng's functions.
    print("\nMengs functions")
    V, RTlnf = Calc_V_f(pressure, temperature, "CO")
    print("CO: V = %f, RTlnf = %f" % (V, RTlnf))
    V, RTlnf = Calc_V_f(pressure, temperature, "CH4")
    print("CH4: V = %f, RTlnf = %f" % (V, RTlnf))
    V, RTlnf = Calc_V_f(pressure, temperature, "H2")
    print("H2: V = %f, RTlnf = %f" % (V, RTlnf))
    print("\n")

    # My classes.
    print("My classes")
    cork: CorkSimple = CorkCO()
    V = cork.volume(temperature, pressure)
    RTlnf = cork.volume_integral(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    print("CO: V = %f, RTlnf = %f" % (V, RTlnf))
    print("CO: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    cork: CorkSimple = CorkCH4()
    V = cork.volume(temperature, pressure)
    # Vcm3 = cork.volume_cm3(temperature, pressure)
    RTlnf = cork.volume_integral(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    print("CH4: V = %f, RTlnf = %f" % (V, RTlnf))
    print("CH4: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    cork: CorkSimple = CorkH2()
    V = cork.volume(temperature, pressure)
    RTlnf = cork.volume_integral(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    print("H2: V = %f, RTlnf = %f" % (V, RTlnf))
    print("H2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    # cork: CorkSimple = CorkSimpleCO2()
    # V = cork.volume(temperature, pressure)
    # RTlnf = cork.volume_integral(temperature, pressure)
    # fugacity = cork.fugacity(temperature, pressure)
    # fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    # print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    # print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    # print("\n")


def test_full_cork(temperature, pressure):
    # Meng's functions.
    # Initially keep the pressure below P0 to avoid the virial contribution, which is formulated
    # differently by Meng based on HP11.
    print("\nMengs functions")
    V, RTlnf = Calc_V_f(pressure, temperature, "CO2")
    print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    fugacity: float = np.exp(RTlnf / (GAS_CONSTANT * temperature))
    fugacity_coeff: float = fugacity / pressure
    print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    # TODO: Get CO2 working first since that is without critical behaviour.
    # V, RTlnf = Calc_V_f(pressure, temperature, "H2O")
    # print("H2O: V = %f, RTlnf = %f" % (V, RTlnf))
    print("\n")

    # My classes.
    print("My classes")
    print("CO2: Full CORK")
    cork = CorkFullCO2()
    # V = cork.volume(temperature, pressure)
    # RTlnf = cork.RTlnf(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    # print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))

    cork = CorkSimpleCO2()
    V = cork.volume(temperature, pressure)
    RTlnf = cork.volume_integral(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    print("CO2: Simple CORK")
    print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    print("\n")


if __name__ == "__main__":
    main()
