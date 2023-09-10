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

import numpy as np
from scipy.constants import kilo
from scipy.optimize import fsolve

from atmodeller import GAS_CONSTANT, debug_logger
from atmodeller.interfaces import GetValueABC

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
                # print(lnlambda_mrk, V_mrk)

            else:
                # print("Psat = %f" % Psat)
                V_init = R * T / P + 10.0 * b
                lnlambda1, V_mrk = Calc_lambda(Psat, T, a_gas, b, V_init)  # type: ignore
                # print(lnlambda1, V_mrk)

                V_init = b / 2.0
                lnlambda2, V_mrk = Calc_lambda(Psat, T, a, b, V_init)
                # print(lnlambda2, V_mrk)

                V_init = R * T / P + b
                lnlambda3, V_mrk = Calc_lambda(P, T, a, b, V_init)
                # print(lnlambda3, V_mrk)

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


@dataclass(kw_only=True)
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
            to zero, which is appropriate for the corresponding states case.
        Tc: Critical temperature in kelvin. Defaults to 1, which effectively means it is unused.
        Pc: Critical pressure in kbar. Defaults to 1, which effectively means it is unused.
        scaling: Scaling depending on the units of the coefficients. Defaults to kilo for the
            Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally scaled by Tc and Pc in the case of corresponding states.
        b_coefficients: As above for the b coefficients.
        c_coefficients: As above for the c coefficients.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar.
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure in kbar.
        scaling: Scaling depending on the units of the coefficients. Defaults to kilo for the
            Holland and Powell data since pressures are in kbar.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of the
            coefficients.
    """

    a_coefficients: tuple[float, float]
    b_coefficients: tuple[float, float]
    c_coefficients: tuple[float, float]
    P0: float = 0  # Default must be zero for corresponding states.
    Pc: float = 1  # Defaults to 1, which effectively means unused.
    Tc: float = 1  # Defaults to 1, which effectively means unused.
    scaling: float = kilo
    GAS_CONSTANT: float = field(init=False, default=GAS_CONSTANT)

    def __post_init__(self):
        self.GAS_CONSTANT /= self.scaling

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
            self.GAS_CONSTANT * temperature
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


@dataclass(kw_only=True)
class MRK(GetValueABC):
    """A Modified Redlich Kwong (MRK) EOS.

    For example, Equation 3, Holland and Powell (1991):
        P = RT/(V-b) - a/(V(V+b)T**0.5)

    where:
        P is pressure.
        T is temperature.
        R is the gas constant.
        a is the Redlich-Kwong function, which is a function of T.
        b is the Redlich-Kwong constant b.

    For a MRK, the a parameter is a function of T and the b parameter is constant.

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of
            a_coefficients and b0.
    """

    a_coefficients: tuple[float, ...]
    b0: float
    scaling: float = kilo
    GAS_CONSTANT: float = field(init=False, default=GAS_CONSTANT)

    def __post_init__(self):
        self.GAS_CONSTANT /= self.scaling

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            Fugacity coefficient evaluated at temperature and pressure.
        """
        pressure_kbar: float = pressure / self.scaling
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure_kbar)

        return fugacity_coefficient

    @abstractmethod
    def a(self, temperature: float) -> float:
        """MRK a parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self) -> float:
        """MRK b parameter.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK b parameter.
        """
        raise NotImplementedError

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity.
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            self.GAS_CONSTANT * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in the same units as pressure.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            fugacity.
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))  # bar
        fugacity /= self.scaling  # to units of pressure for consistency.

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity coefficient.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume.
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume integral.
        """
        ...


@dataclass(kw_only=True)
class MRKExplicit(MRK):
    """A modified Redlich Kwong (MRK) EOS in an explicit form.

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of
            a_coefficients and b0.
    """

    def volume(self, temperature: float, pressure: float) -> float:
        """Convenient volume-explicit equation. Equation 7, Holland and Powell (1991).

        Without complications of critical phenomena the MRK equation can be simplified. Using the
        approximation V /approx RT/P + b leads to an explicit form.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            volume.
        """
        volume: float = (
            self.GAS_CONSTANT * temperature / pressure
            + self.b
            - self.a(temperature)
            * self.GAS_CONSTANT
            * np.sqrt(temperature)
            / (self.GAS_CONSTANT * temperature + self.b * pressure)
            / (self.GAS_CONSTANT * temperature + 2.0 * self.b * pressure)
        )

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume-explicit integral (V dP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        volume_integral: float = (
            self.GAS_CONSTANT * temperature * np.log(1000 * pressure)
            + self.b * pressure
            + self.a(temperature)
            / self.b
            / np.sqrt(temperature)
            * (
                np.log(self.GAS_CONSTANT * temperature + self.b * pressure)
                - np.log(self.GAS_CONSTANT * temperature + 2.0 * self.b * pressure)
            )
        )

        return volume_integral


@dataclass(kw_only=True)
class MRKImplicit(MRK):
    """A modified Redlich Kwong (MRK) EOS in an implicit form.

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of
            a_coefficients and b0.
    """

    @property
    def b(self):
        return self.b0

    def A_factor(self, temperature: float, pressure: float) -> float:
        """A factor in Appendix A of Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            A factor, which is non-dimensional.
        """
        del pressure
        A: float = self.a(temperature) / (self.b * self.GAS_CONSTANT * temperature**1.5)

        return A

    def B_factor(self, temperature: float, pressure: float) -> float:
        """B factor in Appendix A of Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            B factor, which is non-dimensional.
        """
        B: float = self.b * pressure / (self.GAS_CONSTANT * temperature)

        return B

    def compressibility_factor(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Compressibility factor.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            # TODO: Add volume_init.

        Returns:
            Compressibility factor, which is non-dimensional.
        """
        compressibility: float = (
            pressure
            * self.volume(temperature, pressure, volume_init=volume_init)
            / (self.GAS_CONSTANT * temperature)
        )

        return compressibility

    def volume_integral(
        self,
        temperature: float,
        pressure: float,
        *,
        volume_init: float | None = None,
    ) -> float:
        """Volume integral. Equation A.2., Holland and Powell (1991).

        These equations can be applied directly in the absence of critical phenomena in the range
        of temperature and pressure under consideration. Hence these are appropriate for CO2.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.
            volume_init: Initial volume estimate, which defaults to a value to find a root above
                Tc. Other values may be necessary for multi-root systems (e.g. H2O).

        Returns:
            Volume integral.
        """

        z: float = self.compressibility_factor(temperature, pressure, volume_init=volume_init)
        A: float = self.A_factor(temperature, pressure)
        B: float = self.B_factor(temperature, pressure)
        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)
        ln_fugacity: float = np.log(1000 * pressure) + ln_fugacity_coefficient
        volume_integral: float = self.GAS_CONSTANT * temperature * ln_fugacity

        return volume_integral

    def volume(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Solves the MRK equation to get the volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial guess of the volume for the solver. Defaults to a value to locate
                the root above the critical temperature.

        Returns:
            volume.
        """

        if volume_init is None:
            # From Holland and Powell (1991), above Tc there is only one real root.
            volume_init = self.GAS_CONSTANT * temperature / pressure + self.b

        volume_solution: np.ndarray = fsolve(
            self._objective_function_volume,
            volume_init,
            args=(temperature, pressure),
            fprime=self._volume_jacobian,
        )  # type: ignore

        volume: float = volume_solution[0]

        print(volume)

        return volume

    def _objective_function_volume(
        self, volume: float, temperature: float, pressure: float
    ) -> float:
        """Residual function for the MRK volume from Equation A.1, Holland and Powell (1991).

        Args:
            volume: Volume.
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Residual of the MRK volume.
        """

        residual: float = (
            pressure * volume**3
            - self.GAS_CONSTANT * temperature * volume**2
            - (
                self.b * self.GAS_CONSTANT * temperature
                + self.b**2 * pressure
                - self.a(temperature) / np.sqrt(temperature)
            )
            * volume
            - self.a(temperature) * self.b / np.sqrt(temperature)
        )

        return residual

    def _volume_jacobian(self, volume: float, temperature: float, pressure: float):
        """Jacobian of Equation A.1, Holland and Powell (1991).

        Args:
            volume: Volume.
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Jacobian of the MRK volume.
        """

        jacobian: float = (
            3 * pressure * volume**2
            - 2 * self.GAS_CONSTANT * temperature * volume
            - (
                self.b * self.GAS_CONSTANT * temperature
                + self.b**2 * pressure
                - self.a(temperature) / np.sqrt(temperature)
            )
        )
        return jacobian


@dataclass(kw_only=True)
class MRKH2OLiquidHollandPowell1991(MRKImplicit):
    """MRK a parameter for liquid H2O. Equation 6, Holland and Powell (1991)."""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1113.4, -0.88517, 4.53e-3, -1.3183e-5),
    )
    b0: float = field(init=False, default=1.465)
    Tc: float = field(init=False, default=673.0)  # FIXME: Paper says should be 695 K.

    def a(self, temperature: float) -> float:
        """MRK a parameter for liquid H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter for liquid H2O.
        """

        assert temperature <= self.Tc

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (self.Tc - temperature)
            + self.a_coefficients[2] * (self.Tc - temperature) ** 2
            + self.a_coefficients[3] * (self.Tc - temperature) ** 3
        )

        return a


@dataclass(kw_only=True)
class MRKH2OGasHollandPowell1991(MRKImplicit):
    """MRK for gaseous H2O. Equation 6a, Holland and Powell (1991)."""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4,
            5.8487,
            -2.1370e-2,
            6.8133e-5,
        ),
    )
    b0: float = field(init=False, default=1.465)
    Tc: float = field(init=False, default=673.0)  # FIXME: Paper says should be 695 K.

    def a(self, temperature: float) -> float:
        """MRK a parameter for gaseous H2O. Equation 6a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter for gaseous H2O.
        """

        assert temperature <= self.Tc

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (self.Tc - temperature)
            + self.a_coefficients[2] * (self.Tc - temperature) ** 2
            + self.a_coefficients[3] * (self.Tc - temperature) ** 3
        )
        return a


@dataclass(kw_only=True)
class MRKH2OFluidHollandPowell1991(MRKImplicit):
    """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991)."""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4,
            -0.22291,
            -3.8022e-4,
            1.7791e-7,
        ),
    )
    b0: float = field(init=False, default=1.465)
    Tc: float = field(init=False, default=673.0)  # FIXME: Paper says should be 695 K.

    def a(self, temperature: float) -> float:
        """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter supercritical H2O.
        """

        assert temperature >= self.Tc

        a = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (temperature - self.Tc)
            + self.a_coefficients[2] * (temperature - self.Tc) ** 2
            + self.a_coefficients[3] * (temperature - self.Tc) ** 3
        )

        return a


@dataclass(kw_only=True)
class CorkFull(MRKImplicit):
    """A Full Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    The units in Holland and Powell (1991) are K and kbar, so note unit conversions where relevant.

    Constants correspond to Table 1 in Holland and Powell (1991).

    Args:
        Tc: Critical temperature in kelvin.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar.
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.
        a_virial: a coefficients for the virial compensation.
        b_virial: b coefficients for the virial compensation.
        c_virial: c coefficients for the virial compensation.

    Attributes:
        Tc: Critical temperature in kelvin.
        p0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Units are kbar.
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of
            a_coefficients and b0.
        a_virial: a coefficients for the virial compensation.
        b_virial: b coefficients for the virial compensation.
        c_virial: c coefficients for the virial compensation.
        virial: A VirialCompensation instance.
    """

    Tc: float  # kelvin.
    P0: float  # kbar.
    a_virial: tuple[float, float]
    b_virial: tuple[float, float]
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            P0=self.P0,
        )

    def volume(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Volume. Equation 7a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume in kJ kbar^(-1) mol^(-1).
        """
        volume: float = super().volume(temperature, pressure, volume_init=volume_init)

        if pressure > self.P0:
            volume += self.virial.volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume integral in kJ mol^(-1).
        """
        volume_integral: float = super().volume_integral(temperature, pressure)

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CorkFullCO2HollAndPowell1991(CorkFull):
    """Full CORK equation for CO2 from Holland and Powell (1991)."""

    a_coefficients: tuple[float, ...] = field(init=False, default=(741.2, -0.10891, -3.903e-4))
    b0: float = field(init=False, default=3.057)
    a_virial: tuple[float, float] = field(init=False, default=(1.33790e-2, -1.01740e-5))
    b_virial: tuple[float, float] = field(init=False, default=(-2.26924e-1, 7.73793e-5))
    Tc: float = field(init=False, default=304.2)
    P0: float = field(init=False, default=5.0)

    def a(self, temperature: float) -> float:
        """MRK a parameter. Holland and Powell (1991), p270.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter.
        """
        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * temperature
            + self.a_coefficients[2] * temperature**2
        )
        return a


@dataclass(kw_only=True)
class CorkFullCO2(CorkFullCO2HollAndPowell1991):
    """Full CORK equation for CO2 from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.
    """

    a_virial: tuple[float, float] = field(init=False, default=(5.40776e-3, -1.59046e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-1.78198e-1, 2.45317e-5))


@dataclass(kw_only=True)
class CorkFullH2OHollandAndPowell1991(CorkFull):
    """Full CORK equation for H2O from Holland and Powell (1991)."""

    a_coefficients: tuple[float, ...] = field(init=False)
    b0: float = field(init=False, default=1.465)
    a_virial: tuple[float, float] = field(init=False, default=(-3.2297554e-3, 2.2215221e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-3.025650e-2, -5.343144e-6))
    Tc: float = field(init=False, default=673.0)  # FIXME: Paper says should be 695 K.
    P0: float = field(init=False, default=2.0)

    # FIXME: Not ideal to have this and not used.
    def a(self):
        raise NotImplementedError()

    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

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

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity. Appendix A, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            volume integral.
        """

        if temperature >= self.Tc:
            print("temperature >= critical temperature")
            mrk: MRK = MRKH2OFluidHollandPowell1991()
            volume_integral: float = mrk.volume_integral(temperature, pressure)

        elif pressure <= self.Psat(temperature):
            print("pressure <= saturation pressure")
            mrk: MRK = MRKH2OGasHollandPowell1991()
            volume_init = self.GAS_CONSTANT * temperature / pressure + 10 * self.b
            volume_integral: float = mrk.volume_integral(
                temperature, pressure, volume_init=volume_init
            )

        else:  # pressure > self.Psat(temperature):
            print("temperature < critical temperature and pressure > saturation pressure")
            saturation_pressure: float = self.Psat(temperature)
            # See step (1-4) in Appendix A, Holland and Powell (1991).
            mrk: MRK = MRKH2OGasHollandPowell1991()
            volume_init: float = self.GAS_CONSTANT * temperature / pressure + 10 * self.b
            volume_integral1: float = mrk.volume_integral(
                temperature, saturation_pressure, volume_init=volume_init
            )
            mrk = MRKH2OLiquidHollandPowell1991()
            volume_init = self.b / 2
            volume_integral2 = mrk.volume_integral(
                temperature, saturation_pressure, volume_init=volume_init
            )
            mrk = MRKH2OLiquidHollandPowell1991()
            volume_init = self.GAS_CONSTANT * temperature / pressure + self.b
            volume_integral3 = mrk.volume_integral(temperature, pressure, volume_init=volume_init)
            volume_integral = volume_integral1 - volume_integral2 + volume_integral3

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CorkFullH2O(CorkFullH2OHollandAndPowell1991):
    """Full CORK equation for H2O from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.
    """

    a_virial: tuple[float, float] = field(init=False, default=(1.9853e-3, 0))
    b_virial: tuple[float, float] = field(init=False, default=(-8.9090e-2, 0))
    c_virial: tuple[float, float] = field(init=False, default=(8.0331e-2, 0))


@dataclass(kw_only=True)
class CorkCorrespondingStates(MRKExplicit):
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
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure in kbar.
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        a_virial: Constants for virial contribution (d0 and d1 in Table 2).
        b_virial: Constants for virial contribution (c0 and c1 in Table 2).
        c_virial: Constants for vitial contribution (not used).
        virial: Virial contribution object.
            scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of
            a_coefficients and b0. Defaults to kJ/K/mol for the Holland and Powell data.
    """

    Tc: float  # kelvin
    Pc: float  # kbar
    # Universal constants from Table 2, Holland and Powell (1991).
    a_coefficients: tuple[float, float] = field(init=False, default=(5.45963e-5, -8.63920e-6))
    b0: float = field(init=False, default=9.18301e-4)
    a_virial: tuple[float, float] = field(init=False, default=(6.93054e-7, -8.38293e-8))
    b_virial: tuple[float, float] = field(init=False, default=(-3.30558e-5, 2.30524e-6))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            Pc=self.Pc,
            Tc=self.Tc,
        )

    def a(self, temperature: float) -> float:
        """Parameter a in Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a in kJ^2 kbar^(-1) K^(1/2) mol^(-2).
        """
        a: float = (
            self.a_coefficients[0] * self.Tc ** (5.0 / 2.0) / self.Pc
            + self.a_coefficients[1] * self.Tc ** (3.0 / 2.0) / self.Pc * temperature
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

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume. Equation 7a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume in kJ kbar^(-1) mol^(-1).
        """
        volume: float = super().volume(temperature, pressure) + self.virial.volume(
            temperature, pressure
        )

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume integral in kJ mol^(-1).
        """
        volume_integral: float = super().volume_integral(
            temperature, pressure
        ) + self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CorkSimpleCO2(CorkCorrespondingStates):
    """Critical constants for CO2.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=304.2)
    Pc: float = field(init=False, default=0.0738)


@dataclass(kw_only=True)
class CorkCH4(CorkCorrespondingStates):
    """Critical constants for CH4.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=190.6)
    Pc: float = field(init=False, default=0.0460)


@dataclass(kw_only=True)
class CorkH2(CorkCorrespondingStates):
    """Critical constants for H2.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=41.2)
    Pc: float = field(init=False, default=0.0211)


@dataclass(kw_only=True)
class CorkCO(CorkCorrespondingStates):
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
class CorkS2(CorkCorrespondingStates):
    """Critical constants for S2."""

    # Data not in The Properties of Gases and Liquids.  Use S instead?

    Tc: float  # TODO = field(init=False, default=132.9)
    Pc: float  # TODO = field(init=False, default=0.0350)


@dataclass(kw_only=True)
class CorkH2S(CorkCorrespondingStates):
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

    debug_logger()

    pressure: float = 0.1  # 2  # 0.1  # 4  # 1.8200066513507675  # 10
    temperature: float = 600  # 600  # 1500  # 2000

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
    cork: CorkCorrespondingStates = CorkCO()
    V = cork.volume(temperature, pressure)
    RTlnf = cork.volume_integral(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    print("CO: V = %f, RTlnf = %f" % (V, RTlnf))
    print("CO: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    cork: CorkCorrespondingStates = CorkCH4()
    V = cork.volume(temperature, pressure)
    # Vcm3 = cork.volume_cm3(temperature, pressure)
    RTlnf = cork.volume_integral(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    print("CH4: V = %f, RTlnf = %f" % (V, RTlnf))
    print("CH4: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    cork: CorkCorrespondingStates = CorkH2()
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
    V, RTlnf = Calc_V_f(pressure, temperature, "H2O")
    print("H2O: V = %f, RTlnf = %f" % (V, RTlnf))
    fugacity: float = np.exp(RTlnf / (GAS_CONSTANT * temperature))
    fugacity_coeff: float = fugacity / 1000 / pressure
    print("H2O: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))

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

    cork = CorkFullH2O()
    # V = cork.volume(temperature, pressure)
    # RTlnf = cork.RTlnf(temperature, pressure)
    fugacity = cork.fugacity(temperature, pressure)
    fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    # print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    print("H2O: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    Psat = cork.Psat(temperature)
    print("H2O Psat = %f" % Psat)

    # cork = CorkSimpleCO2()
    # V = cork.volume(temperature, pressure)
    # RTlnf = cork.volume_integral(temperature, pressure)
    # fugacity = cork.fugacity(temperature, pressure)
    # fugacity_coeff = cork.fugacity_coefficient(temperature, pressure)
    # print("CO2: Simple CORK")
    # print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    # print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    # print("\n")


if __name__ == "__main__":
    main()
