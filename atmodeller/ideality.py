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
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import kilo

from atmodeller import GAS_CONSTANT
from atmodeller.eos_interfaces import MRKExplicitABC, MRKImplicitABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class VirialCompensation:
    """A compensation term for the increasing deviation of the MRK volumes with pressure.

    General form of the equation from Holland and Powell (1998):

        V_virial = a(P-P0) + b(P-P0)**0.5 + c(P-P0)**0.25

    This form also works for the virial compensation term from Holland and Powell (1991), in which
    case c=0. Pc and Tc are required for gases which are known to obey approximately the principle
    of corresponding states.

    Although this looks similar to an EOS, it's important to remember that it only calculates an
    additional perturbation to the volume and the volume integral of an MRK EOS, and hence it does
    not return a meaningful volume or volume integral by itself.

    Args:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally scaled (internally) by Tc and Pc in the case of corresponding
            states.
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
            may be additionally (internally) scaled by Tc and Pc in the case of corresponding
            states.
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

        Note the scalings by self.Tc and self.Pc to accommodate corresponding states. For example,
        Equation 9 in Holland and Powell (1991).

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

        Note the scalings by self.Tc and self.Pc to accommodate corresponding states. For example,
        Equation 9 in Holland and Powell (1991).

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

        Note the scalings by self.Tc and self.Pc to accommodate corresponding states.

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

        Note that since this EOS is computing a perturbation the volume integral relates to the
        fugacity coefficient and NOT the fugacity as would ordinarily be assumed.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Natural log of the fugacity coefficient.
        """
        ln_fugacity_coefficient: float = self.volume_integral(temperature, pressure) / (
            self.GAS_CONSTANT * temperature
        )

        return ln_fugacity_coefficient

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient of the virial contribution.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Fugacity coefficient.
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
            Volume integral.
        """
        volume_integral: float = (
            self.a(temperature) / 2.0 * (pressure - self.P0) ** 2
            + 2.0 / 3.0 * self.b(temperature) * (pressure - self.P0) ** (3.0 / 2.0)
            + 4.0 / 5.0 * self.c(temperature) * (pressure - self.P0) ** (5.0 / 4.0)
        )

        return volume_integral


@dataclass(kw_only=True)
class MRKH2OLiquidHollandPowell1991(MRKImplicitABC):
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
class MRKH2OGasHollandPowell1991(MRKImplicitABC):
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
class MRKH2OFluidHollandPowell1991(MRKImplicitABC):
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
class CorkFull(MRKImplicitABC):
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
class CorkCorrespondingStates(MRKExplicitABC):
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
