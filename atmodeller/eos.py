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

from atmodeller.eos_interfaces import (
    MRKABC,
    CorkFullABC,
    FugacityModelABC,
    MRKExplicitABC,
    MRKImplicitABC,
    ShiSaxenaABC,
    ShiSaxenaHighPressure,
    ShiSaxenaLowPressure,
    VirialCompensation,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MRKH2OLiquid(MRKImplicitABC):
    """MRK for liquid H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1113.4, -0.88517, 4.53e-3, -1.3183e-5),
    )
    b0: float = field(init=False, default=1.465)
    Tc: float = field(init=False, default=673.0)  # TODO: Check consistency with 695 K of paper.

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
class MRKH2OGas(MRKImplicitABC):
    """MRK for gaseous H2O. Equation 6a, Holland and Powell (1991).

    See base class.
    """

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
    Tc: float = field(init=False, default=673.0)  # TODO: Check consistency with 695 K of paper.

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
class MRKH2OFluid(MRKImplicitABC):
    """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

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
    Tc: float = field(init=False, default=673.0)  # TODO: Check consistency with 695 K of paper.

    def a(self, temperature: float) -> float:
        """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter supercritical H2O.
        """
        assert temperature >= self.Tc

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (temperature - self.Tc)
            + self.a_coefficients[2] * (temperature - self.Tc) ** 2
            + self.a_coefficients[3] * (temperature - self.Tc) ** 3
        )

        return a


@dataclass(kw_only=True)
class CorkFullCO2HollAndPowell1991(CorkFullABC):
    """Full CORK equation for CO2 from Holland and Powell (1991).

    See base class.
    """

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

    See base class.
    """

    a_virial: tuple[float, float] = field(init=False, default=(5.40776e-3, -1.59046e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-1.78198e-1, 2.45317e-5))


@dataclass(kw_only=True)
class CorkFullH2OHollandAndPowell1991(CorkFullABC):
    """Full CORK equation for H2O from Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False, default=(0,)
    )  # TODO: Check not required.
    b0: float = field(init=False, default=1.465)
    a_virial: tuple[float, float] = field(init=False, default=(-3.2297554e-3, 2.2215221e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-3.025650e-2, -5.343144e-6))
    Tc: float = field(init=False, default=673.0)  # FIXME: Paper says should be 695 K.
    P0: float = field(init=False, default=2.0)

    def a(self):
        """Due to critical behaviour a single a parameter cannot be determined for H2O."""
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

    # TODO: volume to override base class.

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral including virial compensation. Appendix A, Holland and Powell (1991).

        Overrides the base class because we might have to integrate across different regions of P-T
        space depending on the input temperature and pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            volume integral.
        """

        if temperature >= self.Tc:
            logger.debug("temperature >= critical temperature")
            mrk: MRKABC = MRKH2OFluid()
            volume_integral: float = mrk.volume_integral(temperature, pressure)

        elif pressure <= self.Psat(temperature):
            logger.debug("pressure <= saturation pressure")
            mrk: MRKABC = MRKH2OGas()
            volume_init = self.GAS_CONSTANT * temperature / pressure + 10 * self.b
            volume_integral: float = mrk.volume_integral(
                temperature, pressure, volume_init=volume_init
            )

        else:  # pressure > self.Psat(temperature):
            logger.debug("temperature < critical temperature and pressure > saturation pressure")
            saturation_pressure: float = self.Psat(temperature)
            # See step (1-4) in Appendix A, Holland and Powell (1991).
            mrk: MRKABC = MRKH2OGas()
            volume_init: float = self.GAS_CONSTANT * temperature / pressure + 10 * self.b
            volume_integral1: float = mrk.volume_integral(
                temperature, saturation_pressure, volume_init=volume_init
            )
            mrk = MRKH2OLiquid()
            volume_init = self.b / 2
            volume_integral2 = mrk.volume_integral(
                temperature, saturation_pressure, volume_init=volume_init
            )
            mrk = MRKH2OLiquid()
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

    See base class.
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
            scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of
            a_coefficients and b0. Defaults to kJ/K/mol for the Holland and Powell data.
        a_virial: Constants for virial contribution (d0 and d1 in Table 2).
        b_virial: Constants for virial contribution (c0 and c1 in Table 2).
        c_virial: Constants for vitial contribution (not used).
        virial: Virial contribution object.
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
        """Volume including virial compensation. Equation 7a, Holland and Powell (1991).

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

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral including virial compensation. Equation 8, Holland and Powell (1991).

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
class CorkSimpleCO2(CorkCorrespondingStates):
    """Critical constants for CO2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=304.2)
    Pc: float = field(init=False, default=0.0738)


@dataclass(kw_only=True)
class CorkCH4(CorkCorrespondingStates):
    """Critical constants for CH4. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=190.6)
    Pc: float = field(init=False, default=0.0460)


@dataclass(kw_only=True)
class CorkH2(CorkCorrespondingStates):
    """Critical constants for H2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=41.2)
    Pc: float = field(init=False, default=0.0211)


@dataclass(kw_only=True)
class CorkCO(CorkCorrespondingStates):
    """Critical constants for CO. See base class.

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
    """Critical constants for S2. See base class.

    Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
    Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.

    http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

    """

    Tc: float = field(init=False, default=208.15)
    Pc: float = field(init=False, default=0.072954)


@dataclass(kw_only=True)
class CorkH2S(CorkCorrespondingStates):
    """Critical constants for H2S. See base class.

    Appendix A.19 in The Properties of Gases and Liquids (2001), 5th edition.
    """

    Tc: float = field(init=False, default=373.4)
    Pc: float = field(init=False, default=0.08963)


@dataclass(kw_only=True)
class ShiSaxenaHighPressureH2(ShiSaxenaHighPressure):
    Tc: float = field(init=False, default=33.25)
    Pc: float = field(init=False, default=12.9696)
    a_coefficients: tuple[float, ...] = field(
        init=False, default=(2.2615, 0, -6.8712e1, 0, -1.0573e4, 0, 0, -1.6936e-1)
    )
    b_coefficients: tuple[float, ...] = field(
        init=False, default=(-2.6707e-4, 0, 2.0173e-1, 0, 4.5759, 0, 0, 3.1452e-5)
    )
    c_coefficients: tuple[float, ...] = field(
        init=False, default=(-2.3376e-9, 0, 3.4091e-7, 0, -1.4188e-3, 0, 0, 3.0117e-10)
    )
    d_coefficients: tuple[float, ...] = field(
        init=False, default=(-3.2606e-15, 0, 2.4402e-12, 0, -2.4027e-9, 0, 0, 0)
    )


@dataclass(kw_only=True)
class ShiSaxenaLowPressureH2(ShiSaxenaLowPressure):
    Tc: float = field(init=False, default=33.25)
    Pc: float = field(init=False, default=12.9696)
    a_coefficients: tuple[float, ...] = field(init=False, default=(1, 0, 0, 0, 0, 0))
    b_coefficients: tuple[float, ...] = field(init=False, default=(0, 0.9827e-1, 0, -0.2709, 0))
    c_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, -0.1030e-2, 0, 0.1427e-1))
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class ShiSaxenaH2(FugacityModelABC):
    low_pressure_eos: ShiSaxenaABC = field(default_factory=ShiSaxenaLowPressureH2)
    high_pressure_eos: ShiSaxenaABC = field(default_factory=ShiSaxenaHighPressureH2)

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Note that the input 'pressure' must ALWAYS be in bar, so it is scaled here using
        'self.scaling' since self.fugacity_coefficient requires the internal units of pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            Fugacity coefficient evaluated at temperature and pressure, which is dimensionaless.
        """
        pressure /= self.scaling

        if pressure >= 1e3:
            return self.high_pressure_eos.get_value(temperature=temperature, pressure=pressure)
        else:
            return self.low_pressure_eos.get_value(temperature=temperature, pressure=pressure)

    def volume(self, temperature: float, pressure: float) -> float:
        if pressure / self.scaling >= 1e3:
            return self.high_pressure_eos.volume(temperature=temperature, pressure=pressure)
        else:
            return self.low_pressure_eos.volume(temperature=temperature, pressure=pressure)

    def volume_integral(self, temperature: float, pressure: float) -> float:
        if pressure / self.scaling >= 1e3:
            return self.high_pressure_eos.volume_integral(
                temperature=temperature, pressure=pressure
            )
        else:
            return self.low_pressure_eos.volume_integral(
                temperature=temperature, pressure=pressure
            )
