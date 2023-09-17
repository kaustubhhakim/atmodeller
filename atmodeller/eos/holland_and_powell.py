"""Fugacity models from Holland and Powell (1991, 1998, 2011).

See the LICENSE file for licensing information.

This module contains concrete classes for the fugacity models presented in Holland and Powell 
(1991, 1998, 2011).

Classes:
    MRKH2OLiquidHP91: MRK for liquid H2O (only) in Holland and Powell (1991).
    MRKH2OGasHP91: MRK for gaseous H2O (only) in Holland and Powell (1991).
    MRKH2OFluidHP91: MRK for fluid H2O (only) in Holland and Powell (1991).
    CORKFullCO2HP91: Full CORK for CO2 in Holland and Powell (1991).
    CORKFullCO2HP98: Full CORK for CO2 in Holland and Powell (1998).
    CORKSimpleCO2HP91: Simple CORK model for CO2 in Holland and Powell (1991).
    CORKFullH2OHP91: Full CORK for H2O in Holland and Powell (1991).
    CORKFullH2OHP98: Full CORK for H2O in Holland and Powell (1998).
    CORKCorrespondingStatesCH4HP91: Corresponding states for CH4 in Holland and Powell (1991).
    CORKCorrespondingStatesH2HP91: Corresponding states for H2 in Holland and Powell (1991).
    CORKCorrespondingStatesCOHP91: Corresponding states for CO in Holland and Powell (1991).
    CORKCorrespondingStatesS2HP11: Corresponding states for S2 in Holland and Powell (2011).
    CORKCorrespondingStatesH2SHP11: Corresponding states for H2S in Holland and Powell (2011).

Example:
    Get the preferred fugacity models for various species from the Holland and Powell models:
    
    ```python
    >>> from atmodeller.eos.holland_and_powell import get_holland_and_powell_fugacity_models
    >>> models = get_holland_and_powell_fugacity_models()
    >>> # list the available species
    >>> models.keys()
    >>> # Get the fugacity model for CO
    >>> co_model = models['CO']
    >>> # Determine the fugacity coefficient at 2000 K and 1000 bar
    >>> fugacity_coefficient = co_model.get_value(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.2664435476696503
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from atmodeller.eos.eos_interfaces import (
    CORKFullABC,
    FugacityModelABC,
    MRKExplicitABC,
    MRKImplicitABC,
    VirialCompensation,
)

logger: logging.Logger = logging.getLogger(__name__)

# For the CORK H2O model there are two different temperatures.
# The critical temperature.
Tc_H2O: float = 695  # K
# This is the temperature at which a_gas = a in Holland and Powell (1991), thereby allowing the
# critical point to be handled by a single a parameter.
Ta_H2O: float = 673  # K


@dataclass(kw_only=True)
class MRKH2OLiquidHP91(MRKImplicitABC):
    """MRK for liquid H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1113.4, -0.88517, 4.53e-3, -1.3183e-5),
    )
    b0: float = field(init=False, default=1.465)
    Ta: float = field(init=False, default=Ta_H2O)

    def a(self, temperature: float) -> float:
        """MRK a parameter for liquid H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter for liquid H2O.
        """
        assert temperature <= self.Ta

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (self.Ta - temperature)
            + self.a_coefficients[2] * (self.Ta - temperature) ** 2
            + self.a_coefficients[3] * (self.Ta - temperature) ** 3
        )

        return a

    def initial_solution_volume(self, *args, **kwargs) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        For the liquid phase a suitably low value must be chosen. See appendix in Holland and
        Powell (1991).

        *args and **kwargs catches unused arguments.

        Returns:
            Initial solution volume.
        """
        del args
        del kwargs
        initial_volume = self.b / 2

        return initial_volume


@dataclass(kw_only=True)
class MRKH2OGasHP91(MRKImplicitABC):
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
    Ta: float = field(init=False, default=Ta_H2O)

    def a(self, temperature: float) -> float:
        """MRK a parameter for gaseous H2O. Equation 6a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter for gaseous H2O.
        """
        assert temperature <= self.Ta

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (self.Ta - temperature)
            + self.a_coefficients[2] * (self.Ta - temperature) ** 2
            + self.a_coefficients[3] * (self.Ta - temperature) ** 3
        )

        return a

    def initial_solution_volume(self, temperature: float, pressure: float) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        See appendix in Holland and Powell (1991).

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            Initial solution volume.
        """
        initial_volume: float = self.GAS_CONSTANT * temperature / pressure + 10 * self.b

        return initial_volume


@dataclass(kw_only=True)
class MRKH2OFluidHP91(MRKImplicitABC):
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
    Ta: float = field(init=False, default=Ta_H2O)
    Tc: float = field(init=False, default=Tc_H2O)

    def a(self, temperature: float) -> float:
        """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter supercritical H2O.
        """
        assert temperature >= self.Ta

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (temperature - self.Ta)
            + self.a_coefficients[2] * (temperature - self.Ta) ** 2
            + self.a_coefficients[3] * (temperature - self.Ta) ** 3
        )

        return a

    def initial_solution_volume(self, temperature: float, pressure: float) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        See appendix in Holland and Powell (1991).

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            Initial solution volume.
        """
        if temperature >= self.Tc:
            initial_volume: float = self.GAS_CONSTANT * temperature / pressure + self.b
        else:
            initial_volume = self.b / 2

        return initial_volume


@dataclass(kw_only=True)
class CORKFullCO2HP91(CORKFullABC):
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
class CORKFullCO2HP98(CORKFullCO2HP91):
    """Full CORK equation for CO2 from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.

    See base class.
    """

    a_virial: tuple[float, float] = field(init=False, default=(5.40776e-3, -1.59046e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-1.78198e-1, 2.45317e-5))


@dataclass(kw_only=True)
class CORKFullH2OHP91(CORKFullABC):
    """Full CORK equation for H2O from Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(0,))  # Not used.
    b0: float = field(init=False, default=1.465)
    a_virial: tuple[float, float] = field(init=False, default=(-3.2297554e-3, 2.2215221e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-3.025650e-2, -5.343144e-6))
    Ta: float = field(init=False, default=Ta_H2O)
    Tc: float = field(init=False, default=Tc_H2O)
    P0: float = field(init=False, default=2.0)
    mrk_fluid: MRKImplicitABC = field(init=False, default_factory=MRKH2OFluidHP91)
    mrk_gas: MRKImplicitABC = field(init=False, default_factory=MRKH2OGasHP91)
    mrk_liquid: MRKImplicitABC = field(init=False, default_factory=MRKH2OLiquidHP91)

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

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            Volume.
        """
        Psat: float = self.Psat(temperature)

        if temperature >= self.Tc:
            logger.debug("temperature >= critical temperature of %f", self.Tc)
            volume: float = self.mrk_fluid.volume(temperature, pressure)

        elif temperature <= self.Ta and pressure <= Psat:
            logger.debug("temperature <= %f and pressure <= %f", self.Ta, Psat)
            volume = self.mrk_gas.volume(temperature, pressure)

        elif temperature < self.Tc and pressure <= Psat:
            logger.debug("temperature < %f and pressure <= %f", self.Tc, Psat)
            volume = self.mrk_fluid.volume(temperature, pressure)

        else:  # temperature < self.Tc and pressure > Psat:
            if temperature <= self.Ta:
                volume = self.mrk_liquid.volume(temperature, pressure)
            else:
                volume = self.mrk_fluid.volume(temperature, pressure)

        if pressure > self.P0:
            volume += self.virial.volume(temperature, pressure)

        return volume

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
        Psat: float = self.Psat(temperature)

        if temperature >= self.Tc:
            logger.debug("temperature >= critical temperature of %f", self.Tc)
            volume_integral: float = self.mrk_fluid.volume_integral(temperature, pressure)

        elif temperature <= self.Ta and pressure <= Psat:
            logger.debug("temperature <= %f and pressure <= %f", self.Ta, Psat)
            volume_integral = self.mrk_gas.volume_integral(temperature, pressure)

        elif temperature < self.Tc and pressure <= Psat:
            logger.debug("temperature < %f and pressure <= %f", self.Tc, Psat)
            volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        else:  # temperature < self.Tc and pressure > Psat:
            if temperature <= self.Ta:
                # To converge to the correct root the actual pressure must be used to compute the
                # initial volume, not Psat.
                volume_init: float = self.GAS_CONSTANT * temperature / pressure + 10 * self.b
                volume_integral = self.mrk_gas.volume_integral(
                    temperature, Psat, volume_init=volume_init
                )
                volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
                volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)
            else:
                volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CORKFullH2OHP98(CORKFullH2OHP91):
    """Full CORK equation for H2O from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.

    See base class.
    """

    a_virial: tuple[float, float] = field(init=False, default=(1.9853e-3, 0))
    b_virial: tuple[float, float] = field(init=False, default=(-8.9090e-2, 0))
    c_virial: tuple[float, float] = field(init=False, default=(8.0331e-2, 0))


@dataclass(kw_only=True)
class CORKCorrespondingStatesHP91(MRKExplicitABC):
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
class CORKSimpleCO2HP91(CORKCorrespondingStatesHP91):
    """Critical constants for CO2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=304.2)  # K
    Pc: float = field(init=False, default=0.0738)  # kbar


@dataclass(kw_only=True)
class CORKCorrespondingStatesCH4HP91(CORKCorrespondingStatesHP91):
    """Critical constants for CH4. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=190.6)  # K
    Pc: float = field(init=False, default=0.0460)  # kbar


@dataclass(kw_only=True)
class CORKCorrespondingStatesH2HP91(CORKCorrespondingStatesHP91):
    """Critical constants for H2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=41.2)  # K
    Pc: float = field(init=False, default=0.0211)  # kbar


@dataclass(kw_only=True)
class CORKCorrespondingStatesCOHP91(CORKCorrespondingStatesHP91):
    """Critical constants for CO. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=132.9)  # K
    Pc: float = field(init=False, default=0.0350)  # kbar


@dataclass(kw_only=True)
class CORKCorrespondingStatesS2HP11(CORKCorrespondingStatesHP91):
    """Critical constants for S2. See base class.

    Holland and Powell (2011) state that the critical constants for S2 are taken from:

        Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
        McGraw-Hill, New York.

    In the fifth edition of this book S2 is not given (only S is), so instead the critical
    constants for S2 are taken from:

        Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
        Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.
        http://www.minsocam.org/ammin/AM77/AM77_1038.pdf
    """

    Tc: float = field(init=False, default=208.15)  # K
    Pc: float = field(init=False, default=0.072954)  # kbar


@dataclass(kw_only=True)
class CORKCorrespondingStatesH2SHP11(CORKCorrespondingStatesHP91):
    """Critical constants for H2S. See base class.

    Appendix A.19 in:

        Poling, Prausnitz, and O'Connell, 2001. The Properties of Gases and Liquids, 5th edition.
        McGraw-Hill, New York. DOI: 10.1036/0070116822.
    """

    Tc: float = field(init=False, default=373.4)  # K
    Pc: float = field(init=False, default=0.08963)  # kbar


def get_holland_and_powell_fugacity_models() -> dict[str, FugacityModelABC]:
    """Gets a dictionary of the preferred fugacity models to use for each species."""
    models: dict[str, FugacityModelABC] = {}
    models["CH4"] = CORKCorrespondingStatesCH4HP91()
    models["CO"] = CORKCorrespondingStatesCOHP91()
    models["CO2"] = CORKFullCO2HP98()
    models["H2"] = CORKCorrespondingStatesH2HP91()
    models["H2O"] = CORKFullH2OHP98()
    models["H2S"] = CORKCorrespondingStatesH2SHP11()
    models["S2"] = CORKCorrespondingStatesS2HP11()

    return models
