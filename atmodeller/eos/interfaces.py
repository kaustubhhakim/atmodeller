"""Interfaces for real gas equations of state.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import GetValueABC
from atmodeller.utilities import debug_decorator

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RealGasABC(GetValueABC):
    """A real gas equation of state (EOS)

    This base class requires a specification for the volume and volume integral. Then the
    fugacity and related quantities can be computed using the standard relation:

    RTlnf = integral(VdP)

    If critical_temperature and critical_pressure are set to their default value of unity, then
    these quantities are effectively not used, and the model coefficients should be in terms of
    the real temperature and pressure. But for corresponding state models, which are formulated in
    terms of a reduced temperature and a reduced pressure, the critical_temperature and
    critical_pressure must be set to appropriate values for the species under consideration.

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
    """

    critical_temperature: float = 1  # Default of one is equivalent to not used
    critical_pressure: float = 1  # Default of one is equivalent to not used
    standard_state_pressure: float = field(init=False, default=1)  # 1 bar

    @debug_decorator(logger)
    def scaled_pressure(self, pressure: float) -> float:
        """Scaled pressure, i.e. a reduced pressure when critical pressure is not unity.

        Args:
            pressure: Pressure in bar

        Returns:
            The scaled (reduced) pressure, which is dimensionless
        """
        scaled_pressure: float = pressure / self.critical_pressure

        return scaled_pressure

    @debug_decorator(logger)
    def scaled_temperature(self, temperature: float) -> float:
        """Scaled temperature, i.e. a reduced temperature when critical temperature is not unity.

        Args:
            temperature: Temperature in kelvin

        Returns:
            The scaled (reduced) temperature, which is dimensionless.
        """
        scaled_temperature: float = temperature / self.critical_temperature

        return scaled_temperature

    @debug_decorator(logger)
    def compressibility_parameter(self, temperature: float, pressure: float, **kwargs) -> float:
        """Compressibility parameter at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar
            **kwargs: Catches unused keyword arguments. Used for overrides in subclasses.

        Returns:
            The compressibility parameter, Z, which is dimensionless.
        """
        del kwargs
        volume: float = self.volume(temperature, pressure)
        volume_ideal: float = self.ideal_volume(temperature, pressure)
        Z: float = volume / volume_ideal

        return Z

    @debug_decorator(logger)
    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Fugacity coefficient evaluated at temperature and pressure, which is dimensionaless.
        """
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)

        return fugacity_coefficient

    @debug_decorator(logger)
    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity.

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Natural log of the fugacity
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT * temperature
        )

        return ln_fugacity

    @debug_decorator(logger)
    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in the same units as the input pressure.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))

        return fugacity

    @debug_decorator(logger)
    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            fugacity coefficient, which is non-dimensional.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    @debug_decorator(logger)
    def ideal_volume(self, temperature: float, pressure: float) -> float:
        """Ideal volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            ideal volume in m^3 mol^(-1)
        """
        volume_ideal: float = GAS_CONSTANT * temperature / pressure

        return volume_ideal

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP).

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        ...


@dataclass(kw_only=True)
class ModifiedRedlichKwongABC(RealGasABC):
    """A Modified Redlich Kwong (MRK) EOS

    For example, Equation 3, Holland and Powell (1991):
        P = RT/(V-b) - a/(V(V+b)T**0.5)

    where:
        P is pressure
        T is temperature
        V is the molar volume
        R is the gas constant
        a is the Redlich-Kwong function, which is a function of T
        b is the Redlich-Kwong constant b

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter
        b0: Coefficient to compute the Redlich-Kwong constant b

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter
        b0: Coefficient to compute the Redlich-Kwong constant b
    """

    a_coefficients: tuple[float, ...]
    b0: float

    @abstractmethod
    def a(self, temperature: float) -> float:
        """MRK a parameter computed from self.a_coefficients

        Args:
            temperature: Temperature in kelvin

        Returns:
            MRK a parameter
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self) -> float:
        """MRK b parameter, which is is independent of temperature, computed from self.b0."""
        raise NotImplementedError


@dataclass(kw_only=True)
class MRKExplicitABC(ModifiedRedlichKwongABC):
    """A Modified Redlich Kwong (MRK) EOS with explicit equations for the volume and its integral.

    See base class.
    """

    @debug_decorator(logger)
    def a(self, temperature: float) -> float:
        """Parameter a in Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a in J^2 bar^(-1) K^(1/2) mol^(-2)
        """
        a: float = (
            self.a_coefficients[0] * self.critical_temperature ** (5.0 / 2)
            + self.a_coefficients[1] * self.critical_temperature ** (3.0 / 2) * temperature
            + self.a_coefficients[2] * self.critical_temperature ** (1.0 / 2) * temperature**2
        )
        a /= self.critical_pressure

        return a

    @property
    @debug_decorator(logger)
    def b(self) -> float:
        """Parameter b in Equation 9, Holland and Powell (1991).

        Returns:
            Parameter b in J bar^(-1) mol^(-1)
        """
        b: float = self.b0 * self.critical_temperature / self.critical_pressure

        return b

    @debug_decorator(logger)
    def volume(self, temperature: float, pressure: float) -> float:
        """Convenient volume-explicit equation. Equation 7, Holland and Powell (1991).

        Without complications of critical phenomena the MRK equation can be simplified using the
        approximation:

            V ~ RT/P + b

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            MRK volume in m^3/mol
        """
        volume: float = (
            GAS_CONSTANT * temperature / pressure
            + self.b
            - self.a(temperature)
            * GAS_CONSTANT
            * np.sqrt(temperature)
            / (GAS_CONSTANT * temperature + self.b * pressure)
            / (GAS_CONSTANT * temperature + 2.0 * self.b * pressure)
        )

        return volume

    @debug_decorator(logger)
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume-explicit integral (VdP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        volume_integral: float = (
            GAS_CONSTANT * temperature * np.log(pressure)
            + self.b * pressure
            + self.a(temperature)
            / self.b
            / np.sqrt(temperature)
            * (
                np.log(GAS_CONSTANT * temperature + self.b * pressure)
                - np.log(GAS_CONSTANT * temperature + 2.0 * self.b * pressure)
            )
        )

        return volume_integral


@dataclass(kw_only=True)
class VirialCompensation(RealGasABC):
    """A compensation term for the increasing deviation of the MRK volumes with pressure

    General form of the equation from Holland and Powell (1998), and also see Holland and Powell
    (1991) Equations 4 and 9:

        V_virial = a(P-P0) + b(P-P0)**0.5 + c(P-P0)**0.25

    This form also works for the virial compensation term from Holland and Powell (1991), in which
    case c=0. critical_pressure and critical_temperature are required for gases which are known to
    obey approximately the principle of corresponding states.

    Although this looks similar to an EOS, it's important to remember that for Holland and Powell
    it only calculates an additional perturbation to the volume and the volume integral of an MRK
    EOS, and hence it does not return a meaningful volume or volume integral by itself.

    Args:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be scaled (internally) by critical parameters for corresponding states
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Defaults to zero, which is
            appropriate for the corresponding states case.
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)

    Attributes:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally (internally) scaled by Tc and Pc in the case of corresponding
            states.
        b_coefficients: As above for the b coefficients
        c_coefficients: As above for the c coefficients
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
    """

    a_coefficients: tuple[float, float]
    b_coefficients: tuple[float, float]
    c_coefficients: tuple[float, float]
    P0: float

    @debug_decorator(logger)
    def a(self, temperature: float) -> float:
        """a parameter in Holland and Powell (1998)

        Args:
            temperature: Temperature in kelvin

        Returns:
            a parameter in J bar^(-2) mol^(-1)
        """
        a: float = (
            self.a_coefficients[0] * self.critical_temperature
            + self.a_coefficients[1] * temperature
        )
        a /= self.critical_pressure**2

        return a

    @debug_decorator(logger)
    def b(self, temperature: float) -> float:
        """b parameter in Holland and Powell (1998)

        Args:
            temperature: Temperature in kelvin

        Returns:
            b parameter in J bar^(-3/2) mol^(-1)
        """
        b: float = (
            self.b_coefficients[0] * self.critical_temperature
            + self.b_coefficients[1] * temperature
        )
        b /= self.critical_pressure ** (3 / 2)

        return b

    @debug_decorator(logger)
    def c(self, temperature: float) -> float:
        """c parameter in Holland and Powell (1998)

        Note the scalings by self.Tc and self.Pc to accommodate corresponding states.

        Args:
            temperature: Temperature in kelvin

        Returns:
            c parameter in J bar^(-5/4) mol^(-1)
        """
        c: float = (
            self.c_coefficients[0] * self.critical_temperature
            + self.c_coefficients[1] * temperature
        )
        c /= self.critical_pressure ** (5 / 4)

        return c

    @debug_decorator(logger)
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume contribution

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume contribution
        """
        volume: float = (
            self.a(temperature) * (pressure - self.P0)
            + self.b(temperature) * (pressure - self.P0) ** 0.5
            + self.c(temperature) * (pressure - self.P0) ** 0.25
        )

        return volume

    @debug_decorator(logger)
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP) contribution

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral contribution
        """
        volume_integral: float = (
            self.a(temperature) / 2.0 * (pressure - self.P0) ** 2
            + 2.0 / 3.0 * self.b(temperature) * (pressure - self.P0) ** (3.0 / 2.0)
            + 4.0 / 5.0 * self.c(temperature) * (pressure - self.P0) ** (5.0 / 4.0)
        )

        return volume_integral


@dataclass(kw_only=True)
class CORKABC(RealGasABC):
    """A Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991)

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)
        P0: Pressure in bar at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data
        mrk: MRK model for computing the MRK contribution
        a_virial: a coefficients for the virial compensation. Defaults to zero coefficients
        b_virial: b coefficients for the virial compensation. Defaults to zero coefficients
        c_virial: c coefficients for the virial compensation. Defaults to zero coefficients

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data
        mrk: MRK model for computing the MRK contribution
        a_virial: a coefficients for the virial compensation
        b_virial: b coefficients for the virial compensation
        c_virial: c coefficients for the virial compensation
        virial: A VirialCompensation instance
    """

    P0: float
    mrk: RealGasABC
    a_virial: tuple[float, float] = field(init=False, default=(0, 0))
    b_virial: tuple[float, float] = field(init=False, default=(0, 0))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            P0=self.P0,
            critical_temperature=self.critical_temperature,
            critical_pressure=self.critical_pressure,
        )

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume including virial compensation. Equation 7a, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume including the virial compensation in m^3 mol^(-1)
        """
        volume: float = self.mrk.volume(temperature, pressure)

        if pressure > self.P0:
            volume += self.virial.volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral including virial compensation. Equation 8, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral including the virial compensation in J mol^(-1)
        """
        volume_integral: float = self.mrk.volume_integral(temperature, pressure)

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class CombinedFugacityModel(RealGasABC):
    """Combines multiple fugacity models for different pressure ranges into a single model.

    Args:
        models: Fugacity models with coefficients specified and ordered by increasing pressure
        upper_pressure_bounds: Upper pressure bound in bar relevant to the fugacity class by
            position

    Attributes:
        models: Fugacity models with coefficients specified and ordered by increasing pressure
        upper_pressure_bounds: Upper pressure bound in bar relevant to the fugacity class by
            position
    """

    models: tuple[RealGasABC, ...]
    upper_pressure_bounds: tuple[float, ...]

    def _get_index(self, pressure: float) -> int:
        """Gets the index of the appropriate fugacity model using the pressure

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant fugacity model
        """
        for index, pressure_high in enumerate(self.upper_pressure_bounds):
            if pressure < pressure_high:
                return index
        # If the pressure is higher than all specified pressure ranges, use the last model.
        return len(self.models) - 1

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        index: int = self._get_index(pressure)
        volume: float = self.models[index].volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        index: int = self._get_index(pressure)
        volume: float = self.models[index].volume_integral(temperature, pressure)

        return volume


@dataclass(frozen=True)
class critical_data:
    """Critical temperature and pressure of a gas species.

    Args:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure in bar

    Attributes:
        Tc: Critical temperature in kelvin
        Pc: Critical pressure in bar
    """

    temperature: float
    pressure: float


# Critical temperature and pressure data for a corresponding states model, based on Table 2 in
# Shi and Saxena (1992) with some additions. For simplicity, we just compile one set of critical
# data, even though they vary a little between studies which could result in subtle differences.
# TODO: Commented values are Kelvin, pressure (kbar) to compare with old test data for Holland and
# Powell
critical_data_dictionary: dict[str, critical_data] = {
    "H2O": critical_data(647.25, 221.1925),
    "CO2": critical_data(304.15, 73.8659),  # 304.2, 0.0738
    "CH4": critical_data(191.05, 46.4069),  # 190.6, 0.0460
    "CO": critical_data(133.15, 34.9571),  # 132.9, 0.0350
    "O2": critical_data(154.75, 50.7638),
    # FIXME: Second set of data give closer to expected, but where did I get that from?
    "H2": critical_data(33.25, 12.9696),  # 41.2, 0.0211
    # Holland and Powell (2011) state that the critical constants for S2 are taken from:
    # Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
    # McGraw-Hill, New York.
    # In the fifth edition of this book S2 is not given (only S is), so instead the critical
    # constants for S2 are taken from:
    # Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
    # Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.
    # http://www.minsocam.org/ammin/AM77/AM77_1038.pdf
    "S2": critical_data(208.15, 72.954),
    "SO2": critical_data(430.95, 78.7295),
    "COS": critical_data(377.55, 65.8612),
    # Appendix A.19 in:
    # Poling, Prausnitz, and O'Connell, 2001. The Properties of Gases and Liquids, 5th edition.
    # McGraw-Hill, New York. DOI: 10.1036/0070116822.
    "H2S": critical_data(373.55, 90.0779),  # 373.4, 0.08963
    "N2": critical_data(126.2, 33.9),  # Saxena and Fei (1987)
    "Ar": critical_data(151, 48.6),  # Saxena and Fei (1987)
}
