"""Base classes for Holland and Powell (1991, 1998, 2011).

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root

from atmodeller import GAS_CONSTANT
from atmodeller.eos.interfaces import FugacityModelABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MRKABC(FugacityModelABC):
    """A Modified Redlich Kwong (MRK) EOS.

    For example, Equation 3, Holland and Powell (1991):
        P = RT/(V-b) - a/(V(V+b)T**0.5)

    where:
        P is pressure.
        T is temperature.
        V is the molar volume.
        R is the gas constant.
        a is the Redlich-Kwong function, which is a function of T.
        b is the Redlich-Kwong constant b.

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: See base class.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: See base class.
        GAS_CONSTANT: See base class.
    """

    a_coefficients: tuple[float, ...]
    b0: float

    @abstractmethod
    def a(self, temperature: float) -> float:
        """MRK a parameter computed from self.a_coefficients.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self) -> float:
        """MRK b parameter, which is is independent of temperature, computed from self.b0."""
        raise NotImplementedError


@dataclass(kw_only=True)
class MRKExplicitABC(MRKABC):
    """A Modified Redlich Kwong (MRK) EOS with explicit equations for the volume and its integral.

    See base class.
    """

    def volume(self, temperature: float, pressure: float) -> float:
        """Convenient volume-explicit equation. Equation 7, Holland and Powell (1991).

        Without complications of critical phenomena the MRK equation can be simplified using the
        approximation:

            V ~ RT/P + b

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            MRK volume.
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
        """Volume-explicit integral (VdP). Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        volume_integral: float = (
            self.GAS_CONSTANT * temperature * np.log(self.scaling * pressure)
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
class MRKImplicitABC(MRKABC):
    """A Modified Redlich Kwong (MRK) EOS in an implicit form.

    See base class.
    """

    @property
    def b(self) -> float:
        """This class is not used for corresponding states which means b0 is the b coefficient."""
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
            volume_init: Initial volume estimate. Defaults to None.

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

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial volume estimate. Defaults to None.

        Returns:
            Volume integral.
        """
        z: float = self.compressibility_factor(temperature, pressure, volume_init=volume_init)
        A: float = self.A_factor(temperature, pressure)
        B: float = self.B_factor(temperature, pressure)
        # Recall that the base class requires a specification of the volume_integral, but the
        # equations in Holland and Powell (1991) are in terms of the fugacity coefficient.
        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)
        ln_fugacity: float = np.log(self.scaling * pressure) + ln_fugacity_coefficient
        volume_integral: float = self.GAS_CONSTANT * temperature * ln_fugacity

        return volume_integral

    @abstractmethod
    def initial_solution_volume(self, temperature: float, pressure: float) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            Initial solution volume.
        """
        ...

    def volume(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Solves the MRK equation numerically to compute the volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial volume estimate. Defaults to a value to find the single root above
                the critical temperature, Tc. Other initial values may be necessary for multi-root
                systems (e.g., relating to the critical behaviour of H2O).

        Returns:
            volume.
        """
        if volume_init is None:
            volume_init = self.initial_solution_volume(temperature, pressure)

        sol = root(
            self._objective_function_volume,
            volume_init,
            args=(temperature, pressure),
            jac=self._volume_jacobian,
        )
        volume: float = sol.x[0]

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
class MRKCriticalBehaviour(FugacityModelABC):
    """A MRK model that accommodates critical behaviour.

    Args:
        mrk_fluid: The MRK for the supercritical fluid.
        mrk_gas: The MRK for the subcritical gas.
        mrk_liquid: The MRK for the subcritical liquid.
        scaling: See base class.
        Ta: Temperature at which a_gas = a in the MRK formulation.
        Tc: Critical temperature.

    Attributes:
        mrk_fluid: The MRK for the supercritical fluid.
        mrk_gas: The MRK for the subcritical gas.
        mrk_liquid: The MRK for the subcritical liquid.
        Ta: Temperature at which a_gas = a in the MRK formulation.
        Tc: Critical temperature.
        scaling: See base class.
        GAS_CONSTANT: See base class.

    """

    mrk_fluid: MRKImplicitABC
    mrk_gas: MRKImplicitABC
    mrk_liquid: MRKImplicitABC
    Ta: float
    Tc: float

    @abstractmethod
    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Saturation curve pressure in kbar.
        """
        ...

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

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral. Appendix A, Holland and Powell (1991).

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
                volume_init: float = (
                    self.mrk_gas.GAS_CONSTANT * temperature / pressure + 10 * self.mrk_gas.b
                )
                volume_integral = self.mrk_gas.volume_integral(
                    temperature, Psat, volume_init=volume_init
                )
                volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
                volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)
            else:
                volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        return volume_integral


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
            significantly, and may be determined from experimental data. Defaults to zero, which is
            appropriate for the corresponding states case.
        Tc: Critical temperature in kelvin. Defaults to 1, which effectively means it is unused.
        Pc: Critical pressure. Defaults to 1, which effectively means it is unused.
        scaling: Scaling depending on the units of the coefficients. Defaults to unity.

    Attributes:
        a_coefficients: Coefficients for a polynomial of the form a = a0 * a1 * T, where a0 and a1
            may be additionally (internally) scaled by Tc and Pc in the case of corresponding
            states.
        b_coefficients: As above for the b coefficients.
        c_coefficients: As above for the c coefficients.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data.
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        scaling: Scaling depending on the units of the coefficients.
        GAS_CONSTANT: Gas constant with the appropriate units depending on the units of the
            coefficients.
    """

    a_coefficients: tuple[float, float]
    b_coefficients: tuple[float, float]
    c_coefficients: tuple[float, float]
    P0: float = 0  # Default must be zero for corresponding states.
    Pc: float = 1  # Defaults to 1, which effectively means unused.
    Tc: float = 1  # Defaults to 1, which effectively means unused.
    scaling: float = 1
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
            pressure: Pressure.

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
            pressure: Pressure.

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
class CORKFullABC(FugacityModelABC):
    """A Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    Args:
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Defaults to zero.
        mrk: Fugacity model for computing the MRK contribution.
        a_virial: a coefficients for the virial compensation. Defaults to zero coefficients.
        b_virial: b coefficients for the virial compensation. Defaults to zero coefficients.
        c_virial: c coefficients for the virial compensation. Defaults to zero coefficients.
        scaling: See base class.

    Attributes:
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data.
        mrk: A FugacityModelABC instance.
        a_virial: a coefficients for the virial compensation. Defaults to zero coefficients.
        b_virial: b coefficients for the virial compensation. Defaults to zero coefficients.
        c_virial: c coefficients for the virial compensation. Defaults to zero coefficients.
        virial: A VirialCompensation instance.
        scaling: See base class.
        GAS_CONSTANT: See base class.
    """

    P0: float = 0
    mrk: FugacityModelABC
    a_virial: tuple[float, float] = field(init=False, default=(0, 0))
    b_virial: tuple[float, float] = field(init=False, default=(0, 0))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))
    virial: VirialCompensation = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.virial = VirialCompensation(
            a_coefficients=self.a_virial,
            b_coefficients=self.b_virial,
            c_coefficients=self.c_virial,
            P0=self.P0,
            scaling=self.scaling,
        )

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume including virial compensation. Equation 7a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume including the virial compensation.
        """
        volume: float = self.mrk.volume(temperature, pressure)

        if pressure > self.P0:
            volume += self.virial.volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral including virial compensation. Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral including the virial compensation.
        """
        volume_integral: float = self.mrk.volume_integral(temperature, pressure)

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral
