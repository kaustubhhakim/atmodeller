"""Interfaces for real gas equations of state.

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

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import GetValueABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class FugacityModelABC(GetValueABC):
    """A fugacity model.

    This base class requires a specification for the volume and volume integral, since then the
    fugacity and related quantities can be computed using the standard relation:

    RTlnf = integral(VdP).

    Args:
        scaling: Scaling depending on the units of pressure (i.e., 1 for bar, kilo for kbar).
            Defaults to unity, meaning that the internal units of pressure are bar.

    Attributes:
        scaling: Scaling depending on the units of pressure.
        GAS_CONSTANT: Gas constant with the appropriate units.
    """

    scaling: float = 1
    GAS_CONSTANT: float = field(init=False, default=GAS_CONSTANT)

    def __post_init__(self):
        """Scales the GAS_CONSTANT to ensure it has the correct units."""
        self.GAS_CONSTANT /= self.scaling

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
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)

        return fugacity_coefficient

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity.

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

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
        """Fugacity in the same units as the input pressure.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity.
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))  # bar
        fugacity /= self.scaling  # to units of input pressure for consistency.

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity coefficient, which is non-dimensional.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        ...


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
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units.
    """

    a_coefficients: tuple[float, ...]
    b0: float
    scaling: float = kilo

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
        """MRK b parameter, which is is independent of temperature, computed from self.b0.

        Returns:
            MRK b parameter.
        """
        raise NotImplementedError

    @abstractmethod
    def volume_MRK(self, temperature: float, pressure: float) -> float:
        """MRK Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        ...

    def volume(self, *args, **kwargs) -> float:
        """Volume is just the MRK volume.

        *args and **kwargs are passed through.
        """
        return self.volume_MRK(*args, **kwargs)

    @abstractmethod
    def volume_integral_MRK(self, temperature: float, pressure: float) -> float:
        """MRK volume integral (VdP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        ...

    def volume_integral(self, *args, **kwargs) -> float:
        """Volume integral is just the MRK volume integral.

        *args and **kwargs are passed through.
        """
        return self.volume_integral_MRK(*args, **kwargs)


@dataclass(kw_only=True)
class MRKExplicitABC(MRKABC):
    """A Modified Redlich Kwong (MRK) EOS with explicit equations for the volume and its integral.

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units.
    """

    def volume_MRK(self, temperature: float, pressure: float) -> float:
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

    def volume_integral_MRK(self, temperature: float, pressure: float) -> float:
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

    Args:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.

    Attributes:
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0.
        GAS_CONSTANT: Gas constant with the appropriate units.
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

    def compressibility_factor_MRK(
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
            * self.volume_MRK(temperature, pressure, volume_init=volume_init)
            / (self.GAS_CONSTANT * temperature)
        )

        return compressibility

    def volume_integral_MRK(
        self,
        temperature: float,
        pressure: float,
        *,
        volume_init: float | None = None,
    ) -> float:
        """Volume integral. Equation A.2., Holland and Powell (1991).

        These equations can be applied directly in the absence of critical phenomena in the range
        of temperature and pressure under consideration.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial volume estimate. Defaults to None.

        Returns:
            Volume integral.
        """
        z: float = self.compressibility_factor_MRK(temperature, pressure, volume_init=volume_init)
        # TODO: REMOVE print("Z = %f" % z)
        A: float = self.A_factor(temperature, pressure)
        # TODO: REMOVE print("A = %f" % A)
        B: float = self.B_factor(temperature, pressure)
        # TODO: REMOVE print("B = %f" % B)
        # Recall that the base class requires a specification of the volume_integral, but the
        # equations in Holland and Powell (1991) go via the fugacity coefficient.
        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)
        ln_fugacity: float = np.log(self.scaling * pressure) + ln_fugacity_coefficient
        # TODO: REMOVE print(ln_fugacity_coefficient)
        volume_integral: float = self.GAS_CONSTANT * temperature * ln_fugacity

        return volume_integral

    def volume_MRK(
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
            # From Holland and Powell (1991), above Tc there is only one real root.
            volume_init = self.GAS_CONSTANT * temperature / pressure + self.b

        volume_solution: np.ndarray = fsolve(
            self._objective_function_volume,
            volume_init,
            args=(temperature, pressure),
            fprime=self._volume_jacobian,
        )  # type: ignore

        volume: float = volume_solution[0]

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
class CorkFullABC(MRKImplicitABC):
    """A Full Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    Args:
        P0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data. Defaults to zero.
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) a parameter.
        b0: Coefficient to compute the Redlich-Kwong constant b.
        scaling: Scaling depending on the units of a_coefficients and b0. Defaults to kilo for
            the Holland and Powell data since pressures are in kbar.
        a_virial: a coefficients for the virial compensation. Defaults to zero coefficients.
        b_virial: b coefficients for the virial compensation. Defaults to zero coefficients.
        c_virial: c coefficients for the virial compensation. Defaults to zero coefficients.

    Attributes:
        p0: Pressure at which the MRK equation begins to overestimate the molar volume
            significantly, and may be determined from experimental data.
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

    P0: float = 0
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
        )

    def volume(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Volume including virial compensation. Equation 7a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial volume estimate. Defaults to None.

        Returns:
            Volume including the virial compensation.
        """
        volume: float = self.volume_MRK(temperature, pressure, volume_init=volume_init)

        if pressure > self.P0:
            volume += self.virial.volume(temperature, pressure)

        return volume

    def volume_integral(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Volume integral including virial compensation. Equation 8, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial volume estimate. Defaults to None.

        Returns:
            Volume integral including the virial compensation.
        """
        volume_integral: float = self.volume_integral_MRK(
            temperature, pressure, volume_init=volume_init
        )

        if pressure > self.P0:
            volume_integral += self.virial.volume_integral(temperature, pressure)

        return volume_integral
