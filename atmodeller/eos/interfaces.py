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
"""Real gas equations of state"""

# Use symbols from the relevant papers for consistency so pylint: disable=C0103

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from atmodeller import GAS_CONSTANT, GAS_CONSTANT_BAR
from atmodeller.utilities import UnitConversion, debug_decorator

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class RealGasProtocol(Protocol):
    def fugacity_coefficient(self, temperature: float, pressure: float) -> float: ...


@dataclass(kw_only=True)
class RealGas(ABC):
    r"""A real gas equation of state (EOS)

    Fugacity is computed using the standard relation:

    .. math::

        R T \ln f = \int V dP

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`f` is fugacity, :math:`V`
    is volume, and :math:`P` is pressure.

    ``critical_temperature`` and ``critical_pressure`` can be non-unity to allow for a
    corresponding states model.

    Args:
        critical_temperature: Critical temperature in K. Defaults to unity, meaning not used.
        critical_pressure: Critical pressure in bar. Defaults to unity, meaning not used.
    """

    critical_temperature: float = 1
    """Critical temperature in K"""
    critical_pressure: float = 1
    """Critical pressure in bar"""
    standard_state_pressure: float = field(init=False, default=1)
    """Standard state pressure in bar"""

    def scaled_pressure(self, pressure: float) -> float:
        """Scaled pressure, i.e. a reduced pressure when critical pressure is not unity

        Args:
            pressure: Pressure in bar

        Returns:
            The scaled (reduced) pressure, which is dimensionless
        """
        scaled_pressure: float = pressure / self.critical_pressure

        return scaled_pressure

    def scaled_temperature(self, temperature: float) -> float:
        """Scaled temperature, i.e. a reduced temperature when critical temperature is not unity

        Args:
            temperature: Temperature in K

        Returns:
            The scaled (reduced) temperature, which is dimensionless
        """
        scaled_temperature: float = temperature / self.critical_temperature

        return scaled_temperature

    def compressibility_parameter(self, temperature: float, pressure: float) -> float:
        """Compressibility parameter

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility parameter, which is dimensionless
        """
        volume: float = self.volume(temperature, pressure)
        volume_ideal: float = self.ideal_volume(temperature, pressure)
        compressibility_parameter: float = volume / volume_ideal

        return compressibility_parameter

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Natural log of the fugacity
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            GAS_CONSTANT * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            fugacity coefficient, which is non-dimensional
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        if fugacity_coefficient == np.inf:
            logger.warning("Fugacity coefficient has blown up (unphysical)")
            logger.warning("Evaluation at temperature = %f, pressure = %f", temperature, pressure)
            logger.warning("Setting fugacity coefficient to unity (ideal gas)")
            fugacity_coefficient = 1

        elif fugacity_coefficient == 0:
            logger.warning("Fugacity coefficient is zero (unphysical)")
            logger.warning("Evaluation at temperature = %f, pressure = %f", temperature, pressure)
            logger.warning("Setting fugacity coefficient to unity (ideal gas)")
            fugacity_coefficient = 1

        elif fugacity_coefficient < 0:
            logger.warning("Fugacity coefficient is negative (unphysical)")
            logger.warning("Evaluation at temperature = %f, pressure = %f", temperature, pressure)
            logger.warning("Setting fugacity coefficient to unity (ideal gas)")
            fugacity_coefficient = 1

        return fugacity_coefficient

    def ideal_volume(self, temperature: float, pressure: float) -> float:
        r"""Ideal volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Ideal volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        volume_ideal: float = GAS_CONSTANT_BAR * temperature / pressure

        return volume_ideal

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        r"""Volume

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        r"""Volume integral

        .. math::

            \int V dP

        This method requires that the units returned are :math:`\mathrm{J}\mathrm{mol}^{-1}`. Hence
        the following conversion is often necessary:

        .. math::

            1\ \mathrm{J} = 10^{-5}\ \mathrm{m}^3\mathrm{bar}

        There are functions to do this conversion in :class:`atmodeller.utilities.UnitConversion`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """


@dataclass(kw_only=True)
class IdealGas(RealGas):
    r"""An ideal gas equation of state:

    .. math::

        R T = P V

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`P` is pressure, and
    :math:`V` is volume.
    """

    @override
    def volume(self, temperature: float, pressure: float) -> float:
        return self.ideal_volume(temperature, pressure)

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        volume_integral: float = GAS_CONSTANT_BAR * temperature * np.log(pressure)
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral


@dataclass(kw_only=True)
class ModifiedRedlichKwongABC(RealGas):
    r"""A Modified Redlich Kwong (MRK) equation of state :cite:p:`{e.g.}HP91{Equation 3}`:

    .. math::

        P = \frac{RT}{V-b} - \frac{a(T)}{V(V+b)\sqrt{T}}

    where :math:`P` is pressure, :math:`T` is temperature, :math:`V` is the molar volume, :math:`R`
    the gas constant, :math:`a(T)` is the Redlich-Kwong function of :math:`T`, and  :math:`b` is
    the Redlich-Kwong constant.

    Args:
        critical_temperature: Critical temperature in K. Defaults to unity, meaning not used.
        critical_pressure: Critical pressure in bar. Defaults to unity, meaning not used.
        a_coefficients: Coefficients for the Modified Redlich Kwong (MRK) `a` parameter
        b0: The Redlich-Kwong constant `b`

    Attributes:
        critical_temperature: Critical temperature in K
        critical_pressure: Critical pressure in bar
        standard_state_pressure: Standard state pressure in bar
    """

    a_coefficients: tuple[float, ...]
    """Coefficients for the Modified Redlich Kwong (MRK) `a` parameter"""
    b0: float
    """The Redlich-Kwong constant `b`"""

    @abstractmethod
    def a(self, temperature: float) -> float:
        r"""MRK `a` parameter computed from :attr:`a_coefficients`.

        Args:
            temperature: Temperature in K

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self) -> float:
        r"""MRK `b` parameter computed from :attr:`b0`

        Units are :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        raise NotImplementedError


@dataclass(kw_only=True)
class MRKExplicitABC(ModifiedRedlichKwongABC):
    """A Modified Redlich Kwong (MRK) EOS in explicit form"""

    @override
    def a(self, temperature: float) -> float:
        r"""MRK `a` parameter from :attr:`a_coefficients` :cite:p:`HP91{Equation 9}`

        Args:
            temperature: Temperature in K

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        a: float = (
            self.a_coefficients[0] * self.critical_temperature ** (5.0 / 2)
            + self.a_coefficients[1] * self.critical_temperature ** (3.0 / 2) * temperature
            + self.a_coefficients[2] * self.critical_temperature ** (1.0 / 2) * temperature**2
        )
        a /= self.critical_pressure

        return a

    @property
    def b(self) -> float:
        r"""MRK `b` parameter computed from :attr:`b0` :cite:p:`HP91{Equation 9}`.

        Units are :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        b: float = self.b0 * self.critical_temperature / self.critical_pressure

        return b

    @override
    def volume(self, temperature: float, pressure: float) -> float:
        r"""Volume-explicit equation :cite:p:`HP91{Equation 7}`

        Without complications of critical phenomena the MRK equation can be simplified using the
        approximation:

        .. math::

            V \sim \frac{RT}{P} + b

        where :math:`V` is volume, :math:`R` is the gas constant, :math:`T` is temperature,
        :math:`P` is pressure, and :math:`b` is :attr:`b`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            MRK volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        volume: float = (
            GAS_CONSTANT_BAR * temperature / pressure
            + self.b
            - self.a(temperature)
            * GAS_CONSTANT_BAR
            * np.sqrt(temperature)
            / (GAS_CONSTANT_BAR * temperature + self.b * pressure)
            / (GAS_CONSTANT_BAR * temperature + 2.0 * self.b * pressure)
        )

        return volume

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        r"""Volume-explicit integral :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        volume_integral: float = (
            GAS_CONSTANT_BAR * temperature * np.log(pressure)
            + self.b * pressure
            + self.a(temperature)
            / self.b
            / np.sqrt(temperature)
            * (
                np.log(GAS_CONSTANT_BAR * temperature + self.b * pressure)
                - np.log(GAS_CONSTANT_BAR * temperature + 2.0 * self.b * pressure)
            )
        )
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral


@dataclass(kw_only=True)
class MRKImplicitABC(ModifiedRedlichKwongABC):
    r"""A Modified Redlich Kwong (MRK) EOS in implicit form

    Args:
        Ta: Temperature at which :math:`a_{\mathrm gas} = a` by constrained fitting :cite:p:`HP91`.
            Defaults to 0.
    """

    # TODO: Is this required here? Is the default value of 0 necessary or risks bugs? Use None?
    Ta: float = 0
    r"""Temperature at which :math:`a_{\mathrm gas} = a` by constrained fitting"""

    @override
    def a(self, temperature: float) -> float:
        r"""MRK `a` parameter from :attr:`a_coefficients` :cite:p:`HP91{Equation 6}`

        Args:
            temperature: Temperature in K

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * self.delta_temperature_for_a(temperature)
            + self.a_coefficients[2] * self.delta_temperature_for_a(temperature) ** 2
            + self.a_coefficients[3] * self.delta_temperature_for_a(temperature) ** 3
        )

        return a

    @abstractmethod
    def delta_temperature_for_a(self, temperature: float) -> float:
        """Temperature difference for the calculation of the `a` parameter

        Args:
            temperature: Temperature in K

        Returns:
            A temperature difference
        """

    @property
    def b(self) -> float:
        """MRK `b` parameter computed from :attr:`b0`.

        :class:`~MRKImplicitABC` is not used for corresponding states models so :attr:`b0` is the
        `b` coefficient.
        """
        return self.b0

    def A_factor(self, temperature: float, pressure: float) -> float:
        """`A` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `A` factor, which is non-dimensional
        """
        del pressure
        A_factor: float = self.a(temperature) / (self.b * GAS_CONSTANT_BAR * temperature**1.5)

        return A_factor

    def B_factor(self, temperature: float, pressure: float) -> float:
        """`B` factor :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            `B` factor, which is non-dimensional
        """
        B_factor: float = self.b * pressure / (GAS_CONSTANT_BAR * temperature)

        return B_factor

    @override
    def volume_integral(
        self,
        temperature: float,
        pressure: float,
    ) -> float:
        r"""Volume integral :cite:p:`HP91{Equation A.2}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        z: float = self.compressibility_parameter(temperature, pressure)
        A: float = self.A_factor(temperature, pressure)
        B: float = self.B_factor(temperature, pressure)
        # The base class requires a specification of the volume_integral, but the equations are in
        # terms of the fugacity coefficient.
        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)
        ln_fugacity: float = np.log(pressure) + ln_fugacity_coefficient
        volume_integral: float = GAS_CONSTANT_BAR * temperature * ln_fugacity
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral

    def volume_roots(self, temperature: float, pressure: float) -> np.ndarray:
        r"""Real and (potentially) physically meaningful volume solutions of the MRK equation

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume solutions of the MRK equation in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        coefficients: list[float] = []
        coefficients.append(-self.a(temperature) * self.b / np.sqrt(temperature))
        coefficients.append(
            -self.b * GAS_CONSTANT_BAR * temperature
            - self.b**2 * pressure
            + self.a(temperature) / np.sqrt(temperature)
        )
        coefficients.append(-GAS_CONSTANT_BAR * temperature)
        coefficients.append(pressure)

        polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="V")
        logger.debug("MRK equation = %s", polynomial)
        volume_roots: np.ndarray = polynomial.roots()
        # Numerical solution could result in a small imaginery component, even though the root is
        # real.
        real_roots: np.ndarray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
        # Physically meaningful volumes must be positive.
        positive_roots: np.ndarray = real_roots[real_roots > 0]
        # In general, several roots could be returned, and subclasses will need to determine which
        # is the correct volume to use depending on the phase (liquid, gas, etc.)
        logger.debug("V = %s", positive_roots)

        return positive_roots


@dataclass(kw_only=True)
class MRKCriticalBehaviour(RealGas):
    """A MRK model that accommodates critical behaviour

    Args:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation
        Tc: Critical temperature

    Attributes:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation
        Tc: Critical temperature
    """

    mrk_fluid: MRKImplicitABC
    mrk_gas: MRKImplicitABC
    mrk_liquid: MRKImplicitABC
    Ta: float
    Tc: float

    @abstractmethod
    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin

        Returns:
            Saturation curve pressure in bar
        """

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume in m^3 mol^(-1)
        """
        Psat: float = self.Psat(temperature)

        if temperature <= self.Ta and pressure <= Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Gas phase")
            volume = self.mrk_gas.volume(temperature, pressure)

        elif temperature <= self.Ta and pressure > Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Liquid phase")
            volume = self.mrk_liquid.volume(temperature, pressure)

        else:
            logger.debug("Fluid phase")
            volume = self.mrk_fluid.volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral. Appendix A, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            volume integral in J mol^(-1)
        """
        Psat: float = self.Psat(temperature)

        if temperature <= self.Ta and pressure <= Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) <= Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Gas phase")
            volume_integral = self.mrk_gas.volume_integral(temperature, pressure)

        elif temperature <= self.Ta and pressure > Psat:
            logger.debug(
                "temperature (%.1f) <= Ta (%.1f) and pressure (%.1f) > Psat (%.1f)",
                temperature,
                self.Ta,
                pressure,
                Psat,
            )
            logger.debug("Performing pressure integration")
            volume_integral = self.mrk_gas.volume_integral(temperature, Psat)
            volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
            volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)

        else:
            logger.debug("Fluid phase")
            volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        return volume_integral


@dataclass(kw_only=True)
class VirialCompensation(RealGas):
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
            may be scaled (internally) by critical parameters for corresponding states.
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

    a_coefficients: tuple[float, ...]
    b_coefficients: tuple[float, ...]
    c_coefficients: tuple[float, ...]
    P0: float

    @debug_decorator(logger)
    def a(self, temperature: float) -> float:
        """a parameter in Holland and Powell (1998)

        This is the d parameter in Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin

        Returns:
            a parameter in m^3 mol^(-1) bar^(-1)
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

        This is the c parameter in Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin

        Returns:
            b parameter in m^3 mol^(-1) bar^(-1/2)
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

        Args:
            temperature: Temperature in kelvin

        Returns:
            c parameter in m^3 mol^(-1) bar^(-1/4)
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
            Volume contribution in m^3 mol^(-1)
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
            Volume integral contribution in J mol^(-1)
        """
        volume_integral: float = (
            self.a(temperature) / 2.0 * (pressure - self.P0) ** 2
            + 2.0 / 3.0 * self.b(temperature) * (pressure - self.P0) ** (3.0 / 2.0)
            + 4.0 / 5.0 * self.c(temperature) * (pressure - self.P0) ** (5.0 / 4.0)
        )
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral


@dataclass(kw_only=True)
class CORK(RealGas):
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
    mrk: RealGas
    a_virial: tuple[float, ...] = (0, 0)
    b_virial: tuple[float, ...] = (0, 0)
    c_virial: tuple[float, ...] = (0, 0)
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
class CombinedEOSModel(RealGas):
    """Combines multiple EOS models for different pressure ranges into a single model.

    Args:
        models: EOS models ordered by increasing pressure from lowest to highest
        upper_pressure_bounds: Upper pressure bound in bar relevant to the EOS by position

    Attributes:
        models: EOS models ordered by increasing pressure from lowest to highest
        upper_pressure_bounds: Upper pressure bound in bar relevant to the EOS by position
    """

    models: tuple[RealGas, ...]
    upper_pressure_bounds: tuple[float, ...]

    def _get_index(self, pressure: float) -> int:
        """Gets the index of the appropriate EOS model using the pressure

        Args:
            pressure: Pressure in bar

        Returns:
            Index of the relevant EOS model
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
class CriticalData:
    """Critical temperature and pressure of a gas species.

    Args:
        Tc: Critical temperature in K
        Pc: Critical pressure in bar

    Attributes:
        Tc: Critical temperature in K
        Pc: Critical pressure in bar
    """

    temperature: float
    pressure: float


# Critical temperature and pressure data for a corresponding states model, based on Table 2 in
# Shi and Saxena (1992) with some additions. For simplicity, we just compile one set of critical
# data, even though they vary a little between studies which could result in subtle differences.

# Holland and Powell use slightly different critical data in their 1991 paper, which makes
# insignificant differences in most cases, but their values are give in comments for completeness.
# However, for H2 their critical values are significantly different and are therefore retained as
# a separate entry.
critical_data_dictionary: dict[str, CriticalData] = {
    "H2O": CriticalData(647.25, 221.1925),
    "CO2": CriticalData(304.15, 73.8659),  # 304.2, 73.8 from Holland and Powell (1991)
    "CH4": CriticalData(191.05, 46.4069),  # 190.6, 46 from Holland and Powell (1991)
    "CO": CriticalData(133.15, 34.9571),  # 132.9, 35 from Holland and Powell (1991)
    "O2": CriticalData(154.75, 50.7638),
    "H2": CriticalData(33.25, 12.9696),
    # Holland and Powell (1991) require different critical parameters
    "H2_Holland": CriticalData(41.2, 21.1),
    # Holland and Powell (2011) state that the critical constants for S2 are taken from:
    # Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
    # McGraw-Hill, New York.
    # In the fifth edition of this book S2 is not given (only S is), so instead the critical
    # constants for S2 are taken from:
    # Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
    # Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.
    # http://www.minsocam.org/ammin/AM77/AM77_1038.pdf
    "S2": CriticalData(208.15, 72.954),
    "SO2": CriticalData(430.95, 78.7295),
    "COS": CriticalData(377.55, 65.8612),
    # Appendix A.19 in:
    # Poling, Prausnitz, and O'Connell, 2001. The Properties of Gases and Liquids, 5th edition.
    # McGraw-Hill, New York. DOI: 10.1036/0070116822.
    "H2S": CriticalData(373.55, 90.0779),  # 373.4, 0.08963
    "N2": CriticalData(126.2, 33.9),  # Saxena and Fei (1987)
    "Ar": CriticalData(151, 48.6),  # Saxena and Fei (1987)
}
