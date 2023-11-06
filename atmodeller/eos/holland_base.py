"""Base classes for the fugacity models from Holland and Powell (1991, 1998, 2011).

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

from atmodeller import GAS_CONSTANT
from atmodeller.eos.interfaces import ModifiedRedlichKwongABC, RealGasABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MRKImplicitABC(ModifiedRedlichKwongABC):
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
        A: float = self.a(temperature) / (self.b * GAS_CONSTANT * temperature**1.5)

        return A

    def B_factor(self, temperature: float, pressure: float) -> float:
        """B factor in Appendix A of Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            B factor, which is non-dimensional.
        """
        B: float = self.b * pressure / (GAS_CONSTANT * temperature)

        return B

    def compressibility_parameter(
        self, temperature: float, pressure: float, *, volume_init: float | None = None
    ) -> float:
        """Compressibility parameter at temperature and pressure

        Overrides the base class because an extra keyword argument is required to ensure the
        correct compressibility parameter is returned.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.
            volume_init: Initial volume estimate. Defaults to None.

        Returns:
            The compressibility parameter, Z
        """
        volume: float = self.volume(temperature, pressure, volume_init=volume_init)
        volume_ideal: float = self.ideal_volume(temperature, pressure)
        Z: float = volume / volume_ideal

        return Z

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
        z: float = self.compressibility_parameter(temperature, pressure, volume_init=volume_init)
        A: float = self.A_factor(temperature, pressure)
        B: float = self.B_factor(temperature, pressure)
        # The base class requires a specification of the volume_integral, but the equations in
        # Holland and Powell (1991) are in terms of the fugacity coefficient.
        ln_fugacity_coefficient: float = z - 1 - np.log(z - B) - A * np.log(1 + B / z)
        ln_fugacity: float = np.log(pressure) + ln_fugacity_coefficient
        volume_integral: float = GAS_CONSTANT * temperature * ln_fugacity

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
            - GAS_CONSTANT * temperature * volume**2
            - (
                self.b * GAS_CONSTANT * temperature
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
            - 2 * GAS_CONSTANT * temperature * volume
            - (
                self.b * GAS_CONSTANT * temperature
                + self.b**2 * pressure
                - self.a(temperature) / np.sqrt(temperature)
            )
        )

        return jacobian


@dataclass(kw_only=True)
class MRKCriticalBehaviour(RealGasABC):
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
            Saturation curve pressure.
        """
        ...

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

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
                volume_init: float = GAS_CONSTANT * temperature / pressure + 10 * self.mrk_gas.b
                volume_integral = self.mrk_gas.volume_integral(
                    temperature, Psat, volume_init=volume_init
                )
                volume_integral -= self.mrk_liquid.volume_integral(temperature, Psat)
                volume_integral += self.mrk_liquid.volume_integral(temperature, pressure)
            else:
                volume_integral = self.mrk_fluid.volume_integral(temperature, pressure)

        return volume_integral
