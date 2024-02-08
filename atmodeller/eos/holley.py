"""Real gas EOSs from Holley et al. (1958)

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.


Compressibility factors and fugacity coefficients calculated from the Beattie-Bridgeman equation
of state for hydrogen, nitrogen, oxygen, carbon dioxide, ammonia, methane, and helium.
C. E. Holley, Jr., W. J. Worlton, and R. K. Zeigler (1958), doi: 10.2172/4289497

https://www.osti.gov/biblio/4289497

Functions:
    get_holley_eos_models: Gets the preferred EOS models to use for each species.

Real gas EOSs (class instances) in this module that can be imported:
    H2_Beattie_holley: Beattie-Bridgeman for H2
    N2_Beattie_holley: Beattie-Bridgeman for N2
    O2_Beattie_holley: Beattie-Bridgeman for O2
    CO2_Beattie_holley: Beattie-Bridgeman for CO2
    NH3_Beattie_holley: Beattie-Bridgeman for NH3
    CH4_Beattie_holley: Beattie-Bridgeman for CH4
    He_Beattie_holley: Beattie-Bridgeman for He

Examples:
    Get the preferred EOS models for various species. Note that the input pressure should always be
    in bar:
    
    ```python
    >>> from atmodeller.eos.holley import get_holley_eos_models
    >>> models = get_holley_eos_models()
    >>> # list the available species
    >>> models.keys()
    >>> # Get the EOS model for He
    >>> he_model = models['He']
    >>> # Determine the fugacity coefficient at 1000 K and 100 bar
    >>> fugacity_coefficient = he_model.get_value(temperature=1000, pressure=100)
    >>> print(fugacity_coefficient)
    1.0165341564229526
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from atmodeller import ATMOSPHERE, GAS_CONSTANT_BAR
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import UnitConversion, debug_decorator

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BeattieBridgeman(RealGas):
    """Beattie-Bridgeman equation

    Compressibility factors and fugacity coefficients calculated from the Beattie-Bridgeman
    equation of state for hydrogen, nitrogen, oxygen, carbon dioxide, ammonia, methane, and helium.
    C. E. Holley, Jr., W. J. Worlton, and R. K. Zeigler (1958), doi: 10.2172/4289497

    https://www.osti.gov/biblio/4289497

    Args:
        A0: A0 constant
        a: a constant
        B0: B0 constant
        b: b constant
        c: c constant

    Attributes:
        A0: A0 constant
        a: a constant
        B0: B0 constant
        b: b constant
        c: c constant
    """

    A0: float
    a: float
    B0: float
    b: float
    c: float

    @debug_decorator(logger)
    def volume_roots(self, temperature: float, pressure: float) -> np.ndarray:
        """Real and (potentially) physically meaningful volume solutions

        E.g. Equation 2, Holley et al. (1958), https://www.osti.gov/biblio/4289497

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume solutions of the Beattie-Bridgeman equation in m^3 mol^(-1)
        """
        coefficients: list[float] = []
        coefficients.append(-GAS_CONSTANT_BAR * self.c * self.b * self.B0 / temperature**2)
        coefficients.append(
            GAS_CONSTANT_BAR * temperature * self.b * self.B0
            + GAS_CONSTANT_BAR * self.c * self.B0 / temperature**2
            - self.a * self.A0
        )
        coefficients.append(
            -GAS_CONSTANT_BAR * temperature * self.B0
            + GAS_CONSTANT_BAR * self.c / temperature**2
            + self.A0
        )
        coefficients.append(-GAS_CONSTANT_BAR * temperature)
        coefficients.append(pressure)

        polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="V")
        logger.debug("Beattie-Bridgeman equation = %s", polynomial)
        volume_roots: np.ndarray = polynomial.roots()
        logger.debug("volume_roots = %s", volume_roots)
        # Numerical solution could result in a small imaginery component, even though the real
        # root is purely real.
        real_roots: np.ndarray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
        # Physically meaningful volumes must be positive.
        positive_roots: np.ndarray = real_roots[real_roots > 0]
        # In general, several roots could be returned, and subclasses will need to determine which
        # is the correct volume to use.
        logger.debug("V = %s", positive_roots)

        return positive_roots

    @debug_decorator(logger)
    def volume(self, *args, **kwargs) -> float:
        """Volume

        The paper doesn't say which root to take, but one root is very small and the maximum root
        gives a volume that agrees with the tabulated compressibility factor for all species.

        Args:
            *args: Positional arguments to pass to self.volume_roots
            **kwargs: Keyword arguments to pass to self.volume_roots

        Returns:
            Volume in m^3 mol^(-1)
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        return np.max(volume_roots)

    @debug_decorator(logger)
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral.

        Equation 11 (Holley et al., 1958) because it includes the limits of integration.  It's
        necessary to multiply ln f by RT to get the volume integral. In practice though, due to the
        choice of integration limits, I don't think this formulation differs from Equation 8.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        # Volume evaluated at T and P conditions
        vol: float = self.volume(temperature, pressure)
        volume_integral: float = (
            GAS_CONSTANT_BAR
            * temperature
            * (
                np.log(GAS_CONSTANT_BAR * temperature / vol)
                + (self.B0 - self.c / temperature**3 - self.A0 / (GAS_CONSTANT_BAR * temperature))
                * 2
                / vol
                - (
                    self.b * self.B0
                    + self.c * self.B0 / temperature**3
                    - self.a * self.A0 / (GAS_CONSTANT_BAR * temperature)
                )
                * 3
                / (2 * vol**2)
                + (self.c * self.b * self.B0 / temperature**3) * 4 / (3 * vol**3)
            )
        )
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral


# Coefficients from Table I, which must be converted to the correct units scheme (SI and pressure
# in bar). Using the original table values below allows easy visual comparison and ensures that
# the base class does not have to figure out how to correctly convert units.

# Converts volumes from litres to m^3
volume_conversion: Callable = UnitConversion.litre_to_m3
# Converts PV**2 coefficient to be in terms of m^3 and bar
A0_conversion: Callable = lambda x: x * ATMOSPHERE * UnitConversion.litre_to_m3() ** 2

H2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.1975),
    a=volume_conversion(-0.00506),
    B0=volume_conversion(0.02096),
    b=volume_conversion(-0.04359),
    c=volume_conversion(0.0504e4),
)
N2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.3445),
    a=volume_conversion(0.02617),
    B0=volume_conversion(0.05046),
    b=volume_conversion(-0.00691),
    c=volume_conversion(4.2e4),
)
O2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.4911),
    a=volume_conversion(0.02562),
    B0=volume_conversion(0.04624),
    b=volume_conversion(0.004208),
    c=volume_conversion(4.8e4),
)
CO2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(5.0065),
    a=volume_conversion(0.07132),
    B0=volume_conversion(0.10476),
    b=volume_conversion(0.07235),
    c=volume_conversion(66e4),
)
NH3_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.3930),
    a=volume_conversion(0.17031),
    B0=volume_conversion(0.03415),
    b=volume_conversion(0.19112),
    c=volume_conversion(476.87e4),
)
CH4_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.2769),
    a=volume_conversion(0.01855),
    B0=volume_conversion(0.05587),
    b=volume_conversion(-0.01587),
    c=volume_conversion(12.83e4),
)
He_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.0216),
    a=volume_conversion(0.05984),
    B0=volume_conversion(0.01400),
    b=0,
    c=volume_conversion(0.004e4),
)


def get_holley_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred EOS models to use for each species.

    Returns:
        Dictionary of prefered EOS models for each species
    """
    models: dict[str, RealGas] = {}
    models["CH4"] = CH4_Beattie_holley
    models["CO2"] = CO2_Beattie_holley
    models["H2"] = H2_Beattie_holley
    models["He"] = He_Beattie_holley
    models["N2"] = N2_Beattie_holley
    models["NH3"] = NH3_Beattie_holley
    models["O2"] = O2_Beattie_holley

    return models
