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
"""Real gas EOSs from :cite:t:`HWZ58`

Examples:
    Get the preferred EOS models for various species from the Holley models::
    
        from atmodeller.eos.holley import get_holley_eos_models
        models = get_holley_eos_models()
        # List the available species
        models.keys()
        # Get the EOS model for He
        he_model = models['He']
        # Determine the fugacity coefficient at 1000 K and 100 bar
        fugacity_coefficient = he_model.fugacity_coefficient(temperature=1000, pressure=100)
        print(fugacity_coefficient)
"""

# Use symbols from the paper for consistency so pylint: disable=C0103

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from atmodeller import ATMOSPHERE, GAS_CONSTANT_BAR
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BeattieBridgeman(RealGas):
    """Beattie-Bridgeman equation

    Args:
        A0: A0 constant
        a: a constant
        B0: B0 constant
        b: b constant
        c: c constant
    """

    A0: float
    """A0 constant"""
    a: float
    """a constant"""
    B0: float
    """B0 constant"""
    b: float
    """b constant"""
    c: float
    """c constant"""

    def volume_roots(self, temperature: float, pressure: float) -> np.ndarray:
        r"""Real and potentially physically meaningful volume solutions :cite:p:`HWZ58{equation 2}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume solutions in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
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

    @override
    def volume(self, *args, **kwargs) -> float:
        r"""Volume

        :cite:t:`HWZ58` doesn't say which root to take, but one root is very small and the maximum
        root gives a volume that agrees with the tabulated compressibility factor for all species.

        Args:
            *args: Positional arguments to pass to :func:`volume_roots`
            **kwargs: Keyword arguments to pass to :func:`volume_roots`

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        return np.max(volume_roots)

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        r"""Volume integral :cite:p:`HWZ58{Equation 11}`.

        It is necessary to multiply :math:`\ln f` by :math:`RT` to obtain the volume integral.
        Due to the choice of integration limits I do not think this formulation differs from
        :cite:p:`HWZ58{Equation 8}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
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
# the base class does not have to deal with unit conversions.

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
"""H2 Beattie-Bridgeman :cite:p:`HWZ58`"""
N2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.3445),
    a=volume_conversion(0.02617),
    B0=volume_conversion(0.05046),
    b=volume_conversion(-0.00691),
    c=volume_conversion(4.2e4),
)
"""N2 Beattie-Bridgeman :cite:p:`HWZ58`"""
O2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.4911),
    a=volume_conversion(0.02562),
    B0=volume_conversion(0.04624),
    b=volume_conversion(0.004208),
    c=volume_conversion(4.8e4),
)
"""O2 Beattie-Bridgeman :cite:p:`HWZ58`"""
CO2_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(5.0065),
    a=volume_conversion(0.07132),
    B0=volume_conversion(0.10476),
    b=volume_conversion(0.07235),
    c=volume_conversion(66e4),
)
"""CO2 Beattie-Bridgeman :cite:p:`HWZ58`"""
NH3_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.3930),
    a=volume_conversion(0.17031),
    B0=volume_conversion(0.03415),
    b=volume_conversion(0.19112),
    c=volume_conversion(476.87e4),
)
"""NH3 Beattie-Bridgeman :cite:p:`HWZ58`"""
CH4_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.2769),
    a=volume_conversion(0.01855),
    B0=volume_conversion(0.05587),
    b=volume_conversion(-0.01587),
    c=volume_conversion(12.83e4),
)
"""CH4 Beattie-Bridgeman :cite:p:`HWZ58`"""
He_Beattie_holley: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.0216),
    a=volume_conversion(0.05984),
    B0=volume_conversion(0.01400),
    b=0,
    c=volume_conversion(0.004e4),
)
"""He Beattie-Bridgeman :cite:p:`HWZ58`"""


def get_holley_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred Holley EOS models for each species.

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
