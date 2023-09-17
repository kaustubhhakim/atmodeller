"""Fugacity coefficients and non-ideal effects.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.constants import kilo

from atmodeller.eos.eos_interfaces import (
    MRKABC,
    CORKFullABC,
    FugacityModelABC,
    MRKExplicitABC,
    MRKImplicitABC,
    ShiSaxenaABC,
    ShiSaxenaHighPressure,
    ShiSaxenaLowPressure,
    VirialCompensation,
)


@dataclass(kw_only=True)
class ShiSaxenaHighPressureH2(ShiSaxenaHighPressure):
    # scaling = kilo
    Tc: float = field(init=False, default=33.25)
    Pc: float = field(init=False, default=12.9696)
    # Coefficients from Saxena and Fei
    # a_coefficients: tuple[float, ...] = field(
    #     init=False, default=(1.6688, 0, -2.0759, 0, -9.6173, 0, 0, -0.1694)
    # )
    # b_coefficients: tuple[float, ...] = field(
    #     init=False, default=(-2.0410e-3, 0, 7.9230e-2, 0, 5.4295e-2, 0, 0, 4.0887e-4)
    # )
    # c_coefficients: tuple[float, ...] = field(
    #     init=False, default=(-2.1693e-7, 0, 1.7406e-6, 0, -2.1885e-4, 0, 0, 5.0897e-5)
    # )
    # d_coefficients: tuple[float, ...] = field(
    #     init=False, default=(-7.1635e-12, 0, 1.6197e-10, 0, -4.8181e-9, 0, 0, 0)
    # )
    # Coefficients from Shi and Saxena
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
class ShiSaxenaCO2(ShiSaxenaHighPressure):
    # scaling = kilo
    Tc: float = field(init=False, default=304.15)
    Pc: float = field(init=False, default=73.8659)
    # Coefficients from Shi and Saxena
    a_coefficients: tuple[float, ...] = field(
        init=False, default=(2.0614, 0, 0, 0, -2.235, 0, 0, -3.941e-1)
    )
    b_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, 5.513e-2, 0, 3.934e-2, 0, 0, 0)
    )
    c_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, -1.894e-6, 0, -1.109e-5, 0, -2.189e-5, 0)
    )
    d_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, 5.053e-11, 0, 0, -6.303e-21, 0, 0)
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
