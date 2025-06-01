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
"""Real gas EOS from :cite:`HWZ58`"""

import logging

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike

from atmodeller.constants import ATMOSPHERE, GAS_CONSTANT_BAR
from atmodeller.eos import ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE, THROW, VOLUME_EPSILON
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.mytypes import OptxSolver
from atmodeller.utilities import ExperimentalCalibration, unit_conversion

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier

logger: logging.Logger = logging.getLogger(__name__)

# Coefficients from Table I, which must be converted to the correct units scheme (SI and pressure
# in bar). Using the original values in the paper also facilitates visual comparison and checking.


def volume_conversion(x: float) -> float:
    """Volume conversion for :cite:t:`HWZ58` units"""
    return x * unit_conversion.litre_to_m3


def A0_conversion(x: float) -> float:
    """:math:`PV^2` conversion for :cite:t:`HWZ58` units"""
    return x * ATMOSPHERE * unit_conversion.litre_to_m3**2


def atm2bar(x: float) -> float:
    """Atmosphere to bar conversion"""
    return unit_conversion.atmosphere_to_bar * x


class BeattieBridgeman(RealGas):
    r"""Beattie-Bridgeman equation :cite:p:`HWZ58{Equation 1}`

    .. math::

        PV^2 = RT\left(1-\frac{c}{VT^3}\right)\left(V+B_0-\frac{bB_0}{V}\right)
        - A_0\left(1-\frac{a}{V}\right)

    Args:
        A0: A0 empirical constant
        a: a empirical constant
        B0: B0 empirical constant
        b: b empirical constant
        c: c empirical constant
    """

    A0: float
    """A0 empirical constant"""
    a: float
    """a empirical constant"""
    B0: float
    """B0 empirical constant"""
    b: float
    """b empirical constant"""
    c: float
    """c empirical constant"""

    @eqx.filter_jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume :cite:p:`HWZ58{Equation 2}`

        .. math::

            PV^4 - RTV^3 - \left(RTB_0 - \frac{Rc}{T^2}-A_0\right)V^2
            +\left(RTbB_0+\frac{RcB_0}{T^2}-aA_0\right)V - \frac{RcbB_0}{T^2}=0

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

        # jax.debug.print("volume = {out}", out=volume)
        # jax.debug.print("temperature = {out}", out=temperature)
        # jax.debug.print("pressure = {out}", out=pressure)

        coeff0: Array = 1 / jnp.square(temperature) * -GAS_CONSTANT_BAR * self.c * self.b * self.B0
        coeff1: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self.c * self.B0
            + GAS_CONSTANT_BAR * temperature * self.b * self.B0
            - self.a * self.A0
        )
        coeff2: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self.c
            - GAS_CONSTANT_BAR * temperature * self.B0
            + self.A0
        )
        coeff3: ArrayLike = -GAS_CONSTANT_BAR * temperature

        residual: Array = (
            coeff0
            + coeff1 * volume
            + coeff2 * jnp.power(volume, 2)
            + coeff3 * jnp.power(volume, 3)
            + pressure * jnp.power(volume, 4)
        )

        return residual

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity :cite:p:`HWZ58{Equation 11}`.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        log_fugacity: Array = (
            jnp.log(GAS_CONSTANT_BAR * temperature / volume)
            + (
                self.B0
                - self.c / jnp.power(temperature, 3)
                - self.A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 2
            / volume
            - (
                self.b * self.B0
                + self.c * self.B0 / jnp.power(temperature, 3)
                - self.a * self.A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 3
            / (2 * jnp.square(volume))
            + (self.c * self.b * self.B0 / jnp.power(temperature, 3))
            * 4
            / (3 * jnp.power(volume, 3))
        )

        return log_fugacity

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Solves the BB equation numerically to compute the volume.

        :cite:t:`HWZ58` doesn't say which root to take, but one real root is very small and the
        maximum real root gives a volume that agrees with the tabulated compressibility factor
        for all species.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
        """
        safe_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure + VOLUME_EPSILON
        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver: OptxSolver = optx.Newton(rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
        sol = optx.root_find(
            self._objective_function, solver, safe_volume, args=kwargs, throw=THROW
        )
        volume = sol.value
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        return self.log_fugacity(temperature, pressure) * GAS_CONSTANT_BAR * temperature


pressure_min: float = atm2bar(0.1)
"""Minimum pressure for :cite:t:`HWZ58`"""

H2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.1975),
    a=volume_conversion(-0.00506),
    B0=volume_conversion(0.02096),
    b=volume_conversion(-0.04359),
    c=volume_conversion(0.0504e4),
)
"""H2 Beattie-Bridgeman :cite:p:`HWZ58`"""
H2_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [H2_beattie_holley58],
    [ExperimentalCalibration(30, 1000, pressure_min, atm2bar(1000.0))],
)
"""H2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

N2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.3445),
    a=volume_conversion(0.02617),
    B0=volume_conversion(0.05046),
    b=volume_conversion(-0.00691),
    c=volume_conversion(4.2e4),
)
"""N2 Beattie-Bridgeman :cite:p:`HWZ58`"""
N2_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [N2_beattie_holley58], [ExperimentalCalibration(70, 1000, pressure_min, atm2bar(1000.0))]
)
"""N2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

O2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.4911),
    a=volume_conversion(0.02562),
    B0=volume_conversion(0.04624),
    b=volume_conversion(0.004208),
    c=volume_conversion(4.8e4),
)
"""O2 Beattie-Bridgeman :cite:p:`HWZ58`"""
O2_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [O2_beattie_holley58], [ExperimentalCalibration(100, 1000, pressure_min, atm2bar(1000.0))]
)
"""O2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

CO2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(5.0065),
    a=volume_conversion(0.07132),
    B0=volume_conversion(0.10476),
    b=volume_conversion(0.07235),
    c=volume_conversion(66e4),
)
"""CO2 Beattie-Bridgeman :cite:p:`HWZ58`"""
CO2_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [CO2_beattie_holley58],
    [ExperimentalCalibration(200, 1000, pressure_min, atm2bar(1000.0))],
)
"""CO2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

NH3_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.3930),
    a=volume_conversion(0.17031),
    B0=volume_conversion(0.03415),
    b=volume_conversion(0.19112),
    c=volume_conversion(476.87e4),
)
"""NH3 Beattie-Bridgeman :cite:p:`HWZ58`"""
NH3_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [NH3_beattie_holley58], [ExperimentalCalibration(300, 1000, pressure_min, atm2bar(500.0))]
)
"""NH3 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

CH4_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.2769),
    a=volume_conversion(0.01855),
    B0=volume_conversion(0.05587),
    b=volume_conversion(-0.01587),
    c=volume_conversion(12.83e4),
)
"""CH4 Beattie-Bridgeman :cite:p:`HWZ58`"""
CH4_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [CH4_beattie_holley58],
    [ExperimentalCalibration(100, 1000, pressure_min, atm2bar(1000.0))],
)
"""CH4 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

He_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.0216),
    a=volume_conversion(0.05984),
    B0=volume_conversion(0.01400),
    b=0.0,
    c=volume_conversion(0.004e4),
)
"""He Beattie-Bridgeman :cite:p:`HWZ58`"""
He_beattie_holley58_bounded: RealGas = CombinedRealGas.create(
    [He_beattie_holley58],
    [ExperimentalCalibration(10, 1000, pressure_min, atm2bar(1000.0))],
)
"""He Beattie-Bridgeman bounded :cite:p:`HWZ58`"""


def get_holley_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of Holley EOS models

    The naming convention is as follows:
        [species]_[eos model]_[citation]

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["CH4_beattie_holley58"] = CH4_beattie_holley58_bounded
    eos_models["CO2_beattie_holley58"] = CO2_beattie_holley58_bounded
    eos_models["H2_beattie_holley58"] = H2_beattie_holley58_bounded
    eos_models["He_beattie_holley58"] = He_beattie_holley58_bounded
    eos_models["N2_beattie_holley58"] = N2_beattie_holley58_bounded
    eos_models["NH3_beattie_holley58"] = NH3_beattie_holley58_bounded
    eos_models["O2_beattie_holley58"] = O2_beattie_holley58_bounded

    return eos_models
