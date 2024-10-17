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

# Use symbols from the paper for consistency so pylint: disable=invalid-name

import logging
import sys
from typing import Any, Callable

import jax.numpy as jnp
import optimistix as optx
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller import ATMOSPHERE, GAS_CONSTANT_BAR
from atmodeller.eos.core import RealGas, RealGasProtocol
from atmodeller.utilities import ExperimentalCalibrationNew, unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


@register_pytree_node_class
class BeattieBridgeman(RealGas):
    r"""Beattie-Bridgeman equation :cite:p:`HWZ58{Equation 1}`.

    .. math::

        PV^2 = RT\left(1-\frac{c}{VT^3}\right)\left(V+B_0-\frac{bB_0}{V}\right)
        - A_0\left(1-\frac{a}{V}\right)

    Args:
        A0: A0 empirical constant determined experimentally
        a: a empirical constant determined experimentally
        B0: B0 empirical constant determined experimentally
        b: b empirical constant determined experimentally
        c: c empirical constant determined experimentally
        calibration: Experimental calibration. Defaults to empty.

    Attributes:
        A0: A0 empirical constant determined experimentally
        a: a empirical constant determined experimentally
        B0: B0 empirical constant determined experimentally
        b: b empirical constant determined experimentally
        c: c empirical constant determined experimentally
        calibration: Experimental calibration. Defaults to empty.
    """

    def __init__(
        self,
        A0: float,
        a: float,
        B0: float,
        b: float,
        c: float,
        calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(),
    ):
        self.A0: float = A0
        self.a: float = a
        self.B0: float = B0
        self.b: float = b
        self.c: float = c
        self._calibration: ExperimentalCalibrationNew = calibration

    @jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume :cite:p:`HWZ58{Equation 2}`

        .. math::

            PV^4 - RTV^3 - \left(RTB_0 - \frac{Rc}{T^2}-A_0\right)V^2
            +\left(RTbB_0+\frac{RcB_0}{T^2}-aA_0\right)V - \frac{RcbB_0}{T^2}=0

        Args:
            volume: Volume
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

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
    @jit
    def log_fugacity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> Array:
        """Log fugacity :cite:p:`HWZ58{Equation 11}`.

        Args:
            temperature: Temperature
            pressure: Pressure

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
                + self.c * self.B0 / temperature**3
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
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Volume

        :cite:t:`HWZ58` doesn't say which root to take, but one real root is very small and the
        maximum real root gives a volume that agrees with the tabulated compressibility parameter
        for all species.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        # Start with a large initial guess, say some factor of the ideal gas volume, to guide the
        # Newton solver to the largest root, which agrees with the tabulated data in the paper.
        # The choice of 10 below is somewhat arbitrary, but based on the calibration data for the
        # Holley model should be comfortably larger than the actual volume.
        scaling_factor: float = 10
        initial_volume = scaling_factor * self.ideal_volume(temperature, pressure)

        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver = optx.Newton(rtol=1.0e-8, atol=1.0e-8)
        sol = optx.root_find(
            self._objective_function,
            solver,
            initial_volume,
            args=kwargs,
        )
        volume: ArrayLike = sol.value

        return volume

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "A0": self.A0,
            "a": self.a,
            "B0": self.B0,
            "b": self.b,
            "c": self.c,
            "calibration": self.calibration,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


# Coefficients from Table I, which must be converted to the correct units scheme (SI and pressure
# in bar). Using the original table values below allows easy visual comparison and ensures that
# the class does not have to deal with unit conversions.

volume_conversion: Callable = lambda x: x * unit_conversion.litre_to_m3
# Converts PV**2 coefficient to be in terms of m^3 and bar
A0_conversion: Callable = lambda x: x * ATMOSPHERE * unit_conversion.litre_to_m3**2

H2_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(0.1975),
    a=volume_conversion(-0.00506),
    B0=volume_conversion(0.02096),
    b=volume_conversion(-0.04359),
    c=volume_conversion(0.0504e4),
    calibration=ExperimentalCalibrationNew(100, 1000, 0.1, 1000),
)
"""H2 Beattie-Bridgeman :cite:p:`HWZ58`"""
N2_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(1.3445),
    a=volume_conversion(0.02617),
    B0=volume_conversion(0.05046),
    b=volume_conversion(-0.00691),
    c=volume_conversion(4.2e4),
    calibration=ExperimentalCalibrationNew(200, 1000, 0.1, 1000),
)
"""N2 Beattie-Bridgeman :cite:p:`HWZ58`"""
O2_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(1.4911),
    a=volume_conversion(0.02562),
    B0=volume_conversion(0.04624),
    b=volume_conversion(0.004208),
    c=volume_conversion(4.8e4),
    calibration=ExperimentalCalibrationNew(200, 1000, 0.1, 1000),
)
"""O2 Beattie-Bridgeman :cite:p:`HWZ58`"""
CO2_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(5.0065),
    a=volume_conversion(0.07132),
    B0=volume_conversion(0.10476),
    b=volume_conversion(0.07235),
    c=volume_conversion(66e4),
    calibration=ExperimentalCalibrationNew(400, 1000, 0.1, 1000),
)
"""CO2 Beattie-Bridgeman :cite:p:`HWZ58`"""
NH3_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(2.3930),
    a=volume_conversion(0.17031),
    B0=volume_conversion(0.03415),
    b=volume_conversion(0.19112),
    c=volume_conversion(476.87e4),
    calibration=ExperimentalCalibrationNew(500, 1000, 0.1, 500),
)
"""NH3 Beattie-Bridgeman :cite:p:`HWZ58`"""
CH4_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(2.2769),
    a=volume_conversion(0.01855),
    B0=volume_conversion(0.05587),
    b=volume_conversion(-0.01587),
    c=volume_conversion(12.83e4),
    calibration=ExperimentalCalibrationNew(200, 1000, 0.1, 1000),
)
"""CH4 Beattie-Bridgeman :cite:p:`HWZ58`"""
He_Beattie_holley: RealGasProtocol = BeattieBridgeman(
    A0=A0_conversion(0.0216),
    a=volume_conversion(0.05984),
    B0=volume_conversion(0.01400),
    b=0,
    c=volume_conversion(0.004e4),
    calibration=ExperimentalCalibrationNew(100, 1000, 0.1, 1000),
)
"""He Beattie-Bridgeman :cite:p:`HWZ58`"""


def get_holley_eos_models() -> dict[str, RealGasProtocol]:
    """Gets a dictionary of the preferred Holley EOS models for each species.

    Returns:
        Dictionary of EOS models for each species
    """
    models: dict[str, RealGasProtocol] = {}
    models["CH4"] = CH4_Beattie_holley
    models["CO2"] = CO2_Beattie_holley
    models["H2"] = H2_Beattie_holley
    models["He"] = He_Beattie_holley
    models["N2"] = N2_Beattie_holley
    models["NH3"] = NH3_Beattie_holley
    models["O2"] = O2_Beattie_holley

    return models
