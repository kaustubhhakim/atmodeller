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
"""Real gas EOSs from :cite:t:`Lide2005`"""

import equinox as eqx
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, ArrayLike

from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos import (
    ABSOLUTE_TOLERANCE,
    PRESSURE_REFERENCE,
    RELATIVE_TOLERANCE,
    THROW,
    VOLUME_EPSILON,
)
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.mytypes import OptxSolver
from atmodeller.utilities import ExperimentalCalibration

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier


class VanderWaals(RealGas):
    r"""Van der Waals EOS

    Args:
        a: a constant in :math:`\mathrm{m}^6 \mathrm{bar} \mathrm{mol}^{-2}`
        b: b constant in :math:`\mathrm{m}^3 \mathrm{mol}^{-1}`
    """

    a: float
    r"""a constant in :math:`\mathrm{m}^6 \mathrm{bar} \mathrm{mol}^{-2}`"""
    b: float
    r"""b constant in :math:`\mathrm{m}^3 \mathrm{mol}^{-1}`"""

    @eqx.filter_jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume

        Args:
            volume: Volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

        coeff0: ArrayLike = -self.a * self.b / pressure
        coeff1: ArrayLike = -self.a / pressure
        coeff2: ArrayLike = -self.b - GAS_CONSTANT_BAR * temperature / pressure
        coeff3: ArrayLike = 1

        residual: Array = (
            coeff3 * jnp.power(volume, 3)
            + coeff2 * jnp.power(volume, 2)
            + coeff1 * volume
            + coeff0
        )

        return residual

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Computes the volume numerically.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3 \mathrm{mol}^{-1}`
        """
        ideal_volume: ArrayLike = GAS_CONSTANT_BAR * temperature / pressure
        # If the ideal volume is around the b constant value then the denominator becomes zero, so
        # shift the volume and add a small epsilon to avoid this.
        safe_volume: ArrayLike = ideal_volume + self.b + VOLUME_EPSILON
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
        r"""Volume integral

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{m}^3\ \mathrm{bar}\ \mathrm{mol}^{-1}`
        """
        vol: ArrayLike = self.volume(temperature, pressure)
        vol0: ArrayLike = self.volume(temperature, PRESSURE_REFERENCE)
        volume_integral: Array = (
            self.b * (vol0 - vol) / ((vol - self.b) * (vol0 - self.b))  # type: ignore
            - jnp.log((vol - self.b) / (vol0 - self.b))
        ) * GAS_CONSTANT_BAR * temperature - 2 * self.a * (1 / vol - 1 / vol0)  # type:ignore

        return volume_integral


experimental_calibration: ExperimentalCalibration = ExperimentalCalibration(pressure_min=1)

# van der Waals cefficients from David R. Lide, ed., CRC Handbook of Chemistry and Physics,
# Internet Version 2005, <http://www.hbcpnetbase.com>, CRC Press, Boca Raton, FL, 2005
H2_lide: RealGas = VanderWaals(2.452e-7, 2.65e-5)
"""H2 van der Waals :cite:p:`Lide2005`"""
H2_lide_bounded: RealGas = CombinedRealGas.create([H2_lide], [experimental_calibration])
"""H2 bounded to data range"""
He_lide: RealGas = VanderWaals(3.46e-8, 2.38e-5)
"""He van der Waals :cite:p:`Lide2005`"""
He_lide_bounded: RealGas = CombinedRealGas.create([He_lide], [experimental_calibration])
"""He bounded to data range"""
N2_lide: RealGas = VanderWaals(1.37e-6, 3.87e-5)
"""N2 van der Waals :cite:p:`Lide2005`"""
N2_lide_bounded: RealGas = CombinedRealGas.create([N2_lide], [experimental_calibration])
"""N2 bounded to data range"""
H4Si_lide: RealGas = VanderWaals(4.38e-6, 5.79e-5)
"""SiH4 van der Waals :cite:p:`Lide2005`"""
H4Si_lide_bounded: RealGas = CombinedRealGas.create([H4Si_lide], [experimental_calibration])
"""SiH4 bounded to data range"""
H2O_lide: RealGas = VanderWaals(5.537e-6, 3.05e-5)
"""H2O van der Waals :cite:p:`Lide2005`"""
H2O_lide_bounded: RealGas = CombinedRealGas.create([H2O_lide], [experimental_calibration])
"""H2O bounded to data range"""
CH4_lide: RealGas = VanderWaals(2.303e-6, 4.31e-5)
"""CH4 van der Waals :cite:p:`Lide2005`"""
CH4_lide_bounded: RealGas = CombinedRealGas.create([CH4_lide], [experimental_calibration])
"""CH4 bounded to data range"""
H3N_lide: RealGas = VanderWaals(4.225e-6, 3.71e-5)
"""NH3 van der Waals :cite:p:`Lide2005`"""
H3N_lide_bounded: RealGas = CombinedRealGas.create([H3N_lide], [experimental_calibration])
"""NH3 bounded to data range"""
CHN_lide: RealGas = VanderWaals(1.29e-5, 8.81e-5)
"""HCN van der Waals :cite:p:`Lide2005`"""
CHN_lide_bounded: RealGas = CombinedRealGas.create([CHN_lide], [experimental_calibration])
"""HCN bounded to data range"""
H4Si_isham: RealGas = VanderWaals(2.478e-6, 3.275e-5)
"""SiH4 van der Waals (Isham) :cite:p:`Lide2005`"""
H4Si_isham_bounded: RealGas = CombinedRealGas.create([H4Si_isham], [experimental_calibration])
"""SiH4 (Isham) bounded to data range"""
OSi_isham: RealGas = VanderWaals(8.698e-6, 8.582e-6)
"""OSi van der Waals (Isham) :cite:p:`Lide2005`"""
OSi_isham_bounded: RealGas = CombinedRealGas.create([OSi_isham], [experimental_calibration])
"""OSi (Isham) bounded to data range"""


def get_vanderwaals_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of van der Waals EOS models.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["H2_vdw_lide05"] = H2_lide_bounded
    eos_models["He_vdw_lide05"] = He_lide_bounded
    eos_models["N2_vdw_lide05"] = N2_lide_bounded
    eos_models["H4Si_vdw_lide05"] = H4Si_lide_bounded
    eos_models["H2O_vdw_lide05"] = H2O_lide_bounded
    eos_models["CH4_vdw_lide05"] = CH4_lide_bounded
    eos_models["H3N_vdw_lide05"] = H3N_lide_bounded
    eos_models["CHN_vdw_lide05"] = CHN_lide_bounded

    return eos_models
