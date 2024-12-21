#
# Copyright 2024 Dan J. Bower, Fabian L. Seidler
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
"""Core classes and functions for solubility laws.

Units for temperature and pressure are K and bar, respectively.
"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer

iron_wustite_buffer: RedoxBufferProtocol = IronWustiteBuffer()


def fO2_temperature_correction(
    fO2: ArrayLike,
    *,
    temperature: ArrayLike,
    pressure: ArrayLike,
    reference_temperature: ArrayLike,
) -> Array:
    """Applies a temperature correction to fO2.

    Some experimentally derived solubility laws operate on absolute fO2, which depends on
    temperature and pressure. A temperature correction has to be applied to maintain the same
    fO2 shift at arbitrary temperature.

    Args:
        fO2: Absolute oxygen fugacity at `temperature`, in bar
        temperature: Temperature in K
        pressure: Absolute pressure in bar
        reference_temperature: Reference temperature, which is usually the temperature at which the
            experiment was performed.

    Returns:
        Adjusted fO2
    """
    logiw_fugacity_at_current_temp: ArrayLike = iron_wustite_buffer.log10_fugacity(
        temperature, pressure
    )
    fo2_shift: Array = jnp.log10(fO2) - logiw_fugacity_at_current_temp

    logiw_fugacity_at_reference_temp: ArrayLike = iron_wustite_buffer.log10_fugacity(
        reference_temperature, pressure
    )
    adjusted_fo2: Array = jnp.power(10, logiw_fugacity_at_reference_temp + fo2_shift)

    return adjusted_fo2
