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

import logging
import sys

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

if sys.version_info < (3, 12):
    pass
else:
    pass

if sys.version_info < (3, 11):
    pass
else:
    pass

logger: logging.Logger = logging.getLogger(__name__)


def fo2_temp_correc(
    fO2: ArrayLike,
    *,
    pressure: ArrayLike,
    temperature: ArrayLike,
    reference_temperature: ArrayLike,
) -> ArrayLike:
    """Apply a temperature correction to fO2.

    Some experimentally derived solubility laws operate on absolute fO2, which depends on
    temperature and pressure. A temperature correction has to be applied to maintain the same
    fO2_shift at arbitrary temperature.

    Args:
        fO2: Absolute oxygen fugacity at `temperature`, in bar.
        pressure: Absolute pressure in bar.
        temperature: Temperature, in K.
        reference_temperature: Reference temperature, usually the temperature at which the
                               experiment was performed.

    """
    logiw_fugacity_at_current_temp: ArrayLike = (
        -28776.8 / temperature
        + 14.057
        + 0.055 * (pressure - 1) / temperature
        - 0.8853 * jnp.log(temperature)
    )
    fo2_shift: Array = jnp.log10(fO2) - logiw_fugacity_at_current_temp

    logiw_fugacity_at_reference_temp: ArrayLike = (
        -28776.8 / reference_temperature
        + 14.057
        + 0.055 * (pressure - 1) / reference_temperature
        - 0.8853 * jnp.log(reference_temperature)
    )
    adjusted_fo2: Array = jnp.power(10, logiw_fugacity_at_reference_temp + fo2_shift)

    return adjusted_fo2
