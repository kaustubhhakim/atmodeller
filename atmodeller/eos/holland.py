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
"""Real gas EOS from :cite:t:`HP91,HP98,HP11`

You will usually want to use the CORK models, since these are the most complete. The MRK models are
nevertheless useful for comparison and understanding the influence of the virial compensation term
that is encapsulated within the CORK model.

Examples:
    Evaluate the fugacity coefficient for the H2O CORK model from :cite:t:`HP98` at 2000 K and
    1000 bar::

        from atmodeller.eos.holland import H2O_CORK_HP98
        model = H2O_CORK_HP98
        fugacity_coefficient = model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)

    Get the preferred EOS models for various species from the Holland and Powell models::

        from atmodeller.eos.holland import get_holland_eos_models
        models = get_holland_eos_models()
        # List the available species
        models.keys()
        # Get the EOS model for CO
        co_model = models['CO']
        # Determine the fugacity coefficient at 2000 K and 1000 bar
        fugacity_coefficient = co_model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)
"""

import logging
from typing import Callable

import jax.numpy as jnp
from jax import Array
from scipy.constants import kilo

from atmodeller.eos.holland_jax import CO2MrkHolland91, H2OMrkHolland91
from atmodeller.eos.interfaces import (
    CORK,
    ExperimentalCalibration,
    RealGas,
)

logger: logging.Logger = logging.getLogger(__name__)

# Common calibration parameters from Holland and Powell (1991)
CALIBRATION_HP91: ExperimentalCalibration = ExperimentalCalibration(400, 1900, 0, 50e3)

CO2_MRK_HP91 = CO2MrkHolland91
"""CO2 MRK :cite:p:`HP91`"""
CO2_MRK_HP98 = CO2MrkHolland91
"""CO2 MRK :cite:p:`HP98`

This is the same as the CO2 MRK model in :cite:t:`HP91`.
"""

H2O_MRK_HP91 = H2OMrkHolland91
"""H2O MRK :cite:p:`HP91`"""
H2O_MRK_HP98 = H2OMrkHolland91
"""H2O MRK :cite:p:`HP98`

This is the same as the H2O MRK model in :cite:t:`HP91`.
"""

# For the Full CORK models, the virial coefficients in the Holland and Powell papers need
# converting to SI units and pressure in bar as follows, where k = kilo = 1000:
#   a_virial = a_virial (Holland and Powell) * 10**(-5) / k
#   b_virial = b_virial (Holland and Powell) * 10**(-5) / k**(1/2)
#   c_virial = c_virial (Holland and Powell) * 10**(-5) / k**(1/4)

_a_conversion: Callable[[tuple[float, ...]], Array] = lambda x: jnp.array(  # noqa: E731
    [y * 1e-5 / kilo for y in x]
)
_b_conversion: Callable[[tuple[float, ...]], Array] = lambda x: jnp.array(  # noqa: E731
    [y * 1e-5 / kilo**0.5 for y in x]
)
_c_conversion: Callable[[tuple[float, ...]], Array] = lambda x: jnp.array(  # noqa: E731
    [y * 1e-5 / kilo**0.25 for y in x]
)

CO2_CORK_HP91: RealGas = CORK(
    P0=5000,
    mrk=CO2_MRK_HP91,  # type:ignore
    a_virial=_a_conversion((1.33790e-2, -1.01740e-5)),
    b_virial=_b_conversion((-2.26924e-1, 7.73793e-5)),
    calibration=CALIBRATION_HP91,
)
"""CO2 CORK :cite:p:`HP91`"""

CO2_CORK_HP98: RealGas = CORK(
    P0=5000,
    mrk=CO2_MRK_HP98,  # type:ignore
    a_virial=_a_conversion((5.40776e-3, -1.59046e-6)),
    b_virial=_b_conversion((-1.78198e-1, 2.45317e-5)),
    calibration=ExperimentalCalibration(400, 1900, 0, 120e3),
)
"""CO2 CORK :cite:p:`HP98`"""

H2O_CORK_HP91: RealGas = CORK(
    P0=2000,
    mrk=H2O_MRK_HP91,  # type:ignore
    a_virial=_a_conversion((-3.2297554e-3, 2.2215221e-6)),
    b_virial=_b_conversion((-3.025650e-2, -5.343144e-6)),
    calibration=ExperimentalCalibration(400, 1700, 0, 50e3),
)
"""H2O CORK :cite:p:`HP91`"""

H2O_CORK_HP98: RealGas = CORK(
    P0=2000,
    mrk=H2O_MRK_HP98,  # type:ignore
    a_virial=_a_conversion((1.9853e-3, 0)),
    b_virial=_b_conversion((-8.9090e-2, 0)),
    c_virial=_c_conversion((8.0331e-2, 0)),
    calibration=ExperimentalCalibration(400, 1700, 0, 120e3),
)
"""H2O CORK :cite:p:`HP98`"""
