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
"""EOS package"""

import importlib.resources
from importlib.abc import Traversable

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory, which is the same as the package directory"""
# ABSOLUTE_TOLERANCE is less than RELATIVE_TOLERANCE because typical volumes are around 1e-6
ABSOLUTE_TOLERANCE: float = 1.0e-12
r"""Absolute tolerance when solving for the volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`"""
RELATIVE_TOLERANCE: float = 1.0e-6
r"""Relative tolerance when solving for the volume in :math:`\mathrm{m}^3\ \mathrm{mol}^{-1}`"""
THROW: bool = False
"""Whether to throw errors. Change to True for debugging purposes."""
PRESSURE_REFERENCE: float = 1.0
"""Standard state pressure in bar"""
VOLUME_EPSILON: float = 1.0e-12
r"""Small volume offset in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`"""

# Expose the public API
from atmodeller.eos._aggregators import CombinedRealGas  # noqa: E402, F401
from atmodeller.eos.core import IdealGas, RealGas  # noqa: E402, F401
from atmodeller.eos.library import get_eos_models  # noqa: E402, F401
