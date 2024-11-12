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
"""Thermodata package level variables"""

# Below may be reinstated if we acquire thermochemical data in a table format.
# import importlib.resources
# from importlib.abc import Traversable

# DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}")
# """Data directory, which is the same as the package directory"""

# Expose public API
from atmodeller.thermodata.core import (
    CondensateActivity,  # noqa: F401
    CriticalData,  # noqa: F401
    SpeciesData,  # noqa: F401
    ThermoCoefficients,  # noqa: F401
)
from atmodeller.thermodata.library import (
    get_thermodata,  # noqa: F401
    select_critical_data,  # noqa: F401
    select_thermodata,  # noqa: F401
)
