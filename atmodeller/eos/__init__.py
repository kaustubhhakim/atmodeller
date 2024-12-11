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

# Expose the public API
from atmodeller.eos._aggregators import (  # noqa: E402, F401
    CombinedRealGas,
    CombinedRealGasRemoveSteps,
    RealGasBounded,
)
from atmodeller.eos.classes import (  # noqa: E402, F401
    BeattieBridgeman,
    Chabrier,
    IdealGas,
)
from atmodeller.eos.library import get_eos_models  # noqa: E402, F401
