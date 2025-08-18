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
"""Common type aliases

This module centralizes type definitions for NumPy arrays, scalar values, and Optimistix solvers.
Having a single place for these aliases improves readability and consistency across the codebase,
whilst also simplifying type checking and documentation.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import optimistix as optx

OptxSolver: TypeAlias = (
    optx.AbstractRootFinder | optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser
)

NpArray: TypeAlias = npt.NDArray
NpBool: TypeAlias = npt.NDArray[np.bool_]
NpFloat: TypeAlias = npt.NDArray[np.float64]
NpInt: TypeAlias = npt.NDArray[np.int_]
Scalar: TypeAlias = int | float
