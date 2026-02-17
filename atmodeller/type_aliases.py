# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Common type aliases

This module centralizes type definitions for NumPy arrays, scalar values, and Optimistix solvers.
Having a single place for these aliases improves readability and consistency across the codebase,
whilst also simplifying type checking and documentation.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from jaxmod.type_aliases import OptxSolver as OptxSolver

NpArray: TypeAlias = npt.NDArray
NpBool: TypeAlias = npt.NDArray[np.bool_]
NpFloat: TypeAlias = npt.NDArray[np.float64]
NpInt: TypeAlias = npt.NDArray[np.int_]
Scalar: TypeAlias = int | float
