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
"""Utilities for tests"""

import logging

# NOTE: Don't import anything from Atmodeller otherwise beartype can't wrap the tests for runtime
# type checking
import numpy as np
import numpy.typing as npt
import pytest
from jaxtyping import ArrayLike

logger: logging.Logger = logging.getLogger("atmodeller.tests")


class Helper:
    """Helper for integral tests"""

    @classmethod
    def isclose(
        cls,
        solution: dict[str, ArrayLike],
        target: dict[str, ArrayLike],
        *,
        log: bool = False,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-6,
    ) -> np.bool_:
        """Determines if the solution is close to a target within tolerance.

        Only the values in `solution` and `target` that have matching keys will be compared.

        Args:
            solution: Dictionary of the solution values
            target: Dictionary of the target values
            log: Compare closeness in log-space. Defaults to False.
            rtol: Relative tolerance. Defaults to 1.0e-6.
            atol: Absolute tolerance. Defaults to 1.0e-6.

        Returns:
            True if the solution is close to the target, otherwise False
        """
        # Find the intersection of keys
        intersection_keys: set[str] = solution.keys() & target.keys()
        logger.info("Keys for comparison = %s", intersection_keys)

        # Create new dictionaries with the intersecting keys
        solution_compare: dict[str, ArrayLike] = {key: solution[key] for key in intersection_keys}
        target_compare: dict[str, ArrayLike] = {key: target[key] for key in intersection_keys}

        target_values: npt.NDArray[np.float64] = np.array(
            list(dict(sorted(target_compare.items())).values())
        )
        logger.debug("target_values = %s", target_values)
        solution_values: npt.NDArray[np.float64] = np.array(
            list(dict(sorted(solution_compare.items())).values())
        )
        logger.debug("solution_values = %s", solution_values)
        if log:
            target_values = np.log10(target_values)
            solution_values = np.log10(solution_values)

        isclose: npt.NDArray[np.bool_] = np.isclose(
            target_values, solution_values, rtol=rtol, atol=atol
        )
        logger.debug("isclose = %s", isclose)

        return isclose.all()


@pytest.fixture
def helper():
    return Helper()
