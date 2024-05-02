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

import numpy as np
import numpy.typing as npt
import pytest

from atmodeller.interior_atmosphere import InteriorAtmosphereSystem

logger: logging.Logger = logging.getLogger(__name__)


class Helper:
    """Helper class for tests"""

    @staticmethod
    def isclose(
        system: InteriorAtmosphereSystem,
        target: dict[str, float],
        *,
        log: bool = False,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-6,
    ) -> np.bool_:

        if len((system.solution_dict())) != len(target):
            return np.bool_(False)

        target_values: npt.NDArray = np.array(list(dict(sorted(target.items())).values()))
        solution_values: npt.NDArray = np.array(
            list(dict(sorted(system.solution_dict().items())).values())
        )
        if log:
            target_values = np.log10(target_values)
            solution_values = np.log10(solution_values)

        isclose: npt.NDArray = np.isclose(target_values, solution_values, rtol=rtol, atol=atol)

        logger.debug("isclose = %s", isclose)

        return isclose.all()


@pytest.fixture(scope="module")
def helper():
    return Helper()
