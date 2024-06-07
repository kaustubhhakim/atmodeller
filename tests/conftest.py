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

# Want to use chemistry symbols so pylint: disable=invalid-name

import logging

import numpy as np
import numpy.typing as npt
import pytest

from atmodeller.constraints import (
    ElementMassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies, Species
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet

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

        if len((system.solution.solution_dict())) != len(target):
            return np.bool_(False)

        target_values: npt.NDArray = np.array(list(dict(sorted(target.items())).values()))
        solution_values: npt.NDArray = np.array(
            list(dict(sorted(system.solution.solution_dict().items())).values())
        )
        if log:
            target_values = np.log10(target_values)
            solution_values = np.log10(solution_values)

        isclose: npt.NDArray = np.isclose(target_values, solution_values, rtol=rtol, atol=atol)

        logger.debug("isclose = %s", isclose)

        return isclose.all()


@pytest.fixture
def helper():
    return Helper()


@pytest.fixture
def graphite_water_condensed() -> InteriorAtmosphereSystem:
    """C and water in equilibrium at 430 K and 10 bar

    This system is convenient for testing several parts of the code, so it is a fixture so it can
    be accessed throughout the test suite.
    """

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    # TODO: Using the 10 bar thermo data pushes the atmodeller result away from FactSage. Why?
    H2O_l = LiquidSpecies("H2O")  # , thermodata_filename="H-066", thermodata_name="Water, 10 Bar")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    C_cr = SolidSpecies("C")

    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])

    planet = Planet()
    planet.surface_temperature = 430
    system = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20

    constraints = SystemConstraints(
        [
            TotalPressureConstraint(10),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system.solve(constraints)

    return system
