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
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from atmodeller.constraints import (
    ActivityConstraint,
    BufferedFugacityConstraint,
    ElementMassConstraint,
    PressureConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies, Planet, SolidSpecies, Species
from atmodeller.reaction_network import InteriorAtmosphereSystem, Solver
from atmodeller.solution import Solution
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer

logger: logging.Logger = logging.getLogger(__name__)


class Helper:
    """Helper class for tests"""

    @classmethod
    def isclose(
        cls,
        solution: Solution,
        target: dict[str, float],
        *,
        log: bool = False,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-6,
    ) -> np.bool_:
        """Determines if the solution is close to a target solution within a tolerance.

        Args:
            solution: Solution
            target: Dictionary of the target values, which should adhere to the format of
                :meth:`solution.Solution.output_solution()`
            rtol: Relative tolerance. Defaults to 1.0e-6.
            atol: Absolute tolerance. Defaults to 1.0e-6.

        Returns:
            True if the solution is close to the target, otherwise False
        """
        output: dict[str, float] = solution.output_solution()

        if len(output) != len(target):
            return np.bool_(False)

        target_values: npt.NDArray[np.float_] = np.array(
            list(dict(sorted(target.items())).values())
        )
        solution_values: npt.NDArray[np.float_] = np.array(
            list(dict(sorted(output.items())).values())
        )
        if log:
            target_values = np.log10(target_values)
            solution_values = np.log10(solution_values)

        isclose: npt.NDArray[np.bool_] = np.isclose(
            target_values, solution_values, rtol=rtol, atol=atol
        )

        logger.debug("isclose = %s", isclose)

        return isclose.all()

    @classmethod
    def isclose_tolerance(
        cls, solution: Solution, target: dict[str, float], log: bool = False, message: str = ""
    ) -> float | None:
        """Writes a log message with the tightest tolerance satisfied.

        Args:
            target: Dictionary of the target values, which should adhere to the format of
                :meth:`solution.Solution.output_solution()`
            message: Message prefix to write to the logger when a tolerance is satisfied. Defaults
                to an empty string.

        Returns:
            The tightest tolerance satisfied
        """
        for log_tolerance in (-6, -5, -4, -3, -2, -1):
            tol: float = 10**log_tolerance
            if cls.isclose(solution, target, log=log, rtol=tol, atol=tol):
                logger.info("%s (tol = %f)".lstrip(), message, tol)
                return tol

        logger.info("%s (no tolerance < 0.1 satisfied)".lstrip(), message)


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
    H2O_l = LiquidSpecies("H2O")  # , thermodata_name="Water, 10 Bar")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    C_cr = SolidSpecies("C")

    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])

    cool_planet: Planet = Planet(surface_temperature=411.75)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    # Specify O, because otherwise a total pressure can give rise to different solutions (with
    # different total O), making it more difficult to compare with a known comparison case.
    o_kg: float = 2.48298883581636e21

    constraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            ElementMassConstraint("O", o_kg),
            ActivityConstraint(H2O_l, 1),
            ActivityConstraint(C_cr, 1),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=cool_planet)
    _, _, solution = system.solve(solver="scipy", constraints=constraints)

    return solution


@pytest.fixture
def generate_regressor_data() -> tuple[InteriorAtmosphereSystem, Path]:
    """Generates data for testing the initial solution regressor"""

    # H2O pressures in bar
    H2O_pressures = [1, 4, 7, 10]
    # fO2 shifts relative to the IW buffer
    delta_IWs = range(-4, 5)

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    species = Species([H2O_g, H2_g, O2_g])

    planet = Planet()
    system = InteriorAtmosphereSystem(species=species, planet=planet)

    # Generate some data
    for H2O_pressure in H2O_pressures:
        for delta_IW in delta_IWs:
            constraints = SystemConstraints(
                [
                    BufferedFugacityConstraint(O2_g, IronWustiteBuffer(delta_IW)),
                    PressureConstraint(H2O_g, H2O_pressure),
                ]
            )
            system.solve(constraints)

    filename = Path("ic_regressor_test_data")
    system.output(filename, to_excel=True, to_pickle=True)

    return system, filename
