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
"""Tests for the initial solution"""

# Convenient to use naming convention so pylint: disable=C0103

import logging

import numpy as np
import numpy.typing as npt

from atmodeller import __version__, debug_logger
from atmodeller.constraints import PressureConstraint, SystemConstraints
from atmodeller.core import GasSpecies, Species
from atmodeller.initial_solution import InitialSolutionDict
from atmodeller.interfaces import InitialSolutionProtocol

logger: logging.Logger = debug_logger()

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

dummy_variable: float = 1
"""Dummy variable used for temperature and pressure arguments when they are not used internally"""


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_no_args_no_constraints_dict():
    """Tests a dict with no arguments and no constraints"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    species: Species = Species([H2O_g, H2_g, CO_g])

    constraints: SystemConstraints = SystemConstraints([])

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(species=species)
    result: npt.NDArray[np.float_] = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )

    assert np.all(result == 1)


def test_no_args_with_constraints_dict():
    """Tests a dict with no arguments, but with constraints and a pressure fill value."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    species: Species = Species([H2O_g, H2_g, CO_g])

    constraints: SystemConstraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        species=species, fill_log10_pressure=2
    )
    result: npt.NDArray[np.float_] = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.array([0.6989700043360189, 2, 2])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_with_args_with_constraints_dict():
    """Tests a dict with arguments, constraints, and a pressure fill value."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    species: Species = Species([H2O_g, H2_g, CO_g, CO2_g])

    constraints: SystemConstraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        {CO_g: 100, H2_g: 1000}, species=species, fill_log10_pressure=4
    )
    result: npt.NDArray[np.float_] = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.array([0.6989700043360189, 3, 2, 4])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)
