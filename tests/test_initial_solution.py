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

from atmodeller import __version__, debug_logger
from atmodeller.core import GasSpecies, Species
from atmodeller.initial_solution import InitialSolutionConstant, InitialSolutionDict
from atmodeller.interfaces import InitialSolutionProtocol

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_constant():
    """Tests a constant initial solution"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    species: Species = Species([H2O_g, H2_g])

    initial_solution: InitialSolutionProtocol = InitialSolutionConstant(10, species=species)

    assert np.array_equal(initial_solution.value, np.array([10, 10]))


def test_dictionary():
    """Tests a dictionary initial solution

    Tests that dictionary fill values are preferred to `fill_value`, as well as that the initial
    solution is aligned with the species order.
    """

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO2")
    species: Species = Species([H2O_g, H2_g, CO_g])

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        {CO_g: 10, H2O_g: 5}, species=species, fill_value=2
    )

    assert np.array_equal(initial_solution.value, np.array([5, 2, 10]))


# TODO: Work in progress, but will need to generate some data to test the regressor.
# def test_regressor():
#     """Tests a regressor initial condition"""

#     H2O_g: GasSpecies = GasSpecies("H2O")
#     H2_g: GasSpecies = GasSpecies("H2")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     species: Species = Species([H2O_g, H2_g, CO_g, CO2_g])

#     output: Output = synthetic_output()

#     initial_solution: InitialSolutionProtocol = InitialSolutionRegressor(
#         output, species=species, species_fill={H2_g: 6}, fill_value=2
#     )

#     expected_value: dict[str, list[float]] = {
#         "H2O_g": [1, 2, 3, 4],
#         "H2_g": [6, 6, 6, 6],
#         "CO_g": [5, 6, 7, 8],
#         "CO2_g": [2, 2, 2, 2],
#     }
#     expected_value_df = pd.DataFrame(expected_value)


# def synthetic_output() -> Output:
#     """Generates synthetic output for testing the initial solution

#     Returns:
#         Synthetic output
#     """

#     output: Output = Output()

#     output["solution"] = [
#         {"H2O_g": 1, "CO_g": 5},
#         {"H2O_g": 2, "CO_g": 6},
#         {"H2O_g": 3, "CO_g": 7},
#         {"H2O_g": 4, "CO_g": 8},
#     ]

#     return output
