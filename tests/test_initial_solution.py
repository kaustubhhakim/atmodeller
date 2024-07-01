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

from atmodeller import __version__, debug_logger
from atmodeller.constraints import PressureConstraint, SystemConstraints
from atmodeller.core import GasSpecies, Species
from atmodeller.initial_solution import InitialSolutionDict
from atmodeller.interfaces import InitialSolutionProtocol

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# def test_constant():
#     """Tests a constant initial solution"""

#     H2O_g: GasSpecies = GasSpecies("H2O")
#     H2_g: GasSpecies = GasSpecies("H2")
#     species: Species = Species([H2O_g, H2_g])

#     initial_solution: InitialSolutionProtocol = InitialSolutionConstant(10, species=species)

#     assert np.array_equal(initial_solution.value, np.array([10, 10]))


def test_dictionary():
    """Tests a dictionary initial solution

    Tests that dictionary fill values are preferred to `fill_value`, as well as that the initial
    solution is aligned with the species order.
    """

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    species: Species = Species([H2O_g, H2_g, CO_g])

    constraints: SystemConstraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        {CO_g: 100, H2_g: 1000}, species=species, fill_log10_pressure=2
    )

    # assert np.array_equal(initial_solution.value, np.array([5, 2, 10]))

    print(initial_solution.get_log10_value(constraints, temperature=1000, pressure=1))
