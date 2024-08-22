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

import pytest
from jax.typing import ArrayLike
from pytest import approx

from atmodeller.eos.interfaces import RealGas

# Tolerances to compare the test results with target output.
RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""


class CheckValues:
    """Helper class with methods to check and confirm values."""

    @staticmethod
    def compressibility(
        temperature: float, pressure: float, eos: RealGas, expected: float, *, rtol=RTOL, atol=ATOL
    ) -> None:
        """Checks the compressibility parameter

        Args:
            temperature: Temperature in K
            pressure: Pressure
            fugacity_model: Fugacity model
            expected: The expected value
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        compressibility: ArrayLike = eos.compressibility_parameter(temperature, pressure)

        assert compressibility == approx(expected, rtol, atol)

    @staticmethod
    def fugacity_coefficient(
        temperature: float, pressure: float, eos: RealGas, expected: float, *, rtol=RTOL, atol=ATOL
    ) -> None:
        """Checks the fugacity coefficient.

        Args:
            temperature: Temperature in K
            pressure: Pressure
            fugacity_model: Fugacity model
            expected: The expected value
            rtol: Relative tolerance
            atol: Absolute tolerance
        """

        fugacity_coeff: ArrayLike = eos.fugacity_coefficient(temperature, pressure)

        assert fugacity_coeff == approx(expected, rtol, atol)

    @staticmethod
    def volume(
        temperature: float, pressure: float, eos: RealGas, expected: float, *, rtol=RTOL, atol=ATOL
    ) -> None:
        """Checks the volume.

        Args:
            temperature: Temperature in K
            pressure: Pressure
            fugacity_model: Fugacity model
            expected: The expected value
        """

        volume: ArrayLike = eos.volume(temperature, pressure)

        assert volume == approx(expected, rtol, atol)


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
