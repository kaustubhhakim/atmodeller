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

from typing import Callable

import numpy.testing as nptest
import pytest
from jax.typing import ArrayLike

from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import RealGasProtocol

# Tolerances to compare the test results with target output.
RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

eos_models: dict[str, RealGasProtocol] = get_eos_models()


class CheckValues:
    """Helper class with methods to check and confirm values"""

    @classmethod
    def _check_property(
        cls,
        property_name: str,
        temperature: ArrayLike,
        pressure: ArrayLike,
        eos: RealGasProtocol,
        expected: ArrayLike,
        *,
        rtol=RTOL,
        atol=ATOL,
    ) -> None:
        """Generalized method to check a property (e.g., compressibility, fugacity, etc.)

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            fugacity_model: Fugacity model
            expected: The expected value
            rtol: Relative tolerance. Defaults to RTOL.
            atol: Absolute tolerance. Dedfaults to ATOL.
        """
        # Dynamically get the method from the eos model based on property_name
        method: Callable = getattr(eos, property_name)
        # Call the method with the provided temperature and pressure
        result: ArrayLike = method(temperature, pressure)

        # Compare the result with the expected value
        nptest.assert_allclose(result, expected, rtol, atol)

    @staticmethod
    def get_eos_model(species_name: str, suffix: str) -> RealGasProtocol:
        """Gets a model for a species

        Args:
            species_name: Species name
            suffix: Model suffix

        Returns:
            EOS model
        """
        return eos_models[f"{species_name}_{suffix}"]

    @classmethod
    def compressibility(cls, *args, **kwargs) -> None:
        """Checks the compressibility parameter"""
        cls._check_property("compressibility_factor", *args, **kwargs)

    @classmethod
    def fugacity(cls, *args, **kwargs) -> None:
        """Checks the fugacity"""
        cls._check_property("fugacity", *args, **kwargs)

    @classmethod
    def fugacity_coefficient(cls, *args, **kwargs) -> None:
        """Checks the fugacity coefficient"""
        cls._check_property("fugacity_coefficient", *args, **kwargs)

    @classmethod
    def volume(cls, *args, **kwargs) -> None:
        """Checks the volume"""
        cls._check_property("volume", *args, **kwargs)

    @classmethod
    def volume_integral(cls, *args, **kwargs) -> None:
        """Checks the volume integral"""
        cls._check_property("volume_integral", *args, **kwargs)


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
