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

from collections.abc import Callable

import numpy as np
import numpy.testing as nptest
import pytest
from jaxtyping import Array, ArrayLike

from atmodeller.eos import RealGas, get_eos_models
from atmodeller.utilities import as_j64

# logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

# Tolerances to compare the test results with target output.
RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""


class CheckValues:
    """Helper class with methods to check and confirm values"""

    def __init__(self) -> None:
        self._eos_models: dict[str, RealGas] = get_eos_models()

    @classmethod
    def _check_property(
        cls,
        property_name: str,
        temperature: ArrayLike,
        pressure: ArrayLike,
        eos: RealGas,
        expected: ArrayLike,
        *,
        rtol=RTOL,
        atol=ATOL,
    ) -> None:
        """Generalized method to check a property (e.g., compressibility, fugacity, etc.)

        Args:
            property_name: Name of the property to check
            temperature: Temperature in K
            pressure: Pressure in bar
            eos: EOS model
            expected: The expected value
            rtol: Relative tolerance. Defaults to RTOL.
            atol: Absolute tolerance. Defaults to ATOL.
        """
        # Dynamically get the method from the eos model based on property_name
        method: Callable = getattr(eos, property_name)
        # Call the method with the provided temperature and pressure avoiding recompilation
        temperature = as_j64(temperature)
        pressure = as_j64(pressure)
        result: ArrayLike = method(temperature, pressure)

        # Compare the result with the expected value
        nptest.assert_allclose(result, expected, rtol, atol)

    @classmethod
    def compressibility(cls, *args, **kwargs) -> None:
        """Checks the compressibility factor"""
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

    @classmethod
    def check_broadcasting(cls, property_name: str, eos: RealGas) -> None:
        """Checks that the EOS model handles broadcasting correctly.

        Args:
            property_name: Name of the property to check
            eos: EOS model
        """
        # Dynamically get the method from the eos model based on property_name
        method: Callable = getattr(eos, property_name)

        # Since the shapes of the arrays are always changing here there's no point in converting to
        # jax arrays in order to avoid recompilation because recompilation will occur anyway due to
        # the changing array shapes.

        # Tests pressure broadcasting
        temperature = 2000
        pressure = np.array([1, 10, 100])
        result: Array = method(temperature, pressure)
        assert result.shape == (3,)

        # Tests temperature broadcasting
        temperature = np.array([1500, 2000])
        pressure = 100
        result = method(temperature, pressure)
        assert result.shape == (2,)

        # Tests both temperature and pressure broadcasting with equal length arrays
        temperature = np.array([1500, 2000])
        pressure = np.array([0.5, 100])
        results = method(temperature, pressure)
        assert results.shape == (2,)

        # Tests both temperature and pressure broadcasting
        temperature = np.array([1500, 2000])[:, None]
        pressure = np.array([1, 10, 100])[None, :]
        result = method(temperature, pressure)
        assert result.shape == (2, 3)

        # Tests both temperature and pressure broadcasting with switched order
        temperature = np.array([1500, 2000])[None, :]
        pressure = np.array([1, 10, 100])[:, None]
        result = method(temperature, pressure)
        assert result.shape == (3, 2)

    def get_eos_model(self, species_name: str, suffix: str) -> RealGas:
        """Gets a model for a species

        Args:
            species_name: Species name
            suffix: Model suffix

        Returns:
            EOS model
        """
        return self._eos_models[f"{species_name}_{suffix}"]


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
