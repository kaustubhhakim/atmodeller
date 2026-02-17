# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for tests"""

import logging

import pytest
from jaxtyping import ArrayLike
from pytest import approx

from atmodeller.interfaces import SolubilityProtocol

logger: logging.Logger = logging.getLogger("atmodeller.tests.solubility")

# Tolerances to compare the test results with target output.
# RTOL: float = 1.0e-8
RTOL: float = 0.61
"""Relative tolerance"""
# ATOL: float = 1.0e-8
ATOL: float = 0.61
"""Absolute tolerance"""


class CheckValues:
    """Helper to check and confirm values"""

    @classmethod
    def concentration(
        cls,
        function_name: str,
        solubility_model: SolubilityProtocol,
        expected_concentration: ArrayLike,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
        *,
        rtol=RTOL,
        atol=ATOL,
    ) -> None:
        concentration: ArrayLike = solubility_model.concentration(
            fugacity, temperature=temperature, pressure=pressure, fO2=fO2
        )
        logger.debug("%s, concentration = %s ppmw", function_name, concentration)

        assert concentration == approx(expected_concentration, rtol, atol)


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
