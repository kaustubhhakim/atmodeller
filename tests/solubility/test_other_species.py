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
"""Tests solubility laws for other species"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

import inspect
import logging

import numpy as np
from jax.typing import ArrayLike

from atmodeller import debug_logger
from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility.other_species import (
    Cl2_ano_dio_for_thomas21,
    Cl2_basalt_thomas21,
    He_basalt_jambon86,
    N2_basalt_bernadou21,
    N2_basalt_dasgupta22,
    N2_basalt_libourel03,
)
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer, RedoxBufferProtocol
from atmodeller.utilities import unit_conversion

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

LOG10_SHIFT: ArrayLike = 0
IW: RedoxBufferProtocol = IronWustiteBuffer(LOG10_SHIFT)

# Test a non-unity fugacity so the exponent is relevant for a power law solubility.
TEST_FUGACITY: ArrayLike = 2
TEST_TEMPERATURE: ArrayLike = 2000
TEST_PRESSURE: ArrayLike = 2  # bar
# Several models are calibrated in the low GPa range, so use this instead
TEST_PRESSURE_GPA: ArrayLike = 2 * unit_conversion.GPa_to_bar  # GPa
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))
TEST_FO2_GPA: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE_GPA))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_PRESSURE_GPA = %e bar", TEST_PRESSURE_GPA)
logger.info("TEST_FO2 = %e bar", TEST_FO2)
logger.info("TEST_FO2_GPA = %e bar", TEST_FO2_GPA)


def test_Cl2_ano_dio_for_thomas(check_values) -> None:
    """Tests Cl in silicate melts :cite:p:`TW21`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = Cl2_ano_dio_for_thomas21
    target_concentration: ArrayLike = 1987252.8978466734

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_Cl2_basalt_thomas(check_values) -> None:
    """Tests Cl in silicate melts :cite:p:`TW21`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = Cl2_basalt_thomas21
    target_concentration: ArrayLike = 1111006.1746003036

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_He_basalt(check_values) -> None:
    """He in tholeittic basalt melt :cite:p:`JWB86`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = He_basalt_jambon86
    target_concentration: ArrayLike = 0.20013

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_N2_basalt_bernadou(check_values) -> None:
    """Tests N2 in basaltic silicate melt :cite:p:`BGF21`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = N2_basalt_bernadou21
    target_concentration: ArrayLike = 0.6844302758688818

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_N2_basalt_dasgupta(check_values) -> None:
    """Tests N2 in silicate melts :cite:p:`DFP22`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = N2_basalt_dasgupta22
    target_concentration: ArrayLike = 0.9502724120546588

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_N2_basalt_libourel(check_values) -> None:
    """Tests N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = N2_basalt_libourel03
    target_concentration: ArrayLike = 0.12236469243947536

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )
