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
"""Tests solubility laws for carbon species"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

import inspect
import logging

import numpy as np
from jax.typing import ArrayLike

from atmodeller import debug_logger
from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility.library import get_solubility_models
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
TEST_FUGACITY: ArrayLike = 2  # bar
TEST_TEMPERATURE: ArrayLike = 2000  # K
TEST_PRESSURE: ArrayLike = 500  # bar, motivated by Dixon experimental range
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))
# Several models are calibrated in the low GPa range, so use this instead for testing
TEST_PRESSURE_GPA: ArrayLike = 2 * unit_conversion.GPa_to_bar  # GPa
TEST_FO2_GPA: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE_GPA))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_PRESSURE_GPA = %e bar", TEST_PRESSURE_GPA)
logger.info("TEST_FO2 = %e bar", TEST_FO2)
logger.info("TEST_FO2_GPA = %e bar", TEST_FO2_GPA)

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()


def test_CH4_basalt_ardia(check_values) -> None:
    """Tests CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["CH4_basalt_ardia13"]
    target_concentration: ArrayLike = 0.0005831884445042942

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_CO_basalt_armstrong(check_values) -> None:
    """Tests volatiles in mafic melts under reduced conditions :cite:p:`AHS15`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["CO_basalt_armstrong15"]
    target_concentration: ArrayLike = 0.027396953726422667

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_CO_basalt_yoshioka(check_values) -> None:
    """Tests carbon in silicate melts :cite:p:`YNN19`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["CO_basalt_yoshioka19"]
    target_concentration: ArrayLike = 0.1098560543306116

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_CO_rhyolite_yoshioka(check_values) -> None:
    """Tests carbon in silicate melts :cite:p:`YNN19`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["CO_rhyolite_yoshioka19"]
    target_concentration: ArrayLike = 1.19271202468211

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE_GPA,
        TEST_FO2_GPA,
    )


def test_CO2_basalt_dixon(check_values) -> None:
    """Tests CO2 in MORB liquids :cite:p:`DSH95`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = solubility_models["CO2_basalt_dixon95"]
    target_concentration: ArrayLike = 0.8527333099685608

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )
