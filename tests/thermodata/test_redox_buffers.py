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
"""Tests redox buffer"""
import logging

from jax.typing import ArrayLike
from pytest import approx

from atmodeller import debug_logger
from atmodeller.thermodata.core import (
    IronWustiteBufferHirschmann,
    IronWustiteBufferHirschmann08,
    IronWustiteBufferHirschmann21,
    RedoxBufferProtocol,
)
from atmodeller.utilities import unit_conversion

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

LOG10_SHIFT: ArrayLike = 0
"""Log10 shift"""
RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

TEST_LOW_TEMPERATURE: ArrayLike = 1000
TEST_LOW_PRESSURE: ArrayLike = 1  # bar
TEST_MEDIUM_TEMPERATURE: ArrayLike = 2000
TEST_MEDIUM_PRESSURE: ArrayLike = 5 * unit_conversion.GPa_to_bar
TEST_HIGH_TEMPERATURE: ArrayLike = 3000
TEST_HIGH_PRESSURE: ArrayLike = 50 * unit_conversion.GPa_to_bar


def test_IW_Hirschmann21() -> None:
    """Tests the Hirschmann (2021) iron-wustite buffer

    The values below agree with the calculator provided in Table S7 of the Excel spreadsheet
    provided in the supplementary materials of the online publication.
    """
    buffer: RedoxBufferProtocol = IronWustiteBufferHirschmann21(LOG10_SHIFT)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_LOW_TEMPERATURE, TEST_LOW_PRESSURE)
    logger.info("log10_fugacity (Low T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-20.816597461900887, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_MEDIUM_TEMPERATURE, TEST_LOW_PRESSURE)
    logger.info("log10_fugacity (Med T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-7.053620769567622, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_HIGH_TEMPERATURE, TEST_LOW_PRESSURE)
    logger.info("log10_fugacity (High T, low P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-2.562340960147525, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_LOW_TEMPERATURE, TEST_MEDIUM_PRESSURE)
    logger.info("log10_fugacity (Low T, med P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-17.97502528402518, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(
        TEST_MEDIUM_TEMPERATURE, TEST_MEDIUM_PRESSURE
    )
    logger.info("log10_fugacity (Med T, med P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-5.619533227830953, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_HIGH_TEMPERATURE, TEST_MEDIUM_PRESSURE)
    logger.info("log10_fugacity (High T, med P) = %s", log10_fugacity)

    assert log10_fugacity == approx(-1.5951793921085295, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_LOW_TEMPERATURE, TEST_HIGH_PRESSURE)
    logger.info("log10_fugacity (Low T, high P) = %s", log10_fugacity)

    assert log10_fugacity == approx(4.232976104931987, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_MEDIUM_TEMPERATURE, TEST_HIGH_PRESSURE)
    logger.info("log10_fugacity (Med T, high P) = %s", log10_fugacity)

    assert log10_fugacity == approx(5.534585436736885, RTOL, ATOL)

    log10_fugacity: ArrayLike = buffer.log10_fugacity(TEST_HIGH_TEMPERATURE, TEST_HIGH_PRESSURE)
    logger.info("log10_fugacity (High T, high P) = %s", log10_fugacity)

    assert log10_fugacity == approx(5.887511178821558, RTOL, ATOL)
