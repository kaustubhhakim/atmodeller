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
"""Tests solubility laws for hydrogen species"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

import inspect
import logging

import numpy as np
from jax.typing import ArrayLike

from atmodeller import debug_logger
from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility.hydrogen_species import (
    H2_andesite_hirschmann12,
    H2_basalt_hirschmann12,
    H2_silicic_melts_gaillard03,
    H2O_ano_dio_newcombe17,
    H2O_basalt_dixon95,
    H2O_basalt_mitchell17,
    H2O_basalt_wilson81,
    H2O_lunar_glass_newcombe17,
    H2O_peridotite_sossi23,
)
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer, RedoxBufferProtocol

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
TEST_PRESSURE: ArrayLike = 10  # bar
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_FO2 = %e bar", TEST_FO2)


def test_H2_andesite_hirschmann(check_values) -> None:
    """Tests H2 in synthetic andesite :cite:p:`HWA12`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2_andesite_hirschmann12
    target_concentration: ArrayLike = 15.545054132817002

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2_basalt_hirschmann(check_values) -> None:
    """Tests H2 in synthetic basalt :cite:p:`HWA12`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2_basalt_hirschmann12
    target_concentration: ArrayLike = 18.13918061563441

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2_silicic_melts_gaillard(check_values) -> None:
    """Tests Fe-H redox exchange in silicate glasses :cite:p:`GSM03`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2_silicic_melts_gaillard03
    target_concentration: ArrayLike = 0.38821933289297966

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_ano_dio_newcombe(check_values) -> None:
    """Tests H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2O_ano_dio_newcombe17
    target_concentration: ArrayLike = 1028.1332598452402

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_basalt_dixon(check_values) -> None:
    """Tests H2O in MORB liquids :cite:p:`DSH95`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2O_basalt_dixon95
    target_concentration: ArrayLike = 1364.7160876900368

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_basalt_mitchell(check_values) -> None:
    """Tests H2O in basaltic melt :cite:p:`MGO17`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2O_basalt_mitchell17
    target_concentration: ArrayLike = 411.7165015844662

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_basalt_wilson(check_values) -> None:
    """Tests H2O in basalt :cite:p:`WH81,HBO64`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2O_basalt_wilson81
    target_concentration: ArrayLike = 349.26853043318124

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_lunar_glass_newcombe(check_values) -> None:
    """Tests H2O in lunar basalt :cite:p:`NBB17`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2O_lunar_glass_newcombe17
    target_concentration: ArrayLike = 965.907863100824

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_H2O_peridotite_sossi(check_values) -> None:
    """Tests H2O in peridotite liquids :cite:p:`STB23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = H2O_peridotite_sossi23
    target_concentration: ArrayLike = 914.9961748553926

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )
