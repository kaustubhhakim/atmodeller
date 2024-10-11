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

from atmodeller import __version__, debug_logger
from atmodeller.solubility.hydrogen_species import (
    H2_andesite_hirschmann,
    H2_basalt_hirschmann,
    H2_silicic_melts_gaillard,
    H2O_ano_dio_newcombe,
    H2O_basalt_dixon,
    H2O_basalt_mitchell,
    H2O_basalt_wilson,
    H2O_lunar_glass_newcombe,
    H2O_peridotite_sossi,
)
from atmodeller.solubility.jax_interfaces import SolubilityProtocol
from atmodeller.solubility.old_hydrogen_species import H2O_peridotite_sossi as old_model
from atmodeller.thermodata.core import IronWustiteBuffer, RedoxBufferProtocol

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

# Test a non-unity fugacity so the exponent is relevant for a power law solubility.
TEST_FUGACITY: ArrayLike = 2
TEST_TEMPERATURE: ArrayLike = 2000
TEST_PRESSURE: ArrayLike = 10

LOG10_SHIFT: ArrayLike = 0
IW: RedoxBufferProtocol = IronWustiteBuffer(LOG10_SHIFT)
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_FO2 = %e bar", TEST_FO2)


def test_H2_andesite_hirschmann() -> None:
    """Tests H2 in synthetic andesite :cite:p:`HWA12`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2_andesite_hirschmann
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 15.545054132817002, rtol=RTOL, atol=ATOL).all()


def test_H2_basalt_hirschmann() -> None:
    """Tests H2 in synthetic basalt :cite:p:`HWA12`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2_basalt_hirschmann
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 18.13918061563441, rtol=RTOL, atol=ATOL).all()


def test_H2_silicic_melts_gaillard() -> None:
    """Tests Fe-H redox exchange in silicate glasses :cite:p:`GSM03`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2_silicic_melts_gaillard
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 0.38821933289297966, rtol=RTOL, atol=ATOL).all()


def test_H2O_ano_dio_newcombe() -> None:
    """Tests H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2O_ano_dio_newcombe
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 1028.1332598452402, rtol=RTOL, atol=ATOL).all()


def test_H2O_basalt_dixon() -> None:
    """Tests H2O in MORB liquids :cite:p:`DSH95`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2O_basalt_dixon
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 1364.7160876900368, rtol=RTOL, atol=ATOL).all()


def test_H2O_basalt_mitchell() -> None:
    """Tests H2O in basaltic melt :cite:p:`MGO17`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2O_basalt_mitchell
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 411.7165015844662, rtol=RTOL, atol=ATOL).all()


def test_H2O_basalt_wilson() -> None:
    """Tests H2O in basalt :cite:p:`WH81,HBO64`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2O_basalt_wilson
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 349.26853043318124, rtol=RTOL, atol=ATOL).all()


def test_H2O_lunar_glass_newcombe() -> None:
    """Tests H2O in lunar basalt :cite:p:`NBB17`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2O_lunar_glass_newcombe
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 965.907863100824, rtol=RTOL, atol=ATOL).all()


def test_H2O_peridotite_sossi() -> None:
    """Tests H2O in peridotite liquids :cite:p:`STB23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    model: SolubilityProtocol = H2O_peridotite_sossi
    concentration: ArrayLike = model.concentration(
        TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
    )
    output_concentration_to_logger(function_name, concentration)

    assert np.isclose(concentration, 914.9961748553926, rtol=RTOL, atol=ATOL).all()

    # solubility2 = old_model().concentration(TEST_FUGACITY)
    # logger.debug("solubility2 = %s", solubility2)


def output_concentration_to_logger(function_name: str, concentration: ArrayLike) -> None:
    """Outputs the concentration to the logger

    Args:
        function_name: Function name
        concentration: Concentration
    """
    logger.debug("%s, concentration = %s ppmw", function_name, concentration)
