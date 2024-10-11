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
from atmodeller.solubility.jax_interfaces import SolubilityProtocol
from atmodeller.solubility.old_carbon_species import CO2_basalt_dixon as old_model
from atmodeller.solubility.other_species import N2_basalt_bernadou
from atmodeller.thermodata.core import IronWustiteBuffer, RedoxBufferProtocol
from atmodeller.utilities import unit_conversion

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

# Test a non-unity fugacity so the exponent is relevant for a power law solubility.
TEST_FUGACITY: ArrayLike = 2
TEST_TEMPERATURE: ArrayLike = 2000
TEST_PRESSURE: ArrayLike = 500  # bar, for Dixon experimental range
# Several models are calibrated in the low GPa range, so use this instead
TEST_PRESSURE_GPA: ArrayLike = 2 * unit_conversion.GPa_to_bar  # GPa

LOG10_SHIFT: ArrayLike = 0
IW: RedoxBufferProtocol = IronWustiteBuffer(LOG10_SHIFT)
TEST_FO2: ArrayLike = np.exp(IW.log_fugacity(TEST_TEMPERATURE, TEST_PRESSURE))

logger.info("TEST_FUGACITY = %e bar", TEST_FUGACITY)
logger.info("TEST_TEMPERATURE = %e K", TEST_TEMPERATURE)
logger.info("TEST_PRESSURE = %e bar", TEST_PRESSURE)
logger.info("TEST_PRESSURE_GPA = %e bar", TEST_PRESSURE_GPA)
logger.info("TEST_FO2 = %e bar", TEST_FO2)


# def test_CH4_basalt_ardia(helper) -> None:
#     """Tests CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`"""

#     function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
#     model: SolubilityProtocol = CH4_basalt_ardia
#     concentration: ArrayLike = model.concentration(
#         TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE_GPA, TEST_FO2
#     )
#     helper.concentration_to_logger(function_name, concentration)

#     assert np.isclose(concentration, 0.0005831884445042942, rtol=RTOL, atol=ATOL).all()

# solubility2 = old_model().concentration(TEST_FUGACITY)
# logger.debug("solubility2 = %s", solubility2)


# def test_CO_basalt_armstrong(helper) -> None:
#     """Tests volatiles in mafic melts under reduced conditions :cite:p:`AHS15`"""

#     function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
#     model: SolubilityProtocol = CO_basalt_armstrong
#     concentration: ArrayLike = model.concentration(
#         TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE_GPA, TEST_FO2
#     )
#     helper.concentration_to_logger(function_name, concentration)

#     assert np.isclose(concentration, 0.027396953726422667, rtol=RTOL, atol=ATOL).all()


# def test_CO_basalt_yoshioka(helper) -> None:
#     """Tests carbon in silicate melts :cite:p:`YNN19`"""

#     function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
#     model: SolubilityProtocol = CO_basalt_yoshioka
#     concentration: ArrayLike = model.concentration(
#         TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE_GPA, TEST_FO2
#     )
#     helper.concentration_to_logger(function_name, concentration)

#     assert np.isclose(concentration, 0.1098560543306116, rtol=RTOL, atol=ATOL).all()


# def test_CO_rhyolite_yoshioka(helper) -> None:
#     """Tests carbon in silicate melts :cite:p:`YNN19`"""

#     function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
#     model: SolubilityProtocol = CO_rhyolite_yoshioka
#     concentration: ArrayLike = model.concentration(
#         TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE_GPA, TEST_FO2
#     )
#     helper.concentration_to_logger(function_name, concentration)

#     assert np.isclose(concentration, 1.19271202468211, rtol=RTOL, atol=ATOL).all()


# def test_CO2_basalt_dixon(helper) -> None:
#     """Tests CO2 in MORB liquids :cite:p:`DSH95`"""

#     function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
#     model: SolubilityProtocol = CO2_basalt_dixon
#     concentration: ArrayLike = model.concentration(
#         TEST_FUGACITY, TEST_TEMPERATURE, TEST_PRESSURE, TEST_FO2
#     )
#     helper.concentration_to_logger(function_name, concentration)

#     assert np.isclose(concentration, 0.8527333099685608, rtol=RTOL, atol=ATOL).all()
