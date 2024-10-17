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
from atmodeller.solubility.core import SolubilityProtocol
from atmodeller.solubility.sulfur_species import (
    S2_andesite_boulliung,
    S2_basalt_boulliung,
    S2_mercury_magma_namur,
    S2_sulfate_andesite_boulliung,
    S2_sulfate_basalt_boulliung,
    S2_sulfate_trachybasalt_boulliung,
    S2_sulfide_andesite_boulliung,
    S2_sulfide_basalt_boulliung,
    S2_sulfide_trachybasalt_boulliung,
    S2_trachybasalt_boulliung,
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


def test_S2_sulfate_andesite_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_sulfate_andesite_boulliung
    target_concentration: ArrayLike = 0.00026017525012647343

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfide_andesite_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in andesite :cite:p:`BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_sulfide_andesite_boulliung
    target_concentration: ArrayLike = 2765.4015601584474

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_andesite_boulliung(check_values) -> None:
    """Tests S in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_andesite_boulliung
    target_concentration: ArrayLike = 2765.4018203336977

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfate_basalt_boulliung(check_values) -> None:
    """Tests S in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_sulfate_basalt_boulliung
    target_concentration: ArrayLike = 0.000205313138905436

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfide_basalt_boulliung(check_values) -> None:
    """Tests S in basalt as sulfide (S^2-) :cite:p:`BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_sulfide_basalt_boulliung
    target_concentration: ArrayLike = 7575.8488900346765

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_basalt_boulliung(check_values) -> None:
    """Tests S in basalt due to sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_basalt_boulliung
    target_concentration: ArrayLike = 7575.849095347816

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfate_trachybasalt_boulliung(check_values) -> None:
    """Tests S as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_sulfate_trachybasalt_boulliung
    target_concentration: ArrayLike = 0.0007003728959148125

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_sulfide_trachybasalt_boulliung(check_values) -> None:
    """Tests S as sulfide (S^2-) in trachybasalt :cite:p:`BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_sulfide_trachybasalt_boulliung
    target_concentration: ArrayLike = 9573.143159494639

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_trachybasalt_boulliung(check_values) -> None:
    """Tests S in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_trachybasalt_boulliung
    target_concentration: ArrayLike = 9573.143859867534

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )


def test_S2_mercury_magma_namur(check_values) -> None:
    """Tests S in reduced mafic silicate melts relevant for Mercury :cite:p:`NCH16`"""

    function_name: str = inspect.currentframe().f_code.co_name  # type: ignore
    solubility_model: SolubilityProtocol = S2_mercury_magma_namur
    target_concentration: ArrayLike = 1827.8623756735988

    check_values.concentration(
        function_name,
        solubility_model,
        target_concentration,
        TEST_FUGACITY,
        TEST_TEMPERATURE,
        TEST_PRESSURE,
        TEST_FO2,
    )
