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
# TODO: Update reference to Chabier.
"""Tests for the EOS models from :cite:t:`HP91,HP98`"""

# Convenient to use acroynms and symbol names so pylint: disable=invalid-name

from __future__ import annotations

import logging

from atmodeller import debug_logger
from atmodeller.eos.chabrier import get_chabrier_eos_models
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import unit_conversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_chabrier_eos_models()
"""EOS models from :cite:t:`CD21`"""


def test_Chabrier_H2_volume_100kbar(check_values) -> None:
    """Test Chabrier"""
    expected: float = 9.005066169376918
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(3000, 100e3, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_Chabrier_H2_volume_1000kbar(check_values) -> None:
    """Test Chabrier 2"""
    expected: float = 3.0100820540769166
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(5000, 1000e3, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)
