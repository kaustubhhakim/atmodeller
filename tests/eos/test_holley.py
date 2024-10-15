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
"""Tests for the EOS models from :cite:t:`HWZ58`"""

# Convenient to use species chemical formulae so pylint: disable=invalid-name

from __future__ import annotations

import logging

from atmodeller import debug_logger
from atmodeller.eos.holley import get_holley_eos_models
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import unit_conversion

# Probably due to rounding of the model parameters in the paper, some compressibilities in the
# table in the paper don't quite match exactly with what we compute. Hence relax the tolerance.
RTOL: float = 1.0e-4
"""Relative tolerance"""
ATOL: float = 1.0e-4
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_holley_eos_models()
"""EOS models from :cite:t:`HWZ58`"""


def test_H2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table II}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, eos_models["H2"], 1.06217, rtol=RTOL, atol=ATOL)


def test_H2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table II}`"""
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["H2"], 1.26294, rtol=RTOL, atol=ATOL)


def test_N2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table III}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, eos_models["N2"], 1.00464, rtol=RTOL, atol=ATOL)


def test_N2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table III}`"""
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["N2"], 1.36551, rtol=RTOL, atol=ATOL)


def test_O2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table IV}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, eos_models["O2"], 0.95454, rtol=RTOL, atol=ATOL)


def test_O2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table IV}`"""
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["O2"], 1.28897, rtol=RTOL, atol=ATOL)


def test_CO2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table V}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(400, pressure, eos_models["CO2"], 0.81853, rtol=RTOL, atol=ATOL)


def test_CO2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table V}`"""
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["CO2"], 1.07058, rtol=RTOL, atol=ATOL)


def test_NH3_low(check_values) -> None:
    """:cite:t:`HWZ58{Table VI}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(400, pressure, eos_models["NH3"], 0.56165, rtol=RTOL, atol=ATOL)


def test_NH3_high(check_values) -> None:
    """:cite:t:`HWZ58{Table VI}`"""
    pressure: float = 500 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["NH3"], 0.93714, rtol=RTOL, atol=ATOL)


def test_CH4_low(check_values) -> None:
    """:cite:t:`HWZ58{Table VII}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, eos_models["CH4"], 0.85583, rtol=RTOL, atol=ATOL)


def test_CH4_high(check_values) -> None:
    """:cite:t:`HWZ58{Table VII}`"""
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["CH4"], 1.36201, rtol=RTOL, atol=ATOL)


def test_He_low(check_values) -> None:
    """:cite:t:`HWZ58{Table VIII}`"""
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, eos_models["He"], 1.05148, rtol=RTOL, atol=ATOL)


def test_He_high(check_values) -> None:
    """:cite:t:`HWZ58{Table VIII}`"""
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, eos_models["He"], 1.14766, rtol=RTOL, atol=ATOL)
