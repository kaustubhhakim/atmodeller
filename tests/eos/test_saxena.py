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
"""Tests for the EOS models from :cite:t:`SF87,SF87a,SF88,SS92`"""

# Convenient to use species chemical formulae so pylint: disable=invalid-name

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.eos.interfaces import RealGas
from atmodeller.eos.saxena import H2_SF87, get_saxena_eos_models
from atmodeller.utilities import UnitConversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_saxena_eos_models()
"""EOS models from :cite:t:`SF87,SF87a,SF88,SS92`"""


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_Ar(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    check_values.compressibility(
        2510, 100e3, eos_models["Ar"], 7.41624600755374, rtol=RTOL, atol=ATOL
    )


def test_CH4(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    check_values.compressibility(
        1912, 159e3, eos_models["CH4"], 17.77499804453072, rtol=RTOL, atol=ATOL
    )


def test_CO2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    check_values.compressibility(
        1167, 184e3, eos_models["CO2"], 33.886349109271734, rtol=RTOL, atol=ATOL
    )


def test_H2_SF87(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    check_values.compressibility(1222, 41.66e3, H2_SF87, 4.975497264839999, rtol=RTOL, atol=ATOL)


def test_N2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    check_values.compressibility(
        1573, 75e3, eos_models["N2"], 10.293087737779091, rtol=RTOL, atol=ATOL
    )


def test_O2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    check_values.compressibility(
        1823, 133e3, eos_models["O2"], 12.409268281002012, rtol=RTOL, atol=ATOL
    )


def test_H2_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    expected: float = 7279.356114821697
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(873, 10, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_medium_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    expected: float = 164.38851468757488
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(873, 500, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    expected: float = 41.97871061892679
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1473, 4000, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure2_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    expected: float = 20.806595067793276
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1073, 10000, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure3_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    expected: float = 71.50153474005484
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(673, 1000, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2S_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 3}`"""
    expected: float = 272.7266232763035
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(673, 200, eos_models["H2S"], expected, rtol=RTOL, atol=ATOL)


def test_H2S_medium_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 3}`"""
    expected: float = 116.55537998390933
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1873, 2000, eos_models["H2S"], expected, rtol=RTOL, atol=ATOL)


def test_SO2_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 2}`"""
    expected: float = 8308.036738813245
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1073, 10, eos_models["SO2"], expected, rtol=RTOL, atol=ATOL)


def test_SO2_high_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 2}`"""
    expected: float = 70.86864302460566
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1873, 4000, eos_models["SO2"], expected, rtol=RTOL, atol=ATOL)
