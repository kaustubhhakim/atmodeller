"""Tests for the Saxena EOS models

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.eos.interfaces import RealGasABC
from atmodeller.eos.saxena import H2_SF87, get_saxena_eos_models
from atmodeller.utilities import UnitConversion

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGasABC] = get_saxena_eos_models()

RTOL: float = 1.0e-8
ATOL: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# Corresponding states


def test_Ar(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(
        2510, 100e3, eos_models["Ar"], 7.41624600755374, rtol=RTOL, atol=ATOL
    )


def test_CH4(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(
        1912, 159e3, eos_models["CH4"], 17.77499804453072, rtol=RTOL, atol=ATOL
    )


def test_CO2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(
        1167, 184e3, eos_models["CO2"], 33.886349109271734, rtol=RTOL, atol=ATOL
    )


def test_H2_SF87(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1222, 41.66e3, H2_SF87, 4.975497264839999, rtol=RTOL, atol=ATOL)


def test_N2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(
        1573, 75e3, eos_models["N2"], 10.293087737779091, rtol=RTOL, atol=ATOL
    )


def test_O2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(
        1823, 133e3, eos_models["O2"], 12.409268281002012, rtol=RTOL, atol=ATOL
    )


# H2


def test_H2_low_pressure_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 7279.356114821697
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(873, 10, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_medium_pressure_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 164.38851468757488
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(873, 500, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 43.46585841223779
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1473, 4000, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure2_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 21.547766750104773
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1073, 10000, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure3_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 71.46244038505347
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(673, 1000, eos_models["H2"], expected, rtol=RTOL, atol=ATOL)


# H2S


def test_H2S_low_pressure_SS92(check_values) -> None:
    """Comparison with Figure 3 in Shi and Saxena (1992)"""
    expected: float = 272.7266232763035
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(673, 200, eos_models["H2S"], expected, rtol=RTOL, atol=ATOL)


def test_H2S_medium_pressure_SS92(check_values) -> None:
    """Comparison with Figure 3 in Shi and Saxena (1992)"""
    expected: float = 116.55537998390933
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1873, 2000, eos_models["H2S"], expected, rtol=RTOL, atol=ATOL)


# SO2


def test_SO2_low_pressure_SS92(check_values) -> None:
    """Comparison with Figure 2 in Shi and Saxena (1992)"""
    expected: float = 8308.036738813245
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1073, 10, eos_models["SO2"], expected, rtol=RTOL, atol=ATOL)


def test_SO2_high_pressure_SS92(check_values) -> None:
    """Comparison with Figure 2 in Shi and Saxena (1992)"""
    expected: float = 70.86864302460566
    expected = UnitConversion.cm3_to_m3(expected)
    check_values.volume(1873, 4000, eos_models["SO2"], expected, rtol=RTOL, atol=ATOL)
