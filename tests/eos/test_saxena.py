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

from __future__ import annotations

import logging

from atmodeller import debug_logger
from atmodeller.eos import get_eos_models
from atmodeller.eos._saxena import H2_SF87
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import unit_conversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGasProtocol] = get_eos_models()
"""EOS models from :cite:t:`SF87,SF87a,SF88,SS92`"""


def test_Ar(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGasProtocol = eos_models["Ar_cs_saxena87"]
    expected: float = 7.41624600755374
    check_values.compressibility(2510, 100e3, model, expected, rtol=RTOL, atol=ATOL)


def test_CH4(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGasProtocol = eos_models["CH4_cs_shi92"]
    expected: float = 17.77499804453072
    check_values.compressibility(1912, 159e3, model, expected, rtol=RTOL, atol=ATOL)


def test_CO2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGasProtocol = eos_models["CO2_cs_shi92"]
    expected: float = 33.886349109271734
    check_values.compressibility(1167, 184e3, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_SF87(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGasProtocol = H2_SF87
    expected: float = 4.975497264839999
    check_values.compressibility(1222, 41.66e3, model, expected, rtol=RTOL, atol=ATOL)


def test_N2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGasProtocol = eos_models["N2_cs_saxena87"]
    expected: float = 10.293087737779091
    check_values.compressibility(1573, 75e3, model, expected, rtol=RTOL, atol=ATOL)


def test_O2(check_values) -> None:
    """:cite:t:`SF87{Table 1}`"""
    model: RealGasProtocol = eos_models["O2_cs_shi92"]
    expected: float = 12.409268281002012
    check_values.compressibility(1823, 133e3, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGasProtocol = eos_models["H2_shi92"]
    expected: float = 7279.356114821697 * unit_conversion.cm3_to_m3
    check_values.volume(873, 10, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_medium_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGasProtocol = eos_models["H2_shi92"]
    expected: float = 164.38851468757488 * unit_conversion.cm3_to_m3
    check_values.volume(873, 500, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGasProtocol = eos_models["H2_shi92"]
    expected: float = 41.97871061892679 * unit_conversion.cm3_to_m3
    check_values.volume(1473, 4000, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure2_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGasProtocol = eos_models["H2_shi92"]
    expected: float = 20.806595067793276 * unit_conversion.cm3_to_m3
    check_values.volume(1073, 10000, model, expected, rtol=RTOL, atol=ATOL)


def test_H2_high_pressure3_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 1}`"""
    model: RealGasProtocol = eos_models["H2_shi92"]
    expected: float = 71.50153474005484 * unit_conversion.cm3_to_m3
    check_values.volume(673, 1000, model, expected, rtol=RTOL, atol=ATOL)


def test_H2S_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 3}`"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    expected: float = 272.7266232763035 * unit_conversion.cm3_to_m3
    check_values.volume(673, 200, model, expected, rtol=RTOL, atol=ATOL)


def test_H2S_medium_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 3}`"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    expected: float = 116.55537998390933 * unit_conversion.cm3_to_m3
    check_values.volume(1873, 2000, model, expected, rtol=RTOL, atol=ATOL)


def test_SO2_low_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 2}`"""
    model: RealGasProtocol = eos_models["SO2_shi92"]
    expected: float = 8308.036738813245 * unit_conversion.cm3_to_m3
    check_values.volume(1073, 10, model, expected, rtol=RTOL, atol=ATOL)


def test_SO2_high_pressure_SS92(check_values) -> None:
    """:cite:t:`SS92{Figure 2}`"""
    model: RealGasProtocol = eos_models["SO2_shi92"]
    expected: float = 70.86864302460566 * unit_conversion.cm3_to_m3
    check_values.volume(1873, 4000, model, expected, rtol=RTOL, atol=ATOL)


# FIXME: Need to confirm that the lower bound behaves sensibly, in terms of all relevant parameters
# but notably the volume integral which is used to compute the fugacity, fugacity coefficient etc.
# def test_lower_extrapolation(check_values) -> None:
#     """Tests the lower bound extrapolation"""
#     model: RealGasProtocol = eos_models["H2S_shi92"]
#     temperature: float = 1000
#     pressure: float = 0.01
#     expected: float = 0.01
#     check_values.fugacity(temperature, pressure, model, expected)


def test_upper_extrapolation(check_values) -> None:
    """Tests the upper bound extrapolation"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    temperature: float = 1073
    pressure: float = 10e3  # Maximum pressure of the highest pressure EOS in this composite EOS
    pressure_above_max: float = 20e3  # To test pressure above the maximum
    # Expected compressibility factor at 3000 K and (the maximum calibrated pressure) 10000 bar
    expected: float = 4.472116811082408
    check_values.compressibility(temperature, pressure, model, expected)
    # The upper bound extrapolation should maintain the same compressibility for higher pressures
    check_values.compressibility(temperature, pressure_above_max, model, expected)


def test_volume_integral_standard_pressure(check_values) -> None:
    """Tests the volume integral"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    temperature: float = 1873
    pressure: float = 1.0
    # At the standard pressure the volume integral is zero by construction
    expected: float = 0.0
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_integral_index0(check_values) -> None:
    """Tests the volume integral for the first EOS"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    # Within ranges of experimental data, otherwise the extrapolated behaviour might be unphysical
    temperature: float = 1073
    pressure: float = 1000.0
    expected: float = 0.619184870412741
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_integral_index1(check_values) -> None:
    """Tests the volume integral for the first and second EOS"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    temperature: float = 1073
    pressure: float = 9999
    expected: float = 1.088570827561088
    check_values.volume_integral(temperature, pressure, model, expected)


def test_volume_integral_index2(check_values) -> None:
    """Tests the volume integral in the extrapolated region above the maximum pressure"""
    model: RealGasProtocol = eos_models["H2S_shi92"]
    temperature: float = 1073
    pressure: float = 20000
    expected: float = 1.365159989447105
    check_values.volume_integral(temperature, pressure, model, expected)
