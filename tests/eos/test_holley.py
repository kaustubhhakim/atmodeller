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

from atmodeller.eos import RealGas
from atmodeller.utilities import unit_conversion

# Probably due to rounding of the model parameters in the paper, some compressibilities in the
# table in the paper don't quite match exactly with what we compute. Hence relax the tolerance.
RTOL: float = 1.0e-4
"""Relative tolerance"""
ATOL: float = 1.0e-4
"""Absolute tolerance"""

MODEL_SUFFIX: str = "beattie_holley58"
"""Suffix of the :cite:t:`HWZ58` models"""


def test_H2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table II}`"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, model, 1.06217, rtol=RTOL, atol=ATOL)


def test_H2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table II}`"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 1.26294, rtol=RTOL, atol=ATOL)


def test_H2_high_fugacity(check_values) -> None:
    """Tests that a fugacity can be calculated"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.fugacity(1000, pressure, model, 1301.672235770893, rtol=RTOL, atol=ATOL)


def test_N2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table III}`"""
    model = check_values.get_eos_model("N2", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, model, 1.00464, rtol=RTOL, atol=ATOL)


def test_N2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table III}`"""
    model: RealGas = check_values.get_eos_model("N2", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 1.36551, rtol=RTOL, atol=ATOL)


def test_O2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table IV}`"""
    model: RealGas = check_values.get_eos_model("O2", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, model, 0.95454, rtol=RTOL, atol=ATOL)


def test_O2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table IV}`"""
    model: RealGas = check_values.get_eos_model("O2", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 1.28897, rtol=RTOL, atol=ATOL)


def test_CO2_low(check_values) -> None:
    """:cite:t:`HWZ58{Table V}`"""
    model: RealGas = check_values.get_eos_model("CO2", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(400, pressure, model, 0.81853, rtol=RTOL, atol=ATOL)


def test_CO2_high(check_values) -> None:
    """:cite:t:`HWZ58{Table V}`"""
    model: RealGas = check_values.get_eos_model("CO2", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 1.07058, rtol=RTOL, atol=ATOL)


def test_NH3_low(check_values) -> None:
    """:cite:t:`HWZ58{Table VI}`"""
    model: RealGas = check_values.get_eos_model("NH3", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(400, pressure, model, 0.56165, rtol=RTOL, atol=ATOL)


def test_NH3_high(check_values) -> None:
    """:cite:t:`HWZ58{Table VI}`"""
    model: RealGas = check_values.get_eos_model("NH3", MODEL_SUFFIX)
    pressure: float = 500 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 0.93714, rtol=RTOL, atol=ATOL)


def test_CH4_low(check_values) -> None:
    """:cite:t:`HWZ58{Table VII}`"""
    model: RealGas = check_values.get_eos_model("CH4", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, model, 0.85583, rtol=RTOL, atol=ATOL)


def test_CH4_high(check_values) -> None:
    """:cite:t:`HWZ58{Table VII}`"""
    model: RealGas = check_values.get_eos_model("CH4", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 1.36201, rtol=RTOL, atol=ATOL)


def test_He_low(check_values) -> None:
    """:cite:t:`HWZ58{Table VIII}`"""
    model: RealGas = check_values.get_eos_model("He", MODEL_SUFFIX)
    pressure: float = 100 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(300, pressure, model, 1.05148, rtol=RTOL, atol=ATOL)


def test_He_high(check_values) -> None:
    """:cite:t:`HWZ58{Table VIII}`"""
    model: RealGas = check_values.get_eos_model("He", MODEL_SUFFIX)
    pressure: float = 1000 * unit_conversion.atmosphere_to_bar
    check_values.compressibility(1000, pressure, model, 1.14766, rtol=RTOL, atol=ATOL)


# def test_volume_with_broadcasting(check_values) -> None:
#     """Tests volume with broadcasting"""
#     model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
#     check_values.check_broadcasting("volume", model)


# def test_fugacity_with_broadcasting(check_values) -> None:
#     """Tests volume with broadcasting"""
#     model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
#     check_values.check_broadcasting("fugacity", model)
