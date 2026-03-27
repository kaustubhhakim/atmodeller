# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the EOS models from :cite:t:`ZD09`"""

from atmodeller.sci_utils import unit_conversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
MODEL_SUFFIX: str = "zhang09"
"""Suffix of the :cite:t:`ZD09` models"""


def test_H2O_volume_low_TP(check_values) -> None:
    """Tests H2O volume at 1203.15 K and 950 MPa :cite:t:`ZD09{Table 6}`"""
    expected: float = 22.20343433408026
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1203.15,
        9500,  # 950 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_H2O_volume_high_TP(check_values) -> None:
    """Tests H2O volume at 1873.15 K and 2500 MPa :cite:t:`ZD09{Table 6}`"""
    expected: float = 19.41089977577485
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1873.15,
        25000,  # 2500 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


def test_H2O_volume_high_TP2(check_values) -> None:
    """Tests H2O volume at 1373.15 K and 3500 MPa :cite:t:`ZD09{Table 6}`"""
    expected: float = 16.02290245403692
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(
        1373.15,
        35000,  # 3500 MPa
        check_values.get_eos_model("H2O", MODEL_SUFFIX),
        expected,
        rtol=RTOL,
        atol=ATOL,
    )


# def test_volume_with_broadcasting(check_values) -> None:
#     """Tests volume with broadcasting"""
#     model: RealGas = check_values.get_eos_model("H2O", MODEL_SUFFIX)
#     check_values.check_broadcasting("volume", model)


# def test_fugacity_with_broadcasting(check_values) -> None:
#     """Tests volume with broadcasting"""
#     model: RealGas = check_values.get_eos_model("H2O", MODEL_SUFFIX)
#     check_values.check_broadcasting("fugacity", model)
