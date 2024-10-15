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
"""Tests for the EOS models from :cite:t:`HP91,HP98`"""

# Convenient to use acroynms and symbol names so pylint: disable=invalid-name

from __future__ import annotations

import logging

from atmodeller import debug_logger
from atmodeller.eos.holland import (
    CH4_CORK_HP91,
    CO2_CORK_HP91,
    CO_CORK_HP91,
    H2_CORK_HP91,
    H2O_CORK_HP91,
    CO2_CORK_simple_HP91,
    CO2_MRK_simple_HP91,
    get_holland_eos_models,
)
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import unit_conversion

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_holland_eos_models()
"""EOS models from :cite:t:`HP91,HP98`"""


def test_CORK_H2O_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 2a}`"""
    expected: float = 47.502083040419844
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(873, 1000, H2O_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CO2_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 7}`"""
    expected: float = 96.13326116472262
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(873, 1000, CO2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CO_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8a}`"""
    expected: float = 131.475184896045
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(1173, 1000, CO_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CO_volume_2kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8a}`"""
    expected: float = 71.32153159834933
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(973, 2000, CO_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CO_volume_4kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8a}`"""
    expected: float = 62.22167162862537
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(1473, 4000, CO_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CH4_volume_1kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8b}`"""
    expected: float = 131.6743085645421
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(1173, 1000, CH4_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CH4_volume_2kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8b}`"""
    expected: float = 72.14376119913776
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(973, 2000, CH4_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_CH4_volume_4kbar(check_values) -> None:
    """:cite:t:`HP91{Figure 8b}`"""
    expected: float = 63.106094264549
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(1473, 4000, CH4_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_H2_volume_500bar(check_values) -> None:
    """:cite:t:`HP91{Figure 8c}`"""
    expected: float = 149.1657987388235
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(773, 500, H2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_H2_volume_1800bar(check_values) -> None:
    """:cite:t:`HP91{Figure 8c}`"""
    expected: float = 55.04174839002075
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(773, 1800, H2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_CORK_H2_volume_10kb(check_values) -> None:
    """:cite:t:`HP91{Figure 8c}`"""
    expected: float = 20.67497630046999
    expected *= unit_conversion.cm3_to_m3
    check_values.volume(773, 10000, H2_CORK_HP91, expected, rtol=RTOL, atol=ATOL)


def test_MRKCO2(check_values) -> None:
    """MRK CO2"""
    check_values.fugacity_coefficient(
        2000, 10e3, CO2_MRK_simple_HP91, 9.80535714428564, rtol=RTOL, atol=ATOL
    )


def test_CorkH2(check_values) -> None:
    """CORK H2"""
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["H2"], 4.672042007568433, rtol=RTOL, atol=ATOL
    )


def test_CorkCO(check_values) -> None:
    """CORK CO"""
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["CO"], 7.737070657107842, rtol=RTOL, atol=ATOL
    )


def test_CorkCH4(check_values) -> None:
    """CORK CH4"""
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["CH4"], 8.013532244610671, rtol=RTOL, atol=ATOL
    )


def test_simple_CorkCO2(check_values) -> None:
    """Simple CORK CO2"""
    check_values.fugacity_coefficient(
        2000, 10e3, CO2_CORK_simple_HP91, 7.120242298956865, rtol=RTOL, atol=ATOL
    )


def test_CorkCO2_at_P0(check_values) -> None:
    """CORK CO2 below P0 so virial contribution excluded"""
    check_values.fugacity_coefficient(
        2000, 2e3, eos_models["CO2"], 1.5754570751655304, rtol=RTOL, atol=ATOL
    )


def test_CorkCO2_above_P0(check_values) -> None:
    """CORK CO2 above P0 so virial contribution included"""
    check_values.fugacity_coefficient(
        2000, 10e3, eos_models["CO2"], 7.144759853226838, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_above_Tc_below_P0(check_values) -> None:
    """CORK H2O above Tc and below P0"""
    check_values.fugacity_coefficient(
        2000, 1e3, eos_models["H2O"], 1.048278616058322, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_above_Tc_above_P0(check_values) -> None:
    """CORK H2O above Tc and above P0"""
    check_values.fugacity_coefficient(
        2000, 5e3, eos_models["H2O"], 1.3444013638026706, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_below_Tc_below_Psat(check_values) -> None:
    """CORK H2O below Tc and below Psat"""
    # Psat = 0.118224 kbar at T = 600 K
    check_values.fugacity_coefficient(
        600, 0.1e3, eos_models["H2O"], 0.7910907770688191, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_below_Tc_above_Psat(check_values) -> None:
    """CORK H2O below Tc and above Psat"""
    # Psat = 0.118224 kbar at T = 600 K
    check_values.fugacity_coefficient(
        600, 1e3, eos_models["H2O"], 0.13704706029361396, rtol=RTOL, atol=ATOL
    )


def test_CorkH2O_below_Tc_above_P0(check_values) -> None:
    """CORK H2O below Tc and above P0"""
    check_values.fugacity_coefficient(
        600, 10e3, eos_models["H2O"], 0.39074941260585533, rtol=RTOL, atol=ATOL
    )
