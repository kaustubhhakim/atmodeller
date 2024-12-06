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
"""Tests for the MRK EOS models from :cite:t:`HP91,HP98`"""

import logging

from atmodeller import debug_logger
from atmodeller.eos import RealGas, get_eos_models

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()

eos_models: dict[str, RealGas] = get_eos_models()


def test_CO2_corresponding_states(check_values) -> None:
    """CO2 corresponding states"""
    model: RealGas = eos_models["CO2_mrk_cs_holland91"]
    check_values.fugacity_coefficient(2000, 2e3, model, 1.5831992703027848, rtol=RTOL, atol=ATOL)


def test_CO2(check_values) -> None:
    """CO2"""
    model: RealGas = eos_models["CO2_mrk_holland91"]
    check_values.fugacity_coefficient(2000, 2e3, model, 1.575457075165528, rtol=RTOL, atol=ATOL)


def test_H2O_above_Tc(check_values) -> None:
    """H2O above Tc"""
    model: RealGas = eos_models["H2O_mrk_fluid_holland91"]
    check_values.fugacity_coefficient(2000, 1e3, model, 1.048278616058322, rtol=RTOL, atol=ATOL)


def test_H2O_below_Tc_below_Psat(check_values) -> None:
    """H2O below Tc and below Psat"""
    # Psat = 0.118224 kbar at T = 600 K
    model: RealGas = eos_models["H2O_mrk_gas_holland91"]
    check_values.fugacity_coefficient(600, 0.1e3, model, 0.7910907770688191, rtol=RTOL, atol=ATOL)


def test_H2O_below_Tc_above_Psat(check_values) -> None:
    """H2O below Tc and above Psat"""
    # Psat = 0.118224 kbar at T = 600 K
    check_values.fugacity_coefficient(
        600, 1e3, eos_models["H2O_mrk_holland91"], 0.13704706029361396, rtol=RTOL, atol=ATOL
    )
