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
"""Tests for the EOS models from :cite:t:`Lide2005`"""

from atmodeller.eos import RealGas

MODEL_SUFFIX: str = "vdw_lide05"
"""Suffix of the :cite:t:`Lide2005` models"""


def test_H2_volume_100bar(check_values) -> None:
    """Tests H2 volume at 300 K and 100 bar"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.volume(300, 100, model, 2.85325715e-4)


def test_H2_volume_1kbar(check_values) -> None:
    """Tests H2 volume at 1000 K and 1000 bar"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.volume(1000, 1000, model, 1.12342095e-4)


def test_volume_with_broadcasting(check_values) -> None:
    """Tests volume with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.check_broadcasting("volume", model)


def test_fugacity_with_broadcasting(check_values) -> None:
    """Tests volume with broadcasting"""
    model: RealGas = check_values.get_eos_model("H2", MODEL_SUFFIX)
    check_values.check_broadcasting("fugacity", model)
