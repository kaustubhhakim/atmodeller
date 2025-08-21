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
"""Tests for the RK49 EOS models :cite:p:`RPP77,C16`"""

from atmodeller.eos import RealGas
from atmodeller.eos._reid_connolly import (
    H3N_rk49_reid87,
    OSi_rk49_connolly16,
    OSi_rk49_connolly16_bounded,
)


def test_OSi_fugacity_coefficient(check_values) -> None:
    """OSi fugacity coefficient"""
    model: RealGas = OSi_rk49_connolly16_bounded
    expected: float = 0.460408741003706
    check_values.fugacity_coefficient(3000, 10e3, model, expected)


def test_OSi_RK49_a(check_values) -> None:
    """OSi RK49 parameter `a` :cite:p:`C16{Table 2}`"""
    model: RealGas = OSi_rk49_connolly16
    expected: float = 44.0972e-5  # published b = 44.0972 J / (Pa mol)
    check_values._check_property("a", 300, 1, model, expected, atol=1e-4, rtol=1e-4)


def test_OSi_RK49_b(check_values) -> None:
    """OSi RK49 parameter `b` :cite:p:`C16{Table 2}`"""
    model: RealGas = OSi_rk49_connolly16
    expected: float = 5.34948e-6  # published a = 5.34948 J / (MPa mol)
    check_values._check_property("a", 300, 1, model, expected, atol=1e-3, rtol=1e-3)


def test_H3N_fugacity_coefficient(check_values) -> None:
    """H3N fugacity coefficient"""
    model: RealGas = H3N_rk49_reid87
    expected: float = 0.867380982251227
    check_values.fugacity_coefficient(500, 100, model, expected)
