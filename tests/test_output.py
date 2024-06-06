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
"""Tests for the output"""

# Want to use chemistry symbols so pylint: disable=invalid-name

from __future__ import annotations

import logging

import pytest

from atmodeller import __version__, debug_logger

logger: logging.Logger = debug_logger()

TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage"""


def test_graphite_water_condensed_output(graphite_water_condensed) -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    system = graphite_water_condensed

    output = system.output(to_dict=True)

    assert 9.89452e18 == pytest.approx(output["C_totals"][0]["atmosphere_mass"])
    assert 9.81055e19 == pytest.approx(output["C_totals"][0]["condensed_mass"])
    assert 3.9873e19 == pytest.approx(output["O_totals"][0]["atmosphere_mass"])
    assert 2.4431158e21 == pytest.approx(output["O_totals"][0]["condensed_mass"])
    assert 2.17398e18 == pytest.approx(output["H_totals"][0]["atmosphere_mass"])
    assert 3.07826e20 == pytest.approx(output["H_totals"][0]["condensed_mass"])
