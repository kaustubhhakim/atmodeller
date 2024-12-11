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
"""Tests for combining EOS models"""

import logging

from atmodeller import debug_logger
from atmodeller.eos import IdealGas, get_eos_models
from atmodeller.eos.aggregators import CombinedRealGas, CombinedRealGasRemoveSteps
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibration

logger: logging.Logger = debug_logger()

eos_models = get_eos_models()


def test_bounded() -> None:
    """TODO"""

    model0: RealGasProtocol = IdealGas()
    model1: RealGasProtocol = eos_models["CH4_cork_cs_holland91"]
    models = [model0, model1]

    experimental_calibration_holland91: ExperimentalCalibration = ExperimentalCalibration(
        100, 4000, 0.1, 50e3
    )
    calibrations = [None, experimental_calibration_holland91]

    a = CombinedRealGas(models, calibrations)
    a = CombinedRealGasRemoveSteps(models, calibrations)

    b = a._get_index(0.01)
    logger.debug("b = %s", b)
    b = a.volume_integral(1000, 0.01)
    logger.debug("b = %s", b)
    fugacity = a.fugacity(1000, 0.01)
    logger.debug("fugacity = %f", fugacity)

    c = a._get_index(1)
    logger.debug("c = %s", c)
    c = a.volume_integral(1000, 10)
    logger.debug("c = %s", c)
    d = a.fugacity(1000, 10)
    logger.debug("fugacity = %f", d)
