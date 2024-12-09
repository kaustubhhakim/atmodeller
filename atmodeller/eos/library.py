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
"""Real gas EOS library

Usage:
    from atmodeller.eos.library import get_eos_models
    eos_models = get_eos_models()
    CH4_beattie = eos_models["CH4_beattie_holley58"]
    # Evaluate fugacity at 10 bar and 800 K
    fugacity = CH4_beattie.fugacity(800, 10)
    print(fugacity)
"""

import logging
from pathlib import Path

from atmodeller.eos._holland_powell import get_holland_eos_models
from atmodeller.eos._holley import get_holley_eos_models
from atmodeller.eos.classes import Chabrier
from atmodeller.eos.core import RealGas, RealGasBounded
from atmodeller.utilities import ExperimentalCalibrationNew

logger: logging.Logger = logging.getLogger(__name__)

# region: Chabrier et al. (2021)

H2_chabrier21: RealGas = Chabrier(Path("TABLE_H_TP_v1"))
"""H2 Chabrier :cite:p:`CD21`"""
# TODO: Update calibration bounds. Kaustubh to do.
H2_chabrier21_bounded: RealGas = RealGasBounded(
    H2_chabrier21, ExperimentalCalibrationNew(100, 4000, 0.1, 50e9)
)
He_chabrier21: RealGas = Chabrier(Path("TABLE_HE_TP_v1"))
"""He :cite:p:`CD21`"""
H2_He_Y0275_chabrier21: RealGas = Chabrier(Path("TABLEEOS_2021_TP_Y0275_v1"))
"""H2HeY0275 :cite:p:`CD21`"""
H2_He_Y0292_chabrier21: RealGas = Chabrier(Path("TABLEEOS_2021_TP_Y0292_v1"))
"""H2HeY0292 :cite:p:`CD21`"""
H2_He_Y0297_chabrier21: RealGas = Chabrier(Path("TABLEEOS_2021_TP_Y0297_v1"))
"""H2HeY0297 :cite:p:`CD21`"""

# endregion


def get_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of EOS models

    The naming convention is as follows:
        [species]_[eos model]_[citation], with an optional suffix of 'bounded'

    'cs' refers to corresponding states and `bounded` means that the EOS is reasonably well-behaved
    outside its calibrated range to mitigate the solver throwing inf/nans.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["H2_chabrier21"] = H2_chabrier21
    eos_models["H2_chabrier21_bounded"] = H2_chabrier21_bounded
    eos_models["H2_He_Y0275_chabrier21"] = H2_He_Y0275_chabrier21
    eos_models["H2_He_Y0292_chabrier21"] = H2_He_Y0292_chabrier21
    eos_models["H2_He_Y0297_chabrier21"] = H2_He_Y0297_chabrier21
    eos_models["He_chabrier21"] = He_chabrier21

    # Merge Holley models
    eos_models |= get_holley_eos_models()
    # Merge Holland and Powell models
    eos_models |= get_holland_eos_models()

    return eos_models
