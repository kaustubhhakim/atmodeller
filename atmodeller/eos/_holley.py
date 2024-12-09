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
"""Real gas EOS from :cite:`HWZ58"""

import logging

from atmodeller.constants import ATMOSPHERE
from atmodeller.eos.classes import BeattieBridgeman
from atmodeller.eos.core import RealGas, RealGasBounded
from atmodeller.interfaces import RealGasProtocol
from atmodeller.utilities import ExperimentalCalibrationNew, unit_conversion

logger: logging.Logger = logging.getLogger(__name__)

# Coefficients from Table I, which must be converted to the correct units scheme (SI and pressure
# in bar). Using the original values in the paper also facilitates visual comparison and checking.


def volume_conversion(x: float) -> float:
    """Volume conversion for :cite:t:`HWZ58` units"""
    return x * unit_conversion.litre_to_m3


def A0_conversion(x: float) -> float:
    """PV**2 conversion for :cite:t:`HWZ58` units"""
    return x * ATMOSPHERE * unit_conversion.litre_to_m3**2


def atm2bar(x: float) -> float:
    """Atmosphere to bar conversion"""
    return unit_conversion.atmosphere_to_bar * x


H2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.1975),
    a=volume_conversion(-0.00506),
    B0=volume_conversion(0.02096),
    b=volume_conversion(-0.04359),
    c=volume_conversion(0.0504e4),
)
"""H2 Beattie-Bridgeman :cite:p:`HWZ58`"""
H2_beattie_holley58_bounded: RealGas = RealGasBounded(
    H2_beattie_holley58,
    ExperimentalCalibrationNew(100, 1000, atm2bar(0.1), atm2bar(1000)),
)
"""H2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

N2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.3445),
    a=volume_conversion(0.02617),
    B0=volume_conversion(0.05046),
    b=volume_conversion(-0.00691),
    c=volume_conversion(4.2e4),
)
"""N2 Beattie-Bridgeman :cite:p:`HWZ58`"""
N2_beattie_holley58_bounded: RealGas = RealGasBounded(
    N2_beattie_holley58, ExperimentalCalibrationNew(200, 1000, atm2bar(0.1), atm2bar(1000))
)
"""N2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

O2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(1.4911),
    a=volume_conversion(0.02562),
    B0=volume_conversion(0.04624),
    b=volume_conversion(0.004208),
    c=volume_conversion(4.8e4),
)
"""O2 Beattie-Bridgeman :cite:p:`HWZ58`"""
O2_beattie_holley58_bounded: RealGas = RealGasBounded(
    O2_beattie_holley58, ExperimentalCalibrationNew(200, 1000, atm2bar(0.1), atm2bar(1000))
)
"""O2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

CO2_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(5.0065),
    a=volume_conversion(0.07132),
    B0=volume_conversion(0.10476),
    b=volume_conversion(0.07235),
    c=volume_conversion(66e4),
)
"""CO2 Beattie-Bridgeman :cite:p:`HWZ58`"""
CO2_beattie_holley58_bounded: RealGas = RealGasBounded(
    CO2_beattie_holley58,
    ExperimentalCalibrationNew(400, 1000, atm2bar(0.1), atm2bar(1000)),
)
"""CO2 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

NH3_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.3930),
    a=volume_conversion(0.17031),
    B0=volume_conversion(0.03415),
    b=volume_conversion(0.19112),
    c=volume_conversion(476.87e4),
)
"""NH3 Beattie-Bridgeman :cite:p:`HWZ58`"""
NH3_beattie_holley58_bounded: RealGas = RealGasBounded(
    NH3_beattie_holley58, ExperimentalCalibrationNew(500, 1000, atm2bar(0.1), atm2bar(500))
)
"""NH3 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

CH4_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(2.2769),
    a=volume_conversion(0.01855),
    B0=volume_conversion(0.05587),
    b=volume_conversion(-0.01587),
    c=volume_conversion(12.83e4),
)
"""CH4 Beattie-Bridgeman :cite:p:`HWZ58`"""
CH4_beattie_holley58_bounded: RealGas = RealGasBounded(
    CH4_beattie_holley58,
    ExperimentalCalibrationNew(200, 1000, atm2bar(0.1), atm2bar(1000)),
)
"""CH4 Beattie-Bridgeman bounded :cite:p:`HWZ58`"""

He_beattie_holley58: RealGas = BeattieBridgeman(
    A0=A0_conversion(0.0216),
    a=volume_conversion(0.05984),
    B0=volume_conversion(0.01400),
    b=0,
    c=volume_conversion(0.004e4),
)
"""He Beattie-Bridgeman :cite:p:`HWZ58`"""
He_beattie_holley58_bounded: RealGas = RealGasBounded(
    He_beattie_holley58,
    ExperimentalCalibrationNew(100, 1000, atm2bar(0.1), atm2bar(1000)),
)
"""He Beattie-Bridgeman bounded :cite:p:`HWZ58`"""


def get_holley_eos_models() -> dict[str, RealGasProtocol]:
    """Gets a dictionary of Holley models

    The naming convention is as follows:
        [species]_[eos model]_[citation], with an optional suffix of 'bounded'

    'cs' refers to corresponding states and `bounded` means that the EOS is reasonably well-behaved
    outside its calibrated range to mitigate the solver throwing inf/nans.

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGasProtocol] = {}
    eos_models["CH4_beattie_holley58"] = CH4_beattie_holley58
    eos_models["CH4_beattie_holley58_bounded"] = CH4_beattie_holley58_bounded
    eos_models["CO2_beattie_holley58"] = CO2_beattie_holley58
    eos_models["CO2_beattie_holley58_bounded"] = CO2_beattie_holley58_bounded
    eos_models["H2_beattie_holley58"] = H2_beattie_holley58
    eos_models["H2_beattie_holley58_bounded"] = H2_beattie_holley58_bounded
    eos_models["He_beattie_holley58"] = He_beattie_holley58
    eos_models["He_beattie_holley58_bounded"] = He_beattie_holley58_bounded
    eos_models["N2_beattie_holley58"] = N2_beattie_holley58
    eos_models["N2_beattie_holley58_bounded"] = N2_beattie_holley58_bounded
    eos_models["NH3_beattie_holley58"] = NH3_beattie_holley58
    eos_models["NH3_beattie_holley58_bounded"] = NH3_beattie_holley58_bounded
    eos_models["O2_beattie_holley58"] = O2_beattie_holley58
    eos_models["O2_beattie_holley58_bounded"] = O2_beattie_holley58_bounded

    return eos_models
