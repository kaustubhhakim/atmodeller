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
from typing import Callable

from scipy.constants import kilo

from atmodeller import ATMOSPHERE
from atmodeller.eos.classes import BeattieBridgeman, Chabrier
from atmodeller.eos.core import CORK, RealGas, RealGasBounded, VirialCompensation
from atmodeller.eos.holland import (
    CO2MrkHolland91,
    H2OMrkFluidHolland91,
    H2OMrkGasHolland91,
    H2OMrkHolland91,
    H2OMrkLiquidHolland91,
    MRKCorrespondingStatesHP91,
)
from atmodeller.thermodata.core import CriticalData
from atmodeller.thermodata.library import select_critical_data
from atmodeller.utilities import ExperimentalCalibrationNew, unit_conversion

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

# region Holland and Powell (1991, 2011)

CO2_mrk_cs_holland91: RealGas = MRKCorrespondingStatesHP91.get_species("CO2_g")
"""CO2 MRK corresponding states :cite:p:`HP91`"""
CH4_mrk_cs_holland91: RealGas = MRKCorrespondingStatesHP91.get_species("CH4_g")
"""CH4 MRK corresponding states :cite:p:`HP91`"""
H2_mrk_cs_holland91: RealGas = MRKCorrespondingStatesHP91.get_species("H2_g_Holland")
"""H2 MRK corresponding states :cite:p:`HP91`"""
CO_mrk_cs_holland91: RealGas = MRKCorrespondingStatesHP91.get_species("CO_g")
"""CO MRK corresponding states :cite:p:`HP91`"""
N2_mrk_cs_holland91: RealGas = MRKCorrespondingStatesHP91.get_species("N2_g")
"""N2 MRK corresponding states :cite:p:`HP91`"""
S2_mrk_cs_holland11: RealGas = MRKCorrespondingStatesHP91.get_species("S2_g")
"""S2 MRK corresponding states :cite:p:`HP11`"""
H2S_mrk_cs_holland11: RealGas = MRKCorrespondingStatesHP91.get_species("H2S_g")
"""H2S MRK corresponding states :cite:p:`HP11`"""
H2O_mrk_fluid_holland91: RealGas = H2OMrkFluidHolland91
"""H2O MRK supercritical fluid :cite:p:`HP91`"""
H2O_mrk_gas_holland91: RealGas = H2OMrkGasHolland91
"""H2O MRK gas :cite:p:`HP91`"""
H2O_mrk_liquid_holland91: RealGas = H2OMrkLiquidHolland91
"""H2O MRK liquid :cite:p:`HP91`"""

virial_compensation: VirialCompensation = VirialCompensation(
    (6.93054e-9, -8.38293e-10), (-3.30558e-7, 2.30524e-8), (0, 0), 0
)
"""Virial compensation for corresponding states :cite:p:`HP91{Table 2}`

In this case it appears `P0` is always zero, even though for the full CORK equations it determines
whether or not the virial contribution is added. The unit conversions to SI and pressure in bar 
mean that every virial coefficient has been multiplied by 1e-2 compared to the values in 
:cite:t:`HP91{Table 2}`.
"""
experimental_calibration_holland91: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
    100, 4000, 0.1, 50e9
)
"""Experimental calibration for :cite:`HP91,HP11` models"""

CH4_cork_cs_holland91: RealGas = CORK(
    CH4_mrk_cs_holland91, virial_compensation, select_critical_data("CH4_g")
)
"""CH4 CORK corresponding states :cite:p:`HP91`"""
CH4_cork_cs_holland91_bounded: RealGas = RealGasBounded(
    CH4_cork_cs_holland91, experimental_calibration_holland91
)
"""CH4 CORK corresponding states bounded :cite:p:`HP91`"""
CO_cork_cs_holland91: RealGas = CORK(
    CO_mrk_cs_holland91, virial_compensation, select_critical_data("CO_g")
)
"""CO CORK corresponding states :cite:p:`HP91`"""
CO_cork_cs_holland91_bounded: RealGas = RealGasBounded(
    CO_cork_cs_holland91, experimental_calibration_holland91
)
"""CO CORK corresponding states bounded :cite:p:`HP91`"""
CO2_cork_cs_holland91: RealGas = CORK(
    CO2_mrk_cs_holland91, virial_compensation, select_critical_data("CO2_g")
)
"""CO2 CORK corresponding states :cite:p:`HP91`"""
CO2_cork_cs_holland91_bounded: RealGas = RealGasBounded(
    CO2_cork_cs_holland91, experimental_calibration_holland91
)
"""CO2 CORK corresponding states bounded :cite:p:`HP91`"""
H2_cork_cs_holland91: RealGas = CORK(
    H2_mrk_cs_holland91, virial_compensation, select_critical_data("H2_g_Holland")
)
"""H2 CORK corresponding states :cite:p:`HP91`"""
H2_cork_cs_holland91_bounded: RealGas = RealGasBounded(
    H2_cork_cs_holland91, experimental_calibration_holland91
)
"""H2 CORK corresponding states bounded :cite:p:`HP91`"""
H2S_cork_cs_holland11: RealGas = CORK(
    H2S_mrk_cs_holland11, virial_compensation, select_critical_data("H2S_g")
)
"""H2S CORK corresponding states :cite:p:`HP91`"""
H2S_cork_cs_holland11_bounded: RealGas = RealGasBounded(
    H2S_cork_cs_holland11, experimental_calibration_holland91
)
"""H2S CORK corresponding states bounded :cite:p:`HP91`"""
N2_cork_cs_holland91: RealGas = CORK(
    N2_mrk_cs_holland91, virial_compensation, select_critical_data("N2_g")
)
"""N2 CORK corresponding states :cite:p:`HP91`"""
N2_cork_cs_holland91_bounded: RealGas = RealGasBounded(
    N2_cork_cs_holland91, experimental_calibration_holland91
)
"""N2 CORK corresponding states bounded :cite:p:`HP91`"""
S2_cork_cs_holland11: RealGas = CORK(
    S2_mrk_cs_holland11, virial_compensation, select_critical_data("S2_g")
)
"""S2 CORK corresponding states :cite:p:`HP91`"""
S2_cork_cs_holland11_bounded: RealGas = RealGasBounded(
    S2_cork_cs_holland11, experimental_calibration_holland91
)
"""S2 CORK corresponding states bounded :cite:p:`HP91`"""

# For the Full CORK models, the virial coefficients in the Holland and Powell papers need
# converting to SI units and pressure in bar as follows, where k = kilo = 1000:
#   a_virial = a_virial (Holland and Powell) * 10**(-5) / k
#   b_virial = b_virial (Holland and Powell) * 10**(-5) / k**(1/2)
#   c_virial = c_virial (Holland and Powell) * 10**(-5) / k**(1/4)

_a_conversion: Callable[[tuple[float, ...]], tuple[float, ...]] = lambda x: tuple(  # noqa: E731
    [y * 1e-5 / kilo for y in x]
)
_b_conversion: Callable[[tuple[float, ...]], tuple[float, ...]] = lambda x: tuple(  # noqa: E731
    [y * 1e-5 / kilo**0.5 for y in x]
)
_c_conversion: Callable[[tuple[float, ...]], tuple[float, ...]] = lambda x: tuple(  # noqa: E731
    [y * 1e-5 / kilo**0.25 for y in x]
)

dummy_critical_data: CriticalData = CriticalData(1.0, 1.0)
"""Dummy critical data

The full CO2 and H2O CORK models are not corresponding states, which can be reproduced by ignoring
the scaling by the critical temperature and pressure, i.e. setting these quantities to unity.
"""
_CO2_virial_compensation_holland91: VirialCompensation = VirialCompensation(
    _a_conversion((1.33790e-2, -1.01740e-5)),
    _b_conversion((-2.26924e-1, 7.73793e-5)),
    (0, 0),
    5000,
)
"""CO2 virial compensation :cite:p:`HP91`"""
CO2_cork_holland91: RealGas = CORK(
    CO2MrkHolland91, _CO2_virial_compensation_holland91, dummy_critical_data
)
"""CO2 cork :cite:p:`HP91`

TODO: ExperimentalCalibrationNew(400, 1900, 0, 50e3)
"""
_H2O_virial_compensation_holland91: VirialCompensation = VirialCompensation(
    _a_conversion((-3.2297554e-3, 2.2215221e-6)),
    _b_conversion((-3.025650e-2, -5.343144e-6)),
    (0, 0),
    2000,
)
"""H2O virial compensation :cite:p:`HP91`"""
H2O_cork_holland91: RealGas = CORK(
    H2OMrkHolland91, _H2O_virial_compensation_holland91, dummy_critical_data
)
"""H2O cork :cite:p:`HP91`

TODO: calibration=ExperimentalCalibration(400, 1700, 0, 50e3),
"""

_CO2_virial_compensation_holland98: VirialCompensation = VirialCompensation(
    _a_conversion((5.40776e-3, -1.59046e-6)),
    _b_conversion((-1.78198e-1, 2.45317e-5)),
    (0, 0),
    5000,
)
"""CO2 virial compensation :cite:p:`HP98`"""
CO2_cork_holland98: RealGas = CORK(
    CO2MrkHolland91, _CO2_virial_compensation_holland98, dummy_critical_data
)
"""CO2 cork :cite:p:`HP98`

TODO: calibration=ExperimentalCalibration(400, 1900, 0, 120e3),
"""
_H2O_virial_compensation_holland98: VirialCompensation = VirialCompensation(
    _a_conversion((1.9853e-3, 0)),
    _b_conversion((-8.9090e-2, 0)),
    _c_conversion((8.0331e-2, 0)),
    2000,
)
H2O_cork_holland98: RealGas = CORK(
    H2OMrkHolland91, _H2O_virial_compensation_holland98, dummy_critical_data
)
"""H2O cork :cite:p:`HP98`

TODO: calibration=ExperimentalCalibration(400, 1700, 0, 120e3),
"""


# end region

# region: Holley et al. (1958)

# Coefficients from Table I, which must be converted to the correct units scheme (SI and pressure
# in bar). Using the original values in the paper also facilitates visual comparison and checking.


def volume_conversion(x: float) -> float:
    """Volume conversion for :cite:t:`HWZ58` units."""
    return x * unit_conversion.litre_to_m3


def A0_conversion(x: float) -> float:
    """PV**2 conversion for :cite:t:`HWZ58` units."""
    return x * ATMOSPHERE * unit_conversion.litre_to_m3**2


def atm2bar(x: float) -> float:
    """Atmosphere to bar conversion."""
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
    eos_models["CH4_beattie_holley58"] = CH4_beattie_holley58
    eos_models["CH4_beattie_holley58_bounded"] = CH4_beattie_holley58_bounded
    eos_models["CH4_cork_cs_holland91"] = CH4_cork_cs_holland91
    eos_models["CH4_cork_cs_holland91_bounded"] = CH4_cork_cs_holland91_bounded
    eos_models["CH4_mrk_cs_holland91"] = CH4_mrk_cs_holland91
    eos_models["CO_cork_cs_holland91"] = CO_cork_cs_holland91
    eos_models["CO_cork_cs_holland91_bounded"] = CO_cork_cs_holland91_bounded
    eos_models["CO_mrk_cs_holland91"] = CO_mrk_cs_holland91
    eos_models["CO2_beattie_holley58"] = CO2_beattie_holley58
    eos_models["CO2_beattie_holley58_bounded"] = CO2_beattie_holley58_bounded
    eos_models["CO2_cork_holland91"] = CO2_cork_holland91
    eos_models["CO2_cork_holland98"] = CO2_cork_holland98
    eos_models["CO2_cork_cs_holland91"] = CO2_cork_cs_holland91
    eos_models["CO2_cork_cs_holland91_bounded"] = CO2_cork_cs_holland91_bounded
    eos_models["CO2_mrk_cs_holland91"] = CO2_mrk_cs_holland91
    eos_models["CO2_mrk_holland91"] = CO2MrkHolland91
    eos_models["H2_beattie_holley58"] = H2_beattie_holley58
    eos_models["H2_beattie_holley58_bounded"] = H2_beattie_holley58_bounded
    eos_models["H2_chabrier21"] = H2_chabrier21
    eos_models["H2_chabrier21_bounded"] = H2_chabrier21_bounded
    eos_models["H2_cork_cs_holland91"] = H2_cork_cs_holland91
    eos_models["H2_cork_cs_holland91_bounded"] = H2_cork_cs_holland91_bounded
    eos_models["H2_mrk_cs_holland91"] = H2_mrk_cs_holland91
    eos_models["H2_He_Y0275_chabrier21"] = H2_He_Y0275_chabrier21
    eos_models["H2_He_Y0292_chabrier21"] = H2_He_Y0292_chabrier21
    eos_models["H2_He_Y0297_chabrier21"] = H2_He_Y0297_chabrier21
    eos_models["H2O_cork_holland91"] = H2O_cork_holland91
    eos_models["H2O_cork_holland98"] = H2O_cork_holland98
    eos_models["H2O_mrk_holland91"] = H2OMrkHolland91
    # Supercritical fluid only
    eos_models["H2O_mrk_fluid_holland91"] = H2O_mrk_fluid_holland91
    # Gas (subcritical) only
    eos_models["H2O_mrk_gas_holland91"] = H2O_mrk_gas_holland91
    # Eventually it might make sense to include the liquid as a condensed activity model
    eos_models["H2O_mrk_liquid_holland91"] = H2O_mrk_liquid_holland91
    eos_models["H2S_cork_cs_holland11"] = H2S_cork_cs_holland11
    eos_models["H2S_cork_cs_holland11_bounded"] = H2S_cork_cs_holland11_bounded
    eos_models["H2S_mrk_cs_holland11"] = H2S_mrk_cs_holland11
    eos_models["He_beattie_holley58"] = He_beattie_holley58
    eos_models["He_beattie_holley58_bounded"] = He_beattie_holley58_bounded
    eos_models["He_chabrier21"] = He_chabrier21
    eos_models["N2_beattie_holley58"] = N2_beattie_holley58
    eos_models["N2_beattie_holley58_bounded"] = N2_beattie_holley58_bounded
    eos_models["N2_cork_cs_holland91"] = N2_cork_cs_holland91
    eos_models["N2_cork_cs_holland91_bounded"] = N2_cork_cs_holland91_bounded
    eos_models["N2_mrk_cs_holland91"] = N2_mrk_cs_holland91
    eos_models["NH3_beattie_holley58"] = NH3_beattie_holley58
    eos_models["NH3_beattie_holley58_bounded"] = NH3_beattie_holley58_bounded
    eos_models["O2_beattie_holley58"] = O2_beattie_holley58
    eos_models["O2_beattie_holley58_bounded"] = O2_beattie_holley58_bounded
    eos_models["S2_cork_cs_holland11"] = S2_cork_cs_holland11
    eos_models["S2_cork_cs_holland11_bounded"] = S2_cork_cs_holland11_bounded
    eos_models["S2_mrk_cs_holland11"] = S2_mrk_cs_holland11

    return eos_models
