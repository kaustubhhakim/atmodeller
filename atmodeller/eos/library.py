#!/usr/bin/env python3
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
"""Real gas EOS library built from the concrete classes"""

import logging
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from atmodeller import ATMOSPHERE
from atmodeller.eos.classes import BeattieBridgeman, Chabrier
from atmodeller.eos.core import RealGas, RealGasBounded
from atmodeller.utilities import ExperimentalCalibrationNew, unit_conversion

logger: logging.Logger = logging.getLogger(__name__)

# region: Chabrier et al. (2021)

H2_chabrier21: RealGas = Chabrier(Path("TABLE_H_TP_v1"), "H2")
"""H2 Chabrier :cite:p:`CD21`"""

# endregion

# region: Holley et al. (1958)

# Coefficients from Table I, which must be converted to the correct units scheme (SI and pressure
# in bar). Using the original values in the paper also facilitates visual comparison and checking.

volume_conversion: Callable = lambda x: x * unit_conversion.litre_to_m3
"""Volume conversion for :cite:t:`HWZ58` units"""
A0_conversion: Callable = lambda x: x * ATMOSPHERE * unit_conversion.litre_to_m3**2
"""PV**2 conversion for :cite:t:`HWZ58` units"""
atm2bar: Callable = lambda x: unit_conversion.atmosphere_to_bar * x
"""Atmosphere to bar"""

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

    Returns:
        Dictionary of EOS models
    """
    models: dict[str, RealGas] = {}
    models["CH4_beattie_holley58"] = CH4_beattie_holley58
    models["CH4_beattie_holley58_bounded"] = CH4_beattie_holley58_bounded
    models["CO2_beattie_holley58"] = CO2_beattie_holley58
    models["CO2_beattie_holley58_bounded"] = CO2_beattie_holley58_bounded
    models["H2_beattie_holley58"] = H2_beattie_holley58
    models["H2_beattie_holley58_bounded"] = H2_beattie_holley58_bounded
    models["H2_chabrier21"] = H2_chabrier21
    models["He_beattie_holley58"] = He_beattie_holley58
    models["He_beattie_holley58_bounded"] = He_beattie_holley58_bounded
    models["N2_beattie_holley58"] = N2_beattie_holley58
    models["N2_beattie_holley58_bounded"] = N2_beattie_holley58_bounded
    models["NH3_beattie_holley58"] = NH3_beattie_holley58
    models["NH3_beattie_holley58_bounded"] = NH3_beattie_holley58_bounded
    models["O2_beattie_holley58"] = O2_beattie_holley58
    models["O2_beattie_holley58_bounded"] = O2_beattie_holley58_bounded

    return models


if __name__ == "__main__":

    model = get_eos_models()["H2_beattie_holley58_bounded"]

    pressures = np.arange(1, 3000, 100)
    temperatures = 1000.0 * np.ones_like(pressures)

    temperature_out = []
    ideal_volume_out = []
    volume_out = []
    fugacity_out = []
    compressibility_factor_out = []
    fugacity_coefficient_out = []

    for nn, pressure in enumerate(pressures):
        temperature = temperatures[nn]
        ideal_volume = model.ideal_volume(temperature, pressure)
        volume = model.volume(temperature, pressure)
        fugacity = model.fugacity(temperature, pressure)
        compressibility_factor = model.compressibility_factor(temperature, pressure)
        fugacity_coefficient = model.fugacity_coefficient(temperature, pressure)
        temperature_out.append(temperature)
        ideal_volume_out.append(ideal_volume)
        volume_out.append(volume)
        fugacity_out.append(fugacity)
        compressibility_factor_out.append(compressibility_factor)
        fugacity_coefficient_out.append(fugacity_coefficient)

    fig, ax = plt.subplots(1, 4)

    ax[0].plot(pressures, ideal_volume_out, "k--")
    ax[0].set_xlabel("Pressure")
    ax[0].set_ylabel("Ideal volume")

    ax[0].plot(pressures, volume_out)
    ax[0].set_xlabel("Pressure")
    ax[0].set_ylabel("Volume")

    ax[1].plot(pressures, fugacity_out)
    ax[1].set_xlabel("Pressure")
    ax[1].set_ylabel("Fugacity")

    ax[2].plot(pressures, compressibility_factor_out)
    ax[2].set_xlabel("Pressure")
    ax[2].set_ylabel("Compressibility factor")

    ax[3].plot(pressures, fugacity_coefficient_out)
    ax[3].set_xlabel("Pressure")
    ax[3].set_ylabel("Fugacity coefficient")

    plt.show()
