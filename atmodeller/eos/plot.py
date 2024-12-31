#!/usr/bin/env python3
#
# Copyright 2024 Dan J. Bower, Fabian L. Seidler
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
"""Plot EOS"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from atmodeller.eos import get_eos_models
from atmodeller.interfaces import RealGasProtocol

logger: logging.Logger = logging.getLogger(__name__)

eos_models: dict[str, RealGasProtocol] = get_eos_models()


def plotter():
    """Plotter for EOS"""

    fig, ax = plt.subplots(1, 1)

    eos_model = eos_models["H2O_cork_holland98"]

    temperature = 2000.0
    log10_pressure = np.linspace(-8, 5, num=1000)
    pressure = 10**log10_pressure

    logger.info("temperature = %s", temperature)
    logger.info("pressure = %s", pressure)

    fugacity_list = []

    for pressure_ in pressure:
        fugacity = eos_model.fugacity(temperature, pressure_)
        fugacity_list.append(fugacity)

    fugacity_coefficient = np.array(fugacity_list) / pressure
    ax.loglog(pressure, fugacity_coefficient)

    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Fugacity coefficient")
    plt.show()


if __name__ == "__main__":
    plotter()
