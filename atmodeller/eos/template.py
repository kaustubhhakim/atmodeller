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
"""Template for creating new real gas equations of state"""

from __future__ import annotations

import importlib.resources
import logging
from contextlib import AbstractContextManager
from importlib.abc import Traversable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from atmodeller.eos.interfaces import RealGasProtocol

logger: logging.Logger = logging.getLogger(__name__)


class SimpleRealGasEquationOfState(RealGasProtocol):
    """A simple real gas equation of state

    This implements the only required method (fugacity_coefficient).
    """

    def __init__(self, *args, **kwargs):
        """Class constructor"""
        logger.info("Creating a simple real gas equation of state")
        del args
        del kwargs
        # Let's get some test data from the data directory in this directory.
        eos_data_directory: Traversable = importlib.resources.files(f"{__package__}.data")
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            eos_data_directory.joinpath("test.dat")
        )
        with data as data_path:
            lookup_data: npt.NDArray = np.loadtxt(data_path)

        # Do stuff with lookup data
        logger.info("Lookup data = %s", lookup_data)

        # Once preprocessing of lookup data (e.g. units) complete, store to self.
        self.lookup_data = lookup_data

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient method must be implemented"""
        # Get fugacity coefficient, possibly from lookup data
        # TODO: Currently just behaves like an ideal gas. Update according to your requirements.
        del temperature
        del pressure
        logger.info("Returning the fugacity coefficient, which is just unity")

        return 1


# If you want a more complete and self-consistent EOS, inherit from the RealGas abstract base
# class.

# Convenient for this template so pylint: disable=C0413
from atmodeller.eos.interfaces import RealGas


class CompleteRealGasEquationOfState(RealGas):
    """A complete real gas equation of state

    Implements more methods to enforce self-consistency and allow interrogation of the EOS.
    """
