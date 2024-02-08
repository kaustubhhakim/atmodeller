"""Package level variables and initialises the package logger

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

__version__: str = "0.1.0"

import importlib.resources
import logging
from importlib.abc import Traversable

from molmass import Formula
from scipy import constants

# Module constants
GAS_CONSTANT: float = constants.gas_constant  # J/K/mol
GAS_CONSTANT_BAR: float = GAS_CONSTANT * 1.0e-5  # m^3 bar/K/mol
GRAVITATIONAL_CONSTANT: float = constants.gravitational_constant  # m^3/kg/s^2
ATMOSPHERE: float = constants.atmosphere / constants.bar  # bar

# Used to determine the JANAF reference state
NOBLE_GASES: list[str] = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]

OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.
OCEAN_MASS_H2: float = OCEAN_MOLES * Formula("H2").mass

DATA_ROOT_PATH: Traversable = importlib.resources.files(f"{__package__}.data")

# Minimum and maximum values of log10(pressure) to prevent the initial solution from giving rise
# to an excessively large total pressure that can cause numerical overflow or underflow.
# Minimum value is guided by typical values of the fO2 for IW atmospheres for Earth-sized planets
INITIAL_SOLUTION_MIN_LOG10: float = -12
INITIAL_SOLUTION_MAX_LOG10: float = 5

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter."""
    fmt: str = (
        "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()] - %(message)s"
    )
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter."""
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Setup the logging for debugging: DEBUG to the console."""
    # Console logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def debug_file_logger() -> logging.Logger:
    """Setup the logging to a file (DEBUG) and to the console (INFO)."""
    # Console logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    # File logger
    file_handler: logging.Handler = logging.FileHandler(f"{__package__}.log")
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger
