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
"""Package level variables and initialises the package logger"""

from __future__ import annotations

__version__: str = "0.2.0"

import logging

import jax
import numpy as jnp
from molmass import Formula
from scipy import constants

jax.config.update("jax_enable_x64", True)
jnp.set_printoptions(precision=15)  # For better clarity in printed output
print("Package initialized with double precision (float64)")

# For debugging
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_debug_infs", False)
jax.config.update("jax_disable_jit", True)

AVOGADRO: float = constants.Avogadro
"""Avogadro constant in 1/mol"""
GAS_CONSTANT: float = constants.gas_constant
"""Gas constant in J/K/mol"""
GAS_CONSTANT_BAR: float = GAS_CONSTANT * 1.0e-5
"""Gas constant in m^3 bar/K/mol"""
GRAVITATIONAL_CONSTANT: float = constants.gravitational_constant
"""Gravitational constant in m^3/kg/s^2"""
ATMOSPHERE: float = constants.atmosphere / constants.bar
"""Atmospheres in 1 bar"""
BOLTZMANN_CONSTANT: float = constants.Boltzmann
"""Boltzmann constant in J/K"""
BOLTZMANN_CONSTANT_BAR: float = BOLTZMANN_CONSTANT * 1e-5
"""Boltzmann constant in bar m^3/K"""
# Used to determine the JANAF reference state
NOBLE_GASES: list[str] = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
"""Noble gases"""
OCEAN_MOLES: float = 7.68894973907177e22
"""Moles of H2 (or H2O) in one present-day Earth ocean"""
OCEAN_MASS_H2: float = OCEAN_MOLES * Formula("H2").mass
"""Mass of H2 in one present-day Earth ocean in grams"""

ENTHALPY_REFERENCE: float = 298.15
"""Enthalpy reference temperature in K"""
PRESSURE_REFERENCE: float = 1.0
"""Standard state pressure in bar"""

# Lower and upper bounds on the hypercube which contains the root
NUMBER_DENSITY_LOWER: float = -100
"""Lower number density"""
NUMBER_DENSITY_UPPER: float = 70
"""Upper number density"""
STABILITY_LOWER: float = -200
"""Lower stability"""
STABILITY_UPPER: float = 10
"""Upper stability"""

TAU: float = 1.0e60
"""Tau scaling factor for condensate stability"""

# Create the package logger.
# https://docs.python.org/3/howto/logging.html#library-config
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def complex_formatter() -> logging.Formatter:
    """Complex formatter"""
    fmt: str = "[%(asctime)s - %(name)-30s - %(lineno)03d - %(levelname)-9s - %(funcName)s()]"
    fmt += " - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def simple_formatter() -> logging.Formatter:
    """Simple formatter for logging

    Returns:
        Formatter for logging
    """
    fmt: str = "[%(asctime)s - %(name)-30s - %(levelname)-9s] - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)

    return formatter


def debug_logger() -> logging.Logger:
    """Sets up debug logging to the console.

    Returns:
        A logger
    """
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return package_logger


def debug_file_logger() -> logging.Logger:
    """Sets up info logging to the console and debug logging to a file.

    Returns:
        A logger
    """
    # Console logger
    package_logger: logging.Logger = logging.getLogger(__name__)
    package_logger.setLevel(logging.DEBUG)
    package_logger.handlers = []
    console_handler: logging.Handler = logging.StreamHandler()
    console_formatter: logging.Formatter = simple_formatter()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    package_logger.addHandler(console_handler)
    # File logger
    file_handler: logging.Handler = logging.FileHandler(f"{__package__}.log")
    file_formatter: logging.Formatter = complex_formatter()
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    package_logger.addHandler(file_handler)

    return package_logger
