"""Package level variables and initialises the package logger."""

from __future__ import annotations

__version__: str = "0.1.0"

import importlib.resources
import logging

from scipy import constants

# Module constants.
GAS_CONSTANT: float = constants.gas_constant  # J/K/mol.
GRAVITATIONAL_CONSTANT: float = constants.gravitational_constant  # m^3/kg/s^2.
OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.

DATA_ROOT_PATH = importlib.resources.files("atmodeller.data")

# Create the package logger.
logger: logging.Logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def debug_logger() -> None:
    """Setup the logging for debugging."""
    logger: logging.Logger = logging.getLogger(__name__)
    logger.handlers = []
    handler: logging.Handler = logging.StreamHandler()
    logger.setLevel(logging.DEBUG)
    # Complex formatter.
    # fmt: str = "[%(asctime)s - %(name)-20s - %(lineno)03d - %(levelname)-9s -
    # %(funcName)s()] %(message)s"
    # datefmt: str = "Y-%m-%d %H:%M:%S"
    fmt: str = "%(asctime)s - %(name)-30s - %(levelname)-9s - %(message)s"
    datefmt: str = "%H:%M:%S"
    formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
