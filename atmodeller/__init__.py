"""Package level variables and initialises the package logger."""

__version__: str = "0.1.0"

import importlib.resources
import logging

# Create the package logger.
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a handler for the logger.
handler: logging.Handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Create a formatter for the log messages.
# Simple formatter.
fmt: str = "%(asctime)s - %(name)-30s - %(levelname)-9s - %(message)s"
datefmt: str = "%H:%M:%S"

# Complex formatter.
# fmt: str = "[%(asctime)s - %(name)-20s - %(lineno)03d - %(levelname)-9s -
# %(funcName)s()] %(message)s"
# datefmt: str = "Y-%m-%d %H:%M:%S"
formatter: logging.Formatter = logging.Formatter(fmt, datefmt=datefmt)
handler.setFormatter(formatter)

# Add the handler to the logger.
logger.addHandler(handler)

logger.info("%s version %s", __name__, __version__)

# Module constants.
GRAVITATIONAL_CONSTANT: float = 6.6743e-11  # SI units.
OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.
GAS_CONSTANT: float = 8.31446261815324  # J/K/mol.

DATA_ROOT_PATH = importlib.resources.files("atmodeller.data")

# pylint: disable=wrong-import-position
from atmodeller.core import (
    BufferedFugacityConstraint,
    FugacityConstraint,
    InteriorAtmosphereSystem,
    MassConstraint,
    SystemConstraint,
)
from atmodeller.thermodynamics import Molecule, Planet
