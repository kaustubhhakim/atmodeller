"""Package level variables and initialises the package logger."""

__version__: str = "0.1.0"

import logging
import logging.config
import subprocess
from pathlib import Path
from typing import Callable

# Assumes Git directory is one above the location of this file.
cwd: Path = Path(__file__).resolve().parent


def _check_git_exists(func: Callable) -> str:
    """Catch exceptions if Git is not available or if the directory is not managed by Git.

    This is necessary because sometimes we copy a directory of .py files (the package) to a
    computer that may not have Git installed or configured.

    Returns:
        Either the string output from func or a string alerting the user that Git is not available.

    """

    try:
        return func()
    except subprocess.CalledProcessError:
        return f"{cwd} is not a git repository"


@_check_git_exists
def get_git_revision_hash() -> str:
    """The long Git hash."""

    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )


@_check_git_exists
def get_git_revision_short_hash() -> str:
    """The short Git hash."""

    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )


@_check_git_exists
def get_git_revision_describe() -> str:
    """The Git version."""

    return (
        subprocess.check_output(["git", "describe", "--always"], cwd=cwd)
        .decode("ascii")
        .strip()
    )


# Create package logger.
# Assumes that logging.conf resides in the same directory as __init__.py.
logging_config: Path = Path(cwd, "logging.conf")
if logging_config.exists():
    logging.config.fileConfig(logging_config)
    package_logger: logging.Logger = logging.getLogger(__package__)
    package_logger.info("%s version %s", __package__, __version__)
    package_logger.debug("%s Git hash: %s", __package__, get_git_revision_hash)
    package_logger.debug(
        "%s Git short hash: %s", __package__, get_git_revision_short_hash
    )
    package_logger.debug("%s Git describe: %s", __package__, get_git_revision_describe)
else:
    raise FileNotFoundError(
        f"Logging configuration file does not exist: {logging_config}"
    )

# Module constants.
GRAVITATIONAL_CONSTANT: float = 6.6743e-11  # SI units.
OCEAN_MOLES: float = (
    7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.
)
GAS_CONSTANT: float = 8.31446261815324  # J/K/mol

# Temperature range used to fit the JANAF data.
TEMPERATURE_JANAF_HIGH = 3000  # K
TEMPERATURE_JANAF_LOW = 1500  # K

from atmodeller.core import InteriorAtmosphereSystem  # type: ignore
from atmodeller.core import Molecule, SystemConstraint
from atmodeller.reaction import MolarMasses
