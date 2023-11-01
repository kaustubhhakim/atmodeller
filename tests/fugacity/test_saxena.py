"""Tests for the Saxena fugacity models.

See the LICENSE file for licensing information.
"""

from atmodeller import __version__, debug_logger
from atmodeller.eos.interfaces import FugacityModelABC
from atmodeller.eos.saxena import get_saxena_fugacity_models

# Tolerances to compare the test results with target output.
rtol: float = 1.0e-8
atol: float = 1.0e-8

debug_logger()

fugacity_models: dict[str, FugacityModelABC] = get_saxena_fugacity_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_Ar(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(2510, 100e3, fugacity_models["Ar"], 7.41624600755374)


def test_N2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1573, 75e3, fugacity_models["N2"], 10.293087737779091)


def test_O2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1823, 133e3, fugacity_models["O2"], 12.409268281002012)
