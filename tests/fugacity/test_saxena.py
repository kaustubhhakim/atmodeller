"""Tests for the Saxena fugacity models.

See the LICENSE file for licensing information.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.eos.interfaces import FugacityModelABC
from atmodeller.eos.saxena import H2SF87, get_saxena_fugacity_models

logger: logging.Logger = debug_logger()

fugacity_models: dict[str, FugacityModelABC] = get_saxena_fugacity_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# These tests only test the pressure regime above 5 kbar.


def test_Ar(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(2510, 100e3, fugacity_models["Ar"], 7.41624600755374)


def test_CH4(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1912, 159e3, fugacity_models["CH4"], 17.77499804453072)


def test_CO2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1167, 184e3, fugacity_models["CO2"], 33.886349109271734)


def test_H2_SF87(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1222, 41.66e3, H2SF87(), 4.975497264839999)


def test_N2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1573, 75e3, fugacity_models["N2"], 10.293087737779091)


def test_O2(check_values) -> None:
    """Comparison with Table 1 in Saxena and Fei (1987)"""
    check_values.compressibility(1823, 133e3, fugacity_models["O2"], 12.409268281002012)
