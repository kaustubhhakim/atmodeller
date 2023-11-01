"""Tests for the Saxena fugacity models.

See the LICENSE file for licensing information.
"""

import logging

from atmodeller import __version__, debug_logger
from atmodeller.eos.interfaces import FugacityModelABC
from atmodeller.eos.saxena import H2SF87, H2LowPressureSS92, get_saxena_fugacity_models
from atmodeller.utilities import UnitConversion

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


# H2 model


def test_H2_low_pressure_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 7279.356114821697
    expected = UnitConversion.cm3_to_J_per_bar(expected)
    check_values.volume(873, 10, H2LowPressureSS92(), expected)


def test_H2_medium_pressure_SS92(check_values) -> None:
    """Comparison with Figure 1 in Shi and Saxena (1992)"""
    expected: float = 164.388310378618
    expected = UnitConversion.cm3_to_J_per_bar(expected)
    check_values.volume(873, 500, H2LowPressureSS92(), expected)


# def test_H2_high_pressure_SS92(check_values) -> None:
#     """Comparison with Figure 1 in Shi and Saxena (1992)"""
#     expected: float = 164.388310378618
#     expected = UnitConversion.cm3_to_J_per_bar(expected)
#     check_values.volume(1473, 4000, H2HighPressureSS92(), expected)


# H2S


def test_H2S_low_pressure_SS92(check_values) -> None:
    """Comparison with Figure 3 in Shi and Saxena (1992)"""
    expected: float = 272.7266232763035
    expected = UnitConversion.cm3_to_J_per_bar(expected)
    check_values.volume(673, 200, fugacity_models["H2S"], expected)


def test_H2S_medium_pressure_SS92(check_values) -> None:
    """Comparison with Figure 3 in Shi and Saxena (1992)"""
    expected: float = 116.55537998390933
    expected = UnitConversion.cm3_to_J_per_bar(expected)
    check_values.volume(1873, 2000, fugacity_models["H2S"], expected)


# SO2


def test_SO2_low_pressure_SS92(check_values) -> None:
    """Comparison with Figure 2 in Shi and Saxena (1992)"""
    expected: float = 8308.036738813245
    expected = UnitConversion.cm3_to_J_per_bar(expected)
    check_values.volume(1073, 10, fugacity_models["SO2"], expected)


def test_SO2_high_pressure_SS92(check_values) -> None:
    """Comparison with Figure 2 in Shi and Saxena (1992)"""
    expected: float = 70.86864302460566
    expected = UnitConversion.cm3_to_J_per_bar(expected)
    check_values.volume(1873, 4000, fugacity_models["SO2"], expected)
