"""Tests for the Holley et al. EOS models

See the LICENSE file for licensing information.
"""

import logging
from typing import Type

from atmodeller import ATMOSPHERE, __version__, debug_logger
from atmodeller.eos.holley import get_holley_eos_models
from atmodeller.interfaces import RealGasABC, ThermodynamicData, ThermodynamicDataBase

logger: logging.Logger = debug_logger()

thermodynamic_data: Type[ThermodynamicDataBase] = ThermodynamicData

eos_models: dict[str, RealGasABC] = get_holley_eos_models()

rtol: float = 1.0e-8
atol: float = 1.0e-8


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# Hydrogen from Table II
def test_H2_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(300, pressure, eos_models["H2"], 1.06217)


def test_H2_high(check_values) -> None:
    pressure: float = 1000 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["H2"], 1.26294)


# Nitrogen from Table III
def test_N2_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(300, pressure, eos_models["N2"], 1.00464)


def test_N2_high(check_values) -> None:
    pressure: float = 1000 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["N2"], 1.36551)


# Oxygen from Table IV
def test_O2_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(300, pressure, eos_models["O2"], 0.95454)


def test_O2_high(check_values) -> None:
    pressure: float = 1000 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["O2"], 1.28897)


# Carbon dioxide from Table V
def test_CO2_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(400, pressure, eos_models["CO2"], 0.81853)


def test_CO2_high(check_values) -> None:
    pressure: float = 1000 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["CO2"], 1.07058)


# Ammonia from Table VI
def test_NH3_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(400, pressure, eos_models["NH3"], 0.56165)


def test_NH3_high(check_values) -> None:
    pressure: float = 500 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["NH3"], 0.93714)


# Methane from Table VII
def test_CH4_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(300, pressure, eos_models["CH4"], 0.85583)


def test_CH4_high(check_values) -> None:
    pressure: float = 1000 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["CH4"], 1.36201)


# Helium from Table VIII
def test_He_low(check_values) -> None:
    pressure: float = 100 * ATMOSPHERE
    check_values.compressibility(300, pressure, eos_models["He"], 1.05148)


def test_He_high(check_values) -> None:
    pressure: float = 1000 * ATMOSPHERE
    check_values.compressibility(1000, pressure, eos_models["He"], 1.14766)
