# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helper module with physical, chemical, and mathematical constants for scientific modelling."""

from typing import Optional

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike
from molmass import Formula
from scipy import constants
from scipy.constants import atmosphere, bar, kilo, mega

from atmodeller.jax_utils import Scalar

AVOGADRO: float = constants.Avogadro
r"""Avogadro constant in :math:`\mathrm{mol}^{-1}`"""

GAS_CONSTANT: float = constants.gas_constant
r"""Gas constant in :math:`\mathrm{J}\ \mathrm{K}^{-1}\ \mathrm{mol}^{-1}`"""

GAS_CONSTANT_BAR: float = GAS_CONSTANT * 1.0e-5
r"""Gas constant in :math:`\mathrm{m}^3\ \mathrm{bar}^{-1}\ \mathrm{K}^{-1}\ \mathrm{mol}^{-1}`"""

GRAVITATIONAL_CONSTANT: float = constants.gravitational_constant
r"""Gravitational constant in :math:`\mathrm{m}^3\ \mathrm{kg}^{-1}\ \mathrm{s}^{-2}`"""

ATMOSPHERE: float = constants.atmosphere / constants.bar
"""Atmospheres in 1 bar"""

BOLTZMANN_CONSTANT: float = constants.Boltzmann
r"""Boltzmann constant in :math:`\mathrm{J}\ \mathrm{K}^{-1}`"""

BOLTZMANN_CONSTANT_BAR: float = BOLTZMANN_CONSTANT * 1e-5
r"""Boltzmann constant in :math:`\mathrm{bar}\ \mathrm{m}^3\ \mathrm{K}^{-1}`"""

EARTH_MASS: float = 5.9722e24
r"""Mass of Earth in kg"""

OCEAN_MOLES: float = 7.68894973907177e22
r"""Moles of :math:`\mathrm{H}_2` or :math:`\mathrm{H}_2\mathrm{O}` in present-day Earth's ocean"""

OCEAN_MASS_H2: float = OCEAN_MOLES * Formula("H2").mass / 1e3
r"""Mass of :math:`\mathrm{H}_2` in one present-day Earth ocean in kg"""

OCEAN_MASS_H2O: float = OCEAN_MOLES * Formula("H2O").mass / 1e3
r"""Mass of :math:`\mathrm{H}_2\mathrm{O}` in one present-day Earth ocean in kg"""


class UnitConversion(eqx.Module):
    """Unit conversions"""

    # Pressure
    atmosphere_to_bar: float = atmosphere / bar
    bar_to_Pa: float = 1.0e5
    bar_to_MPa: float = 1.0e-1
    bar_to_GPa: float = 1.0e-4
    Pa_to_bar: float = 1.0e-5
    MPa_to_bar: float = 1.0e1
    GPa_to_bar: float = 1.0e4

    # Concentration / fraction
    fraction_to_ppm: float = mega
    ppm_to_fraction: float = 1 / mega
    ppm_to_percent: float = 100 / mega
    percent_to_ppm: float = 1.0e4

    # Mass / volume
    g_to_kg: float = 1 / kilo
    cm3_to_m3: float = 1.0e-6
    m3_to_cm3: float = 1.0e6
    litre_to_m3: float = 1.0e-3

    # Energy / work
    m3_bar_to_J: float = 1.0e5
    J_to_m3_bar: float = 1.0e-5


# Single instance for convenient access
unit_conversion: UnitConversion = UnitConversion()


class ExperimentalCalibration(eqx.Module):
    r"""Experimental calibration

    Args:
        temperature_min: Minimum calibrated temperature. Defaults to ``None``.
        temperature_max: Maximum calibrated temperature. Defaults to ``None``.
        pressure_min: Minimum calibrated pressure. Defaults to ``None``.
        pressure_max: Maximum calibrated pressure. Defaults to ``None``.
        log10_fO2_min: Minimum calibrated :math:`\log_{10} f\rm{O}_2`. Defaults to ``None``.
        log10_fO2_max: Maximum calibrated :math:`\log_{10} f\rm{O}_2`. Defaults to ``None``.
    """

    temperature_min: Optional[float]
    """Minimum calibrated temperature"""
    temperature_max: Optional[float]
    """Maximum calibrated temperature"""
    pressure_min: Optional[float]
    """Minimum calibrated pressure"""
    pressure_max: Optional[float]
    """Maximum calibrated pressure"""
    log10_fO2_min: Optional[float]
    r"""Minimum calibrated :math:`\log_{10} f\rm{O}_2`"""
    log10_fO2_max: Optional[float]
    r"""Maximum calibrated :math:`\log_{10} f\rm{O}_2`"""

    def __init__(
        self,
        temperature_min: Optional[Scalar] = None,
        temperature_max: Optional[Scalar] = None,
        pressure_min: Optional[Scalar] = None,
        pressure_max: Optional[Scalar] = None,
        log10_fO2_min: Optional[Scalar] = None,
        log10_fO2_max: Optional[Scalar] = None,
    ):
        self.temperature_min = float(temperature_min) if temperature_min is not None else None
        self.temperature_max = float(temperature_max) if temperature_max is not None else None
        self.pressure_min = float(pressure_min) if pressure_min is not None else None
        self.pressure_max = float(pressure_max) if pressure_max is not None else None
        self.log10_fO2_min = float(log10_fO2_min) if log10_fO2_min is not None else None
        self.log10_fO2_max = float(log10_fO2_max) if log10_fO2_max is not None else None


def bulk_silicate_earth_abundances() -> dict[str, dict[str, float]]:
    """Bulk silicate Earth element masses in kg

    Hydrogen, carbon, and nitrogen from :cite:t:`SKG21`, sulfur from :cite:t:`H16`, and chlorine
    from :cite:t:`KHK17`

    Returns:
        A dictionary of Earth BSE element masses in kg
    """
    earth_bse: dict[str, dict[str, float]] = {
        "H": {"min": 1.852e20, "max": 1.894e21},
        "C": {"min": 1.767e20, "max": 3.072e21},
        "S": {"min": 8.416e20, "max": 1.052e21},
        "N": {"min": 3.493e18, "max": 1.052e19},
        "Cl": {"min": 7.574e19, "max": 1.431e20},
    }

    for _, values in earth_bse.items():
        values["mean"] = np.mean((values["min"], values["max"]))  # type: ignore

    return earth_bse


def earth_oceans_to_hydrogen_mass(number_of_earth_oceans: ArrayLike = 1) -> ArrayLike:
    """Converts Earth oceans to hydrogen mass.

    Args:
        number_of_earth_oceans: Number of Earth oceans. Defaults to ``1`` kg.

    Returns:
        Hydrogen mass in kg
    """
    return number_of_earth_oceans * OCEAN_MASS_H2
