# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Helper module with physical, chemical, and mathematical constants for scientific modelling."""

import equinox as eqx
from molmass import Formula
from scipy import constants
from scipy.constants import atmosphere, bar, kilo, mega

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
