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
from molmass import Formula
from scipy import constants

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
OCEAN_MOLES: float = 7.68894973907177e22
"""Moles of H2 or H2O in one present-day Earth ocean"""
OCEAN_MASS_H2: float = OCEAN_MOLES * Formula("H2").mass
"""Mass of H2 in one present-day Earth ocean in grams"""
OCEAN_MASS_H2O: float = OCEAN_MOLES * Formula("H2O").mass
"""Mass of H2O in one present-day Earth ocean in grams"""
