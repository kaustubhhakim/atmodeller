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
import numpy as np
from molmass import Formula
from scipy import constants

# Thermodynamic standard state
TEMPERATURE_REFERENCE: float = 298.15
"""Enthalpy reference temperature in K (:math:`T_r` in the JANAF tables) :cite:p:`MZG02,Cha98`"""
PRESSURE_REFERENCE: float = 1.0
"""Standard state pressure in bar"""
GAS_STATE: str = "g"
"""Suffix to identify gases as per JANAF convention for the state of aggregation"""

# Initial solution guess
INITIAL_LOG_NUMBER_DENSITY: float = 50.0
"""Initial log number density

Empiricially determined. This value is mid-range for Earth-like planets.
"""
INITIAL_LOG_STABILITY: float = -30.0
"""Initial log stability.

Empirically determined.
"""

# Maximum x for which exp(x) is finite in 64-bit precision (to prevent overflow)
MAX_EXP_INPUT: float = np.log(np.finfo(np.float64).max)
# Minimum x for which exp(x) is non-zero in 64-bit precision
MIN_EXP_INPUT: float = np.log(np.finfo(np.float64).tiny)

# Lower and upper bounds on the hypercube which contains the root
LOG_NUMBER_DENSITY_LOWER: float = -170.0
"""Lower log number density for a species

For a gas species this corresponds to ``3.17E-77`` bar and ``3.16E-78`` bar at ``3000`` K and
``298`` K, respectively.
"""
LOG_NUMBER_DENSITY_UPPER: float = 80.0
"""Upper log number density for a species

For a gas species this corresponds to ``2294896`` GPa and ``227960`` GPa at ``3000`` K and ``298`` 
K, respectively. However, the choice of this upper limit is actually motivated by condensed
species.
"""
LOG_STABILITY_LOWER: float = -700.0  # basically the same as MIN_EXP_INPUT
"""Lower stability for a species

Derived to ensure that the exponential function exp(x) does not underflow to zero
"""
LOG_STABILITY_UPPER: float = 35.0
"""Upper stability for a species

Empirically determined.
"""
TAU_MAX: float = 1.0e-3
"""Maximum tau scaling factor for species stability when using the tau cascade solver"""
TAU: float = 1.0e-25
"""Desired (i.e. final/minimium) tau scaling factor for species stability :cite:p:`LKK16`.

Tau effectively controls the minimum non-zero number density of unstable species. Formally, it
defines the number density of an unstable pure condensate with an activity of ``1/e``, which
corresponds to a log stability of zero.

This value is typically appropriate for condensate stability only, but if you additionally apply 
stability criteria to gas species you should reduce this value, maybe as low as ``1e-60`` to 
``1e-72`` if you want to ensure you do not truncated O2 at low temperatures. Hence you can override
this default using an argument to :class:`atmodeller.classes.InteriorAtmosphere`.
"""
TAU_NUM: int = 2
"""Number of tau values to solve between :const:`TAU_MAX` and :const:`TAU` (inclusive) for the tau 
cascade solver

Empirically determined. Basically, once a solution has been found for :const:`TAU_MAX` the solver 
can immediately proceed to :const:`TAU`. This usually solves within a few steps on the first 
attempt.
"""

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
OCEAN_MASS_H2: float = OCEAN_MOLES * Formula("H2").mass / 1e3
"""Mass of H2 in one present-day Earth ocean in kilograms"""
OCEAN_MASS_H2O: float = OCEAN_MOLES * Formula("H2O").mass / 1e3
"""Mass of H2O in one present-day Earth ocean in kilograms"""
