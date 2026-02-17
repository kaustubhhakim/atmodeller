# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical and numerical constants

This module defines reference thermodynamic conditions and numerical limits.
"""

import numpy as np

# Thermodynamic standard state
TEMPERATURE_REFERENCE: float = 298.15
"""Enthalpy reference temperature in K (:math:`T_r` in the JANAF tables) :cite:p:`MZG02,Cha98`"""
STANDARD_PRESSURE: float = 1.0
"""Standard state pressure in bar"""
STANDARD_FUGACITY: float = STANDARD_PRESSURE
"""Standard fugacity for gases in bar"""
STANDARD_CONCENTRATION: float = 1.0
"""Standard state concentration for dissolved species in ppmw"""
GAS_STATE: str = "g"
"""Suffix to identify gases as per JANAF convention for the state of aggregation"""
LIQUID_STATE: str = "l"
"""Suffix to identify liquids as per JANAF convention for the state of aggregation"""
SOLID_STATE: str = "s"
"""Suffix to identify solids as per JANAF convention for the state of aggregation"""
DISSOLVED_STATE: str = "di"
"""Suffix to identify dissolved species"""

# Initial solution guess
INITIAL_LOG_NUMBER_MOLES: float = 50.0
"""Initial log number of moles

Empiricially determined. This value is mid-range for Earth-like planets.
"""
INITIAL_LOG_STABILITY: float = -30.0
"""Initial log stability

Empirically determined
"""

# Maximum x for which exp(x) is finite in 64-bit precision (to prevent overflow)
MAX_EXP_INPUT: float = np.log(np.finfo(np.float64).max)
# Minimum x for which exp(x) is non-zero in 64-bit precision
MIN_EXP_INPUT: float = np.log(np.finfo(np.float64).tiny)

# Lower and upper bounds on the hypercube which contains the root. These are somewhat empirically
# calibrated to bound the expected values for typical models, but in principle could require
# adjustment for edge cases.
LOG_NUMBER_MOLES_LOWER: float = -200.0
"""Lower log number of moles for a species"""
LOG_NUMBER_MOLES_UPPER: float = 80.0
"""Upper log number of moles for a species"""
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

Tau effectively controls the minimum non-zero number of moles of unstable species. Formally, it
defines the number of moles of an unstable pure condensate with an activity of ``1/e``, which
corresponds to a log stability of zero.
"""
TAU_NUM: int = 2
"""Number of tau values to solve between :const:`TAU_MAX` and :const:`TAU` (inclusive) for the tau 
cascade solver

Empirically determined. Basically, once a solution has been found for :const:`TAU_MAX` the solver 
can immediately proceed to :const:`TAU`. This usually solves within a few steps on the first 
attempt.
"""
