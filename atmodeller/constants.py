# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Physical and numerical constants

This module defines reference thermodynamic conditions and numerical limits.
"""

# Thermodynamic standard state
TEMPERATURE_REFERENCE: float = 298.15
"""Enthalpy reference temperature (K) (:math:`T_r` in the JANAF tables) :cite:p:`MZG02,Cha98`"""
STANDARD_PRESSURE: float = 1.0
"""Standard state pressure (bar)"""
STANDARD_FUGACITY: float = STANDARD_PRESSURE
"""Standard fugacity for gases (bar)"""
GAS_STATE: str = "g"
"""Suffix to identify gases as per JANAF convention for the state of aggregation"""
LIQUID_STATE: str = "l"
"""Suffix to identify liquids as per JANAF convention for the state of aggregation"""
SOLID_STATE: str = "s"
"""Suffix to identify solids as per JANAF convention for the state of aggregation"""
DISSOLVED_STATE: str = "d"
"""Suffix to identify dissolved species for output purposes"""

# Numerical floor for dissolved-species solubility in dissolution reactions
DISSOLUTION_PPMW_FLOOR: float = 1.0e-20
"""Minimum dissolved concentration (ppmw) used before taking logs in dissolution reactions.

Numerical floor for dissolved-species solubility in dissolution reactions. This prevents ``log(0)``
while keeping the floor large enough to avoid excessive solver stiffness in underflow-prone 
regimes.
"""

# Initial solution guess
INITIAL_LOG_NUMBER_MOLES: float = 45.0
"""Initial log number of moles

Empirically determined. This value is mid-range for Earth-like planets.
"""
INITIAL_LOG_STABILITY: float = -30.0
"""Initial log stability

Empirically determined
"""

# Lower and upper bounds on the hypercube which contains the root. These are somewhat empirically
# calibrated to bound the expected values for typical models, but in principle could require
# adjustment for edge cases.
LOG_NUMBER_MOLES_LOWER: float = -200.0
"""Lower log number of moles for a species"""
LOG_NUMBER_MOLES_UPPER: float = 80.0
"""Upper log number of moles for a species"""
LOG_STABILITY_LOWER: float = -700.0
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
"""Desired (i.e. final/minimum) tau scaling factor for species stability :cite:p:`LKK16`.

Tau effectively controls the minimum non-zero number of moles of unstable species. Formally, it
defines the number of moles of an unstable pure condensate with an activity of ``1/e``, which
corresponds to a log stability of zero.
"""
TAU_NUM: int = 2
"""Number of tau values solved between :const:`TAU_MAX` and :const:`TAU` (inclusive).

Used by the tau-cascade solver. Empirically, once a solution has been found for
:const:`TAU_MAX`, the solver can typically proceed directly to :const:`TAU` and converge within a
few steps on the first attempt.
"""
