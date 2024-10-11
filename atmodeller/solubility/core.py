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
"""Core classes and functionality"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike


class SolubilityProtocol(Protocol):
    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike: ...


@jit
def power_law(fugacity: ArrayLike, constant: float, exponent: float) -> Array:
    """Power law

    Args:
        fugacity: Fugacity
        constant: Constant for the power law
        exponent: Exponent for the power law

    Returns:
        Evaluated power law
    """
    return constant * jnp.power(fugacity, exponent)


class SolubilityPowerLaw(NamedTuple):
    """A solubility power law

    Args:
        constant: Constant
        exponent: Exponent
    """

    constant: float
    """Constant"""
    exponent: float
    """Exponent"""

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del temperature
        del pressure
        del fO2

        return power_law(fugacity, self.constant, self.exponent)


@jit
def power_law_log10(fugacity: ArrayLike, log10_constant: float, log10_exponent: float) -> Array:
    """Power law with log10 constant and exponent

    Args:
        fugacity: Fugacity
        log10_constant: Log10 constant for the power law
        log10_exponent: Log10 exponent for the power law

    Returns:
        Evaluated power law
    """
    return jnp.power(10, (log10_constant + log10_exponent * jnp.log10(fugacity)))


class SolubilityPowerLawLog10(NamedTuple):
    """A solubility power law with log10 coefficients

    Args:
        log10_constant: Log10 constant
        log10_exponent: Log10 exponent
    """

    log10_constant: float
    """Constant"""
    log10_exponent: float
    """Exponent"""

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del temperature
        del pressure
        del fO2

        return power_law_log10(fugacity, self.log10_constant, self.log10_exponent)


class NoSolubility(NamedTuple):
    """No solubility"""

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del fugacity
        del temperature
        del pressure
        del fO2

        # Must be 0.0 (float) for JAX array type compliance
        return 0.0
