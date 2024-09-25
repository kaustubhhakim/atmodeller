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
"""JAX interfaces for solubility laws"""

from __future__ import annotations

import logging
from functools import partial
from typing import NamedTuple, Protocol

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

logger: logging.Logger = logging.getLogger(__name__)


class SolubilityProtocol(Protocol):
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: float,
        pressure: ArrayLike,
        **kwargs,
    ) -> ArrayLike: ...


@partial(jit, static_argnames=["constant", "exponent"])
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

    def concentration(self, fugacity: ArrayLike, **kwargs) -> ArrayLike:
        """Concentration

        Args:
            fugacity: Fugacity
            **kwargs: Arbitrary unused keyword arguments

        Returns:
            Concentration
        """
        del kwargs

        return power_law(fugacity, self.constant, self.exponent)


class NoSolubility(NamedTuple):
    """No solubility"""

    def concentration(self, *args, **kwargs) -> ArrayLike:
        """No concentration"""
        del args
        del kwargs

        # Must be 0.0 (float) for JAX array type compliance
        return 0.0
