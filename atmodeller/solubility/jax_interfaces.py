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
from typing import NamedTuple

import jax.numpy as jnp
from jax.typing import ArrayLike

logger: logging.Logger = logging.getLogger(__name__)


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
            **kwargs: Arbitrary keyword arguments

        Returns:
            Concentration
        """
        del kwargs

        return self.constant * jnp.power(fugacity, self.exponent)


class NoSolubility(NamedTuple):
    """No solubility"""

    def concentration(self, *args, **kwargs) -> ArrayLike:
        """No concentration"""
        del args
        del kwargs

        return 0.0
