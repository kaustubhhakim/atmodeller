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

import sys
from typing import Protocol

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.utilities import PyTreeNoData

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


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


@register_pytree_node_class
class SolubilityPowerLaw:
    """A solubility power law

    Args:
        constant: Constant
        exponent: Exponent

    Attributes:
        constant: Constant
        exponent: Exponent
    """

    def __init__(self, constant: float, exponent: float):
        self.constant: float = constant
        self.exponent: float = exponent

    @jit
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

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {"constant": self.constant, "exponent": self.exponent}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class SolubilityPowerLawLog10:
    """A solubility power law with log10 coefficients

    Args:
        log10_constant: Log10 constant
        log10_exponent: Log10 exponent

    Attributes:
        log10_constant: Log10 constant
        log10_exponent: Log10 exponent
    """

    def __init__(self, log10_constant: float, log10_exponent: float):
        self.log10_constant: float = log10_constant
        self.log10_exponent: float = log10_exponent

    @jit
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

        return jnp.power(10, (self.log10_constant + self.log10_exponent * jnp.log10(fugacity)))

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {"log10_constant": self.log10_constant, "log10_exponent": self.log10_exponent}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class NoSolubility(PyTreeNoData):
    """No solubility"""

    @jit
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
