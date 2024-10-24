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
"""Abstract and concrete classes for solubility laws"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

import sys
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.interfaces import SolubilityProtocol
from atmodeller.utilities import PyTreeNoData, power_law

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class Solubility(ABC, SolubilityProtocol):
    """Solubility interface

    :meth:`~Solubility.jax_concentration` is defined in order to allow arguments to be passed by
    position to lax.switch.
    """

    @abstractmethod
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ):
        """Concentration in ppmw

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K
            pressure: Pressure in bar
            fO2: fO2 in bar

        Returns:
            Concentration in ppmw
        """

    @jit
    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ):
        """Wrapper to pass concentration arguments by position to use with JAX lax.switch"""
        return self.concentration(fugacity, temperature=temperature, pressure=pressure, fO2=fO2)


@register_pytree_node_class
class NoSolubility(PyTreeNoData, Solubility):
    """No solubility"""

    @jit
    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,  # Convenient to use fO2 so pylint: disable=invalid-name
    ) -> ArrayLike:
        del fugacity
        del temperature
        del pressure
        del fO2

        # Must be 0.0 (float) for JAX array type compliance
        return 0.0


@register_pytree_node_class
class SolubilityPowerLaw(Solubility):
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
    def concentration(self, fugacity: ArrayLike, **kwargs) -> ArrayLike:
        del kwargs

        return power_law(fugacity, self.constant, self.exponent)

    @jit
    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> ArrayLike:
        del temperature
        del pressure
        del fO2
        return self.concentration(fugacity)

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {"constant": self.constant, "exponent": self.exponent}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class SolubilityPowerLawLog10(Solubility):
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
    def concentration(self, fugacity: ArrayLike, **kwargs) -> ArrayLike:
        del kwargs

        return jnp.power(10, (self.log10_constant + self.log10_exponent * jnp.log10(fugacity)))

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {"log10_constant": self.log10_constant, "log10_exponent": self.log10_exponent}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)
