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
"""Core classes and functions for thermochemical data"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

import sys
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


phase_mapping: dict[str, int] = {"g": 0, "l": 1, "cr": 2, "alpha": 3, "beta": 4}
"""Mapping from the JANAF phase string to an integer code"""
inverse_phase_mapping: dict[int, str] = {value: key for key, value in phase_mapping.items()}
"""Inverse mapping from the integer code to a JANAF phase string"""


# TODO: First get buffer working, then can reimplement this if required.
# class ConstantBuffer(NamedTuple):
#     """Constant buffer

#     Args:
#         log10_fugacity: Log10 fugacity
#     """

#     log10_fugacity: ArrayLike

#     def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
#         del temperature
#         del pressure

#         return self.log10_fugacity * jnp.log(10)


class CondensateActivity(NamedTuple):
    """Activity of a stable condensate"""

    activity: ArrayLike = 1.0

    def log_activity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> ArrayLike:
        del temperature
        del pressure

        return jnp.log(self.activity)


@register_pytree_node_class
class ThermoCoefficients:
    """Coefficients for thermochemical data

    Coefficients are available at https://ntrs.nasa.gov/citations/20020085330

    Args:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in the range
        T_max: Maximum temperature(s) in the range

    Attributes:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in the range
        T_max: Maximum temperature(s) in the range
    """

    def __init__(
        self,
        b1: tuple[float, ...],
        b2: tuple[float, ...],
        cp_coeffs: tuple[tuple[float, float, float, float, float, float, float], ...],
        T_min: tuple[float, ...],
        T_max: tuple[float, ...],
    ):
        self._b1 = b1
        self._b2 = b2
        self._cp_coeffs = cp_coeffs
        self._T_min = T_min
        self._T_max = T_max

    @jit
    def _cp_over_R(self, cp_coefficients: ArrayLike, temperature: ArrayLike) -> Array:
        """Heat capacity relative to the gas constant (R)

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            temperature: Temperature

        Returns:
            Heat capacity (J/K/mol) relative to R
        """
        temperature_terms: Array = jnp.stack(
            [
                jnp.power(temperature, -2),
                jnp.power(temperature, -1),
                jnp.ones_like(temperature),
                temperature,
                jnp.power(temperature, 2),
                jnp.power(temperature, 3),
                jnp.power(temperature, 4),
            ]
        )

        heat_capacity: Array = jnp.dot(cp_coefficients, temperature_terms)
        # jax.debug.print("heat_capacity = {out}", out=heat_capacity)

        return heat_capacity

    @jit
    def _S_over_R(
        self, cp_coefficients: ArrayLike, b2: ArrayLike, temperature: ArrayLike
    ) -> Array:
        """Entropy relative to the gas constant (R)

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b2: Entropy integration constant
            temperature: Temperature

        Returns:
            Entropy (J/K/mol) relative to R
        """
        temperature_terms: Array = jnp.stack(
            [
                -jnp.power(temperature, -2) / 2,
                -jnp.power(temperature, -1),
                jnp.log(temperature),
                temperature,
                jnp.power(temperature, 2) / 2,
                jnp.power(temperature, 3) / 3,
                jnp.power(temperature, 4) / 4,
            ]
        )

        entropy: Array = jnp.dot(cp_coefficients, temperature_terms) + b2
        # jax.debug.print("entropy = {out}", out=entropy)

        return entropy

    @jit
    def _H_over_RT(
        self, cp_coefficients: ArrayLike, b1: ArrayLike, temperature: ArrayLike
    ) -> Array:
        """Enthalpy relative to RT

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            temperature: Temperature

        Returns:
            Enthalpy (J/mol) relative to RT
        """
        temperature_terms: Array = jnp.stack(
            [
                -jnp.power(temperature, -2),
                jnp.log(temperature) / temperature,
                jnp.ones_like(temperature),
                temperature / 2,
                jnp.power(temperature, 2) / 3,
                jnp.power(temperature, 3) / 4,
                jnp.power(temperature, 4) / 5,
            ]
        )

        enthalpy: Array = jnp.dot(cp_coefficients, temperature_terms) + b1 / temperature
        # jax.debug.print("enthalpy = {out}", out=enthalpy)

        return enthalpy

    @jit
    def _G_over_RT(
        self, cp_coefficients: ArrayLike, b1: ArrayLike, b2: ArrayLike, temperature: ArrayLike
    ) -> Array:
        """Gibbs energy relative to RT

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            b2: Entropy integration constant
            temperature: Temperature

        Returns:
            Gibbs energy relative to RT
        """
        enthalpy: Array = self._H_over_RT(cp_coefficients, b1, temperature)
        entropy: Array = self._S_over_R(cp_coefficients, b2, temperature)
        # No temperature multiplication is correct since the return is Gibbs energy relative to RT
        gibbs: Array = enthalpy - entropy

        return gibbs

    @jit
    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Array:
        """Gets Gibbs energy over RT

        This is calculated using data from the appropriate temperature range.

        Args:
            temperature: Temperature

        Returns:
            Gibbs energy over RT
        """
        # This assumes the temperature is within one of the ranges and will produce unexpected
        # output if the temperature is outside the ranges
        T_min_array: Array = jnp.asarray(self._T_min)
        T_max_array: Array = jnp.asarray(self._T_max)
        bool_mask: Array = (T_min_array <= temperature) & (temperature <= T_max_array)
        index: Array = jnp.argmax(bool_mask)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Array = jnp.take(jnp.array(self._cp_coeffs), index, axis=0)
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index)
        b1_for_index: Array = jnp.take(jnp.array(self._b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        b2_for_index: Array = jnp.take(jnp.array(self._b2), index)
        # jax.debug.print("b2_for_index = {out}", out=b2_for_index)
        gibbs_for_index: Array = self._G_over_RT(
            cp_coeffs_for_index, b1_for_index, b2_for_index, temperature
        )

        return gibbs_for_index

    def tree_flatten(self) -> tuple[tuple, dict[str, tuple]]:
        children: tuple = ()
        aux_data = {
            "b1": self._b1,
            "b2": self._b2,
            "cp_coeffs": self._cp_coeffs,
            "T_min": self._T_min,
            "T_max": self._T_max,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


class SpeciesData(NamedTuple):
    """Species data

    Args:
        composition: Composition
        phase_code: Phase code
        molar_mass: Molar mass
        thermodata: Thermodynamic data
    """

    composition: ImmutableMap[str, tuple[int, float, float]]
    """Composition"""
    phase_code: int
    """Phase code"""
    molar_mass: float
    """Molar mass"""
    thermodata: ThermoCoefficients
    """Thermodynamic data"""

    @classmethod
    def create(
        cls,
        formula: str,
        phase: str,
        thermodata: ThermoCoefficients,
    ) -> Self:
        """Creates an instance

        Args:
            formula: Formula
            phase: Phase
            thermodata: Thermodynamic data

        Returns:
            An instance
        """
        mformula: Formula = Formula(formula)
        composition: ImmutableMap[str, tuple[int, float, float]] = ImmutableMap(
            mformula.composition().asdict()
        )
        molar_mass: float = mformula.mass * unit_conversion.g_to_kg
        phase_code: int = phase_mapping[phase]

        return cls(composition, phase_code, molar_mass, thermodata)

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    def formula(self) -> Formula:
        """Formula object"""
        formula: str = ""
        for element, values in self.composition.items():
            count: int = values[0]
            formula += element
            if count > 1:
                formula += str(count)

        return Formula(formula)

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Array:
        """Gets Gibbs energy over RT

        This is calculated using data from the appropriate temperature range.

        Args:
            temperature: Temperature

        Returns:
            Gibbs energy over RT
        """
        return self.thermodata.get_gibbs_over_RT(temperature)

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return f"{self.hill_formula}_{self.phase}"

    @property
    def hill_formula(self) -> str:
        """Hill formula"""
        return self.formula().formula

    @property
    def phase(self) -> str:
        """JANAF phase"""
        return inverse_phase_mapping[self.phase_code]


class CriticalData(NamedTuple):
    """Critical temperature and pressure of a gas species.

    Args:
        temperature: Critical temperature in K
        pressure: Critical pressure in bar
    """

    temperature: float
    """Critical temperature in K"""
    pressure: float
    """Critical pressure in bar"""
