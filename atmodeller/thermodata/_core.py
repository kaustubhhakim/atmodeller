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

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller.utilities import as_j64, unit_conversion


class CondensateActivity(eqx.Module):
    """Activity of a stable condensate"""

    activity: Array = eqx.field(converter=as_j64, default=1.0)

    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        del temperature
        del pressure

        return jnp.log(self.activity)


class ThermoCoefficients(eqx.Module):
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

    b1: tuple[float, ...]
    b2: tuple[float, ...]
    cp_coeffs: tuple[tuple[float, float, float, float, float, float, float], ...]
    T_min: tuple[float, ...]
    T_max: tuple[float, ...]

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
        T_min_array: Array = jnp.asarray(self.T_min)
        T_max_array: Array = jnp.asarray(self.T_max)
        # Temperature must be a float array since JAX cannot raise integers to negative powers.
        temperature = jnp.asarray(temperature, dtype=jnp.float64)
        bool_mask: Array = (T_min_array <= temperature) & (temperature <= T_max_array)
        index: Array = jnp.argmax(bool_mask)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Array = jnp.take(jnp.array(self.cp_coeffs), index, axis=0)
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index)
        b1_for_index: Array = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        b2_for_index: Array = jnp.take(jnp.array(self.b2), index)
        # jax.debug.print("b2_for_index = {out}", out=b2_for_index)
        gibbs_for_index: Array = self._G_over_RT(
            cp_coeffs_for_index, b1_for_index, b2_for_index, temperature
        )

        return gibbs_for_index


class SpeciesData(eqx.Module):
    """Species data

    Args:
        formula: Formula
        phase: Phase
        thermodata: Thermodynamic data
    """

    formula: str
    """Formula"""
    phase: str
    """Phase"""
    thermodata: ThermoCoefficients
    """Thermodynamic data"""
    composition: ImmutableMap[str, tuple[int, float, float]] = eqx.field(init=False)
    """Composition"""
    hill_formula: str = eqx.field(init=False)
    """Hill formula"""
    molar_mass: float = eqx.field(init=False)
    """Molar mass"""

    def __post_init__(self):
        mformula: Formula = Formula(self.formula)
        self.composition = ImmutableMap(mformula.composition().asdict())
        self.hill_formula = mformula.formula
        self.molar_mass = mformula.mass * unit_conversion.g_to_kg

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return f"{self.hill_formula}_{self.phase}"

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Array:
        """Gets Gibbs energy over RT

        This is calculated using data from the appropriate temperature range.

        Args:
            temperature: Temperature

        Returns:
            Gibbs energy over RT
        """
        return self.thermodata.get_gibbs_over_RT(temperature)


class CriticalData(eqx.Module):
    """Critical temperature and pressure of a gas species

    Args:
        temperature: Critical temperature in K
        pressure: Critical pressure in bar
    """

    temperature: float = 1.0
    """Critical temperature in K"""
    pressure: float = 1.0
    """Critical pressure in bar"""
