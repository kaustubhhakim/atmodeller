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

# Convenient to use chemical formulas so pylint: disable=C0103

import sys
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array, jit
from jax.typing import ArrayLike
from molmass import Formula

from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

Href: float = 298.15
"""Enthalpy reference temperature in K"""
Pref: float = 1.0
"""Standard state pressure in bar"""
phase_mapping: dict[str, int] = {"g": 0, "l": 1, "cr": 2}
"""Mapping from the JANAF phase string to an integer code"""
inverse_phase_mapping: dict[int, str] = {value: key for key, value in phase_mapping.items()}
"""Inverse mapping from the integer code to a JANAF phase string"""


# For all activity models recall that the log_number_density argument is scaled
class ActivityProtocol(Protocol):

    def log_activity(
        self,
        log_number_density: ArrayLike,
        species_index: Array,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> ArrayLike: ...


class RedoxBufferProtocol(Protocol):

    @property
    def log10_shift(self) -> ArrayLike: ...

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...


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

#         return self.log10_fugacity * np.log(10)


# TODO: Add in Hirschmann 2021 high pressure extension
class IronWustiteBuffer(NamedTuple):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
    """

    log10_shift: ArrayLike = 0.0

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        # TODO: Eventually make this a jitted function that this method calls.
        log10_fugacity: Array = (
            -0.8853 * jnp.log(temperature)
            - 28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
        )

        return log10_fugacity + self.log10_shift

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        return self.log10_fugacity(temperature, pressure) * np.log(10)


class CondensateActivity(NamedTuple):
    """Activity of a stable condensate"""

    activity: ArrayLike = 1.0

    def log_activity(
        self,
        log_number_density: ArrayLike,
        species_index: Array,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> ArrayLike:
        del log_number_density
        del species_index
        del temperature
        del pressure

        return jnp.log(self.activity)


class IdealGasActivity(NamedTuple):
    """Activity of an ideal gas"""

    def log_activity(
        self,
        log_number_density: ArrayLike,
        species_index: Array,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> ArrayLike:
        del temperature
        del pressure

        return jnp.take(log_number_density, species_index)


class ThermoData(NamedTuple):
    """Thermochemical data"""

    b1: tuple[float, ...]
    """Enthalpy constant(s) of integration"""
    b2: tuple[float, ...]
    """Entropy constant(s) of integration"""
    cp_coeffs: tuple[tuple[float, float, float, float, float, float, float], ...]
    """Heat capacity coefficients"""
    T_min: tuple[float, ...]
    """Minimum temperature(s) in the range"""
    T_max: tuple[float, ...]
    """Maximum temperature(s) in the range"""

    # TODO: Add a class method to spawn these objects using coefficients from:
    # "NASA Glenn Coefficients for Calculating Thermodynamic Properties of Individual Species"
    # Might involve using a LLM to tabulated the coefficients from the PDF? Or can an electronic
    # version of the tabulated data be obtained?


class SpeciesData(NamedTuple):
    """Species data

    Args:
        composition: Composition
        phase_code: Phase code
        molar_mass: Molar mass
        thermodata: Thermodynamic data
    """

    # Because composition is a dict this object is not hashable. Python does not have a frozendict,
    # so other options would be:
    # https://flax.readthedocs.io/en/latest/api_reference/flax.core.frozen_dict.html
    # https://github.com/GalacticDynamics/xmmutablemap
    composition: dict[str, tuple[int, float, float]]
    """Composition"""
    phase_code: int
    """Phase code"""
    molar_mass: float
    """Molar mass"""
    thermodata: ThermoData
    """Thermodynamic data"""

    @classmethod
    def create(
        cls,
        formula: str,
        phase: str,
        thermodata: ThermoData,
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
        composition: dict[str, tuple[int, float, float]] = mformula.composition().asdict()
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


@jit
def cp_over_R(cp_coefficients: Array, temperature: ArrayLike) -> Array:
    """Heat capacity relative to the gas constant (R)

    Args:
        cp_coefficients: Coefficients for the heat capacity
        temperature: Temperature in K

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
def S_over_R(cp_coefficients: Array, b2: ArrayLike, temperature: ArrayLike) -> Array:
    """Entropy relative to the gas constant (R)

    Args:
        cp_coefficients: Coefficients for the heat capacity
        b2: Entropy integration constant
        temperature: Temperature in K

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
def H_over_RT(cp_coefficients: Array, b1: ArrayLike, temperature: ArrayLike) -> Array:
    """Enthalpy relative to RT

    Args:
        cp_coefficients: Coefficients for the heat capacity
        b1: Enthalpy integration constant
        temperature: Temperature in K

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
def G_over_RT(
    cp_coefficients: Array, b1: ArrayLike, b2: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gibbs energy relative to RT

    Args:
        cp_coefficients: Coefficients for the heat capacity
        b1: Enthalpy integration constant
        b2: Entropy integration constant
        temperature: Temperature in K

    Returns:
        Gibbs energy relative to RT
    """
    enthalpy: Array = H_over_RT(cp_coefficients, b1, temperature)
    entropy: Array = S_over_R(cp_coefficients, b2, temperature)
    # No temperature multiplication is correct since the return is Gibbs energy relative to RT
    gibbs: Array = enthalpy - entropy

    return gibbs


@jit
def get_index_for_temperature(
    temperature_min: ArrayLike, temperature_max: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets the index of the thermodynamic data for a given temperature

    Args:
        temperature_min: Minimum temperature of range
        temperature_max: Maximum temperature of range
        temperature: Target temperature

    Returns:
        Index of the temperature range
    """
    # TODO: This assumes the temperature is within one of the ranges and will produce unexpected
    # output if the temperature is outside the ranges
    T_min_array: Array = jnp.asarray(temperature_min)
    T_max_array: Array = jnp.asarray(temperature_max)
    bool_mask: Array = (T_min_array <= temperature) & (temperature <= T_max_array)
    index: Array = jnp.argmax(bool_mask)
    # jax.debug.print("index = {out}", out=index)

    return index


@jit
def get_gibbs_over_RT(thermodata: ThermoData, temperature: ArrayLike) -> Array:
    """Gets Gibbs energy over RT

    Args:
        thermodata: Thermodynamic data
        temperature: Temperature

    Returns:
        Gibbs energy over RT
    """
    index: Array = get_index_for_temperature(thermodata.T_min, thermodata.T_max, temperature)
    # jax.debug.print("index = {out}", out=index)
    cp_coeffs: Array = jnp.take(jnp.array(thermodata.cp_coeffs), index, axis=0)
    # jax.debug.print("cp_coeffs = {out}", out=cp_coeffs)
    b1: Array = jnp.take(jnp.array(thermodata.b1), index)
    # jax.debug.print("b1 = {out}", out=b1)
    b2: Array = jnp.take(jnp.array(thermodata.b2), index)
    # jax.debug.print("b2 = {out}", out=b2)
    gibbs: Array = G_over_RT(cp_coeffs, b1, b2, temperature)

    return gibbs
