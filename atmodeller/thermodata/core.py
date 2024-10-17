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
from typing import NamedTuple, Protocol

import jax.numpy as jnp
import numpy as np
from jax import Array, jit, lax
from jax.typing import ArrayLike
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

Href: float = 298.15
"""Enthalpy reference temperature in K"""
Pref: float = 1.0
"""Standard state pressure in bar"""
phase_mapping: dict[str, int] = {"g": 0, "l": 1, "cr": 2, "alpha": 3, "beta": 4}
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


# TODO: Eventually rename to ExperimentalCalibration once the old class is removed.
class ExperimentalCalibrationNew(NamedTuple):
    """Experimental calibration

    Args:
        temperature_min: Minimum calibrated temperature
        temperature_max: Maximum calibrated temperature
        pressure_min: Minimum calibrated pressure
        pressure_max: Maximum calibrated pressure
        log10_fO2_min: Minimum calibrated log10 fO2
        log10_fO2_max: Maximum calibrated log10 fO2
    """

    temperature_min: float | None = None
    temperature_max: float | None = None
    pressure_min: float | None = None
    pressure_max: float | None = None
    log10_fO2_min: float | None = None
    log10_fO2_max: float | None = None


class RedoxBufferProtocol(Protocol):
    """Redox buffer protocol"""

    def __init__(self, log10_shift: ArrayLike, calibration: ExperimentalCalibrationNew): ...

    @property
    def calibration(self) -> ExperimentalCalibrationNew: ...

    @property
    def log10_shift(self) -> ArrayLike: ...

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...

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

#         return self.log10_fugacity * jnp.log(10)


class IronWustiteBufferHirschmann08(NamedTuple):
    """Iron-wustite buffer :cite:p:`OP93,HGD08`

    Experimental calibration is provided in the abstract of :cite:t:`HGD08`.

    Args:
        log10_shift: Log10 shift relative to the buffer. Defaults to zero.
        calibration: Experimental calibration
    """

    log10_shift: ArrayLike = 0
    calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
        pressure_max=27.5 * unit_conversion.GPa_to_bar
    )

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity
        """
        log10_fugacity: Array = (
            -0.8853 * jnp.log(temperature)
            - 28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
        )

        return log10_fugacity + self.log10_shift

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        return self.log10_fugacity(temperature, pressure) * jnp.log(10)


class IronWustiteBufferHirschmann21(NamedTuple):
    """Iron-wustite buffer :cite:p:`H21`

    Regarding the calibration, :cite:t:`H21` states that: "It extrapolates smoothly to higher
    temperature, though not calibrated above 3000 K. Extrapolation to lower temperatures (<1000 K)
    or higher pressures (>100 GPa) is not recommended.
    """

    log10_shift: ArrayLike = 0
    calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
        temperature_min=1000, pressure_max=100 * unit_conversion.GPa_to_bar
    )
    a: tuple[float, ...] = (6.844864, 1.175691e-1, 1.143873e-3, 0, 0)
    b: tuple[float, ...] = (5.791364e-4, -2.891434e-4, -2.737171e-7, 0.0, 0.0)
    c: tuple[float, ...] = (-7.971469e-5, 3.198005e-5, 0, 1.059554e-10, 2.014461e-7)
    d: tuple[float, ...] = (-2.769002e4, 5.285977e2, -2.919275, 0, 0)
    e: tuple[float, ...] = (8.463095, -3.000307e-3, 7.213445e-5, 0, 0)
    f: tuple[float, ...] = (1.148738e-3, -9.352312e-5, 5.161592e-7, 0, 0)
    g: tuple[float, ...] = (-7.448624e-4, -6.329325e-6, 0, -1.407339e-10, 1.830014e-4)
    h: tuple[float, ...] = (-2.782082e4, 5.285977e2, -8.473231e-1, 0, 0)

    def _evaluate_m(self, pressure: ArrayLike, coefficients: tuple[float, ...]) -> Array:
        """Evaluates an m parameter

        Args:
            pressure: Pressure in GPa
            coefficients: Coefficients

        Return:
            m parameter
        """
        m: Array = (
            coefficients[4] * jnp.power(pressure, 0.5)
            + coefficients[0]
            + coefficients[1] * pressure
            + coefficients[2] * jnp.power(pressure, 2)
            + coefficients[3] * jnp.power(pressure, 3)
        )

        return m

    def _evaluate_fO2(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        coefficients: tuple[tuple[float, ...], ...],
    ) -> Array:
        """Evaluates the fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa
            coefficients: Coefficients

        Returns:
            log10fO2
        """
        log10fO2: Array = (
            self._evaluate_m(pressure, coefficients[0])
            + self._evaluate_m(pressure, coefficients[1]) * temperature
            + self._evaluate_m(pressure, coefficients[2]) * temperature * jnp.log(temperature)
            + self._evaluate_m(pressure, coefficients[3]) / temperature
        )

        return log10fO2

    def _fcc_bcc_iron(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """log10fO2 for fcc and bcc iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for fcc and bcc iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self.a, self.b, self.c, self.d)
        )

        return log10fO2

    def _hcp_iron(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """log10fO2 for hcp iron

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Return:
            log10fO2 for hcp iron
        """
        log10fO2: Array = self._evaluate_fO2(
            temperature, pressure, (self.e, self.f, self.g, self.h)
        )

        return log10fO2

    def _use_hcp(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Check to use hcp iron formulation for fO2

        Args:
            temperature: Temperature in K
            pressure: Pressure in GPa

        Returns:
            True/False whether to use the hcp iron formulation
        """
        x: tuple[float, ...] = (-18.64, 0.04359, -5.069e-6)
        threshold: Array = x[2] * jnp.power(temperature, 2) + x[1] * temperature + x[0]

        return jnp.array(pressure) > threshold

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity
        """
        pressure_GPa: ArrayLike = pressure * unit_conversion.bar_to_GPa

        def hcp_case() -> Array:
            return self._hcp_iron(temperature, pressure_GPa)

        def fcc_bcc_case() -> Array:
            return self._fcc_bcc_iron(temperature, pressure_GPa)

        buffer_value: Array = lax.cond(
            self._use_hcp(temperature, pressure_GPa), hcp_case, fcc_bcc_case
        )

        return buffer_value + self.log10_shift

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        return self.log10_fugacity(temperature, pressure) * jnp.log(10)


class IronWustiteBufferHirschmann(NamedTuple):
    """Composite iron-wustite buffer using :cite:t:`OP93,HGD08` and :cite:t:`H21`"""

    log10_shift: ArrayLike = 0
    calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(
        pressure_max=100 * unit_conversion.GPa_to_bar
    )

    def low_temperature_buffer(self) -> RedoxBufferProtocol:
        """Gets the low temperature buffer"""
        return IronWustiteBufferHirschmann08(self.log10_shift)

    def high_temperature_buffer(self) -> RedoxBufferProtocol:
        """Gets the high temperature buffer"""
        return IronWustiteBufferHirschmann21(self.log10_shift)

    def log10_fugacity_low_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> ArrayLike:
        """Log10 fugacity for the low temperature buffer

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity for the low temperature buffer
        """
        return self.low_temperature_buffer().log10_fugacity(temperature, pressure)

    def log10_fugacity_high_temperature(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> ArrayLike:
        """Log10 fugacity for the high temperature buffer

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity for the high temperature buffer
        """
        return self.high_temperature_buffer().log10_fugacity(temperature, pressure)

    def _use_low_temperature(self, temperature: ArrayLike) -> Array:
        """Check to use the low temperature buffer for fO2

        Args:
            temperature: Temperature

        Returns:
            True/False whether to use the low temperature formulation
        """
        return (
            jnp.asarray(temperature) < self.high_temperature_buffer().calibration.temperature_min
        )

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log10 fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log10 fugacity
        """

        def low_temperature_case() -> ArrayLike:
            return self.log10_fugacity_low_temperature(temperature, pressure)

        def high_temperature_case() -> ArrayLike:
            return self.log10_fugacity_high_temperature(temperature, pressure)

        # Shift has already been added by the component buffers
        fO2: ArrayLike = lax.cond(
            self._use_low_temperature(temperature), low_temperature_case, high_temperature_case
        )

        return fO2

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Gets the log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        return self.log10_fugacity(temperature, pressure) * jnp.log(10)


IronWustiteBuffer = IronWustiteBufferHirschmann


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
    composition: ImmutableMap[str, tuple[int, float, float]]
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


# TODO: Clean up this function which back-computes the log10 fO2 shift for a given fugacity. Will
# be required for output.
# def solve_for_log10_dIW(
#     target_fugacity: float, temperature: float, pressure: ArrayLike = 1.0, **kwargs
# ) -> float:
#     """Solves for the log10 shift relative to the default Iron-wustite buffer

#     The shift is report relative to the standard state defined at temperature and 1 bar pressure.
#     If desired, the shift can be reported relative to a standard state defined at an alternative
#     pressure.

#     Args:
#         target_fugacity: Target fugacity in bar
#         temperature: Temperature in K
#         pressure: Pressure defining the standard state in bar. Defaults to 1 bar.
#         **kwargs: Arbitrary keyword arguments

#     Returns:
#         The required log10_shift to match the target fugacity
#     """
#     buffer: RedoxBufferProtocol = IronWustiteBuffer()

#     def objective_function(log10_shift: ArrayLike, args):
#         """Objective function

#         Args:
#             log10_shift: Log10 shift
#             args: Optional arguments (not used)

#         Returns:
#             Residual of the objective function
#         """
#         del args
#         buffer.log10_shift = log10_shift
#         calculated_log10_fugacity: ArrayLike = buffer.get_log10_value(
#             temperature, pressure, penalty=False, **kwargs
#         )

#         return calculated_log10_fugacity - jnp.log10(target_fugacity)

#     solver = optx.Bisection(rtol=1.0e-8, atol=1.0e-8)
#     sol = optx.root_find(
#         objective_function, solver, jnp.array(-20.0), options=dict(lower=-100, upper=100)
#     )

#     # Success is indicated by no message
#     if optx.RESULTS[sol.result] == "":
#         value: float = sol.value.item()
#     else:
#         raise ValueError("Root finding did not converge")

#     return value
