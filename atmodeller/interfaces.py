# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interfaces for thermodynamic models and constraints.

This module defines formal protocol classes (via :class:`typing.Protocol`) that specify the
expected interfaces for different thermodynamic components.
"""

from typing import Optional, Protocol, runtime_checkable

import equinox as eqx
from jaxmod.units import unit_conversion
from jaxtyping import Array, ArrayLike, Bool
from molmass import Formula


class ChemicalSpeciesData(eqx.Module):
    """General data container for an individual species

    Args:
        formula: Formula
        state: State of aggregation, typically follows JANAF convention: 'g' for gas, 'l' for
            liquid, 's' for solid.
    """

    formula: str
    """Formula"""
    state: str
    """State of aggregation. Defaults to an empty string."""
    composition: dict[str, tuple[int, float, float]]
    """Composition"""
    hill_formula: str
    """Hill formula"""
    molar_mass: float = eqx.field(converter=float)
    """Molar mass"""

    def __init__(self, formula: str, state: str = ""):
        self.formula = formula
        self.state = state
        mformula: Formula = Formula(self.formula)
        self.composition = mformula.composition().asdict()
        self.hill_formula = mformula.formula
        self.molar_mass = mformula.mass * unit_conversion.g_to_kg

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and state of aggregation"""
        return f"{self.hill_formula}_{self.state}"


@runtime_checkable
class ActivityProtocol(Protocol):
    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class SpeciesProtocol(Protocol):
    @property
    def activity(self) -> ActivityProtocol: ...
    @property
    def data(self) -> ChemicalSpeciesData: ...

    @property
    def number_solution(self) -> int: ...

    @property
    def solve_for_stability(self) -> bool: ...


@runtime_checkable
class FugacityConstraintProtocol(Protocol):
    def active(self) -> Bool[Array, "..."]: ...

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class RedoxBufferProtocol(FugacityConstraintProtocol, Protocol):
    evaluation_pressure: Optional[float]

    @property
    def log10_shift(self) -> Array: ...

    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> Array: ...

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array: ...


@runtime_checkable
class SolubilityProtocol(Protocol):
    """Solubility protocol

    :meth:`~SolubilityProtocol.jax_concentration` is defined in order to allow arguments to be
    passed by position to lax.switch.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: Optional[ArrayLike] = None,
        pressure: Optional[ArrayLike] = None,
        fO2: Optional[ArrayLike] = None,
    ) -> Array: ...

    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> Array: ...


@runtime_checkable
class ThermodynamicStateProtocol(Protocol):
    @property
    def mass(self) -> Array: ...

    @property
    def melt_fraction(self) -> Array: ...

    @property
    def melt_mass(self) -> Array: ...

    @property
    def solid_mass(self) -> Array: ...

    @property
    def temperature(self) -> Array: ...

    def get_pressure(self, gas_mass: Array) -> Array: ...

    def asdict(self) -> dict: ...
