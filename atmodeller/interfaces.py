# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Interfaces for thermodynamic models and constraints.

This module defines formal protocol classes (via :class:`typing.Protocol`) that specify the
expected interfaces for different thermodynamic components.

It also contains :class:`ChemicalSpeciesData`, a concrete :class:`equinox.Module` that holds
per-species formula and composition data. It lives here rather than in
:mod:`atmodeller.containers.py`` because :mod:`containers.py` imports from this module — moving it
there would create a circular import.
"""

from typing import Optional, Protocol, TypeVar, runtime_checkable

import equinox as eqx
from jaxmod.units import unit_conversion
from jaxtyping import Array, ArrayLike, Bool, Float
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
    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Log activity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log activity, which is dimensionless
        """
        ...


@runtime_checkable
class SpeciesProtocol(Protocol):
    """Protocol for a chemical species that participates in reactions"""

    @property
    def activity(self) -> ActivityProtocol:
        """Activity model"""
        ...

    @property
    def data(self) -> ChemicalSpeciesData:
        """Chemical species data"""
        ...

    @property
    def number_solution(self) -> int:
        """Number of solution quantities"""
        ...

    @property
    def solve_for_stability(self) -> bool:
        """Whether to solve for stability"""
        ...

    @property
    def include_in_phase_mass(self) -> bool:
        """Whether the species is included in phase-level mass, mole, and fraction aggregations"""
        ...


TSpecies_co = TypeVar("TSpecies_co", bound=SpeciesProtocol, covariant=True)


@runtime_checkable
class FugacityConstraintProtocol(Protocol):
    def active(self) -> Bool[Array, "..."]:
        """True if the constraint is active, otherwise False"""
        ...

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        """Log fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        ...


@runtime_checkable
class RedoxBufferProtocol(FugacityConstraintProtocol, Protocol):
    evaluation_pressure: Optional[float]
    """Pressure at which to evaluate the buffer, or None to use the total pressure"""

    @property
    def log10_shift(self) -> Float[Array, "..."]:
        """Log10 shift relative to the buffer"""
        ...

    def log10_fugacity_buffer(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> Float[Array, "..."]:
        """Log10 fugacity at the unshifted buffer

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log10 fugacity at the buffer
        """
        ...

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Float[Array, "..."]:
        """Log10 fugacity including any shift

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log10 fugacity
        """
        ...


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
    ) -> Float[Array, "..."]:
        r"""Concentration in ppmw

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K. Defaults to ``None`` for not used.
            pressure: Pressure in bar. Defaults to ``None`` for not used.
            fO2: Oxygen fugacity in bar. Defaults to ``None`` for not used.

        Returns:
            Concentration in ppmw
        """
        ...

    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> Float[Array, "..."]:
        """Wrapper to pass concentration arguments by position to use with JAX lax.switch

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K
            pressure: Pressure in bar
            fO2: Oxygen fugacity in bar

        Returns:
            Concentration in ppmw
        """
        ...


@runtime_checkable
class ThermodynamicStateProtocol(Protocol):
    @property
    def mass(self) -> Float[Array, "..."]:
        """Total mass in kg"""
        ...

    @property
    def melt_fraction(self) -> Float[Array, "..."]:
        """Melt mass fraction"""
        ...

    @property
    def melt_mass(self) -> Float[Array, "..."]:
        """Melt mass in kg"""
        ...

    @property
    def melt_moles(self) -> Float[Array, "..."]:
        """Moles of melt"""
        ...

    @property
    def molar_mass(self) -> Float[Array, "..."]:
        """Molar mass of the background in kg/mol"""
        ...

    @property
    def solid_mass(self) -> Float[Array, "..."]:
        """Solid mass in kg"""
        ...

    @property
    def solid_moles(self) -> Float[Array, "..."]:
        """Moles of solid"""
        ...

    @property
    def temperature(self) -> Float[Array, "..."]:
        """Temperature in K"""
        ...

    def get_pressure(
        self, optional_gas_mass: Optional[Float[Array, "..."]] = None
    ) -> Float[Array, "..."]:
        """Pressure in bar"""
        ...

    def asdict(
        self, gas_mass: Optional[Float[Array, "..."]] = None
    ) -> dict[str, Float[Array, "..."]]:
        """Dictionary representation"""
        ...
