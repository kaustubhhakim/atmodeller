# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Thermodynamic state representations for planetary modeling.

This module provides JAX/Equinox-compatible classes for representing the thermodynamic state of a
system or planet, including temperature, pressure, mass, melt fraction, and molar mass. All fields
are stored as JAX arrays for seamless use with JAX transformations (jit, grad, vmap) and to avoid
unnecessary recompilation.

Key features:

- **JAX/Equinox compatibility:** All fields are JAX arrays, using Equinox modules and field
  converters.
- **Protocol adherence:** Classes are designed to comply with
  :class:`~atmodeller.interfaces.ThermodynamicStateProtocol` for interoperability.
- **Planetary modeling:** Includes a generic :class:`ThermodynamicState` and a
  :class:`ThinAtmospherePlanet` model with Earth-like defaults and surface pressure calculation.
- **Convenience methods:** Properties for melt/solid mass and moles, surface gravity, and area;
  dictionary export for downstream use.

Classes:

- :class:`ThermodynamicState`: Generic thermodynamic state for any system.
- :class:`ThinAtmospherePlanet`: Earth-like planet with a thin atmosphere and surface pressure
  calculation.
- :class:`Planet`: Alias for :class:`ThinAtmospherePlanet` (for future extensibility).

All classes are designed for use in JAX-based scientific workflows and can be extended for more
complex planetary models.
"""

from abc import abstractmethod
from collections.abc import Iterable
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float
from scipy.constants import gravitational_constant

from atmodeller import override
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.jax_utils import FloatArray, as_j64
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase
from atmodeller.reactions import PhaseSystem, ReactionSystem
from atmodeller.sci_utils import EARTH_MASS, EARTH_RADIUS, SIO2_MOLAR_MASS, unit_conversion


class BaseThermodynamicState(eqx.Module):
    reaction_system: ReactionSystem
    """Reaction system representing the thermodynamic state of the system"""
    temperature: FloatArray
    """Temperature in K"""
    pressure: FloatArray
    """Pressure in bar. Should not be used directly; use :meth:`get_pressure` instead"""

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None: ...

    @property
    def phase_system(self) -> PhaseSystem:
        """Phase system representing the thermodynamic state of the planetary body"""
        return self.reaction_system.phase_system

    @abstractmethod
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Pressure in bar"""
        ...

    @abstractmethod
    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary representation."""
        ...


class ThermodynamicState(BaseThermodynamicState):
    """A generic thermodynamic state

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the system
        temperature: Temperature in K
        pressure: Pressure in bar
    """

    reaction_system: ReactionSystem
    """Reaction system representing the thermodynamic state of the system"""
    temperature: FloatArray
    """Temperature in K"""
    pressure: FloatArray
    """Pressure in bar. Should not be used directly; use :meth:`get_pressure` instead"""

    # For helpful typing information
    @override
    def __init__(
        self, reaction_system: ReactionSystem, temperature: ArrayLike, pressure: ArrayLike
    ):
        self.reaction_system = reaction_system
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)

    @classmethod
    def create(
        cls,
        gas_species: Iterable[ChemicalSpecies],
        pressure: ArrayLike,
        *,
        mass: ArrayLike = 1.0,
        melt_fraction: ArrayLike = 0.0,
        temperature: ArrayLike = 2000,
        molar_mass: ArrayLike = SIO2_MOLAR_MASS,
        melt_species: Iterable[SpeciesProtocol] = (),
        solid_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
    ) -> Self:
        r"""Creates a new instance.

        Args:
            gas_species: Iterable of species in the gas phase
            pressure: Pressure in bar
            mass: Mass in kg. Defaults to ``1`` kg (i.e., reference unit mass).
            melt_fraction: Melt fraction. Defaults to ``1`` (i.e., fully molten).
            temperature: Temperature in K. Defaults to ``2000``.
            molar_mass: Molar mass. Defaults to :data:`~atmodeller.sci_utils.SIO2_MOLAR_MASS`.
            melt_species: Iterable of species in the melt phase. Defaults to an empty tuple.
            solid_species: Iterable of species in the solid phase. Defaults to an empty tuple.
            condensates: Iterable of pure phases representing condensates in the system. Defaults
                to an empty tuple.

        Returns:
            An instance
        """
        background_melt_mass: ArrayLike = mass * melt_fraction
        background_solid_mass: ArrayLike = mass * (1 - melt_fraction)

        gas: GasPhase = GasPhase(gas_species)
        melt: MeltPhase = MeltPhase(melt_species, background_melt_mass, molar_mass)
        solid: SolidPhase = SolidPhase(solid_species, background_solid_mass, molar_mass)
        phase_system = PhaseSystem(gas, melt=melt, solid=solid, condensates=tuple(condensates))
        reaction_system: ReactionSystem = ReactionSystem(phase_system)

        return cls(reaction_system, temperature, pressure)

    @override
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        Returns:
            Pressure in bar
        """
        del log_number_moles

        return self.pressure

    @override
    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary representation.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            A dictionary of the values
        """
        del log_number_moles

        # FIXME: This breaks because of nested phase system
        # base_dict: dict[str, Array] = asdict(self)
        base_dict = {}
        # TODO: Reinstate these outputs?
        # base_dict["mantle_mass"] = self.mantle_mass
        # base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        # base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["temperature"] = self.temperature
        base_dict["pressure"] = self.pressure

        return base_dict


class BasePlanet(BaseThermodynamicState):
    """A planet.

    Default values are for a fully molten Earth.

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface in m
        metallic_core_mass: Metallic core mass in kg
        temperature: Temperature in K
        pressure: Pressure in bar
        background_planet_mass: Planet mass in kg from only the background melt and solid mass,
            plus the metallic core mass. This value will only be equal to the actual planet mass if
            ``include_in_phase_mass`` is ``False`` for all species in the melt and solid phases.
    """

    reaction_system: ReactionSystem
    """Reaction system representing the thermodynamic state of the planetary body"""
    surface_radius: FloatArray
    """Radius of the surface in m"""
    metallic_core_mass: FloatArray
    """Metallic core mass in kg"""
    temperature: FloatArray
    """Temperature in K"""
    pressure: FloatArray
    """Pressure in bar"""
    background_planet_mass: FloatArray
    """Planet mass in kg from only the background melt and solid mass, plus the core mass"""

    # For helpful typing information
    @override
    def __init__(
        self,
        reaction_system: ReactionSystem,
        surface_radius: ArrayLike,
        metallic_core_mass: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        background_planet_mass: ArrayLike,
    ):
        self.reaction_system = reaction_system
        self.surface_radius = as_j64(surface_radius)
        self.metallic_core_mass = as_j64(metallic_core_mass)
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.background_planet_mass = as_j64(background_planet_mass)

    @classmethod
    def create(
        cls,
        gas_species: Iterable[ChemicalSpecies],
        *,
        planet_mass: ArrayLike = EARTH_MASS,
        core_mass_fraction: ArrayLike = 0.295334691460966,
        mantle_melt_fraction: ArrayLike = 1.0,
        surface_radius: ArrayLike = EARTH_RADIUS,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = jnp.nan,
        molar_mass: ArrayLike = 60e-3,
        melt_species: Iterable[SpeciesProtocol] = (),
        solid_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
    ) -> Self:
        r"""Creates a new instance.

        Args:
            gas_species: Iterable of species in the gas phase
            planet_mass: Mass of the planet in kg. Defaults to
                :data:`~atmodeller.sci_utils.EARTH_MASS`.
            core_mass_fraction: Mass fraction of the iron core relative to the planetary mass.
                Defaults to ``0.295334691460966`` kg kg\ :sup:`-1` (Earth).
            mantle_melt_fraction: Mantle melt fraction. Defaults to ``1.0``.
            surface_radius: Radius of the planetary surface in m. Defaults to
                :data:`~atmodeller.sci_utils.EARTH_RADIUS`.
            temperature: Temperature in K. Defaults to ``2000`` K.
            pressure: Pressure in bar. Defaults to ``NaN`` to solve for the mechanical pressure
                balance at the surface.
            molar_mass: Molar mass. Defaults to :data:`~atmodeller.sci_utils.SIO2_MOLAR_MASS`.
            melt_species: Iterable of species in the melt phase. Defaults to an empty tuple.
            solid_species: Iterable of species in the solid phase. Defaults to an empty tuple.
            condensates: Iterable of pure phases representing condensates in the system. Defaults
                to an empty tuple.

        Returns:
            An instance
        """
        mantle_mass: ArrayLike = planet_mass * (1 - core_mass_fraction)
        background_melt_mass: ArrayLike = mantle_mass * mantle_melt_fraction
        background_solid_mass: ArrayLike = mantle_mass * (1 - mantle_melt_fraction)
        metallic_core_mass: ArrayLike = planet_mass * core_mass_fraction

        gas: GasPhase = GasPhase(gas_species)
        melt: MeltPhase = MeltPhase(melt_species, background_melt_mass, molar_mass)
        solid: SolidPhase = SolidPhase(solid_species, background_solid_mass, molar_mass)
        phase_system = PhaseSystem(gas, melt=melt, solid=solid, condensates=tuple(condensates))
        reaction_system: ReactionSystem = ReactionSystem(phase_system)

        return cls(
            reaction_system, surface_radius, metallic_core_mass, temperature, pressure, planet_mass
        )

    @property
    def surface_area(self) -> FloatArray:
        """Surface area"""
        return 4.0 * jnp.pi * jnp.square(self.surface_radius)

    def get_mantle_melt_mass(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the mass of the molten mantle.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Mantle melt mass in kg
        """
        log_number_moles_melt: Float[Array, "... melt_species"] = log_number_moles[
            ..., self.phase_system.melt_slice
        ]
        melt_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.melt.get_log_phase_mass(log_number_moles_melt)
        )
        melt_mass_squeeze: FloatArray = jnp.squeeze(melt_mass, axis=-1)

        return melt_mass_squeeze

    def get_mantle_solid_mass(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the mass of the solid mantle.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Solid mantle mass in kg
        """
        log_number_moles_solid: Float[Array, "... solid_species"] = log_number_moles[
            ..., self.phase_system.solid_slice
        ]
        solid_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.solid.get_log_phase_mass(log_number_moles_solid)
        )
        solid_mass_squeeze: FloatArray = jnp.squeeze(solid_mass, axis=-1)

        return solid_mass_squeeze

    def get_planet_mass(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the planet mass.

        Computes the planet mass from the mass of the condensed phases and the metallic core.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Planet mass in kg
        """
        mantle_melt_mass: FloatArray = self.get_mantle_melt_mass(log_number_moles)
        mantle_solid_mass: FloatArray = self.get_mantle_solid_mass(log_number_moles)
        planet_mass = mantle_melt_mass + mantle_solid_mass + self.metallic_core_mass

        return planet_mass

    def get_surface_gravity(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        r"""Gets the surface gravity.

        Computes the surface gravity from the mass of condensed phases and the radius of the
        planet.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Surface gravity in m s\ :sup:`-2`
        """
        planet_mass: FloatArray = self.get_planet_mass(log_number_moles)
        # jax.debug.print("planet_mass = {out}", out=planet_mass_squeeze)

        surface_gravity: FloatArray = (
            gravitational_constant * planet_mass / jnp.square(self.surface_radius)
        )
        # jax.debug.print("surface_gravity = {out}", out=surface_gravity)

        return surface_gravity

    @override
    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary representation.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            A dictionary of the values
        """
        base_dict = {}
        base_dict["mantle_melt_mass"] = self.get_mantle_melt_mass(log_number_moles)
        base_dict["mantle_solid_mass"] = self.get_mantle_solid_mass(log_number_moles)
        base_dict["mantle_mass"] = base_dict["mantle_melt_mass"] + base_dict["mantle_solid_mass"]
        base_dict["metallic_core_mass"] = self.metallic_core_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.get_surface_gravity(log_number_moles)
        base_dict["planet_mass"] = self.get_planet_mass(log_number_moles)
        base_dict["temperature"] = self.temperature
        base_dict["pressure"] = self.get_pressure(log_number_moles)

        return base_dict


class ThinAtmospherePlanet(BasePlanet):
    """A planet with a thin atmosphere.

    In this context, "thin atmosphere" means the atmosphere is shallow compared with the planetary
    radius, so gravity is approximated as constant throughout the atmospheric column. Surface
    gravity is therefore set by the planetary body (condensed phases plus metallic core), and the
    atmosphere is treated as non-self-gravitating in the pressure balance.

    Default values are for a fully molten Earth.

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface in m
        metallic_core_mass: Metallic core mass in kg
        temperature: Temperature in K
        pressure: Pressure in bar
        background_planet_mass: Planet mass in kg from only the background melt and solid mass, plus
            the metallic core mass. This value will only be equal to the actual planet mass if
            ``include_in_phase_mass`` is ``False`` for all species in the melt and solid phases.
    """

    @override
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        A pressure is used if specified, otherwise the default behaviour is to compute the
        pressure from the mechanical pressure balance at the planetary surface assuming the thin
        atmosphere approximation. That is, gravity is computed from the planetary body and treated
        as constant throughout the atmospheric column.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Pressure in bar
        """
        pressure_specified: Bool[Array, "..."] = ~jnp.isnan(self.pressure)

        log_number_moles_gas: Float[Array, "... gas_species"] = log_number_moles[
            ..., self.phase_system.gas_slice
        ]
        # jax.debug.print("log_number_moles_gas = {out}", out=log_number_moles_gas)
        gas_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.gas.get_log_phase_mass(log_number_moles_gas)
        )
        gas_mass_squeeze: FloatArray = jnp.squeeze(gas_mass, axis=-1)
        # jax.debug.print("gas_mass = {out}", out=gas_mass_squeeze)

        surface_gravity: FloatArray = self.get_surface_gravity(log_number_moles)
        mechanical_pressure: FloatArray = (
            gas_mass_squeeze * surface_gravity / self.surface_area * unit_conversion.Pa_to_bar
        )
        # jax.debug.print("mechanical_pressure = {out}", out=mechanical_pressure)

        pressure: FloatArray = jnp.where(pressure_specified, self.pressure, mechanical_pressure)
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure


class PressureScalingLawPlanet(BasePlanet):
    """A planet with a scaling law for the atmospheric pressure.

    A pressure is used if specified, otherwise it is computed from the scaling law:

    .. math::
        P_{\\text{surface}} = 1 \\times 10^6 \\frac{M_{\\text{atm}}}{M_{\\text{p}}} \\left( \\frac{M_\\text{p}}{M_{\\text{Earth}}}\\right)^{2/3}

    Default values are for a fully molten Earth.

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface in m
        metallic_core_mass: Metallic core mass in kg
        temperature: Temperature in K
        pressure: Pressure in bar
        background_planet_mass: Planet mass in kg from only the background melt and solid mass,
            plus the metallic core mass. This value will only be equal to the actual planet mass if
            ``include_in_phase_mass`` is ``False`` for all species in the melt and solid phases.
    """

    @override
    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        A pressure is used if specified, otherwise it is computed from the scaling law.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Pressure in bar
        """
        pressure_specified: Bool[Array, "..."] = ~jnp.isnan(self.pressure)

        log_number_moles_gas: Float[Array, "... gas_species"] = log_number_moles[
            ..., self.phase_system.gas_slice
        ]
        # jax.debug.print("log_number_moles_gas = {out}", out=log_number_moles_gas)
        gas_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.gas.get_log_phase_mass(log_number_moles_gas)
        )
        gas_mass_squeeze: FloatArray = jnp.squeeze(gas_mass, axis=-1)

        scaling_law: FloatArray = (
            1e6
            * (gas_mass_squeeze / self.get_planet_mass(log_number_moles))
            * (self.get_planet_mass(log_number_moles) / EARTH_MASS) ** (2 / 3)
        )

        pressure: FloatArray = jnp.where(pressure_specified, self.pressure, scaling_law)
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure


# The only planet supported so far is one with a thin atmosphere
Planet = ThinAtmospherePlanet
