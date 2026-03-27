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

from collections.abc import Iterable
from dataclasses import asdict
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxmod.constants import GRAVITATIONAL_CONSTANT
from jaxmod.type_aliases import FloatArray
from jaxmod.units import unit_conversion
from jaxtyping import Array, ArrayLike, Bool, Float

from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.jaxhelper import as_j64
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase
from atmodeller.reactions import PhaseSystem, ReactionSystem


@runtime_checkable
class ThermodynamicStateProtocol(Protocol):
    reaction_system: ReactionSystem

    @property
    def phase_system(self) -> PhaseSystem: ...

    @property
    def pressure(self) -> Float[Array, "..."]:
        """Pressure in bar

        Note:
            This should not be used directly; use :meth:`get_pressure` instead to ensure the
            correct pressure is used based on the state of the system.
        """
        ...

    @property
    def temperature(self) -> Float[Array, "..."]:
        """Temperature in K"""
        ...

    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> Float[Array, "..."]:
        """Pressure in bar"""
        ...

    def asdict(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> dict[str, Float[Array, "..."]]:
        """Dictionary representation"""
        ...


class ThermodynamicState(eqx.Module):
    """A generic thermodynamic state

    This must adhere to :class:`~atmodeller.interfaces.ThermodynamicStateProtocol`.

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
    """Pressure in bar"""

    # For helpful typing information
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
        molar_mass: ArrayLike = 60e-3,
        melt_species: Iterable[SpeciesProtocol] = (),
        solid_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
    ) -> "ThermodynamicState":
        r"""Creates a new instance.

        Args:
            gas_species: Iterable of species in the gas phase
            pressure: Pressure in bar
            mass: Mass in kg. Defaults to ``1`` kg (reference unit mass).
            melt_fraction: Melt fraction. Defaults to ``1.0``.
            surface_radius: Radius of the planetary surface in m. Defaults to ``6371000`` m
                (Earth).
            temperature: Temperature in K. Defaults to ``2000`` K.
            molar_mass: Molar mass. Defaults to 60 g mol\ :sup:`-1`, which is a typical value for
                silicate melts based on SiO\ :sub:`2`.
            melt_species: Iterable of species in the melt phase
            solid_species: Iterable of species in the solid phase
            condensates: Iterable of pure phases representing condensates in the system

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

    @property
    def phase_system(self) -> PhaseSystem:
        """Phase system representing the thermodynamic state of the planetary body"""
        return self.reaction_system.phase_system

    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        """Gets the pressure.

        Returns:
            Pressure in bar
        """
        del log_number_moles

        return self.pressure

    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary of the values as NumPy arrays.

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
        # base_dict["metallic_core_mass"] = self.metallic_core_mass
        # base_dict["surface_area"] = self.surface_area
        # base_dict["surface_gravity"] = self.get_surface_gravity(log_number_moles)
        base_dict["temperature"] = self.temperature
        base_dict["pressure"] = self.pressure

        return base_dict


class ThinAtmospherePlanet(eqx.Module):
    """A new planet class

    This must adhere to :class:`~atmodeller.interfaces.ThermodynamicStateProtocol`.

    Args:
        reaction_system: Reaction system representing the thermodynamic state of the planetary body
        surface_radius: Radius of the surface in m
        metallic_core_mass: Metallic core mass in kg
        temperature: Temperature in K
        pressure: Pressure in bar
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

    # For helpful typing information
    def __init__(
        self,
        reaction_system: ReactionSystem,
        surface_radius: ArrayLike = 6371000,
        metallic_core_mass: ArrayLike = 1.7637387774048892e24,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = jnp.nan,
    ):
        self.reaction_system = reaction_system
        self.surface_radius = as_j64(surface_radius)
        self.metallic_core_mass = as_j64(metallic_core_mass)
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)

    @classmethod
    def create(
        cls,
        gas_species: Iterable[ChemicalSpecies],
        *,
        planet_mass: ArrayLike = 5.972e24,
        core_mass_fraction: ArrayLike = 0.295334691460966,
        mantle_melt_fraction: ArrayLike = 1.0,
        surface_radius: ArrayLike = 6371000,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = jnp.nan,
        molar_mass: ArrayLike = 60e-3,
        melt_species: Iterable[SpeciesProtocol] = (),
        solid_species: Iterable[SpeciesProtocol] = (),
        condensates: Iterable[PurePhase] = (),
    ):
        r"""Creates a new instance.

        Args:
            gas_species: Iterable of species in the gas phase
            planet_mass: Mass of the planet in kg. Defaults to ``5.972e24`` kg (Earth).
            core_mass_fraction: Mass fraction of the iron core relative to the planetary mass.
                Defaults to ``0.295334691460966`` kg kg\ :sup:`-1
            mantle_melt_fraction: Mantle melt fraction. Defaults to ``1.0``.
            surface_radius: Radius of the planetary surface in m. Defaults to ``6371000`` m
                (Earth).
            temperature: Temperature in K. Defaults to ``2000`` K.
            pressure: Pressure in bar. Defaults to ``jnp.nan`` to solve for the mechanical pressure
                balance at the surface.
            molar_mass: Molar mass. Defaults to 60 g mol\ :sup:`-1`, which is a typical value for
                silicate melts based on SiO\ :sub:`2`.
            melt_species: Iterable of species in the melt phase
            solid_species: Iterable of species in the solid phase
            condensates: Iterable of pure phases representing condensates in the system

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

        return cls(reaction_system, surface_radius, metallic_core_mass, temperature, pressure)

    @property
    def phase_system(self) -> PhaseSystem:
        """Phase system representing the thermodynamic state of the planetary body"""
        return self.reaction_system.phase_system

    @property
    def surface_area(self) -> FloatArray:
        """Surface area"""
        return 4.0 * jnp.pi * jnp.square(self.surface_radius)

    def get_surface_gravity(self, log_number_moles: Float[Array, "... n_species"]) -> FloatArray:
        r"""Gets the surface gravity.

        Computes the surface gravity from the mass of condensed phases and the radius of the
        planet.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            Surface gravity in m s\ :sup:`-2`
        """
        # Melt mass
        log_number_moles_melt: Float[Array, "... melt_species"] = log_number_moles[
            ..., self.phase_system.melt_slice
        ]
        # jax.debug.print("log_number_moles_melt = {out}", out=log_number_moles_melt)
        melt_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.melt.get_log_phase_mass(log_number_moles_melt)
        )
        # jax.debug.print("melt_mass = {out}", out=melt_mass)

        # Solid mass
        log_number_moles_solid: Float[Array, "... solid_species"] = log_number_moles[
            ..., self.phase_system.solid_slice
        ]
        # jax.debug.print("log_number_moles_solid = {out}", out=log_number_moles_solid)
        solid_mass: Float[Array, "... 1"] = jnp.exp(
            self.phase_system.solid.get_log_phase_mass(log_number_moles_solid)
        )
        # jax.debug.print("solid_mass = {out}", out=solid_mass)

        planet_mass = melt_mass + solid_mass + self.metallic_core_mass[..., None]
        planet_mass_squeeze: FloatArray = jnp.squeeze(planet_mass, axis=-1)
        # jax.debug.print("planet_mass_squeeze = {out}", out=planet_mass_squeeze)

        surface_gravity: FloatArray = (
            GRAVITATIONAL_CONSTANT * planet_mass_squeeze / jnp.square(self.surface_radius)
        )
        # jax.debug.print("surface_gravity = {out}", out=surface_gravity)

        return surface_gravity

    def get_pressure(self, log_number_moles: Float[Array, "... n_species"]) -> Float[Array, "..."]:
        """Gets the pressure.

        A pressure is used if specified, otherwise the default behaviour is to compute the
        pressure from the mechanical pressure balance at the planetary surface assuming the thin
        atmosphere approximation. That is, the surface gravity is computed from the mass of the
        planet alone and is assumed to act on all the mass of the atmosphere.

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
        gas_mass_squeeze: Float[Array, "..."] = jnp.squeeze(gas_mass, axis=-1)
        # jax.debug.print("gas_mass = {out}", out=gas_mass_squeeze)

        surface_gravity: Float[Array, "..."] = self.get_surface_gravity(log_number_moles)
        mechanical_pressure: Float[Array, "..."] = (
            gas_mass_squeeze * surface_gravity / self.surface_area * unit_conversion.Pa_to_bar
        )
        # jax.debug.print("mechanical_pressure = {out}", out=mechanical_pressure)

        pressure: Float[Array, "..."] = jnp.where(
            pressure_specified, self.pressure, mechanical_pressure
        )
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure

    def asdict(self, log_number_moles: Float[Array, "... n_species"]) -> dict[str, Array]:
        """Gets a dictionary of the values as NumPy arrays.

        Args:
            log_number_moles: Log number of moles for all species in the system

        Returns:
            A dictionary of the values
        """
        # FIXME: This breaks because of nested phase system
        # base_dict: dict[str, Array] = asdict(self)
        base_dict = {}
        # TODO: Reinstate these outputs?
        # base_dict["mantle_mass"] = self.mantle_mass
        # base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        # base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["metallic_core_mass"] = self.metallic_core_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.get_surface_gravity(log_number_moles)
        base_dict["temperature"] = self.temperature
        base_dict["pressure"] = self.pressure

        return base_dict


class ThinAtmospherePlanetPrevious(eqx.Module):
    r"""A planet with a thin atmosphere.

    In this context, "thin atmosphere" means that the surface gravity is determined solely by the
    mass of the planet (i.e., the solid/liquid body), and is not compensated or significantly
    altered by the presence of an extended or massive atmosphere above. This is appropriate for
    cases where the atmospheric mass is much less than the planetary mass, so the gravitational
    acceleration at the surface is set by the planet alone, not by any self-gravitating atmospheric
    shell.

    This must adhere to :class:`~atmodeller.interfaces.ThermodynamicStateProtocol`.

    Default values are for a fully molten Earth.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to ``5.972e24`` kg (Earth).
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to ``0.295334691460966`` kg kg\ :sup:`-1` (Earth).
        mantle_melt_fraction: Mass fraction of the mantle that is molten in kg kg\ :sup:`-1`.
            Defaults to ``1.0``.
        surface_radius: Radius of the planetary surface in m. Defaults to ``6371000`` m (Earth).
        temperature: Temperature in K. Defaults to ``2000`` K.
        pressure: Pressure in bar. Defaults to ``np.nan`` to solve for the mechanical pressure
            balance at the surface.
        molar_mass: Molar mass of the silicate in kg mol\ :sup:`-1`. Defaults to
            60 g mol\ :sup:`-1`, which is a typical value for silicate melts based on SiO\ :sub:`2`.
    """

    planet_mass: Float[Array, "..."]
    """Mass of the planet in kg"""
    core_mass_fraction: Float[Array, "..."]
    r"""Mass fraction of the core relative to the planetary mass in kg kg\ :sup:`-1`"""
    mantle_melt_fraction: Float[Array, "..."]
    r"""Mass fraction of the molten mantle in kg kg\ :sup:`-1`"""
    surface_radius: Float[Array, "..."]
    """Radius of the surface in m"""
    temperature: Float[Array, "..."]
    """Temperature in K"""
    pressure: Float[Array, "..."]
    """Pressure in bar"""
    molar_mass: Float[Array, "..."]
    r"""Molar mass of the silicate in kg mol\ :sup:`-1`"""

    def __init__(
        self,
        planet_mass: ArrayLike = 5.972e24,
        core_mass_fraction: ArrayLike = 0.295334691460966,
        mantle_melt_fraction: ArrayLike = 1.0,
        surface_radius: ArrayLike = 6371000,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = jnp.nan,
        molar_mass: ArrayLike = 60e-3,
    ):
        self.planet_mass = as_j64(planet_mass)
        self.core_mass_fraction = as_j64(core_mass_fraction)
        self.mantle_melt_fraction = as_j64(mantle_melt_fraction)
        self.surface_radius = as_j64(surface_radius)
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.molar_mass = as_j64(molar_mass)

    @property
    def mantle_mass(self) -> Float[Array, "..."]:
        """Mantle mass in kg"""
        return self.planet_mass * self.mantle_mass_fraction

    @property
    def mantle_moles(self) -> Float[Array, "..."]:
        """Moles of the mantle"""
        return self.mantle_mass / self.molar_mass

    @property
    def mantle_mass_fraction(self) -> Float[Array, "..."]:
        r"""Mantle mass fraction in kg kg\ :sup:`-1`"""
        return 1 - self.core_mass_fraction

    @property
    def mantle_melt_mass(self) -> Float[Array, "..."]:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_melt_moles(self) -> Float[Array, "..."]:
        """Moles of the molten mantle"""
        return self.mantle_melt_mass / self.molar_mass

    @property
    def mantle_solid_mass(self) -> Float[Array, "..."]:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def mantle_solid_moles(self) -> Float[Array, "..."]:
        """Moles of the solid mantle"""
        return self.mantle_solid_mass / self.molar_mass

    @property
    def surface_area(self) -> Float[Array, "..."]:
        """Surface area"""
        return 4.0 * jnp.pi * jnp.square(self.surface_radius)

    @property
    def surface_gravity(self) -> Float[Array, "..."]:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / jnp.square(self.surface_radius)

    # The following properties ensure compliance with ThermodynamicStateProtocol
    @property
    def mass(self) -> Float[Array, "..."]:
        """Mantle mass in kg (alias for :attr:`mantle_mass`)"""
        return self.mantle_mass

    @property
    def melt_fraction(self) -> Float[Array, "..."]:
        r"""Mantle melt fraction in kg kg\ :sup:`-1` (alias for :attr:`mantle_melt_fraction`)"""
        return self.mantle_melt_fraction

    @property
    def melt_mass(self) -> Float[Array, "..."]:
        """Mass of the molten mantle in kg (alias for :attr:`mantle_melt_mass`)"""
        return self.mantle_melt_mass

    @property
    def melt_moles(self) -> Float[Array, "..."]:
        """Moles of the molten mantle (alias for :attr:`mantle_melt_moles`)"""
        return self.mantle_melt_moles

    @property
    def solid_mass(self) -> Float[Array, "..."]:
        """Mass of the solid mantle in kg (alias for :attr:`mantle_solid_mass`)"""
        return self.mantle_solid_mass

    @property
    def solid_moles(self) -> Float[Array, "..."]:
        """Moles of the solid mantle (alias for :attr:`mantle_solid_moles`)"""
        return self.mantle_solid_moles

    def get_pressure(self, gas_mass: Float[Array, "..."]) -> Float[Array, "..."]:
        """Gets the pressure.

        A pressure is used if specified, otherwise the default behaviour is to compute the
        pressure from the mechanical pressure balance at the planetary surface assuming the thin
        atmosphere approximation. That is, the surface gravity is computed from the mass of the
        planet alone and is assumed to act on all the mass of the atmosphere.

        Args:
            gas_mass: Gas mass in kg

        Returns:
            Pressure in bar
        """
        pressure_specified: Bool[Array, "..."] = ~jnp.isnan(self.pressure)

        mechanical_pressure: Float[Array, "..."] = (
            gas_mass * self.surface_gravity / self.surface_area * unit_conversion.Pa_to_bar
        )
        # jax.debug.print("mechanical_pressure = {out}", out=mechanical_pressure)

        pressure: Float[Array, "..."] = jnp.where(
            pressure_specified, self.pressure, mechanical_pressure
        )
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure

    def asdict(self, gas_mass: Float[Array, "..."]) -> dict[str, Array]:
        """Gets a dictionary of the values as NumPy arrays.

        Args:
            gas_mass: Gas mass in kg

        Returns:
            A dictionary of the values
        """
        base_dict: dict[str, Array] = asdict(self)
        base_dict["pressure"] = self.get_pressure(gas_mass)
        base_dict["mantle_mass"] = self.mantle_mass
        base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        return base_dict


# The only planet supported so far is one with a thin atmosphere
Planet = ThinAtmospherePlanet
