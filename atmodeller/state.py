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

from dataclasses import asdict
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxmod.constants import GRAVITATIONAL_CONSTANT
from jaxmod.units import unit_conversion
from jaxmod.utils import as_j64
from jaxtyping import Array, ArrayLike, Bool, Float


class ThermodynamicState(eqx.Module):
    r"""A generic thermodynamic state

    This must adhere to :class:`~atmodeller.interfaces.ThermodynamicStateProtocol`.

    Args:
        temperature: Temperature in K
        pressure: Pressure in bar
        mass: Mass in kg. Defaults to ``1`` kg.
        melt_fraction: Melt fraction by weight in kg kg\ :sup:`-1`. Defaults to ``1``.
        molar_mass: Molar mass of the silicate in kg mol\ :sup:`-1`. Defaults to
            60 g mol\ :sup:`-1`, which is a typical value for silicate melts based on SiO\ :sub:`2`.
    """

    temperature: Float[Array, "..."]
    """Temperature in K"""
    pressure: Float[Array, "..."]
    """Pressure in bar"""
    mass: Float[Array, "..."]
    """Mass in kg"""
    melt_fraction: Float[Array, "..."]
    r"""Mass fraction of melt in kg kg\ :sup:`-1`"""
    molar_mass: Float[Array, "..."]
    r"""Molar mass of the silicate in kg mol\ :sup:`-1`"""

    def __init__(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass: ArrayLike = 1,
        melt_fraction: ArrayLike = 1,
        molar_mass: ArrayLike = 60e-3,
    ):
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.mass = as_j64(mass)
        self.melt_fraction = as_j64(melt_fraction)
        self.molar_mass = as_j64(molar_mass)

    @property
    def melt_mass(self) -> Float[Array, "..."]:
        """Mass of the melt in kg"""
        return self.mass * self.melt_fraction

    @property
    def melt_moles(self) -> Float[Array, "..."]:
        """Moles of the melt"""
        return self.melt_mass / self.molar_mass

    @property
    def solid_mass(self) -> Float[Array, "..."]:
        """Mass of the solid in kg"""
        return self.mass * (1.0 - self.melt_fraction)

    @property
    def solid_moles(self) -> Float[Array, "..."]:
        """Moles of the solid"""
        return self.solid_mass / self.molar_mass

    def get_pressure(self, gas_mass: Optional[Float[Array, "..."]] = None) -> Float[Array, "..."]:
        """Gets the pressure.

        Args:
            gas_mass: Gas mass in kg. Unused but required by the interface. Defaults to ``None``.

        Returns:
            Pressure in bar
        """
        del gas_mass

        return self.pressure

    def asdict(
        self, gas_mass: Optional[Float[Array, "..."]] = None
    ) -> dict[str, Float[Array, "..."]]:
        """Gets a dictionary of the values as NumPy arrays.

        Args:
            gas_mass: Gas mass in kg. Unused but required by the interface. Defaults to ``None``.

        Returns:
            A dictionary of the values
        """
        del gas_mass

        base_dict: dict[str, Float[Array, "..."]] = asdict(self)
        base_dict["melt_mass"] = self.melt_mass
        base_dict["solid_mass"] = self.solid_mass

        return base_dict


class ThinAtmospherePlanet(eqx.Module):
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
