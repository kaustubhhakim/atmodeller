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
"""Pytree containers that can be use by JAX

The leaves of pytrees must be JAX-compliant types, which excludes strings. So the preferred
approach is to encode the data as JAX-compatible types and provide properties and methods that can
reconstruct other desired quantities, notably strings and other objects. This ensures that similar 
functionality can remain together whilst accommodating the requirements of JAX-compliant pytrees.
"""
from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from typing import Callable, NamedTuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array, jit, lax
from jax.typing import ArrayLike
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller import (
    AVOGADRO,
    GRAVITATIONAL_CONSTANT,
    LOG_NUMBER_DENSITY_LOWER,
    LOG_NUMBER_DENSITY_UPPER,
    LOG_STABILITY_LOWER,
    LOG_STABILITY_UPPER,
)
from atmodeller.engine import get_log_number_density_from_log_pressure
from atmodeller.eos.classes import IdealGas
from atmodeller.interfaces import (
    ActivityProtocol,
    FugacityConstraintProtocol,
    SolubilityProtocol,
)
from atmodeller.solubility.library import NoSolubility
from atmodeller.thermodata import select_thermodata
from atmodeller.thermodata.core import CondensateActivity, SpeciesData
from atmodeller.utilities import OptxSolver, unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)

# region: Containers for traced parameters


class TracedParameters(NamedTuple):
    """Traced parameters

    Args:
        planet: Planet
        fugacity_constraints: Fugacity constraints
        mass_constraints: Mass constraints
    """

    planet: Planet
    """Planet"""
    fugacity_constraints: FugacityConstraints
    """Fugacity constraints"""
    mass_constraints: MassConstraints
    """Mass constraints"""


class Planet(NamedTuple):
    """Planet properties

    Default values are for a fully molten Earth.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to Earth.
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        surface_radius: Radius of the planetary surface in m. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
    """

    planet_mass: ArrayLike = 5.972e24
    """Mass of the planet in kg"""
    core_mass_fraction: ArrayLike = 0.295334691460966
    """Mass fraction of the core relative to the planetary mass in kg/kg"""
    mantle_melt_fraction: ArrayLike = 1.0
    """Mass fraction of the molten mantle in kg/kg"""
    surface_radius: ArrayLike = 6371000.0
    """Radius of the surface in m"""
    surface_temperature: ArrayLike = 2000.0
    """Temperature of the surface in K"""

    @property
    def mantle_mass(self) -> ArrayLike:
        """Mantle mass"""
        return self.planet_mass * (1.0 - self.core_mass_fraction)

    @property
    def mantle_melt_mass(self) -> ArrayLike:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_solid_mass(self) -> ArrayLike:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def surface_area(self) -> ArrayLike:
        """Surface area"""
        return 4.0 * jnp.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> ArrayLike:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2

    def asdict(self) -> dict[str, Array]:
        """Gets a dictionary of the values

        Returns:
            A dictionary of the values
        """
        base_dict: dict[str, ArrayLike] = self._asdict()
        base_dict["mantle_mass"] = self.mantle_mass
        base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        # Always return JAX arrays which are required for subsequent output operations
        converted_dict: dict[str, Array] = {
            key: jnp.asarray(value) for key, value in base_dict.items()
        }

        return converted_dict

    def vmap_axes(self) -> Self:
        """Gets vmap axes.

        Returns:
            vmap axes
        """
        vmap_axes: list[int | None] = []
        for field in self._fields:
            value: ArrayLike = getattr(self, field)
            if jnp.isscalar(value):
                vmap_axis: int | None = None
            else:
                vmap_axis = 0
            vmap_axes.append(vmap_axis)

        return Planet(*vmap_axes)  # type: ignore - container types are for data not axes


class ConstantFugacityConstraint(NamedTuple):
    """A constant fugacity constraint

    This must adhere to FugacityConstraintProtocol

    Args:
        fugacity: Fugacity
    """

    fugacity: ArrayLike

    @property
    def value(self) -> ArrayLike:
        return self.fugacity

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        del temperature
        del pressure
        return jnp.log(self.fugacity)


class FugacityConstraints(NamedTuple):
    """Fugacity constraints

    These are applied as constraints on the gas activity.

    Args:
        constraints: Fugacity constraints
    """

    constraints: ImmutableMap[str, FugacityConstraintProtocol]
    """Fugacity constraints"""

    @classmethod
    def create(
        cls,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
    ) -> Self:
        """Creates an instance

        Args:
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                None.

        Returns:
            An instance
        """
        if fugacity_constraints is None:
            init_dict: dict[str, FugacityConstraintProtocol] = {}
        else:
            init_dict = dict(fugacity_constraints)

        init_map: ImmutableMap[str, FugacityConstraintProtocol] = ImmutableMap(init_dict)

        return cls(init_map)

    def asdict(self, temperature: ArrayLike, pressure: ArrayLike) -> dict[str, Array]:
        """Gets a dictionary of the evaluated fugacity constraints.

        `temperature` and `pressure` should have a size equal to the number of solutions, which
        will ensure that the returned dictionary also has the same size. For this reason there is
        no need to vmap.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            A dictionary of the evaluated fugacity constraints
        """
        # This contains evaluated fugacity constraints in columns
        log_fugacity: Array = jnp.atleast_2d(self.log_fugacity(temperature, pressure)).T

        # Split the evaluated fugacity constraints by column
        out: dict[str, Array] = {
            f"{key}_fugacity": jnp.exp(log_fugacity[:, ii])
            for ii, key in enumerate(self.constraints)
        }

        return out

    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        fugacity_funcs: list[Callable] = [
            constraint.log_fugacity for constraint in self.constraints.values()
        ]

        def apply_fugacity_function(
            index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike
        ) -> Array:
            return lax.switch(
                index,
                fugacity_funcs,
                temperature,
                pressure,
            )

        vmap_apply_function: Callable = jax.vmap(apply_fugacity_function, in_axes=(0, None, None))
        indices: Array = jnp.arange(len(self.constraints))
        log_fugacity: Array = vmap_apply_function(indices, temperature, pressure)

        return log_fugacity

    @jit
    def log_number_density(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log number density

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log number density
        """
        log_fugacity: Array = self.log_fugacity(temperature, pressure)
        log_number_density: Array = get_log_number_density_from_log_pressure(
            log_fugacity, temperature
        )

        return log_number_density

    def vmap_axes(self) -> Self:
        """Gets vmap axes.

        Returns:
            vmap axes
        """
        constraints_vmap: dict[str, FugacityConstraintProtocol] = {}

        for key, constraint in self.constraints.items():
            if jnp.isscalar(constraint.value):
                vmap_axis: int | None = None
            else:
                vmap_axis = 0
            constraints_vmap[key] = type(constraint)(vmap_axis)  # type: ignore - container

        return FugacityConstraints(ImmutableMap(constraints_vmap))  # type: ignore - container

    def __bool__(self) -> bool:
        return bool(self.constraints)


class MassConstraints(NamedTuple):
    """Mass constraints

    Args:
        log_molecules: Log number of molecules of the species
    """

    log_molecules: ImmutableMap[str, ArrayLike]
    """Log number of molecules"""

    @classmethod
    def create(cls, mass_constraints: Mapping[str, ArrayLike] | None = None) -> Self:
        """Creates an instance

        Args:
            mass_constraints: Mapping of element name and mass constraint in kg. Defaults to None.

        Returns:
            An instance
        """
        if mass_constraints is None:
            init_dict: dict[str, ArrayLike] = {}
        else:
            init_dict = dict(mass_constraints)

        sorted_mass: dict[str, ArrayLike] = {k: init_dict[k] for k in sorted(init_dict)}
        log_number_of_molecules: dict[str, ArrayLike] = {}

        for element, mass_constraint in sorted_mass.items():
            molar_mass: ArrayLike = Formula(element).mass * unit_conversion.g_to_kg
            log_number_of_molecules_: Array = (
                jnp.log(mass_constraint) + jnp.log(AVOGADRO) - jnp.log(molar_mass)
            )
            log_number_of_molecules[element] = log_number_of_molecules_

        init_map: ImmutableMap[str, ArrayLike] = ImmutableMap(log_number_of_molecules)

        return cls(init_map)

    def asdict(self) -> dict[str, Array]:
        """Gets a dictionary of the values

        Returns:
            A dictionary of the values
        """
        out: dict[str, Array] = {
            f"{key}_molecules": jnp.exp(jnp.asarray(value))
            for key, value in self.log_molecules.items()
        }

        return out

    @jit
    def log_number_density(self, log_atmosphere_volume: ArrayLike) -> Array:
        """Log number density

        Args:
            log_atmosphere_volume: Log volume of the atmosphere

        Returns:
            Log number density
        """
        log_molecules: Array = jnp.array(list(self.log_molecules.values()))
        log_number_density: Array = log_molecules - log_atmosphere_volume

        return log_number_density

    def vmap_axes(self) -> Self:
        """Gets vmap axes.

        Returns:
            vmap axes
        """
        log_molecules_vmap: dict[str, int | None] = {}

        for key, log_molecules in self.log_molecules.items():
            if jnp.isscalar(log_molecules):
                vmap_axis: int | None = None
            else:
                vmap_axis = 0
            log_molecules_vmap[key] = vmap_axis

        return MassConstraints(ImmutableMap(log_molecules_vmap))  # type: ignore - container

    def __bool__(self) -> bool:
        return bool(self.log_molecules)


# endregion


# region: Containers for fixed parameters
class FixedParameters(NamedTuple):
    """Parameters that are always fixed for a calculation

    This container and all objects within it must be hashable.

    Args:
        species: Tuple of species
        formula_matrix; Formula matrix
        formula_matrix_constraints: Formula matrix for applying mass constraints
        reaction_matrix: Reaction matrix
        fugacity_matrix: Fugacity constraint matrix
        gas_species_indices: Indices of gas species
        fugacity_species_indices: Indices of species to constrain the fugacity
        diatomic_oxygen_index: Index of diatomic oxygen
        molar_masses: Molar masses of all species
        tau: Tau factor for species stability
    """

    species: tuple[Species, ...]
    """Tuple of species """
    formula_matrix: tuple[tuple[float, ...], ...]
    """Formula matrix"""
    formula_matrix_constraints: tuple[tuple[float, ...], ...]
    """Formula matrix for applying mass constraints"""
    reaction_matrix: tuple[tuple[float, ...], ...]
    """Reaction matrix"""
    fugacity_matrix: tuple[tuple[float, ...], ...]
    """Fugacity constraint matrix"""
    gas_species_indices: tuple[int, ...]
    """Indices of gas species"""
    fugacity_species_indices: tuple[int, ...]
    """Indices of species to constrain the fugacity"""
    diatomic_oxygen_index: int
    """Index of diatomic oxygen"""
    molar_masses: tuple[float, ...]
    """Molar masses of all species"""
    tau: float
    """Tau factor for species"""


class Species(NamedTuple):
    """Species

    Args:
        data: Species data
        activity: Activity
        solubility: Solubility
    """

    data: SpeciesData
    activity: ActivityProtocol
    solubility: SolubilityProtocol

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return self.data.name

    @classmethod
    def create_condensed(
        cls,
        species_name: str,
        activity: ActivityProtocol = CondensateActivity(),
    ) -> Self:
        """Creates a condensate

        Args:
            species_name: Species name, as it appears in the species dictionary
            activity: Activity. Defaults to unity activity.

        Returns:
            A condensed species
        """
        species_data: SpeciesData = select_thermodata(species_name)

        return cls(species_data, activity, NoSolubility())

    @classmethod
    def create_gas(
        cls,
        species_name: str,
        activity: ActivityProtocol = IdealGas(),
        solubility: SolubilityProtocol = NoSolubility(),
    ) -> Self:
        """Creates a gas species

        Args:
            species_name: Species name, as it appears in the species dictionary
            activity: Activity. Defaults to an ideal gas.
            solubility: Solubility. Defaults to no solubility.

        Returns:
            A gas species
        """
        species_data: SpeciesData = select_thermodata(species_name)

        return cls(species_data, activity, solubility)


class Solution(NamedTuple):
    """Solution

    Args:
        log_number_density: Log number density of species
        stability: Stability of species
    """

    log_number_density: ArrayLike
    """Log number density of species"""
    stability: ArrayLike
    """Stability of species"""

    @classmethod
    def create(cls, log_number_density: ArrayLike, stability: ArrayLike) -> Self:
        """Creates an instance.

        Args:
            log_number_density: Log number density
            stability: Stability

        Returns:
            An instance
        """
        return cls(log_number_density, stability)

    @property
    def data(self) -> Array:
        """Combined data in a single array"""
        return jnp.concatenate((self.log_number_density, self.stability))


class SolverParameters(NamedTuple):
    """Solver parameters

    Args:
        solver: Solver
        throw: How to report any failures
        max_steps: The maximum number of steps the solver can take
        lower: Lower bound on the hypercube which contains the root
        upper: Upper bound on the hypercube which contains the root
    """

    solver: OptxSolver
    """Solver"""
    throw: bool = True
    """How to report any failures"""
    max_steps: int = 256
    """Maximum number of steps the solver can take"""
    lower: tuple[float, ...] = ()
    """Lower bound on the hypercube which contains the root"""
    upper: tuple[float, ...] = ()
    """Upper bound on the hypercube which contains the root"""

    @classmethod
    def create(
        cls,
        species: tuple[Species, ...],
        solver_class: Type[OptxSolver] = optx.Newton,
        rtol: float = 1.0e-8,
        atol: float = 1.0e-8,
        throw: bool = True,
        max_steps: int = 256,
        norm: Callable = optx.rms_norm,
    ) -> Self:
        """Creates an instance

        Args:
            species: A tuple of species
            solver_class: Solver class. Defaults to optimistix Newton.
            rtol: Relative tolerance. Defaults to 1.0e-8.
            atol: Absolute tolerance. Defaults to 1.0e-8.
            throw. How to report any failures. Defaults to True.
            max_steps: The maximum number of steps the solver can take. Defaults to 256.
            norm: The norm. Defaults to optimistix RMS norm.

        Returns:
            An instance
        """
        solver: OptxSolver = solver_class(rtol=rtol, atol=atol, norm=norm)
        lower: tuple[float, ...] = cls._get_hypercube_bound(
            species, LOG_NUMBER_DENSITY_LOWER, LOG_STABILITY_LOWER
        )
        upper: tuple[float, ...] = cls._get_hypercube_bound(
            species, LOG_NUMBER_DENSITY_UPPER, LOG_STABILITY_UPPER
        )

        return cls(solver, throw=throw, max_steps=max_steps, lower=lower, upper=upper)

    @classmethod
    def _get_hypercube_bound(
        cls,
        species: tuple[Species, ...],
        log_number_density_bound: float,
        stability_bound: float,
    ) -> tuple[float, ...]:
        """Gets the bound on the hypercube

        Args:
            species: Tuple of species
            log_number_density_bound: Bound on the log number density
            stability_bound: Bound on the stability

        Returns:
            Bound on the hypercube which contains the root
        """
        num_species: int = len(species)
        log_number_density: ArrayLike = log_number_density_bound * np.ones(num_species)

        bound: tuple[float, ...] = tuple(
            np.concatenate(
                (
                    log_number_density,
                    stability_bound * np.ones(num_species),
                )
            ).tolist()
        )

        return bound


# endregion
