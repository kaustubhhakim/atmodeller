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
"""Pytree containers that can be use by JAX"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from typing import Callable, Iterator, Literal, Type

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array, lax
from jax.tree_util import Partial
from jax.typing import ArrayLike
from lineax import QR, AbstractLinearSolver
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller import (
    LOG_NUMBER_DENSITY_LOWER,
    LOG_NUMBER_DENSITY_UPPER,
    LOG_STABILITY_LOWER,
    LOG_STABILITY_UPPER,
)
from atmodeller.constants import AVOGADRO, GRAVITATIONAL_CONSTANT
from atmodeller.eos.core import IdealGas
from atmodeller.interfaces import (
    ActivityProtocol,
    FugacityConstraintProtocol,
    SolubilityProtocol,
)
from atmodeller.solubility.library import NoSolubility
from atmodeller.thermodata import CondensateActivity, SpeciesData, select_thermodata
from atmodeller.utilities import (
    OptxSolver,
    get_log_number_density_from_log_pressure,
    unit_conversion,
)

try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self

logger: logging.Logger = logging.getLogger(__name__)

# Required for multistart and helpful for reproducibility and debugging
np.random.seed(0)


class SpeciesCollection(eqx.Module):
    """A collection of species

    Args:
        species: Species
    """

    data: tuple[Species, ...] = eqx.field(converter=tuple)

    @classmethod
    def create(cls, species_names: Iterable[str]) -> Self:
        """Creates an instance

        Args:
            species_names: A list or tuple of species names. This must match the available species
                in Atmodeller, but a complete list is returned if any of the entries are incorrect.

        Returns
            An instance
        """
        species_list: list[Species] = []
        for species_ in species_names:
            if species_[-1] == "g":
                species_to_add: Species = Species.create_gas(species_)
            else:
                species_to_add: Species = Species.create_condensed(species_)
            species_list.append(species_to_add)

        return cls(species_list)

    @property
    def number(self) -> int:
        """Number of species"""
        return len(self.data)

    def get_condensed_species_indices(self) -> Array:
        """Gets the indices of condensed species

        Returns:
            Indices of the condensed species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.data):
            if species_.data.phase != "g":
                indices.append(nn)

        return jnp.array(indices, dtype=jnp.int_)

    def get_diatomic_oxygen_index(self) -> int:
        """Gets the species index corresponding to diatomic oxygen.

        Returns:
            Index of diatomic oxygen, or the first index if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.data):
            if species_.data.hill_formula == "O2":
                # logger.debug("Found O2 at index = %d", nn)
                return nn

        # TODO: Bad practice to return the first index because it could be wrong and therefore give
        # rise to spurious results, but an index must be passed to evaluate the species solubility
        # that may depend on fO2. Otherwise, a precheck could be be performed in which all the
        # solubility laws chosen by the user are checked to see if they depend on fO2. And if so,
        # and fO2 is not included in the model, an error is raised.
        return 0

    def get_gas_species_indices(self) -> Array:
        """Gets the indices of gas species

        Returns:
            Indices of the gas species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.data):
            if species_.data.phase == "g":
                indices.append(nn)

        return jnp.array(indices, dtype=jnp.int_)

    def get_lower_bound(self: SpeciesCollection) -> Array:
        """Gets the lower bound for truncating the solution during the solve"""
        return self._get_hypercube_bound(LOG_NUMBER_DENSITY_LOWER, LOG_STABILITY_LOWER)

    def get_upper_bound(self: SpeciesCollection) -> Array:
        """Gets the upper bound for truncating the solution during the solve"""
        return self._get_hypercube_bound(LOG_NUMBER_DENSITY_UPPER, LOG_STABILITY_UPPER)

    def get_molar_masses(self) -> Array:
        """Gets the molar masses of all species.

        Returns:
            Molar masses of all species
        """
        molar_masses: Array = jnp.array([species_.data.molar_mass for species_ in self.data])
        # logger.debug("molar_masses = %s", molar_masses)

        return molar_masses

    def get_species_names(self) -> tuple[str, ...]:
        """Gets the names of all species.

        Returns:
            Species names
        """
        return tuple([species_.name for species_ in self.data])

    def get_stability_species_indices(self) -> Array:
        """Gets the indices of species to solve for stability

        Returns:
            Indices of the species to solve for stability
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.data):
            if species_.solve_for_stability:
                indices.append(nn)

        return jnp.array(indices, dtype=jnp.int_)

    def get_stability_species_mask(self) -> Array:
        """Gets the stability species mask

        Returns:
            Mask for the species to solve for the stability
        """
        stability_bool: Array = jnp.array(
            [species.solve_for_stability for species in self.data], dtype=jnp.bool_
        )

        return stability_bool

    def get_unique_elements_in_species(self) -> tuple[str, ...]:
        """Gets unique elements.

        Args:
            species: A list of species

        Returns:
            Unique elements in the species ordered alphabetically
        """
        elements: list[str] = []
        for species_ in self.data:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        # logger.debug("unique_elements_in_species = %s", sorted_elements)

        return tuple(sorted_elements)

    def number_of_stability(self: SpeciesCollection) -> int:
        """Number of stability solutions"""
        return len(self.get_stability_species_indices())

    def _get_hypercube_bound(
        self: SpeciesCollection, log_number_density_bound: float, stability_bound: float
    ) -> Array:
        """Gets the bound on the hypercube

        Args:
            log_number_density_bound: Bound on the log number density
            stability_bound: Bound on the stability

        Returns:
            Bound on the hypercube which contains the root
        """
        log_number_density: ArrayLike = log_number_density_bound * np.ones(self.number)

        bound: ArrayLike = np.concatenate(
            (
                log_number_density,
                stability_bound * np.ones(self.number_of_stability()),
            )
        )

        return jnp.array(bound)

    def __getitem__(self, index: int) -> Species:
        return self.data[index]

    def __iter__(self) -> Iterator[Species]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class TracedParameters(eqx.Module):
    """Traced parameters

    These are parameters that should be traced, inasmuch as they may be updated by the user for
    repeat calculations.

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


class Planet(eqx.Module):
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
    surface_radius: ArrayLike = 6371000
    """Radius of the surface in m"""
    surface_temperature: ArrayLike = 2000
    """Temperature of the surface in K"""

    @property
    def mantle_mass(self) -> ArrayLike:
        """Mantle mass"""
        return self.planet_mass * self.mantle_mass_fraction

    @property
    def mantle_mass_fraction(self) -> ArrayLike:
        """Mantle mass fraction"""
        return 1 - self.core_mass_fraction

    @property
    def mantle_melt_mass(self) -> ArrayLike:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_solid_mass(self) -> ArrayLike:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def mass(self) -> ArrayLike:
        """Mass"""
        return self.mantle_mass

    @property
    def melt_mass(self) -> ArrayLike:
        """Mass of the melt"""
        return self.mantle_melt_mass

    @property
    def solid_mass(self) -> ArrayLike:
        """Mass of the solid"""
        return self.mantle_solid_mass

    @property
    def surface_area(self) -> ArrayLike:
        """Surface area"""
        return 4.0 * jnp.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> ArrayLike:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2

    @property
    def temperature(self) -> ArrayLike:
        """Temperature"""
        return self.surface_temperature

    def asdict(self) -> dict[str, ArrayLike]:
        """Gets a dictionary of the values

        Returns:
            A dictionary of the values
        """
        base_dict: dict[str, ArrayLike] = asdict(self)
        base_dict["mantle_mass"] = self.mass
        base_dict["mantle_melt_mass"] = self.melt_mass
        base_dict["mantle_solid_mass"] = self.solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        return base_dict


class NormalisedMass(eqx.Module):
    """Normalised mass for conventional outgassing

    This is not currently used, but it is a placeholder for future development.

    Default values are for a unit mass (1 kg) system.

    Args:
        melt_fraction: Melt fraction. Defaults to 0.3 for 30%.
        temperature: Temperature. Defaults to 1400 K.
        mass: Total mass. Defaults to 1 kg for a unit mass system.
    """

    melt_fraction: ArrayLike = 0.3
    """Mass fraction of melt in kg/kg"""
    temperature: ArrayLike = 1400
    """Temperature in K"""
    mass: ArrayLike = 1.0
    """Total mass"""

    @property
    def melt_mass(self) -> ArrayLike:
        """Mass of the melt"""
        return self.mass * self.melt_fraction

    @property
    def solid_mass(self) -> ArrayLike:
        """Mass of the solid"""
        return self.mass * (1 - self.melt_fraction)


class ConstantFugacityConstraint(eqx.Module):
    """A constant fugacity constraint

    This must adhere to FugacityConstraintProtocol

    Args:
        fugacity: Fugacity
    """

    fugacity: ArrayLike

    @property
    def value(self) -> ArrayLike:
        return self.fugacity

    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        del temperature
        del pressure
        log_fugacity_value: ArrayLike = jnp.log(self.fugacity)

        return log_fugacity_value


class FugacityConstraints(eqx.Module):
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

    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        # Partial to avoid the late-binding closure issue
        fugacity_funcs: list[Callable] = [
            Partial(constraint.log_fugacity) for constraint in self.constraints.values()
        ]
        # jax.debug.print("fugacity_funcs = {out}", out=fugacity_funcs)

        # Temperature must be a float array to ensure branches have have identical types
        temperature = jnp.asarray(temperature, dtype=jnp.float_)

        def apply_fugacity_function(
            index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike
        ) -> Array:
            # jax.debug.print("index = {out}", out=index)
            return lax.switch(
                index,
                fugacity_funcs,
                temperature,
                pressure,
            )

        vmap_apply_function: Callable = eqx.filter_vmap(
            apply_fugacity_function, in_axes=(0, None, None)
        )
        indices: Array = jnp.arange(len(self.constraints))
        log_fugacity: Array = vmap_apply_function(indices, temperature, pressure)
        # jax.debug.print("log_fugacity = {out}", out=log_fugacity)

        return log_fugacity

    @eqx.filter_jit
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

    def __bool__(self) -> bool:
        return bool(self.constraints)


class MassConstraints(eqx.Module):
    """Mass constraints of elements

    Args:
        log_abundance: Log number of atoms of elements
    """

    log_abundance: ImmutableMap[str, ArrayLike]
    """Log number of atoms of elements"""

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
        log_abundance: dict[str, ArrayLike] = {}

        for element, mass_constraint in sorted_mass.items():
            molar_mass: ArrayLike = Formula(element).mass * unit_conversion.g_to_kg
            log_abundance_: ArrayLike = (
                np.log(mass_constraint) + np.log(AVOGADRO) - np.log(molar_mass)
            )
            log_abundance[element] = log_abundance_

        init_map: ImmutableMap[str, ArrayLike] = ImmutableMap(log_abundance)

        return cls(init_map)

    def asdict(self) -> dict[str, ArrayLike]:
        """Gets a dictionary of the values

        Returns:
            A dictionary of the values
        """
        out: dict[str, ArrayLike] = {
            f"{key}_number": jnp.exp(jnp.asarray(value))
            for key, value in self.log_abundance.items()
        }

        return out

    @eqx.filter_jit
    def log_number_density(self, log_atmosphere_volume: ArrayLike) -> Array:
        """Log number density

        Args:
            log_atmosphere_volume: Log volume of the atmosphere

        Returns:
            Log number density
        """
        log_abundance: Array = jnp.array(list(self.log_abundance.values()))
        log_number_density: Array = log_abundance - log_atmosphere_volume

        return log_number_density

    @eqx.filter_jit
    def log_maximum_number(self) -> Array:
        """Log of the maximum abundance

        Returns:
            Log of the maximum abundance
        """
        log_abundance: Array = jnp.array(list(self.log_abundance.values()))

        return jnp.max(log_abundance)

    def __bool__(self) -> bool:
        return bool(self.log_abundance)


class FixedParameters(eqx.Module):
    """Parameters that are always fixed for a calculation

    This container and all objects within it must be hashable.

    Args:
        species: Collection of species
        formula_matrix; Formula matrix
        formula_matrix_constraints: Formula matrix for applying mass constraints
        reaction_matrix: Reaction matrix
        reaction_stability_matrix: Reaction stability matrix
        stability_species_indices: Indices of species to solve for stability
        fugacity_matrix: Fugacity constraint matrix
        gas_species_indices: Indices of gas species
        condensed_specie_indices: Indices of condensed species
        fugacity_species_indices: Indices of species to constrain the fugacity
        diatomic_oxygen_index: Index of diatomic oxygen
        molar_masses: Molar masses of all species
        tau: Tau factor for species stability
    """

    species: SpeciesCollection
    """Collection of species"""
    formula_matrix: npt.NDArray[np.int_]
    """Formula matrix"""
    formula_matrix_constraints: npt.NDArray[np.int_]
    """Formula matrix for applying mass constraints"""
    reaction_matrix: npt.NDArray[np.float_]
    """Reaction matrix"""
    reaction_stability_matrix: npt.NDArray[np.float_]
    """Reaction stability matrix"""
    stability_species_indices: Array
    """Indices of species to solve for stability"""
    fugacity_matrix: Array
    """Fugacity constraint matrix"""
    gas_species_indices: Array
    """Indices of gas species"""
    condensed_species_indices: Array
    """Indices of condensed species"""
    fugacity_species_indices: Array
    """Indices of species to constrain the fugacity"""
    diatomic_oxygen_index: int
    """Index of diatomic oxygen"""
    molar_masses: Array
    """Molar masses of all species"""
    tau: float
    """Tau factor for species"""


class Species(eqx.Module):
    """Species

    Args:
        data: Species data
        activity: Activity
        solubility: Solubility
        solve_for_stability: Solve for stability
    """

    data: SpeciesData
    activity: ActivityProtocol
    solubility: SolubilityProtocol
    solve_for_stability: bool

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return self.data.name

    @classmethod
    def create_condensed(
        cls,
        species_name: str,
        activity: ActivityProtocol = CondensateActivity(),
        solve_for_stability: bool = True,
    ) -> Self:
        """Creates a condensate

        Args:
            species_name: Species name, as it appears in the species dictionary
            activity: Activity. Defaults to unity activity.
            solve_for_stability. Solve for stability. Defaults to True.

        Returns:
            A condensed species
        """
        species_data: SpeciesData = select_thermodata(species_name)

        return cls(species_data, activity, NoSolubility(), solve_for_stability)

    @classmethod
    def create_gas(
        cls,
        species_name: str,
        activity: ActivityProtocol = IdealGas(),
        solubility: SolubilityProtocol = NoSolubility(),
        solve_for_stability: bool = False,
    ) -> Self:
        """Creates a gas species

        Args:
            species_name: Species name, as it appears in the species dictionary
            activity: Activity. Defaults to an ideal gas.
            solubility: Solubility. Defaults to no solubility.
            solve_for_stability. Solve for stability. Defaults to False.

        Returns:
            A gas species
        """
        species_data: SpeciesData = select_thermodata(species_name)

        return cls(species_data, activity, solubility, solve_for_stability)


class SolverParameters(eqx.Module):
    """Solver parameters

    Args:
        solver: Solver. Defaults to optx.Newton
        atol: Absolute tolerance. Defaults to 1.0e-6.
        rtol: Relative tolerance. Defaults to 1.0e-6.
        linear_solver: Linear solver. Defaults to lineax.QR.
        norm: Norm. Defaults to optx.rms_norm.
        throw: How to report any failures. Defaults to False.
        max_steps: The maximum number of steps the solver can take. Defaults to 256
        jac: Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian.
            Can be either fwd or bwd. Defaults to fwd.
        multistart: Number of multistarts. Defaults to 10.
        multistart_perturbation: Perturbation for multistart. Defaults to 30.
    """

    solver: Type[OptxSolver] = optx.Newton
    """Solver"""
    atol: float = 1.0e-6
    """Absolute tolerance"""
    rtol: float = 1.0e-6
    """Relative tolerance"""
    linear_solver: Type[AbstractLinearSolver] = QR
    """Linear solver"""
    norm: Callable = optx.max_norm
    """Norm""" ""
    throw: bool = False
    """How to report any failures"""
    max_steps: int = 256
    """Maximum number of steps the solver can take"""
    jac: Literal["fwd", "bwd"] = "fwd"
    """Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian"""
    multistart: int = 10
    """Number of multistarts"""
    multistart_perturbation: float = 30.0
    """Perturbation for multistart"""
    solver_instance: OptxSolver = eqx.field(init=False)
    """Solver instance"""

    def __post_init__(self):
        self.solver_instance = self.solver(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            linear_solver=self.linear_solver(),  # type: ignore because there is a parameter
        )
