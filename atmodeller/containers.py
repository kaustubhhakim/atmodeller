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
"""Containers"""

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Float64, Integer
from lineax import AbstractLinearSolver
from molmass import Formula

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
from atmodeller.mytypes import NpArray, NpFloat, OptxSolver
from atmodeller.solubility.library import NoSolubility
from atmodeller.thermodata import CondensateActivity, SpeciesData, select_thermodata
from atmodeller.utilities import (
    all_not_nan,
    as_j64,
    get_batch_size,
    get_log_number_density_from_log_pressure,
    to_hashable,
    unit_conversion,
)

logger: logging.Logger = logging.getLogger(__name__)


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
    ) -> "Species":
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
    ) -> "Species":
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


class SpeciesCollection(eqx.Module):
    """A collection of species

    Args:
        species: Species
    """

    data: tuple[Species, ...] = eqx.field(converter=tuple)

    @classmethod
    def create(cls, species_names: Iterable[str]) -> "SpeciesCollection":
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

    def active_stability(self) -> Bool[Array, " species_dim"]:
        """Active species stability

        Returns:
            True for species stabilities that are to be solved for, otherwise False
        """
        mask: Bool[Array, " species_dim"] = jnp.array(
            [species.solve_for_stability for species in self.data], dtype=bool
        )

        return mask

    def get_condensed_species_names(self) -> tuple[str, ...]:
        """Condensed species names

        Returns:
            Condensed species names
        """
        condensed_names: list[str] = [
            species.name for species in self.data if species.data.phase != "g"
        ]

        return tuple(condensed_names)

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

    def get_gas_species_mask(self) -> Bool[Array, " species_dim"]:
        """Gets the gas species mask

        Returns:
            Mask for the gas species
        """
        gas_species_mask: Bool[Array, " species_dim"] = jnp.array(
            [species.data.phase == "g" for species in self.data], dtype=bool
        )

        return gas_species_mask

    def get_gas_species_names(self) -> tuple[str, ...]:
        """Gas species names

        Returns:
            Gas species names
        """
        gas_names: list[str] = [species.name for species in self.data if species.data.phase == "g"]

        return tuple(gas_names)

    def get_lower_bound(self) -> Array:
        """Gets the lower bound for truncating the solution during the solve

        Returns:
            Lower bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(LOG_NUMBER_DENSITY_LOWER, LOG_STABILITY_LOWER)

    def get_upper_bound(self) -> Array:
        """Gets the upper bound for truncating the solution during the solve

        Returns:
            Upper bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(LOG_NUMBER_DENSITY_UPPER, LOG_STABILITY_UPPER)

    def get_molar_masses(self) -> Float[Array, " species_dim"]:
        """Gets the molar masses of all species.

        Returns:
            Molar masses of all species
        """
        molar_masses: Float[Array, " species_dim"] = jnp.array(
            [species_.data.molar_mass for species_ in self.data]
        )
        # logger.debug("molar_masses = %s", molar_masses)

        return molar_masses

    def get_species_names(self) -> tuple[str, ...]:
        """Gets the unique names of all species.

        Unique names by combining Hill notation and phase

        Returns:
            Species names
        """
        return tuple([species_.name for species_ in self.data])

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

    def _get_hypercube_bound(
        self, log_number_density_bound: float, stability_bound: float
    ) -> Array:
        """Gets the bound on the hypercube

        Args:
            log_number_density_bound: Bound on the log number density
            stability_bound: Bound on the stability

        Returns:
            Bound on the hypercube which contains the root
        """
        bound: ArrayLike = np.concatenate(
            (
                log_number_density_bound * np.ones(self.number),
                stability_bound * np.ones(self.number),
            )
        )

        return jnp.array(bound)

    def __getitem__(self, index: int) -> Species:
        return self.data[index]

    def __iter__(self) -> Iterator[Species]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


class Planet(eqx.Module):
    """Planet properties

    Default values are for a fully molten Earth.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to Earth (5.972e24 kg).
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to Earth (0.3 kg/kg).
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1 kg/kg.
        surface_radius: Radius of the planetary surface in m. Defaults to Earth (6371 km).
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
    """

    planet_mass: Array = eqx.field(converter=as_j64, default=5.972e24)
    """Mass of the planet in kg"""
    core_mass_fraction: Array = eqx.field(converter=as_j64, default=0.295334691460966)
    """Mass fraction of the core relative to the planetary mass in kg/kg"""
    mantle_melt_fraction: Array = eqx.field(converter=as_j64, default=1.0)
    """Mass fraction of the molten mantle in kg/kg"""
    surface_radius: Array = eqx.field(converter=as_j64, default=6371000)
    """Radius of the surface in m"""
    surface_temperature: Array = eqx.field(converter=as_j64, default=2000)
    """Temperature of the surface in K"""

    @property
    def mantle_mass(self) -> Array:
        """Mantle mass"""
        return self.planet_mass * self.mantle_mass_fraction

    @property
    def mantle_mass_fraction(self) -> Array:
        """Mantle mass fraction"""
        return 1 - self.core_mass_fraction

    @property
    def mantle_melt_mass(self) -> Array:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_solid_mass(self) -> Array:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def mass(self) -> Array:
        """Mass"""
        return self.mantle_mass

    @property
    def melt_mass(self) -> Array:
        """Mass of the melt"""
        return self.mantle_melt_mass

    @property
    def solid_mass(self) -> Array:
        """Mass of the solid"""
        return self.mantle_solid_mass

    @property
    def surface_area(self) -> Array:
        """Surface area"""
        return 4.0 * jnp.pi * jnp.square(self.surface_radius)

    @property
    def surface_gravity(self) -> Array:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / jnp.square(self.surface_radius)

    @property
    def temperature(self) -> Array:
        """Temperature"""
        return self.surface_temperature

    def asdict(self) -> dict[str, NpArray]:
        """Gets a dictionary of the values as NumPy arrays.

        Returns:
            A dictionary of the values
        """
        base_dict: dict[str, ArrayLike] = asdict(self)
        base_dict["mantle_mass"] = self.mass
        base_dict["mantle_melt_mass"] = self.melt_mass
        base_dict["mantle_solid_mass"] = self.solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        # Convert all values to NumPy arrays
        base_dict_np: dict[str, NpArray] = {k: np.asarray(v) for k, v in base_dict.items()}

        return base_dict_np


class ConstantFugacityConstraint(eqx.Module):
    """A constant fugacity constraint

    This must adhere to FugacityConstraintProtocol

    Args:
        fugacity: Fugacity. Defaults to nan.
    """

    fugacity: Array = eqx.field(converter=as_j64, default=np.nan)
    """Fugacity"""

    def active(self) -> Bool[Array, ""]:
        """Is the fugacity constraint active.

        Returns:
            True if the fugacity constraint is active, otherwise False
        """
        return all_not_nan(self.fugacity)

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        del temperature
        del pressure

        return jnp.log(self.fugacity)


class FugacityConstraints(eqx.Module):
    """Fugacity constraints

    These are applied as constraints on the gas activity.

    Args:
        constraints: Fugacity constraints
        species: Species corresponding to the columns of `constraints`
    """

    constraints: tuple[FugacityConstraintProtocol, ...]
    """Fugacity constraints"""
    species: tuple[str, ...]
    """Species corresponding to the entries of constraints"""

    @classmethod
    def create(
        cls,
        species: SpeciesCollection,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
    ) -> "FugacityConstraints":
        """Creates an instance

        Args:
            species: Species
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                None.

        Returns:
            An instance
        """
        fugacity_constraints_: Mapping[str, FugacityConstraintProtocol] = (
            fugacity_constraints if fugacity_constraints is not None else {}
        )

        # All unique species
        unique_species: tuple[str, ...] = species.get_species_names()

        constraints: list[FugacityConstraintProtocol] = []

        for species_name in unique_species:
            if species_name in fugacity_constraints_:
                constraints.append(fugacity_constraints_[species_name])
            else:
                constraints.append(ConstantFugacityConstraint(np.nan))

        return cls(tuple(constraints), unique_species)

    def active(self) -> Bool[Array, " species_dim"]:
        """Active fugacity constraints

        Returns:
            Mask indicating whether fugacity constraints are active or not
        """
        mask: list[Array] = [
            jnp.atleast_1d(constraint.active()) for constraint in self.constraints
        ]

        return jnp.concatenate(mask)

    def asdict(self, temperature: ArrayLike, pressure: ArrayLike) -> dict[str, NpArray]:
        """Gets a dictionary of the evaluated fugacity constraints as NumPy Arrays

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            A dictionary of the evaluated fugacity constraints
        """
        log_fugacity_list: list[NpFloat] = []

        for constraint in self.constraints:
            log_fugacity: NpFloat = np.asarray(constraint.log_fugacity(temperature, pressure))
            log_fugacity_list.append(log_fugacity)

        out: dict[str, NpArray] = {
            f"{key}_fugacity": np.exp(log_fugacity_list[idx])
            for idx, key in enumerate(self.species)
            if not np.all(np.isnan(log_fugacity_list[idx]))
        }

        return out

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        # NOTE: Must avoid the late-binding closure issue
        fugacity_funcs: list[Callable] = [
            to_hashable(constraint.log_fugacity) for constraint in self.constraints
        ]
        # jax.debug.print("fugacity_funcs = {out}", out=fugacity_funcs)

        # Temperature must be a float array to ensure branches have have identical types
        temperature = jnp.asarray(temperature, dtype=jnp.float64)

        def apply_fugacity(index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike) -> Array:
            # jax.debug.print("index = {out}", out=index)
            return lax.switch(
                index,
                fugacity_funcs,
                temperature,
                pressure,
            )

        indices: Array = jnp.arange(len(self.constraints))
        vmap_fugacity: Callable = eqx.filter_vmap(apply_fugacity, in_axes=(0, None, None))
        log_fugacity: Array = vmap_fugacity(indices, temperature, pressure)
        # jax.debug.print("log_fugacity = {out}", out=log_fugacity)

        return log_fugacity

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


class NormalisedMass(eqx.Module):
    """Normalised mass for conventional outgassing

    This is not currently used, but it is a placeholder for future development.

    Default values are for a unit mass (1 kg) system.

    Args:
        melt_fraction: Melt fraction. Defaults to 0.3 for 30%.
        temperature: Temperature. Defaults to 1400 K.
        mass: Total mass. Defaults to 1 kg for a unit mass system.
    """

    melt_fraction: Array = eqx.field(converter=as_j64, default=0.3)
    """Mass fraction of melt in kg/kg"""
    temperature: Array = eqx.field(converter=as_j64, default=1400)
    """Temperature in K"""
    mass: Array = eqx.field(converter=as_j64, default=1.0)
    """Total mass"""

    @property
    def melt_mass(self) -> Array:
        """Mass of the melt"""
        return self.mass * self.melt_fraction

    @property
    def solid_mass(self) -> Array:
        """Mass of the solid"""
        return self.mass * (1 - self.melt_fraction)


class MassConstraints(eqx.Module):
    """Mass constraints of elements

    Args:
        log_abundance: Log number of atoms
        elements: Elements corresponding to the columns of `log_abundance`
    """

    log_abundance: Float64[Array, "batch_dim el_dim"] = eqx.field(converter=as_j64)
    elements: tuple[str, ...]

    @classmethod
    def create(
        cls,
        species: SpeciesCollection,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        batch_size: int = 1,
    ) -> "MassConstraints":
        """Creates an instance

        Args:
            species: Species
            mass_constraints: Mapping of element name and mass constraint in kg. Defaults to None.
            batch_size: Total batch size, which is required for broadcasting

        Returns:
            An instance
        """
        mass_constraints_: Mapping[str, ArrayLike] = (
            mass_constraints if mass_constraints is not None else {}
        )

        # All unique elements in alphabetical order
        unique_elements: tuple[str, ...] = species.get_unique_elements_in_species()

        # Determine the maximum length of any array in mass_constraints_
        max_len: int = get_batch_size(mass_constraints_)

        # Initialise to all nans assuming that there are no mass constraints
        log_abundance: NpFloat = np.full((max_len, len(unique_elements)), np.nan, dtype=np.float64)

        # Populate mass constraints
        for nn, element in enumerate(unique_elements):
            if element in mass_constraints_.keys():
                molar_mass: ArrayLike = Formula(element).mass * unit_conversion.g_to_kg
                log_abundance_: ArrayLike = (
                    np.log(mass_constraints_[element]) + np.log(AVOGADRO) - np.log(molar_mass)
                )
                log_abundance[:, nn] = log_abundance_

        # Broadcast, which avoids JAX recompilation if mass constraints change since otherwise the
        # shape of self.log_abundance can vary between a 1-D and 2-D array which forces
        # recompilation of solve.
        log_abundance = np.broadcast_to(log_abundance, (batch_size, len(unique_elements)))
        # jax.debug.print("log_abundance = {out}", out=log_abundance)

        return cls(log_abundance, unique_elements)

    def asdict(self) -> dict[str, NpArray]:
        """Gets a dictionary of the values as NumPy arrays

        Returns:
            A dictionary of the values
        """
        abundance: NpArray = np.exp(self.log_abundance)
        out: dict[str, NpArray] = {
            f"{element}_number": abundance[:, idx]
            for idx, element in enumerate(self.elements)
            if not np.all(np.isnan(abundance[:, idx]))
        }

        return out

    def active(self) -> Bool[Array, " el_dim"]:
        """Active mass constraints

        Returns:
            Mask indicating whether elemental mass constraints are active or not
        """
        return all_not_nan(self.log_abundance)

    def log_number_density(self, log_atmosphere_volume: ArrayLike) -> Array:
        """Log number density

        Args:
            log_atmosphere_volume: Log volume of the atmosphere

        Returns:
            Log number density
        """
        log_number_density: Array = self.log_abundance - log_atmosphere_volume

        return log_number_density


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


class FixedParameters(eqx.Module):
    """Parameters that are always fixed for a calculation

    Args:
        species: Collection of species
        formula_matrix; Formula matrix
        reaction_matrix: Reaction matrix
        stability_species_mask: Mask of species to solve for stability
        gas_species_mask: Mask of gas species
        diatomic_oxygen_index: Index of diatomic oxygen
        molar_masses: Molar masses of all species
    """

    species: SpeciesCollection
    """Collection of species"""
    formula_matrix: Integer[Array, "el_dim species_dim"]
    """Formula matrix"""
    # TODO: Currently breaks with "react_dim species_dim" because reaction_matrix might be empty.
    reaction_matrix: Float[Array, "..."]
    """Reaction matrix"""
    gas_species_mask: Array
    """Mask of gas species"""
    diatomic_oxygen_index: int
    """Index of diatomic oxygen"""
    molar_masses: Array
    """Molar masses of all species"""

    def active_reactions(self) -> Bool[Array, " react_dim"]:
        """Active reactions

        Returns:
            True for all reactions
        """
        return jnp.ones(self.reaction_matrix.shape[0], dtype=bool)

    def active_stability(self) -> Bool[Array, " species_dim"]:
        """Active species stability

        Returns:
            True for species stabilities that are to be solved for, otherwise False
        """
        return self.species.active_stability()


class SolverParameters(eqx.Module):
    """Solver parameters

    Args:
        solver: Solver. Defaults to optx.Newton
        atol: Absolute tolerance. Defaults to 1.0e-6.
        rtol: Relative tolerance. Defaults to 1.0e-6.
        linear_solver: Linear solver. Defaults to AutoLinearSolver(well_posed=False).
        norm: Norm. Defaults to optx.rms_norm.
        throw: How to report any failures. Defaults to False.
        max_steps: The maximum number of steps the solver can take. Defaults to 256
        jac: Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian.
            Can be either fwd or bwd. Defaults to fwd.
        multistart: Number of multistarts. Defaults to 10.
        multistart_perturbation: Perturbation for multistart. Defaults to 30.
    """

    solver: type[OptxSolver] = optx.Newton
    """Solver"""
    atol: float = 1.0e-6
    """Absolute tolerance"""
    rtol: float = 1.0e-6
    """Relative tolerance"""
    linear_solver: AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None)
    """Linear solver
    
    https://docs.kidger.site/lineax/api/solvers/   
    """
    norm: Callable = optx.max_norm
    """Norm""" ""
    throw: bool = False
    """How to report any failures"""
    max_steps: int = 512
    """Maximum number of steps the solver can take"""
    jac: Literal["fwd", "bwd"] = "fwd"
    """Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian"""
    multistart: int = 10
    """Number of multistarts"""
    multistart_perturbation: float = 30.0
    """Perturbation for multistart"""

    def get_solver_instance(self) -> OptxSolver:
        return self.solver(
            rtol=self.rtol,
            atol=self.atol,
            norm=self.norm,
            linear_solver=self.linear_solver,  # type: ignore because there is a parameter
            # For debugging LM solver. Not valid for all solvers (e.g. Newton)
            # verbose=frozenset({"step_size", "y", "loss", "accepted"}),
        )
