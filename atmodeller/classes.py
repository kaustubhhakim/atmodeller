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
"""Classes"""

import logging
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jaxtyping import ArrayLike

from atmodeller import TAU
from atmodeller.jax_containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Parameters,
    Planet,
    Solution,
    SolverParameters,
    Species,
)
from atmodeller.jax_engine import get_log_extended_activity, solve
from atmodeller.jax_utilities import unscale_number_density
from atmodeller.thermodata.jax_thermo import RedoxBufferProtocol
from atmodeller.utilities import partial_rref

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


# TODO: Make flag to check if vmap active in initialise_solve, then push solve etc. through
# appropriate function
class InteriorAtmosphereABC(ABC):
    """Interior atmosphere base

    Args:
        species: List of species
        log_scaling: Log scaling for the number density
        tau: Tau factor for species stability. Defaults to TAU.
    """

    # Attributes that are set during initialise_solve and included for typing
    fixed: FixedParameters
    fugacity_constraints: FugacityConstraints
    initial_solution: Solution
    mass_constraints: MassConstraints
    parameters: Parameters
    planet: Planet
    _solver: Callable

    def __init__(self, species: list[Species], log_scaling: float, tau: float = TAU):
        self.species: list[Species] = species
        self.log_scaling: float = log_scaling
        self.tau: float = tau
        self.solver_parameters: SolverParameters = SolverParameters.create(
            self.species, self.log_scaling
        )
        logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @abstractmethod
    def get_solver(self) -> Callable:
        """Gets the solver."""

    @abstractmethod
    def solve(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Solves the system with processing"""

    def initialise_solve(
        self,
        planet: Planet,
        initial_number_density: npt.NDArray[np.float_],
        initial_stability: npt.NDArray[np.float_],
        fugacity_constraints: Mapping[str, RedoxBufferProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
    ) -> Callable:
        """Initialises the solve.

        Args:
            planet: Planet
            initial_number_density: Initial number density
            initial_stability: Initial stability
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.

        Returns:
            Solver callable
        """
        self.planet = planet
        self.fugacity_constraints = FugacityConstraints.create(
            self.log_scaling, fugacity_constraints
        )
        logger.debug("fugacity_constraints = %s", self.fugacity_constraints)

        self.mass_constraints = MassConstraints.create(self.log_scaling, mass_constraints)
        logger.debug("mass_constraints = %s", self.mass_constraints)

        self.fixed = self.get_fixed_parameters(self.fugacity_constraints, self.mass_constraints)
        self.initial_solution = Solution.create(
            initial_number_density, initial_stability, self.log_scaling
        )
        logger.debug("initial_solution = %s", self.initial_solution)
        self.parameters = Parameters(
            fixed=self.fixed,
            planet=self.planet,
            fugacity_constraints=self.fugacity_constraints,
            mass_constraints=self.mass_constraints,
        )
        logger.debug("parameters = %s", self.parameters)

        # TODO: Check if fugacity or mass constraints or planet are vmapped, and select appropriate
        # solver accordingly. Similarly, set solve function.
        self._solver = self.get_solver()

        # Compile
        start_time = time.time()
        # self._solver is callable so pylint: disable=not-callable
        self._solver(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        compile_time = end_time - start_time
        logger.info("Compile time: %.6f seconds", compile_time)

        return self._solver

    def solve_raw_output(self) -> Array:
        """Solves the system and returns the raw (unprocessed) solution

        Returns:
            Solution
        """
        start_time = time.time()
        out: Array = self._solver(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Execution time: %.6f seconds", execution_time)
        logger.debug("out = %s", out)

        return out

    def get_diatomic_oxygen_index(self) -> int:
        """Gets the species index corresponding to diatomic oxygen.

        Returns:
            Index of diatomic oxygen, or the first index if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.species):
            if species_.data.hill_formula == "O2":
                logger.info("Found O2 at index = %d", nn)
                return nn

        # TODO: Bad practice to return the first index because it could be wrong and therefore give
        # rise to spurious results, but an index must be passed to evaluate the species solubility
        # that may depend on fO2.
        return 0

    def get_fixed_parameters(
        self, fugacity_constraints: FugacityConstraints, mass_constraints: MassConstraints
    ) -> FixedParameters:
        """Gets fixed parameters.

        Args:
            fugacity_constraints: Fugacity constraints
            mass_constraints: Mass constraints

        Returns:
            Fixed parameters
        """
        reaction_matrix: Array = jnp.array(self.get_reaction_matrix())
        gas_species_indices: Array = jnp.array(self.get_gas_species_indices())
        molar_masses: Array = jnp.array(self.get_molar_masses())
        diatomic_oxygen_index: Array = jnp.array(self.get_diatomic_oxygen_index())

        # Formula matrix rows for elements that are constrained by mass constraints
        formula_matrix_np: npt.NDArray[np.int_] = self.get_formula_matrix()
        unique_elements: list[str] = self.get_unique_elements_in_species()
        indices: list[int] = []
        for element in mass_constraints.log_molecules.keys():
            index: int = unique_elements.index(element)
            indices.append(index)
        formula_matrix_np = formula_matrix_np[indices, :]
        formula_matrix: Array = jnp.array(formula_matrix_np)

        # Fugacity constraint matrix and indices
        number_fugacity_constraints: int = len(fugacity_constraints.constraints)
        fugacity_species_indices: list[int] = []
        species_names: list[str] = self.get_species_names()
        for species_name in fugacity_constraints.constraints.keys():
            index: int = species_names.index(species_name)
            fugacity_species_indices.append(index)
        fugacity_matrix: Array = jnp.array(np.identity(number_fugacity_constraints))
        fugacity_species_indices_jnp = jnp.array(fugacity_species_indices)

        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=formula_matrix,
            reaction_matrix=reaction_matrix,
            fugacity_matrix=fugacity_matrix,
            gas_species_indices=gas_species_indices,
            fugacity_species_indices=fugacity_species_indices_jnp,
            diatomic_oxygen_index=diatomic_oxygen_index,
            molar_masses=molar_masses,
            tau=self.tau,
            log_scaling=self.log_scaling,
        )

        return fixed_parameters

    def get_formula_matrix(self) -> npt.NDArray[np.int_]:
        """Gets the formula matrix.

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            Formula matrix
        """
        unique_elements: list[str] = self.get_unique_elements_in_species()
        formula_matrix: npt.NDArray[np.int_] = np.zeros(
            (len(unique_elements), len(self.species)), dtype=jnp.int_
        )

        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(self.species):
                count: int = 0
                try:
                    count = species_.data.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def get_gas_species_indices(self) -> npt.NDArray[np.int_]:
        """Gets the indices of gas species

        Returns:
            Indices of the gas species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.species):
            if species_.data.phase == "g":
                indices.append(nn)

        return np.array(indices)

    def get_unique_elements_in_species(self) -> list[str]:
        """Gets unique elements

        Args:
            species: A list of species

        Returns:
            Unique elements in the species ordered alphabetically
        """
        elements: list[str] = []
        for species_ in self.species:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        logger.debug("unique_elements_in_species = %s", sorted_elements)

        return sorted_elements

    def get_wrapped_jit_solver(self) -> Callable:
        """Gets the jit solver with solver parameters set.

        Returns:
            jit solver with solver parameters set
        """

        def wrapped_jit_solver(solution: Solution, parameters: Parameters) -> Callable:
            return solve(solution, parameters, self.solver_parameters)

        return wrapped_jit_solver

    def get_molar_masses(self) -> npt.NDArray[np.float_]:
        """Gets the molar masses of all species.

        Returns:
            Molar masses of all species
        """
        molar_masses: npt.NDArray[np.float_] = np.array(
            [species_.data.molar_mass for species_ in self.species]
        )
        logger.debug("molar_masses = %s", molar_masses)

        return molar_masses

    def get_processed_output(
        self, raw_out: Array, axis: int, extended_activity_func: Callable
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Gets processed output

        Args:
            raw_out: Raw output from the solver
            axis: Axis along which to separate the number density from stability
            extended_activity_func: Function to evaluate the extended activity

        Returns:
            Number density, extended activity
        """
        scaled_number_density, stability = jnp.split(raw_out, 2, axis=axis)
        unscaled_number_density: Array = unscale_number_density(
            scaled_number_density, self.log_scaling
        )
        number_density_np: npt.NDArray[np.float_] = np.array(unscaled_number_density)
        logger.info("log_number_density = %s", number_density_np)
        extended_activity: Array = extended_activity_func(
            self.parameters,
            unscaled_number_density,
            stability,
        )
        extended_activity_np: npt.NDArray[np.float_] = np.array(extended_activity)
        logger.info("log_extended_activity = %s", extended_activity_np)

        return number_density_np, extended_activity_np

    def get_reaction_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction matrix.

        Returns:
            A matrix of linearly independent reactions or None # TODO: Still return None?
        """
        # TODO: Would prefer to always return an array even in the absence of reactions?
        # if self._species.number == 1:
        #    logger.debug("Only one species therefore no reactions")
        #    return None

        transpose_formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix().T
        reaction_matrix: npt.NDArray[np.float_] = partial_rref(transpose_formula_matrix)

        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        reactions: dict[int, str] = {}
        # TODO: Would like to avoid below if possible
        # if self.reaction_matrix is not None:
        for reaction_index in range(reaction_matrix.shape[0]):
            reactants: str = ""
            products: str = ""
            for species_index, species_ in enumerate(self.species):
                coeff: float = reaction_matrix[reaction_index, species_index].item()
                if coeff != 0:
                    if coeff < 0:
                        reactants += f"{abs(coeff)} {species_.data.name} + "
                    else:
                        products += f"{coeff} {species_.data.name} + "

            reactants = reactants.rstrip(" + ")
            products = products.rstrip(" + ")
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

        return reactions

    def get_species_names(self) -> list[str]:
        """Gets the names of all species

        Returns:
            Species names
        """
        return [species_.name for species_ in self.species]


class InteriorAtmosphere(InteriorAtmosphereABC):
    """Interior atmosphere single system

    Args:
        species: List of species
        log_scaling: Log scaling for the number density
        tau: Tau factor for species stability. Defaults to TAU.
    """

    @override
    def get_solver(self) -> Callable:
        return self.get_wrapped_jit_solver()

    @override
    def solve(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Solves the system.

        Returns:
            Number density, extended activity
        """
        out: Array = self.solve_raw_output()
        number_density_np, extended_activity_np = self.get_processed_output(
            out, axis=0, extended_activity_func=get_log_extended_activity
        )

        return number_density_np, extended_activity_np


class InteriorAtmosphereBatch(InteriorAtmosphereABC):
    """Interior atmosphere batch system

    Args:
        species: A list of species
        log_scaling: Log scaling for the number density
        tau: Tau factor for species stability. Defaults to TAU.
    """

    # For typing
    parameters_vmap: Parameters

    @override
    def get_solver(self) -> Callable:
        # Define the structures to vectorize.
        self.parameters_vmap: Parameters = Parameters(
            fixed=None,  # type: ignore
            planet=self.planet.vmap_axes(),  # type: ignore
            fugacity_constraints=self.fugacity_constraints.vmap_axes(),  # type: ignore
            mass_constraints=self.mass_constraints.vmap_axes(),  # type: ignore
        )
        solver: Callable = jax.jit(
            jax.vmap(self.get_wrapped_jit_solver(), in_axes=(None, self.parameters_vmap))
        )

        return solver

    def solve(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Solves the system.

        Returns:
            Number density, extended activity
        """
        out: Array = self.solve_raw_output()
        # Must vmap the function to enable it to be used in batch mode.
        vmap_get_log_extended_activity: Callable = jax.vmap(
            get_log_extended_activity, in_axes=(self.parameters_vmap, 0, 0)
        )
        number_density_np, extended_activity_np = self.get_processed_output(
            out, axis=1, extended_activity_func=vmap_get_log_extended_activity
        )

        return number_density_np, extended_activity_np
