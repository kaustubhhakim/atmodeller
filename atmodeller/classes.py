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
import time
from collections.abc import Mapping
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.tree_util import tree_flatten
from jaxtyping import ArrayLike

from atmodeller import BOLTZMANN_CONSTANT_BAR, TAU
from atmodeller.containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Planet,
    Solution,
    SolverParameters,
    Species,
    TracedParameters,
)
from atmodeller.engine import solve
from atmodeller.jax_output import JaxOutput
from atmodeller.thermodata.redox_buffers import RedoxBufferProtocol
from atmodeller.utilities import partial_rref, unscale_number_density

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere

    Args:
        species: Tuple of species
        scaling: Scaling for the number density
        tau: Tau factor for species stability. Defaults to TAU.
    """

    # Attributes that are set during initialise_solve are included for typing
    traced_parameters: TracedParameters
    planet: Planet
    fugacity_constraints: FugacityConstraints
    mass_constraints: MassConstraints
    fixed_parameters: FixedParameters
    initial_solution: Solution
    _solver: Callable
    # Used for vmapping (if relevant)
    traced_parameters_vmap: TracedParameters

    def __init__(self, species: tuple[Species, ...], scaling: float, tau: float = TAU):
        self.species: tuple[Species, ...] = species
        self.log_scaling: float = np.log(scaling)
        self.tau: float = tau
        self.solver_parameters: SolverParameters = SolverParameters.create(
            self.species, self.log_scaling
        )
        logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def is_batch(self) -> bool:
        """Returns if any parameters are batched, thereby necessitating a vmap solve"""
        vmap_axes: list = [
            self.planet.vmap_axes(),
            self.fugacity_constraints.vmap_axes(),
            self.mass_constraints.vmap_axes(),
        ]
        leaves, _ = tree_flatten(vmap_axes)

        # Check if any of the axes should be vmapped, which is defined by an entry of zero
        contains_zero: bool = any(np.array(leaves) == 0)

        return contains_zero

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

        self.fixed_parameters = self.get_fixed_parameters(
            self.fugacity_constraints, self.mass_constraints
        )
        logger.debug("fixed_parameters = %s", self.fixed_parameters)

        self.initial_solution = Solution.create(
            initial_number_density, initial_stability, self.log_scaling
        )
        logger.debug("initial_solution = %s", self.initial_solution)
        self.traced_parameters = TracedParameters(
            planet=self.planet,
            fugacity_constraints=self.fugacity_constraints,
            mass_constraints=self.mass_constraints,
        )
        logger.debug("traced_parameters = %s", self.traced_parameters)

        if self.is_batch:
            self._solver = self._get_solver_vmap()
        else:
            self._solver = self._get_solver_single()

        # Compile
        start_time = time.time()
        self._solver(self.initial_solution, self.traced_parameters).block_until_ready()
        end_time = time.time()
        compile_time = end_time - start_time
        logger.info("Compile time: %.6f seconds", compile_time)

        return self._solver

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
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        gas_species_indices: tuple[int, ...] = self.get_gas_species_indices()
        molar_masses: tuple[float, ...] = self.get_molar_masses()
        diatomic_oxygen_index: int = self.get_diatomic_oxygen_index()

        # Formula matrix rows for elements that are constrained by mass constraints
        formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix()
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
        indices: list[int] = []
        for element in mass_constraints.log_molecules.keys():
            index: int = unique_elements.index(element)
            indices.append(index)
        formula_matrix = formula_matrix[indices, :]

        # Fugacity constraint matrix and indices
        number_fugacity_constraints: int = len(fugacity_constraints.constraints)
        fugacity_species_indices: list[int] = []
        species_names: tuple[str, ...] = self.get_species_names()
        for species_name in fugacity_constraints.constraints.keys():
            index: int = species_names.index(species_name)
            fugacity_species_indices.append(index)
        fugacity_matrix: npt.NDArray[np.float_] = np.identity(number_fugacity_constraints)

        # For fixed parameters all objects must be hashable.
        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=tuple(map(tuple, formula_matrix)),
            reaction_matrix=tuple(map(tuple, reaction_matrix)),
            fugacity_matrix=tuple(map(tuple, fugacity_matrix)),
            gas_species_indices=gas_species_indices,
            fugacity_species_indices=tuple(fugacity_species_indices),
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
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
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

    def get_gas_species_indices(self) -> tuple[int, ...]:
        """Gets the indices of gas species

        Returns:
            Indices of the gas species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.species):
            if species_.data.phase == "g":
                indices.append(nn)

        return tuple(indices)

    def get_unique_elements_in_species(self) -> tuple[str, ...]:
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

        return tuple(sorted_elements)

    def get_wrapped_jit_solver(self) -> Callable:
        """Gets the jit solver with solver parameters set.

        Returns:
            jit solver with solver parameters set
        """

        def wrapped_jit_solver(
            solution: Solution, traced_parameters: TracedParameters
        ) -> Callable:
            return solve(
                solution, traced_parameters, self.fixed_parameters, self.solver_parameters
            )

        return wrapped_jit_solver

    def get_molar_masses(self) -> tuple[float, ...]:
        """Gets the molar masses of all species.

        Returns:
            Molar masses of all species
        """
        molar_masses: tuple[float, ...] = tuple(
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
            self.traced_parameters,
            self.fixed_parameters,
            unscaled_number_density,
            stability,
        )
        extended_activity_np: npt.NDArray[np.float_] = np.array(extended_activity)
        logger.info("log_extended_activity = %s", extended_activity_np)

        return number_density_np, extended_activity_np

    def get_reaction_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction matrix.

        Returns:
            A matrix of linearly independent reactions or an empty array if no reactions
        """
        if len(self.species) == 1:
            logger.debug("Only one species therefore no reactions")
            return np.array([])

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
        if reaction_matrix.size != 0:
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

    def get_species_names(self) -> tuple[str, ...]:
        """Gets the names of all species

        Returns:
            Species names
        """
        return tuple([species_.name for species_ in self.species])

    def _get_solver_single(self) -> Callable:
        """Gets the solver for a single solve."""
        return self.get_wrapped_jit_solver()

    def _get_solver_vmap(self) -> Callable:
        """Gets the solver for a batch solve."""
        # Define the structures to vectorize.
        self.traced_parameters_vmap = TracedParameters(
            planet=self.planet.vmap_axes(),  # type: ignore
            fugacity_constraints=self.fugacity_constraints.vmap_axes(),  # type: ignore
            mass_constraints=self.mass_constraints.vmap_axes(),  # type: ignore
        )
        logger.debug("traced_parameters_vmap = %s", self.traced_parameters_vmap)

        solver: Callable = jax.jit(
            jax.vmap(self.get_wrapped_jit_solver(), in_axes=(None, self.traced_parameters_vmap))
        )

        return solver

    def solve_raw_output(
        self,
        initial_solution: Solution | None = None,
        traced_parameters: TracedParameters | None = None,
    ) -> tuple[Array, Solution, TracedParameters]:
        """Solves the system and returns the raw (unprocessed) solution

        Args:
            initial_solution: Initial solution. Defaults to None, meaning that the initial solution
                used to initialise the solver is used.
            traced_parameters: Traced parameters. Defaults to None, meaning that the traced
                parameters used to initialise the solver are used.

        Returns:
            Solution, initial solution, traced parameters
        """
        if initial_solution is None:
            initial_solution_: Solution = self.initial_solution
        else:
            initial_solution_ = initial_solution

        if traced_parameters is None:
            traced_parameters_: TracedParameters = self.traced_parameters
        else:
            traced_parameters_ = traced_parameters

        start_time = time.time()
        out: Array = self._solver(initial_solution_, traced_parameters_).block_until_ready()
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Execution time: %.6f seconds", execution_time)
        logger.debug("out = %s", out)

        return out, initial_solution_, traced_parameters_

    def solve(
        self,
        initial_solution: Solution | None = None,
        traced_parameters: TracedParameters | None = None,
    ) -> dict[str, ArrayLike]:
        """Solves the system and returns the processed solution

        Args:
            initial_solution: Initial solution. Defaults to None, meaning that the initial solution
                used to initialise the solver is used.
            traced_parameters: Traced parameters. Defaults to None, meaning that the traced
                parameters used to initialise the solver are used.

        Returns:
            Number density, extended activity
        """
        solution, initial_solution_, traced_parameters_ = self.solve_raw_output(
            initial_solution, traced_parameters
        )

        output: JaxOutput = JaxOutput(solution, self, initial_solution_, traced_parameters_)
        quick_look: dict[str, ArrayLike] = output.quick_look()

        logger.info("output_dict = %s", pprint.pformat(quick_look))

        return quick_look

    def output_dict(
        self,
        number_density: npt.NDArray[np.float_],
        extended_activity: npt.NDArray[np.float_],
    ) -> dict[str, ArrayLike]:
        """Output dictionary to quickly assess the solution.

        This is intended for a quick first glance of the output with convenient units and to ease
        comparison with test or benchmark data.

        Args:
            number_density: Log number density
            extended_activity: Log extended activity
        """

        def collapse_single_entry_values(
            input_dictionary: dict[str, ArrayLike]
        ) -> dict[str, ArrayLike]:
            for key, value in input_dictionary.items():
                if value.size == 1:  # type: ignore
                    input_dictionary[key] = float(value[0])  # type: ignore

            return input_dictionary

        output_dict: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self.species):
            species_density: ArrayLike = np.exp(np.atleast_2d(number_density)[:, nn])
            species_activity: ArrayLike = np.exp(np.atleast_2d(extended_activity)[:, nn])
            if nn in self.get_gas_species_indices():
                # TODO: Check this works for real gas EOS as well
                # Convert gas number densities to pressure/fugacity in bar
                species_density *= BOLTZMANN_CONSTANT_BAR * self.planet.surface_temperature
                species_activity *= BOLTZMANN_CONSTANT_BAR * self.planet.surface_temperature

            output_dict[species_.name] = species_density
            output_dict[f"{species_.name}_activity"] = species_activity

        return collapse_single_entry_values(output_dict)
