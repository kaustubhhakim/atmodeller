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

from atmodeller import TAU
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
from atmodeller.output import Output
from atmodeller.thermodata.redox_buffers import RedoxBufferProtocol
from atmodeller.utilities import partial_rref

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere

    Args:
        species: Tuple of species
        tau: Tau factor for species stability. Defaults to TAU.
        solver_parameters: Solver parameters. Defaults to None to use defaults.
    """

    # Attributes that are set during initialise_solve are included for typing
    traced_parameters: TracedParameters
    planet: Planet
    fugacity_constraints: FugacityConstraints
    mass_constraints: MassConstraints
    fixed_parameters: FixedParameters
    initial_solution: Solution
    _solver: Callable

    def __init__(
        self,
        species: tuple[Species, ...],
        tau: float = TAU,
        solver_parameters: SolverParameters | None = None,
    ):
        self.species: tuple[Species, ...] = species
        self.tau: float = tau
        if solver_parameters is None:
            solver_parameters_: SolverParameters = SolverParameters.create(self.species)
        else:
            solver_parameters_ = solver_parameters
        self.solver_parameters: SolverParameters = solver_parameters_
        logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def is_batch(self) -> bool:
        """Returns if any parameters are batched, thereby necessitating a vmap solve"""
        leaves, _ = tree_flatten(self.get_traced_parameters_vmap())
        # Check if any of the axes should be vmapped, which is defined by an entry of zero
        contains_zero: bool = any(np.array(leaves) == 0)

        return contains_zero

    def initialise_solve(
        self,
        planet: Planet,
        initial_log_number_density: npt.NDArray[np.float_],
        initial_log_stability: npt.NDArray[np.float_],
        fugacity_constraints: Mapping[str, RedoxBufferProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
    ) -> Callable:
        """Initialises the solve.

        Args:
            planet: Planet
            initial_log_number_density: Initial log number density
            initial_log_stability: Initial log stability
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.

        Returns:
            Solver callable
        """
        self.planet = planet
        self.fugacity_constraints = FugacityConstraints.create(fugacity_constraints)
        logger.debug("fugacity_constraints = %s", self.fugacity_constraints)

        self.mass_constraints = MassConstraints.create(mass_constraints)
        logger.debug("mass_constraints = %s", self.mass_constraints)

        self.fixed_parameters = self.get_fixed_parameters(
            self.fugacity_constraints, self.mass_constraints
        )
        logger.debug("fixed_parameters = %s", self.fixed_parameters)

        self.initial_solution = Solution.create(initial_log_number_density, initial_log_stability)
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

    def get_condensed_species_indices(self) -> tuple[int, ...]:
        """Gets the indices of condensed species

        Returns:
            Indices of the condensed species
        """
        indices: list[int] = []
        for nn, species_ in enumerate(self.species):
            if species_.data.phase != "g":
                indices.append(nn)

        return tuple(indices)

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

        # The complete formula matrix is not required for the calculation but it is used for
        # computing output quantities. So calculate and store it.
        formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix()

        # Formula matrix for elements that are constrained by mass constraints
        unique_elements: tuple[str, ...] = self.get_unique_elements_in_species()
        indices: list[int] = []
        for element in mass_constraints.log_molecules.keys():
            index: int = unique_elements.index(element)
            indices.append(index)
        formula_matrix_constraints: npt.NDArray[np.int_] = formula_matrix.copy()
        formula_matrix_constraints = formula_matrix_constraints[indices, :]
        logger.info("formula_matrix_constraints = %s", formula_matrix_constraints)

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
            formula_matrix_constraints=tuple(map(tuple, formula_matrix_constraints)),
            reaction_matrix=tuple(map(tuple, reaction_matrix)),
            fugacity_matrix=tuple(map(tuple, fugacity_matrix)),
            gas_species_indices=gas_species_indices,
            fugacity_species_indices=tuple(fugacity_species_indices),
            diatomic_oxygen_index=diatomic_oxygen_index,
            molar_masses=molar_masses,
            tau=self.tau,
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
        solver: Callable = jax.jit(
            jax.vmap(
                self.get_wrapped_jit_solver(), in_axes=(None, self.get_traced_parameters_vmap())
            )
        )

        return solver

    def get_traced_parameters_vmap(self) -> TracedParameters:
        """Gets the vmapping axes for tracer parameters

        Returns:
            Vmapping for tracer parameters
        """
        traced_parameters_vmap: TracedParameters = TracedParameters(
            planet=self.planet.vmap_axes(),  # type: ignore
            fugacity_constraints=self.fugacity_constraints.vmap_axes(),  # type: ignore
            mass_constraints=self.mass_constraints.vmap_axes(),  # type: ignore
        )

        return traced_parameters_vmap

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

        output: Output = Output(solution, self, initial_solution_, traced_parameters_)

        # TODO: Remove. Now unified class.
        # if self.is_batch:
        #    output: Output = OutputBatch(solution, self, initial_solution_, traced_parameters_)
        # else:
        #    output = OutputSingle(solution, self, initial_solution_, traced_parameters_)

        # output.output_to_logger()

        # TODO: Implement output options
        output.asdict()
        # output.to_dataframes()
        # output.to_excel()

        quick_look: dict[str, ArrayLike] = output.quick_look()

        logger.info("output_dict = %s", pprint.pformat(quick_look))

        return quick_look
