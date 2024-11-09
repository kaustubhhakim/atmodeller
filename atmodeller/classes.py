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

from __future__ import annotations

import logging
import pprint
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.tree_util import tree_flatten
from jax.typing import ArrayLike

from atmodeller import INITIAL_LOG_NUMBER_DENSITY, INITIAL_LOG_STABILITY, TAU
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
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.utilities import partial_rref

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SolutionArguments:
    """Container for the solution arguments

    Args:
        planet: Planet
        initial_log_number_density: Initial log number density
        initial_log_stability: Initial log stability
        fugacity_constraints: Fugacity constraints
        mass_constraints: Mass constraints
        solver_parameters: Solver parameters
    """

    planet: Planet
    initial_log_number_density: ArrayLike
    initial_log_stability: ArrayLike
    fugacity_constraints: FugacityConstraints
    mass_constraints: MassConstraints
    solver_parameters: SolverParameters

    @classmethod
    def create_with_defaults(
        cls,
        species: tuple[Species, ...],
        planet: Planet | None = None,
        initial_log_number_density: ArrayLike | None = None,
        initial_log_stability: ArrayLike | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> Self:
        """Creates an instance with defaults applied if arguments are not specified.

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density: Initial log number density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.
            solver_parameters: Solver parameters. Defaults to None.

        Returns:
            An instance
        """
        if planet is None:
            planet_: Planet = Planet()
        else:
            planet_ = planet

        if initial_log_number_density is None:
            initial_log_number_density_: ArrayLike = INITIAL_LOG_NUMBER_DENSITY * jnp.ones(
                len(species), dtype=jnp.float_
            )
        else:
            initial_log_number_density_ = initial_log_number_density

        if initial_log_stability is None:
            initial_log_stability_: ArrayLike = INITIAL_LOG_STABILITY * jnp.ones(
                len(species), dtype=jnp.float_
            )
        else:
            initial_log_stability_ = initial_log_stability

        fugacity_constraints_: FugacityConstraints = FugacityConstraints.create(
            fugacity_constraints
        )
        mass_constraints_: MassConstraints = MassConstraints.create(mass_constraints)

        if solver_parameters is None:
            solver_parameters_: SolverParameters = SolverParameters.create(species)
        else:
            solver_parameters_ = solver_parameters

        return cls(
            planet_,
            initial_log_number_density_,
            initial_log_stability_,
            fugacity_constraints_,
            mass_constraints_,
            solver_parameters_,
        )

    def get_initial_solution(self) -> Solution:
        """Gets the initial solution

        Returns:
            Initial solution
        """
        return Solution.create(self.initial_log_number_density, self.initial_log_stability)

    def get_traced_parameters(self) -> TracedParameters:
        """Gets traced parameters

        Returns:
            Traced parameters
        """
        return TracedParameters(self.planet, self.fugacity_constraints, self.mass_constraints)

    def override(
        self,
        planet: Planet | None = None,
        initial_log_number_density: ArrayLike | None = None,
        initial_log_stability: ArrayLike | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
    ) -> SolutionArguments:
        """Overrides values

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.

        Returns:
            An instance
        """
        self_asdict: dict[str, Any] = asdict(self)
        to_merge: dict[str, Any] = {}

        if planet is not None:
            to_merge["planet"] = planet
        if initial_log_number_density is not None:
            to_merge["initial_log_number_density"] = initial_log_number_density
        if initial_log_stability is not None:
            to_merge["initial_log_stability"] = initial_log_stability
        if fugacity_constraints:
            to_merge["fugacity_constraints"] = FugacityConstraints.create(fugacity_constraints)
        if mass_constraints:
            to_merge["mass_constraints"] = MassConstraints.create(mass_constraints)

        merged_dict: dict[str, Any] = self_asdict | to_merge

        return SolutionArguments(**merged_dict)


class InteriorAtmosphere:
    """Interior atmosphere

    Args:
        species: Tuple of species
        tau: Tau factor for species stability. Defaults to TAU.
    """

    # Set during initialise_solve
    _solution_args: SolutionArguments
    _solver: Callable

    def __init__(self, species: tuple[Species, ...], tau: float = TAU):
        self.species: tuple[Species, ...] = species
        self.tau: float = tau
        logger.info("species = %s", [species.name for species in self.species])
        logger.info("reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def is_batch(self) -> bool:
        """Returns if any parameters are batched, thereby necessitating a vmap solve"""
        leaves, _ = tree_flatten(self.get_traced_parameters_vmap())
        # Check if any of the axes should be vmapped, which is defined by an entry of zero
        contains_zero: bool = any(np.array(leaves) == 0)

        return contains_zero

    @property
    def solution_args(self) -> SolutionArguments:
        """Solution arguments"""
        return self._solution_args

    def initialise_solve(
        self,
        *,
        planet: Planet | None = None,
        initial_log_number_density: ArrayLike | None = None,
        initial_log_stability: ArrayLike | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> Callable:
        """Initialises the solve.

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density: Initial log number density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.
            solver_parameters: Solver parameters, which can only be set here during the
                solver initialisation. Defaults to None.

        Returns:
            Solver callable
        """
        self._solution_args = SolutionArguments.create_with_defaults(
            self.species,
            planet,
            initial_log_number_density,
            initial_log_stability,
            fugacity_constraints,
            mass_constraints,
            solver_parameters,
        )

        if self.is_batch:
            self._solver = self._get_solver_vmap()
        else:
            self._solver = self._get_solver_single()

        initial_solution: Solution = self.solution_args.get_initial_solution()
        traced_parameters: TracedParameters = self.solution_args.get_traced_parameters()

        # Compile
        start_time = time.time()
        self._solver(initial_solution, traced_parameters).block_until_ready()
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
                logger.debug("Found O2 at index = %d", nn)
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
        stability_matrix: npt.NDArray[np.float_] = self.get_stability_matrix()
        gas_species_indices: tuple[int, ...] = self.get_gas_species_indices()
        condensed_species_indices: tuple[int, ...] = self.get_condensed_species_indices()
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

        # Fugacity constraint matrix and indices
        number_fugacity_constraints: int = len(fugacity_constraints.constraints)
        fugacity_species_indices: list[int] = []
        species_names: tuple[str, ...] = self.get_species_names()
        for species_name in fugacity_constraints.constraints.keys():
            index: int = species_names.index(species_name)
            fugacity_species_indices.append(index)
        fugacity_matrix: npt.NDArray[np.float_] = np.identity(number_fugacity_constraints)

        # For fixed parameters all objects must be hashable because it is a static argument
        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=tuple(map(tuple, formula_matrix)),
            formula_matrix_constraints=tuple(map(tuple, formula_matrix_constraints)),
            reaction_matrix=tuple(map(tuple, reaction_matrix)),
            stability_matrix=tuple(map(tuple, stability_matrix)),
            fugacity_matrix=tuple(map(tuple, fugacity_matrix)),
            gas_species_indices=gas_species_indices,
            condensed_species_indices=condensed_species_indices,
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
        """Gets unique elements.

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
        """Gets the jit solver with fixed and solver parameters set.

        Returns:
            jit solver with fixed and solver parameters set
        """
        fixed_parameters: FixedParameters = self.get_fixed_parameters(
            self.solution_args.fugacity_constraints, self.solution_args.mass_constraints
        )
        solver_parameters: SolverParameters = self.solution_args.solver_parameters

        def wrapped_jit_solver(
            solution: Solution,
            traced_parameters: TracedParameters,
        ) -> Callable:
            return solve(solution, traced_parameters, fixed_parameters, solver_parameters)

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

    def get_stability_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the stability matrix.

        Returns:
            A matrix for the stability of species
        """
        reaction_matrix: npt.NDArray[np.float_] = self.get_reaction_matrix()
        mask: npt.NDArray[np.bool_] = np.zeros_like(reaction_matrix, dtype=bool)

        if reaction_matrix.size > 0:
            # Find the species to solve for stability
            stability_bool: npt.NDArray[np.bool_] = np.array(
                [species.solve_for_stability for species in self.species], dtype=bool
            )
            mask[:, stability_bool] = True

            stability_matrix: npt.NDArray[np.float_] = reaction_matrix * mask
        else:
            stability_matrix = reaction_matrix

        logger.debug("stability_matrix = %s", stability_matrix)

        return stability_matrix

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
        """Gets the names of all species.

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
                self.get_wrapped_jit_solver(),
                in_axes=(None, self.get_traced_parameters_vmap()),
            )
        )

        return solver

    def get_traced_parameters_vmap(self) -> TracedParameters:
        """Gets the vmapping axes for tracer parameters.

        Returns:
            Vmapping for tracer parameters
        """
        traced_parameters_vmap: TracedParameters = TracedParameters(
            planet=self.solution_args.planet.vmap_axes(),
            fugacity_constraints=self.solution_args.fugacity_constraints.vmap_axes(),
            mass_constraints=self.solution_args.mass_constraints.vmap_axes(),
        )

        return traced_parameters_vmap

    def solve(
        self,
        *,
        planet: Planet | None = None,
        initial_log_number_density: ArrayLike | None = None,
        initial_log_stability: ArrayLike | None = None,
        fugacity_constraints: Mapping[str, FugacityConstraintProtocol] | None = None,
        mass_constraints: Mapping[str, ArrayLike] | None = None,
        quick_look: bool = False,
    ) -> Output:
        """Solves the system and returns the processed solution.

        Args:
            planet: Planet. Defaults to None.
            initial_log_number_density: Initial log number density. Defaults to None.
            initial_log_stability: Initial log stability. Defaults to None.
            fugacity_constraints: Fugacity constraints. Defaults to None.
            mass_constraints: Mass constraints. Defaults to None.
            quick_look: Show the solution via the logger. Defaults to False.

        Returns:
            Output
        """
        self._solution_args: SolutionArguments = self.solution_args.override(
            planet,
            initial_log_number_density,
            initial_log_stability,
            fugacity_constraints,
            mass_constraints,
        )

        initial_solution: Solution = self.solution_args.get_initial_solution()
        traced_parameters: TracedParameters = self.solution_args.get_traced_parameters()

        # Execute
        start_time = time.time()
        solution: Array = self._solver(initial_solution, traced_parameters).block_until_ready()
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Execution time: %.6f seconds", execution_time)

        output: Output = Output(solution, self, initial_solution, traced_parameters)

        if quick_look:
            quick_look_dict: dict[str, ArrayLike] = output.quick_look()
            logger.info("quick_look = %s", pprint.pformat(quick_look_dict))

        return output
