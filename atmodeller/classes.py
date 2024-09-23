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
import time
from collections.abc import KeysView
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array

from atmodeller import TAU
from atmodeller.jax_containers import (
    Constraints,
    Parameters,
    Planet,
    Solution,
    SolverParameters,
    SpeciesData,
)
from atmodeller.jax_engine import get_log_extended_activity, solve_set_solver
from atmodeller.jax_utilities import pytrees_stack, unscale_number_density
from atmodeller.utilities import partial_rref, unique_elements_in_species

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere system

    Args:
        species: A list of species
        log_scaling: Log scaling for the numerical solution
    """

    def __init__(self, species: list[SpeciesData], log_scaling: float):
        self.species: list[SpeciesData] = species
        self.log_scaling: float = log_scaling
        self.formula_matrix: Array = jnp.array(self.get_formula_matrix())
        self.reaction_matrix: Array = jnp.array(self.get_reaction_matrix())
        self.solver_parameters: SolverParameters = SolverParameters.create(
            self.species, self.log_scaling
        )

    def initialise_single(
        self,
        planet: Planet,
        mass_constraints: dict[str, float],
        initial_number_density: npt.NDArray[np.float_],
        initial_stability: npt.NDArray[np.float_],
        tau: float = TAU,
    ) -> None:
        """Initialises a solve for a single system

        Args:
            planet: Planet
            mass_constraints: Mass constraints
            initial_number_density: Initial number density
            initial_stability: Initial stability
            tau: Tau factor for species stability
        """
        self.planet: Planet = planet
        logger.debug("planet = %s", self.planet)
        self.constraints: Constraints = Constraints.create(
            self.species, mass_constraints, self.log_scaling
        )
        logger.debug("constraints = %s", self.constraints)
        self.initial_solution: Solution = Solution.create(
            initial_number_density, initial_stability, self.log_scaling
        )
        logger.debug("initial_solution = %s", self.initial_solution)
        self.parameters: Parameters = Parameters(
            formula_matrix=self.formula_matrix,
            reaction_matrix=self.reaction_matrix,
            species=self.species,
            planet=self.planet,
            constraints=self.constraints,
            tau=tau,
            log_scaling=self.log_scaling,
        )

        start_time = time.time()
        self._solve = solve_set_solver(self.solver_parameters)
        self._solve(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        compile_time = end_time - start_time
        logger.info("Compile time: %.6f seconds", compile_time)

    def stack_mass_constraints(
        self, mass_constraints_list: list[dict[str, float]]
    ) -> dict[str, Array]:

        stacked_mass_constraints: dict[str, Array] = {}
        keys: KeysView[str] = mass_constraints_list[0].keys()

        for key in keys:
            stacked_mass_constraints[key] = jnp.array([d[key] for d in mass_constraints_list])

        return stacked_mass_constraints

    def initialise_batch(
        self,
        planet_list: list[Planet],
        mass_constraints_list: list[dict[str, float]],
        initial_number_density: npt.NDArray[np.float_],
        initial_stability: npt.NDArray[np.float_],
        tau: float = TAU,
    ) -> None:
        """Initialises a solve for a batch system

        Args:
            planet_batch: A list of planets
            mass_constraints_batch: A list of mass constraints
            initial_number_density: Initial number density
            initial_stability: Initial stability
            tau: Tau factor for species stability
        """
        planets_batch = pytrees_stack(planet_list)
        # TODO: Temporary hack to get something working
        mass_constraints_batch = (
            mass_constraints_list  # self.stack_mass_constraints(mass_constraints_list)
        )
        self.constraints: Constraints = Constraints.create(
            self.species, mass_constraints_batch, self.log_scaling
        )
        constraints_vmap: Constraints = Constraints(species=None, log_molecules=0)  # type: ignore
        self.initial_solution: Solution = Solution.create(
            initial_number_density, initial_stability, self.log_scaling
        )
        self.parameters: Parameters = Parameters(
            formula_matrix=self.formula_matrix,
            reaction_matrix=self.reaction_matrix,
            species=self.species,
            planet=planets_batch,
            constraints=self.constraints,
            tau=tau,
            log_scaling=self.log_scaling,
        )
        parameters_vmap: Parameters = Parameters(
            formula_matrix=None,  # type: ignore
            reaction_matrix=None,  # type: ignore
            species=None,  # type: ignore
            planet=0,  # type: ignore
            constraints=constraints_vmap,  # type: ignore
            tau=None,  # type: ignore
            log_scaling=None,  # type: ignore
        )

        self._solve: Callable = jax.jit(
            jax.vmap(solve_set_solver(self.solver_parameters), in_axes=(None, parameters_vmap))
        )
        start_time = time.time()
        self._solve(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        compile_time = end_time - start_time
        logger.info("Compile time: %.6f seconds", compile_time)

    def get_formula_matrix(self) -> npt.NDArray[np.int_]:
        """Formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Args:
            species: A list of species

        Returns:
            The formula matrix
        """
        unique_elements: tuple[str, ...] = unique_elements_in_species(self.species)
        formula_matrix: npt.NDArray[np.int_] = np.zeros(
            (len(unique_elements), len(self.species)), dtype=jnp.int_
        )

        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(self.species):
                count: int = 0
                try:
                    count = species_.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def get_reaction_matrix(self) -> npt.NDArray[np.float_]:
        """Reaction matrix

        Args:
            species: A list of species

        Returns:
            A matrix of linearly independent reactions or None # TODO: Still return None?
        """
        # TODO: Would prefer to always return an array even in the absence of reactions?
        # if self._species.number == 1:
        #    logger.debug("Only one species therefore no reactions")
        #    return None

        transpose_formula_matrix: npt.NDArray[np.int_] = self.get_formula_matrix().T
        reaction_matrix: npt.NDArray[np.float_] = partial_rref(transpose_formula_matrix)

        # logger.debug("species = %s", species)
        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary

        Args:
            species: A list of species

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
                        reactants += f"{abs(coeff)} {species_.name} + "
                    else:
                        products += f"{coeff} {species_.name} + "

            reactants = reactants.rstrip(" + ")
            products = products.rstrip(" + ")
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

        return reactions

    def solve(self) -> tuple[npt.NDArray[np.float_], ...]:
        start_time = time.time()
        out: Array = self._solve(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Execution time: %.6f seconds", execution_time)

        logger.info("out = %s", out)

        # Split the array into two arrays along the middle column
        # FIXME: Needs to work with 1-D and 2-D data
        scaled_number_density, stability = jnp.split(out, 2, axis=1)
        unscaled_number_density: Array = unscale_number_density(
            scaled_number_density, self.log_scaling
        )
        number_density_numpy: npt.NDArray[np.float_] = np.array(unscaled_number_density)
        logger.debug("log_number_density = %s", number_density_numpy)
        extended_activity: Array = get_log_extended_activity(
            unscaled_number_density, stability, self.parameters
        )
        extended_activity_numpy: npt.NDArray[np.float_] = np.array(extended_activity)
        logger.debug("log_activity = %s", extended_activity_numpy)

        return number_density_numpy, extended_activity_numpy
