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
    FixedParameters,
    Parameters,
    Planet,
    Solution,
    SolverParameters,
    Species,
)
from atmodeller.jax_engine import get_log_extended_activity, solve
from atmodeller.jax_utilities import pytrees_stack, unscale_number_density
from atmodeller.utilities import partial_rref, unique_elements_in_species2

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphere:
    """Interior atmosphere system

    Args:
        species: A list of species
        log_scaling: Log scaling for the numerical solution
    """

    def __init__(self, species: list[Species], log_scaling: float):
        self.species: list[Species] = species
        self.log_scaling: float = log_scaling
        self.solver_parameters: SolverParameters = SolverParameters.create(
            self.species, self.log_scaling
        )
        logger.info("reactions = %s", pprint.pformat(self.reactions()))

    def initialise_single(
        self,
        planet: Planet,
        mass_constraints: dict[str, float],
        initial_number_density: npt.NDArray[np.float_],
        initial_stability: npt.NDArray[np.float_],
        tau: float = TAU,
    ) -> None:
        """Initialises the solution for a single system

        Args:
            planet: Planet
            mass_constraints: Mass constraints
            initial_number_density: Initial number density
            initial_stability: Initial stability
            tau: Tau factor for species stability. Defaults to TAU.
        """
        logger.debug("planet = %s", planet)
        constraints: Constraints = Constraints.create(
            self.species, mass_constraints, self.log_scaling
        )
        logger.debug("constraints = %s", constraints)
        self.initial_solution: Solution = Solution.create(
            initial_number_density, initial_stability, self.log_scaling
        )
        logger.debug("initial_solution = %s", self.initial_solution)
        fixed: FixedParameters = self.get_fixed_parameters(tau)
        self.parameters: Parameters = Parameters(
            fixed=fixed,
            planet=planet,
            constraints=constraints,
        )

        start_time = time.time()
        self._solve: Callable = self.get_jit_solver()
        self._solve(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        compile_time = end_time - start_time
        logger.info("Compile time: %.6f seconds", compile_time)

    def stack_mass_constraints(
        self, mass_constraints_list: list[dict[str, float]]
    ) -> dict[str, Array]:
        """Stacks the mass constraints into a pytree

        Args:
            mass_constraints_list: A list of mass constraints

        Returns:
            A pytree of mass constraints
        """

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
            planet_list: A list of planets
            mass_constraints_list: A list of mass constraints
            initial_number_density: Initial number density
            initial_stability: Initial stability
            tau: Tau factor for species stability
        """
        # Stack the planets and mass constraints into pytrees
        planets_batch = pytrees_stack(planet_list)
        mass_constraints_batch = self.stack_mass_constraints(mass_constraints_list)
        constraints: Constraints = Constraints.create(
            self.species, mass_constraints_batch, self.log_scaling
        )
        self.initial_solution: Solution = Solution.create(
            initial_number_density, initial_stability, self.log_scaling
        )
        fixed: FixedParameters = self.get_fixed_parameters(tau)
        self.parameters: Parameters = Parameters(
            fixed=fixed,
            planet=planets_batch,
            constraints=constraints,
        )
        # Define the structures to vectorize.
        constraints_vmap: Constraints = Constraints(species=None, log_molecules=0)  # type: ignore
        parameters_vmap: Parameters = Parameters(
            fixed=None,  # type: ignore
            planet=0,  # type: ignore
            constraints=constraints_vmap,  # type: ignore
        )

        self._solve: Callable = jax.jit(
            jax.vmap(self.get_jit_solver(), in_axes=(None, parameters_vmap))
        )
        start_time = time.time()
        self._solve(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        compile_time = end_time - start_time
        logger.info("Compile time: %.6f seconds", compile_time)

    def get_fixed_parameters(self, tau: float = TAU) -> FixedParameters:
        """Gets fixed parameters

        Args:
            tau: Tau factor for species stability. Defaults to TAU.

        Returns:
            Fixed parameters
        """
        formula_matrix: Array = jnp.array(self.get_formula_matrix())
        reaction_matrix: Array = jnp.array(self.get_reaction_matrix())
        gas_species_indices: Array = jnp.array(self.get_gas_species_indices())
        molar_masses: Array = jnp.array(self.get_molar_masses())
        diatomic_oxygen_index: Array = jnp.array(self.get_diatomic_oxygen_index())

        fixed_parameters: FixedParameters = FixedParameters(
            species=self.species,
            formula_matrix=formula_matrix,
            reaction_matrix=reaction_matrix,
            gas_species_indices=gas_species_indices,
            diatomic_oxygen_index=diatomic_oxygen_index,
            molar_masses=molar_masses,
            tau=tau,
            log_scaling=self.log_scaling,
        )

        return fixed_parameters

    def get_diatomic_oxygen_index(self) -> int:
        """Gets the species index corresponding to diatomic oxygen

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

    def get_molar_masses(self) -> npt.NDArray[np.float_]:
        """Gets the molar masses of all species

        Returns:
            Molar masses of all species
        """
        molar_masses: npt.NDArray[np.float_] = np.array(
            [species_.data.molar_mass for species_ in self.species]
        )
        logger.debug("molar_masses = %s", molar_masses)

        return molar_masses

    def get_formula_matrix(self) -> npt.NDArray[np.int_]:
        """Gets the formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Args:
            species: A list of species

        Returns:
            The formula matrix
        """
        unique_elements: tuple[str, ...] = unique_elements_in_species2(self.species)
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

    def get_reaction_matrix(self) -> npt.NDArray[np.float_]:
        """Gets the reaction matrix

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

    def get_jit_solver(self) -> Callable:
        """Gets the jit solver

        Returns:
            The jit solver with solver parameters set
        """

        def wrapped_jit_solver(solution: Solution, parameters: Parameters) -> Callable:
            return solve(solution, parameters, self.solver_parameters)

        return wrapped_jit_solver

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
                        reactants += f"{abs(coeff)} {species_.data.name} + "
                    else:
                        products += f"{coeff} {species_.data.name} + "

            reactants = reactants.rstrip(" + ")
            products = products.rstrip(" + ")
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

        return reactions

    def solve(self) -> tuple[npt.NDArray[np.float_], ...]:
        """Solves the system.

        Returns:
            Number density, extended activity
        """
        start_time = time.time()
        out: Array = self._solve(self.initial_solution, self.parameters).block_until_ready()
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("Execution time: %.6f seconds", execution_time)
        logger.debug("out = %s", out)

        # Split the array into two arrays along the middle column
        axis: int = 0 if out.ndim == 1 else 1
        scaled_number_density, stability = jnp.split(out, 2, axis=axis)
        unscaled_number_density: Array = unscale_number_density(
            scaled_number_density, self.log_scaling
        )
        number_density_numpy: npt.NDArray[np.float_] = np.array(unscaled_number_density)
        logger.info("log_number_density = %s", number_density_numpy)
        extended_activity: Array = get_log_extended_activity(
            unscaled_number_density, stability, self.parameters
        )
        extended_activity_numpy: npt.NDArray[np.float_] = np.array(extended_activity)
        logger.info("log_activity = %s", extended_activity_numpy)

        return number_density_numpy, extended_activity_numpy
