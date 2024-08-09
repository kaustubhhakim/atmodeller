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
"""Reaction network"""

# Convenient to use chemical formulae so pylint: disable=C0103

from __future__ import annotations

import logging
import pprint
import sys
from typing import Callable

import jax
import jax.numpy as jnp
import optimistix as optx
from scipy.optimize import OptimizeResult, root

from atmodeller import BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.core import Planet, Species
from atmodeller.initial_solution import InitialSolutionDict, InitialSolutionProtocol
from atmodeller.solution import Solution

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class ReactionNetwork:
    """A chemical reaction network.

    Args:
        species: Species
        planet: Planet
        constraints: Constraints
    """

    def __init__(self, *, species: Species, planet: Planet):
        self._species: Species = species
        self.planet = planet
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self._species.names)
        self.reaction_matrix: jnp.ndarray | None = self.get_reaction_matrix()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        """Number of reactions"""
        if self._species.number == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def temperature(self) -> float:
        return self.planet.surface_temperature

    def formula_matrix(self) -> jnp.ndarray:
        """Gets the formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            The formula matrix
        """
        formula_matrix: jnp.ndarray = jnp.zeros(
            (len(self._species.elements()), len(self._species.data)), dtype=jnp.int_
        )
        for element_index, element in enumerate(self._species.elements()):
            for species_index, species in enumerate(self._species.data):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                formula_matrix = formula_matrix.at[element_index, species_index].set(count)

        return formula_matrix

    def get_reaction_matrix(self) -> jnp.ndarray | None:
        """Gets the reaction matrix

        Returns:
            A matrix of linearly independent reactions or None
        """
        if self._species.number == 1:
            logger.debug("Only one species therefore no reactions")
            return None

        transpose_formula_matrix: jnp.ndarray = self.formula_matrix().T

        return partial_rref(transpose_formula_matrix)

    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary

        Returns:
            Reactions as a dictionary
        """
        reactions: dict[int, str] = {}
        if self.reaction_matrix is not None:
            for reaction_index in range(self.number_reactions):
                reactants: str = ""
                products: str = ""
                for species_index, species in enumerate(self._species.data):
                    coeff: float = self.reaction_matrix[reaction_index, species_index].item()
                    if coeff != 0:
                        if coeff < 0:
                            reactants += f"{abs(coeff)} {species.name} + "
                        else:
                            products += f"{coeff} {species.name} + "

                reactants = reactants.rstrip(" + ")
                products = products.rstrip(" + ")
                reaction: str = f"{reactants} = {products}"
                reactions[reaction_index] = reaction

        return reactions

    def _get_lnKp(self, reaction_index: int, pressure: jnp.ndarray) -> jnp.ndarray | float:
        """Gets the natural log of the equilibrium constant in terms of partial pressures.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            Natural log of the equilibrium constant in terms of partial pressures
        """
        gibbs_energy: jnp.ndarray | float = self._get_reaction_gibbs_energy_of_formation(
            reaction_index, pressure
        )
        lnKp: jnp.ndarray | float = -gibbs_energy / (GAS_CONSTANT * self.temperature())

        return lnKp

    def _get_log10Kp(self, reaction_index: int, pressure: jnp.ndarray) -> jnp.ndarray | float:
        """Gets the log10 of the equilibrium constant in terms of partial pressures.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            log10 of the equilibrium constant in terms of partial pressures
        """
        lnKp: jnp.ndarray | float = self._get_lnKp(reaction_index, pressure)
        log10Kp: jnp.ndarray | float = lnKp / jnp.log(10)

        return log10Kp

    def _get_delta_n(self, reaction_index: int) -> jnp.ndarray:
        """Gets the difference in the moles of gas products compared to the moles of gas reactants.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`

        Returns:
            The difference in the moles of gas products compared to the moles of gas reactants
        """
        assert self.reaction_matrix is not None

        delta_n = jnp.sum(
            jnp.array(
                [
                    self.reaction_matrix[reaction_index, gas_index]
                    for gas_index in self._species.gas_species()
                ]
            )
        )

        return delta_n

    def _get_lnKc(self, reaction_index: int, pressure: jnp.ndarray) -> jnp.ndarray | float:
        """Gets the natural log of the equilibrium constant in terms of number densities.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            Natural log of the equilibrium constant in terms of number densities
        """
        lnKp: jnp.ndarray | float = self._get_lnKp(reaction_index, pressure)
        delta_n: jnp.ndarray | float = self._get_delta_n(reaction_index)
        lnKc: jnp.ndarray | float = lnKp - delta_n * (
            jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(self.temperature())
        )

        return lnKc

    def _get_log10Kc(self, reaction_index: int, pressure: jnp.ndarray) -> jnp.ndarray | float:
        """Gets the log10 of the equilibrium constant in terms of number densities.

        Args:
            reaction_index: Row index of the reaction as it appears in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            log10 of the equilibrium constant in terms of number densities
        """
        lnKc: jnp.ndarray | float = self._get_lnKc(reaction_index, pressure)
        log10Kc: jnp.ndarray | float = lnKc / jnp.log(10)

        return log10Kc

    def _get_reaction_gibbs_energy_of_formation(
        self, reaction_index: int, pressure: jnp.ndarray
    ) -> jnp.ndarray | float:
        r"""Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            The Gibb's free energy of the reaction in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        gibbs_energy: float = 0
        assert self.reaction_matrix is not None
        for species_index, species in enumerate(self._species):
            assert species.thermodata is not None
            # TODO: Convert to JAX
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * species.thermodata.get_formation_gibbs(
                temperature=self.temperature(), pressure=pressure
            )

        return gibbs_energy

    def get_coefficient_matrix(self, constraints: SystemConstraints) -> jnp.ndarray:
        """Gets the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """
        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        coeff: jnp.ndarray = jnp.zeros((nrows, self._species.number))
        if self.reaction_matrix is not None:
            coeff = coeff.at[0 : self.number_reactions].set(self.reaction_matrix)

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index: int = self._species.species_index(constraint.species)
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff = coeff.at[row_index, species_index].set(1.0)

        logger.debug("Coefficient matrix = \n%s", coeff)

        return coeff

    def get_right_hand_side(
        self, pressure: jnp.ndarray, constraints: SystemConstraints
    ) -> jnp.ndarray:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            pressure: Pressure in bar
            constraints: Constraints

        Returns:
            The right-hand side vector of values
        """
        nrows: int = self.number_reactions + constraints.number_reaction_network_constraints
        rhs: jnp.ndarray = jnp.zeros(nrows, dtype=jnp.float_)

        # Reactions
        for reaction_index in range(self.number_reactions):
            # log10Kc: float = self._get_log10Kc(
            #    reaction_index
            # )
            # FIXME: Hack to try jax
            log10Kc = 17.47976865
            rhs = rhs.at[reaction_index].set(log10Kc)

        # Constraints
        for index, constraint in enumerate(constraints.reaction_network_constraints):
            row_index: int = self.number_reactions + index
            # pylint: disable=line-too-long
            # logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs = rhs.at[row_index].set(
                constraint.get_log10_value(temperature=self.temperature(), pressure=pressure)
            )

        logger.debug("_right_hand_side = %s", rhs)

        return rhs

    def _assemble_log_fugacity_coefficients(
        self,
        *,
        temperature: float,
        pressure: float,
    ) -> jnp.ndarray:
        """Assembles the fugacity coefficient vector on the left-hand side of the equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The log10(fugacity coefficient) vector
        """
        # Initialise to ideal behaviour.
        fugacity_coefficients: jnp.ndarray = jnp.ones_like(self._species)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for gas_species_index, gas_species in self._species.gas_species().items():
            fugacity_coefficient: float = gas_species.eos.fugacity_coefficient(
                temperature=temperature, pressure=pressure
            )
            fugacity_coefficients = fugacity_coefficients.at[gas_species_index].set(
                fugacity_coefficient
            )

        log_fugacity_coefficients: jnp.ndarray = jnp.log10(fugacity_coefficients)
        logger.debug("Log10 fugacity coefficient vector = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    def objective_function(self, solution_array: jnp.ndarray, args) -> jnp.ndarray:
        """Objective function

        Args:
            solution_array: Array of the solution
            constraints: System constraints

        Returns:
            Residual array
        """
        self.solution = Solution.create_from_species(species=self._species)
        self.solution.planet = self.planet
        self.solution.value = solution_array

        pressure: jnp.ndarray = self.solution.atmosphere.pressure()

        constraints = args[0]

        # FIXME: Commented out for simplicity
        # log_fugacity_coefficients: jnp.ndarray = self._assemble_log_fugacity_coefficients(
        #    temperature=temperature, pressure=pressure
        # )
        residual: jnp.ndarray = (
            # FIXME: Commented out for simplicity
            # coefficient_matrix.dot(log_fugacity_coefficients)
            self.get_coefficient_matrix(constraints).dot(self.solution.get_reaction_array())
            - self.get_right_hand_side(pressure, constraints)
        )

        return residual

    # def jacobian(self) -> Callable:
    #     return jax.jacobian(self.objective_function)

    # def solve(
    #     self,
    #     initial_solution: InitialSolutionProtocol | None = None,
    #     *,
    #     method: str = "hybr",
    #     tol: float | None = None,
    #     **options,
    # ) -> tuple(OptimizeResult, Solution):

    #     if initial_solution is None:
    #         initial_solution = InitialSolutionDict(species=self._species)
    #     assert initial_solution is not None

    #     initial_solution_guess: jnp.ndarray = initial_solution.get_log10_value(
    #         self.constraints,
    #         temperature=self.temperature(),
    #         pressure=1,
    #     )

    #     sol = root(
    #         self.objective_function,
    #         initial_solution_guess,
    #         method=method,
    #         jac=self.jacobian(),
    #         tol=tol,
    #         options=options,
    #     )

    #     return sol, self.solution

    def solve_optimistix(
        self,
        initial_solution: InitialSolutionProtocol | None = None,
        *,
        constraints: SystemConstraints,
        method: str = "hybr",
        tol: float | None = None,
        **options,
    ) -> tuple[OptimizeResult, Solution]:

        if initial_solution is None:
            initial_solution = InitialSolutionDict(species=self._species)
        assert initial_solution is not None

        initial_solution_guess: jnp.ndarray = initial_solution.get_log10_value(
            constraints,
            temperature=self.temperature(),
            pressure=1,
        )

        solver = optx.Newton(rtol=1e-8, atol=1e-8)
        sol = optx.root_find(
            self.objective_function,
            solver,
            initial_solution_guess,
            args=(constraints,),
        )

        solution: Solution = Solution.create_from_species(species=self._species)
        solution.planet = self.planet
        solution.value = jnp.array(sol.value)

        return sol, solution


class ReactionNetworkWithCondensateStability(ReactionNetwork):
    """A reaction network with condensate stability

    This automatically determines condensate stability using the extended law of mass-action
    (xLMA) equations :cite:p:`LKS17`. Also see :cite:p:`KSP24`.

    Args:
        species: Species

    Attributes:
        species: Species
        reaction_matrix: The reaction stoichiometry matrix
    """

    def get_activity_modifier(self) -> jnp.ndarray:
        """Gets the activity modifier matrix for condensate stability

        Returns:
            Activity modifier matrix
        """
        coefficient_matrix: jnp.ndarray = self.get_coefficient_matrix()
        activity_modifier: jnp.ndarray = jnp.zeros_like(coefficient_matrix)
        for condensed_index in self._species.condensed_species():
            activity_modifier = activity_modifier.at[:, condensed_index].set(
                coefficient_matrix[:, condensed_index]
            )

        logger.debug("activity_modifier = %s", activity_modifier)

        return activity_modifier

    def get_equilibrium_modifier(self) -> jnp.ndarray:
        """Gets the equilibrium constant modifier matrix for condensate stability

        Returns:
            Equilibrium constant modifier matrix
        """
        activity_modifier: jnp.ndarray = self.get_activity_modifier()
        equilibrium_modifier: jnp.ndarray = jnp.array(activity_modifier)
        equilibrium_modifier = equilibrium_modifier.at[self.number_reactions :, :].set(0)

        logger.debug("equilibrium_modifier = %s", equilibrium_modifier)

        return equilibrium_modifier

    @override
    def get_residual(
        self,
        *,
        temperature: float,
        pressure: jnp.ndarray,
        reaction_array: jnp.ndarray,
        activity_modifier: jnp.ndarray,
        equilibrium_modifier: jnp.ndarray,
        stability_array: jnp.ndarray,
    ) -> jnp.ndarray:

        residual_reaction: jnp.ndarray = super().get_residual(
            temperature=temperature,
            pressure=pressure,
            reaction_array=reaction_array,
        )

        residual_reaction += activity_modifier.dot(10**stability_array)
        residual_reaction -= equilibrium_modifier.dot(10**stability_array)

        # TODO: Move stability residual also into here

        return residual_reaction


def partial_rref(matrix: jnp.ndarray) -> jnp.ndarray:
    """Computes the partial reduced row echelon form to determine linear components

    Args:
        matrix: The matrix to compute the reduced row echelon form

    Returns:
        A matrix of linear components
    """
    nrows, ncols = matrix.shape

    # Augment the matrix with the identity matrix
    augmented_matrix = jnp.hstack((matrix, jnp.eye(nrows)))
    logger.debug("augmented_matrix = \n%s", augmented_matrix)

    # Forward elimination
    for i in range(ncols):
        # Check if the pivot element is zero and swap rows to get a non-zero pivot element.
        if augmented_matrix[i, i] == 0:
            nonzero_rows = jnp.nonzero(augmented_matrix[i:, i])[0]
            if nonzero_rows.size > 0:
                nonzero_row = nonzero_rows[0] + i
                # Swap rows
                augmented_matrix = augmented_matrix.at[i, :].set(augmented_matrix[nonzero_row, :])
                augmented_matrix = augmented_matrix.at[nonzero_row, :].set(augmented_matrix[i, :])

        # Perform row operations to eliminate values below the pivot.
        for j in range(i + 1, nrows):
            ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix = augmented_matrix.at[j, :].set(
                augmented_matrix[j, :] - ratio * augmented_matrix[i, :]
            )

    logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

    # Backward substitution
    for i in range(ncols - 1, -1, -1):
        # Normalize the pivot row.
        augmented_matrix = augmented_matrix.at[i, :].set(
            augmented_matrix[i, :] / augmented_matrix[i, i]
        )
        # Eliminate values above the pivot.
        for j in range(i - 1, -1, -1):
            if augmented_matrix[j, i] != 0:
                ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix = augmented_matrix.at[j, :].set(
                    augmented_matrix[j, :] - ratio * augmented_matrix[i, :]
                )

    logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

    # Extract the reduced matrix and component matrix
    reduced_matrix = augmented_matrix[:, :ncols]
    component_matrix = augmented_matrix[ncols:, ncols:]
    logger.debug("reduced_matrix = \n%s", reduced_matrix)
    logger.debug("component_matrix = \n%s", component_matrix)

    return component_matrix
