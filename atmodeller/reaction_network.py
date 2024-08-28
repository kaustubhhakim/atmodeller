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
from typing import Protocol

import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import ArrayLike

from atmodeller import BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.core import Planet, Species
from atmodeller.solution import Solution

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class ResidualProtocol(Protocol):
    """Protocol for some system to solve"""

    _species: Species
    _planet: Planet

    def get_residual(self, solution: Solution, constraints: SystemConstraints) -> Array: ...

    @property
    def species(self) -> Species:
        return self._species

    @property
    def planet(self) -> Planet:
        return self._planet

    def temperature(self) -> float:
        return self.planet.surface_temperature


class ReactionNetwork(ResidualProtocol):
    """A chemical reaction network.

    Args:
        species: Species
        planet: Planet
    """

    def __init__(self, species: Species, planet: Planet):
        logger.info("Creating a reaction network")
        self._species: Species = species
        logger.info("Species = %s", self._species.names)
        self._planet: Planet = planet
        self.reaction_matrix: Array | None = self.get_reaction_matrix()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        """Number of reactions"""
        if self._species.number == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def formula_matrix(self) -> Array:
        """Gets the formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            The formula matrix
        """
        formula_matrix: Array = jnp.zeros(
            (len(self._species.elements()), len(self._species.data)), dtype=jnp.int_
        )
        for element_index, element in enumerate(self._species.elements()):
            for species_index, species in enumerate(self._species.data):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                formula_matrix = formula_matrix.at[element_index, species_index].set(count)

        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def get_reaction_matrix(self) -> Array | None:
        """Gets the reaction matrix

        Returns:
            A matrix of linearly independent reactions or None
        """
        if self._species.number == 1:
            logger.debug("Only one species therefore no reactions")
            return None

        transpose_formula_matrix: Array = self.formula_matrix().T

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

    def _get_lnKp(self, reaction_index: int, pressure: ArrayLike) -> Array:
        """Gets the natural log of the equilibrium constant in terms of partial pressures.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            Natural log of the equilibrium constant in terms of partial pressures
        """
        gibbs_energy: ArrayLike = self._get_reaction_gibbs_energy_of_formation(
            reaction_index, pressure
        )
        lnKp: ArrayLike = -1 * gibbs_energy / (GAS_CONSTANT * self.temperature())

        return lnKp

    def _get_log10Kp(self, reaction_index: int, pressure: ArrayLike) -> Array:
        """Gets the log10 of the equilibrium constant in terms of partial pressures.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            log10 of the equilibrium constant in terms of partial pressures
        """
        lnKp: Array = self._get_lnKp(reaction_index, pressure)
        log10Kp: Array = lnKp / jnp.log(10)

        return log10Kp

    def _get_delta_n(self, reaction_index: int) -> Array:
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

    def _get_lnKc(self, reaction_index: int, pressure: ArrayLike) -> Array:
        """Gets the natural log of the equilibrium constant in terms of number densities.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            Natural log of the equilibrium constant in terms of number densities
        """
        lnKp: ArrayLike = self._get_lnKp(reaction_index, pressure)
        delta_n: ArrayLike = self._get_delta_n(reaction_index)
        lnKc: ArrayLike = lnKp - delta_n * (
            jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(self.temperature())
        )

        return lnKc

    def _get_log10Kc(self, reaction_index: int, pressure: ArrayLike) -> Array:
        """Gets the log10 of the equilibrium constant in terms of number densities.

        Args:
            reaction_index: Row index of the reaction as it appears in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            log10 of the equilibrium constant in terms of number densities
        """
        lnKc: Array = self._get_lnKc(reaction_index, pressure)
        log10Kc: Array = lnKc / jnp.log(10)

        return log10Kc

    def _get_reaction_gibbs_energy_of_formation(
        self, reaction_index: int, pressure: ArrayLike
    ) -> Array:
        r"""Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in :attr:`reaction_matrix`
            pressure: Pressure in bar

        Returns:
            The Gibb's free energy of the reaction in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        gibbs_energy: Array = jnp.zeros(())

        assert self.reaction_matrix is not None

        for species_index, species in enumerate(self._species):
            assert species.thermodata is not None
            # Get Gibb's formation energy for the current species
            formation_gibbs: ArrayLike = species.thermodata.get_formation_gibbs(
                temperature=self.temperature(), pressure=pressure
            )
            gibbs_energy = (
                gibbs_energy
                + self.reaction_matrix[reaction_index, species_index] * formation_gibbs
            )

        return gibbs_energy

    def get_coefficient_matrix(self, *, constraints: SystemConstraints) -> Array:
        """Gets the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """
        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        coeff: Array = jnp.zeros((nrows, self._species.number))
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

    def _get_right_hand_side(
        self,
        pressure: ArrayLike,
        *,
        constraints: SystemConstraints,
    ) -> Array:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            pressure: Pressure in bar
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = self.number_reactions + constraints.number_reaction_network_constraints
        rhs: Array = jnp.zeros(nrows, dtype=jnp.float_)

        # Reactions
        for reaction_index in range(self.number_reactions):
            log10Kc: Array = self._get_log10Kc(reaction_index, pressure)
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

    def _get_log_fugacity_coefficients(
        self,
        pressure: ArrayLike,
    ) -> Array:
        """Assembles the log10 fugacity coefficients.

        Args:
            pressure: Pressure in bar

        Returns:
            The log10(fugacity coefficient) vector
        """
        # Initialise to ideal behaviour.
        fugacity_coefficients: Array = jnp.ones(self._species.number, dtype=jnp.float_)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for gas_species_index, gas_species in self._species.gas_species().items():
            fugacity_coefficient: ArrayLike = gas_species.eos.fugacity_coefficient(
                temperature=self.temperature(), pressure=pressure
            )
            fugacity_coefficients = fugacity_coefficients.at[gas_species_index].set(
                fugacity_coefficient
            )

        log_fugacity_coefficients: Array = jnp.log10(fugacity_coefficients)
        logger.debug("Log10 fugacity coefficients = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    @override
    def get_residual(self, solution: Solution, constraints: SystemConstraints) -> Array:
        """Gets the residual

        Args:
            solution: Solution
            constraints: Constraints for the system of equations

        Returns:
            Residual array
        """
        pressure: Array = solution.atmosphere.pressure()
        residual: Array = (
            self.get_coefficient_matrix(constraints=constraints).dot(
                self._get_log_fugacity_coefficients(pressure)
            )
            + self.get_coefficient_matrix(constraints=constraints).dot(
                solution.get_reaction_array()
            )
            - self._get_right_hand_side(pressure, constraints=constraints)
        )

        return residual


class ReactionNetworkWithCondensateStability(ReactionNetwork):
    """A chemical reaction network with condensate stability

    This automatically determines condensate stability using the extended law of mass-action
    (xLMA) equations :cite:p:`LKS17`. Also see :cite:p:`KSP24`.

    Args:
        species: Species
        planet: Planet
    """

    def get_activity_modifier(self, *, constraints: SystemConstraints) -> Array:
        """Gets the activity modifier matrix for condensate stability

        Args:
            constraints: Constraints for the system of equations

        Returns:
            Activity modifier matrix
        """
        coefficient_matrix: Array = self.get_coefficient_matrix(constraints=constraints)
        activity_modifier: Array = jnp.zeros_like(coefficient_matrix)
        condensed_indices: Array = jnp.array(
            tuple(self._species.condensed_species().keys()), dtype=jnp.int_
        )
        activity_modifier = activity_modifier.at[:, condensed_indices].set(
            coefficient_matrix[:, condensed_indices]
        )

        logger.debug("activity_modifier = %s", activity_modifier)

        return activity_modifier

    def get_equilibrium_modifier(self, *, constraints: SystemConstraints) -> Array:
        """Gets the equilibrium constant modifier matrix for condensate stability

        Args:
            constraints: Constraints for the system of equations

        Returns:
            Equilibrium constant modifier matrix
        """
        activity_modifier: Array = self.get_activity_modifier(constraints=constraints)
        equilibrium_modifier: Array = activity_modifier.at[self.number_reactions :, :].set(0)

        logger.debug("equilibrium_modifier = %s", equilibrium_modifier)

        return equilibrium_modifier

    def get_stability_array(self, solution: Solution) -> Array:
        """Gets the condensate stability array

        Args:
            solution: Solution

        Returns:
            The condensate stability array
        """
        stability_array: Array = jnp.zeros(self._species.number, dtype=jnp.float_)
        for condensed_species, collection in solution.condensed.items():
            index: int = self._species.species_index(condensed_species)
            stability_array = stability_array.at[index].set(collection.stability.value)

        return stability_array

    @override
    def get_residual(self, solution: Solution, constraints: SystemConstraints) -> Array:
        stability_array: Array = self.get_stability_array(solution)
        residual: Array = super().get_residual(solution, constraints)
        activity_modifier: Array = self.get_activity_modifier(constraints=constraints)
        equilibrium_modifier: Array = self.get_equilibrium_modifier(constraints=constraints)

        residual = residual + jnp.dot(activity_modifier, jnp.power(10, stability_array))
        residual = residual - jnp.dot(equilibrium_modifier, jnp.power(10, stability_array))

        # Auxiliary condensate stability equations
        auxiliary_residual: Array = jnp.zeros(
            len(self._species.condensed_species()), dtype=jnp.float_
        )
        for index, collection in enumerate(solution.condensed.values()):
            value: Array = (
                collection.stability.value - collection.tauc.value + collection.abundance.value
            )
            auxiliary_residual = auxiliary_residual.at[index].set(value)

        # TODO: Have to do this here because above modifies the residual and therefore relies on
        # the same array length. This should be cleaned up.
        pressure: Array = solution.atmosphere.pressure()
        pressure_residual: Array = jnp.zeros(
            len(constraints.total_pressure_constraint), dtype=jnp.float_
        )
        if len(constraints.total_pressure_constraint) == 1:
            constraint = constraints.total_pressure_constraint[0]
            value: Array = jnp.log10(
                solution.atmosphere.number_density()
            ) - constraint.get_log10_value(temperature=self.temperature(), pressure=pressure)
            pressure_residual = pressure_residual.at[0].set(value)

        return jnp.concatenate((residual, auxiliary_residual, pressure_residual))


class ReactionNetworkWithMassBalance(ResidualProtocol):
    """A reaction network with condensate stability and mass balance

    Args:
        species: Species
        planet: Planet
    """

    def __init__(self, species: Species, planet: Planet):
        logger.info("Creating a reaction network with mass balance")
        self._species: Species = species
        self._planet: Planet = planet
        self._reaction_network = ReactionNetworkWithCondensateStability(species, planet)

    def get_residual(self, solution: Solution, constraints: SystemConstraints) -> Array:
        """Gets the residual

        Args:
            solution: Solution
            constraints: Constraints for the system of equations

        Returns:
            Residual array
        """
        reaction_residual: jnp.ndarray = self._reaction_network.get_residual(solution, constraints)

        number_residual: Array = jnp.zeros(len(constraints.mass_constraints), dtype=jnp.float_)
        for index, constraint in enumerate(constraints.mass_constraints):
            value: Array = (
                solution.log10_number_density(element=constraint.element)
                - constraint.log10_number_of_molecules
                + solution.atmosphere.log10_volume()
            )
            number_residual = number_residual.at[index].set(value)

        # Concatenate all residuals
        residual: Array = jnp.concatenate(
            (
                reaction_residual,
                number_residual,
            )
        )
        logger.debug("residual = %s", residual)

        return residual


def partial_rref(matrix: Array) -> Array:
    """Computes the partial reduced row echelon form to determine linear components.

    Args:
        matrix: The matrix to compute the reduced row echelon form.

    Returns:
        A matrix of linear components.
    """
    nrows, ncols = matrix.shape

    # Augment the matrix with the identity matrix
    augmented_matrix: Array = jnp.hstack((matrix, jnp.eye(nrows)))
    logger.debug("augmented_matrix = \n%s", augmented_matrix)

    def swap_rows(matrix: Array, row1: ArrayLike, row2: ArrayLike) -> Array:
        """Swaps two rows in the matrix."""
        # Extract rows to swap
        row1_data: Array = lax.dynamic_slice(matrix, (row1, 0), (1, matrix.shape[1]))
        row2_data: Array = lax.dynamic_slice(matrix, (row2, 0), (1, matrix.shape[1]))
        # Update matrix
        matrix = lax.dynamic_update_slice(matrix, row2_data, (row1, 0))
        matrix = lax.dynamic_update_slice(matrix, row1_data, (row2, 0))

        return matrix

    # Forward elimination
    def forward_step(i: int, matrix: Array) -> Array:
        # Check if the pivot element is zero and swap rows to get a non-zero pivot element.
        pivot_value: Array = lax.dynamic_slice(matrix, (i, i), (1, 1))
        if pivot_value == 0:
            nonzero_rows: Array = jnp.nonzero(matrix[i:, i])[0]
            if nonzero_rows.size > 0:
                nonzero_row: Array = nonzero_rows[0] + i
                matrix = swap_rows(matrix, i, nonzero_row)

        # Perform row operations to eliminate values below the pivot.
        for j in range(i + 1, nrows):
            pivot: Array = lax.dynamic_slice(matrix, (i, i), (1, 1))[0, 0]
            ratio: Array = lax.dynamic_slice(matrix, (j, i), (1, 1))[0, 0] / pivot
            row_i: Array = lax.dynamic_slice(matrix, (i, 0), (1, ncols + nrows))
            row_j: Array = lax.dynamic_slice(matrix, (j, 0), (1, ncols + nrows))
            matrix = lax.dynamic_update_slice(matrix, row_j - ratio * row_i, (j, 0))

        return matrix

    for i in range(ncols):
        augmented_matrix = forward_step(i, augmented_matrix)

    logger.debug("augmented_matrix after forward step = \n%s", augmented_matrix)

    # Backward substitution
    def backward_step(i: int, matrix: Array):
        # Normalize the pivot row.
        pivot: Array = lax.dynamic_slice(matrix, (i, i), (1, 1))[0, 0]
        normalized_row = lax.dynamic_slice(matrix, (i, 0), (1, ncols + nrows)) / pivot
        matrix = lax.dynamic_update_slice(matrix, normalized_row, (i, 0))
        # Eliminate values above the pivot.
        for j in range(i - 1, -1, -1):
            if lax.dynamic_slice(matrix, (j, i), (1, 1))[0, 0] != 0:
                # Scaled pivot row
                pivot = lax.dynamic_slice(matrix, (i, i), (1, 1))[0, 0]
                ratio: Array = lax.dynamic_slice(matrix, (j, i), (1, 1))[0, 0] / pivot
                row_i: Array = lax.dynamic_slice(matrix, (i, 0), (1, ncols + nrows))
                row_j: Array = lax.dynamic_slice(matrix, (j, 0), (1, ncols + nrows))
                matrix = lax.dynamic_update_slice(matrix, row_j - ratio * row_i, (j, 0))

        return matrix

    for i in range(ncols - 1, -1, -1):
        augmented_matrix = backward_step(i, augmented_matrix)

    logger.debug("augmented_matrix after backward step = \n%s", augmented_matrix)

    reduced_matrix = lax.dynamic_slice(augmented_matrix, (0, 0), (nrows, ncols))
    component_matrix = lax.dynamic_slice(augmented_matrix, (ncols, ncols), (nrows - ncols, nrows))
    logger.debug("reduced_matrix = \n%s", reduced_matrix)
    logger.debug("component_matrix = \n%s", component_matrix)

    return component_matrix
