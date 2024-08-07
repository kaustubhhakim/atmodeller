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

import copy
import logging
import pprint
import sys

import jax
import jax.numpy as jnp

from atmodeller import BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.core import Species
from atmodeller.solution import Solution

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class ReactionNetwork:
    """Determines the reactions to solve a chemical network.

    Args:
        species: Species
    """

    def __init__(self, species: Species):
        self._species: Species = species
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self._species.names)
        self.reaction_matrix: jnp.ndarray = self.get_reaction_matrix()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        """Number of reactions"""
        if self._species.number == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def formula_matrix(self) -> jnp.ndarray:
        """Gets the formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            The formula matrix
        """
        matrix: jnp.ndarray = jnp.zeros((len(self._species.elements()), len(self._species.data)))
        for element_index, element in enumerate(self._species.elements()):
            for species_index, species in enumerate(self._species.data):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                # Update the matrix using JAX's indexed update
                matrix = matrix.at[element_index, species_index].set(count)

        return matrix

    def get_reaction_matrix(self) -> jnp.ndarray:
        """Gets the reaction matrix

        Returns:
            A matrix of linearly independent reactions or None
        """
        # if self._species.number == 1:
        #    logger.debug("Only one species therefore no reactions")
        #    return None

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
                    coeff: float = self.reaction_matrix[reaction_index, species_index]
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

    def _get_lnKp(self, reaction_index: int, *, temperature: float, pressure: float) -> float:
        """Gets the natural log of the equilibrium constant in terms of partial pressures.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Natural log of the equilibrium constant in terms of partial pressures
        """
        gibbs_energy: float = self._get_reaction_gibbs_energy_of_formation(
            reaction_index, temperature=temperature, pressure=pressure
        )
        lnKp: float = -gibbs_energy / (GAS_CONSTANT * temperature)

        return lnKp

    def _get_log10Kp(self, reaction_index: int, *, temperature: float, pressure: float) -> float:
        """Gets the log10 of the equilibrium constant in terms of partial pressures.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            log10 of the equilibrium constant in terms of partial pressures
        """
        lnKp: float = self._get_lnKp(reaction_index, temperature=temperature, pressure=pressure)
        log10Kp: float = lnKp / jnp.log(10)

        return log10Kp

    def _get_delta_n(self, reaction_index: int) -> float:
        """Gets the difference in the moles of gas products compared to the moles of gas reactants.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`

        Returns:
            The difference in the moles of gas products compared to the moles of gas reactants
        """
        assert self.reaction_matrix is not None

        delta_n: float = 0
        for gas_index in self._species.gas_species():
            delta_n += self.reaction_matrix[reaction_index, gas_index]

        return delta_n

    def _get_lnKc(self, reaction_index: int, *, temperature: float, pressure: float) -> float:
        """Gets the natural log of the equilibrium constant in terms of number densities.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Natural log of the equilibrium constant in terms of number densities
        """
        lnKp: float = self._get_lnKp(reaction_index, temperature=temperature, pressure=pressure)
        delta_n: float = self._get_delta_n(reaction_index)
        lnKc: float = lnKp - delta_n * (jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature))

        return lnKc

    def _get_log10Kc(self, reaction_index: int, *, temperature: float, pressure: float) -> float:
        """Gets the log10 of the equilibrium constant in terms of number densities.

        Args:
            reaction_index: Row index of the reaction as it appears in :attr:`reaction_matrix`
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            log10 of the equilibrium constant in terms of number densities
        """
        lnKc: float = self._get_lnKc(reaction_index, temperature=temperature, pressure=pressure)
        log10Kc: float = lnKc / jnp.log(10)

        return log10Kc

    def _get_reaction_gibbs_energy_of_formation(
        self, reaction_index: int, *, temperature: float, pressure: float
    ) -> float:
        r"""Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in :attr:`reaction_matrix`
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The Gibb's free energy of the reaction in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        gibbs_energy: float = 0
        assert self.reaction_matrix is not None
        for species_index, species in enumerate(self._species):
            assert species.thermodata is not None
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * species.thermodata.get_formation_gibbs(temperature=temperature, pressure=pressure)

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

    def _assemble_right_hand_side_values(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
    ) -> jnp.ndarray:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = self.number_reactions + constraints.number_reaction_network_constraints
        rhs: jnp.ndarray = jnp.zeros(nrows)

        # Reactions
        for reaction_index in range(self.number_reactions):
            # log10Kc: float = self._get_log10Kc(
            #    reaction_index,
            #    temperature=temperature,
            #    pressure=pressure,
            # )
            # FIXME: Hack to try jax
            log10Kc = 10.0
            rhs = rhs.at[reaction_index].set(log10Kc)

        # Constraints
        for index, constraint in enumerate(constraints.reaction_network_constraints):
            row_index: int = self.number_reactions + index
            # pylint: disable=line-too-long
            # logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs = rhs.at[row_index].set(
                constraint.get_log10_value(temperature=temperature, pressure=pressure)
            )

        logger.debug("RHS vector = %s", rhs)

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

    def get_residual_jax(self, x: jnp.ndarray) -> jnp.ndarray:

        solution = Solution.create_from_species(species=self._species)
        solution.value = x

        reaction_list: list = []
        for collection in solution.gas_solution.values():
            reaction_list.append(collection.gas_abundance.value)
        for collection in solution.condensed_solution.values():
            reaction_list.append(collection.activity.value)

        coefficient_matrix = self.get_coefficient_matrix(self.constraints)
        reaction_array = jnp.array(reaction_list)

        solution.planet = self.planet
        # pressure = solution.atmosphere.pressure()
        pressure = jnp.sum(10**x) * self.planet.surface_temperature * BOLTZMANN_CONSTANT_BAR
        logger.warning("pressure = %s", pressure)

        residual = self.get_residual(
            temperature=self.planet.surface_temperature,
            pressure=pressure,
            constraints=self.constraints,
            coefficient_matrix=coefficient_matrix,
            reaction_array=reaction_array,
        )

        logger.warning("residual = %s", residual)

        return residual

    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: jnp.ndarray,
        reaction_array: jnp.ndarray,
        **kwargs,
    ) -> jnp.ndarray:
        """The residual array of the reaction network

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations
            coefficient_matrix: Coefficient matrix
            reaction_array: Reaction array
            **kwargs: Catches additional keyword arguments used by child classes

        Returns:
            The residual vector of the reaction network
        """
        del kwargs
        # FIXME: Commented out for simplicity
        # log_fugacity_coefficients: jnp.ndarray = self._assemble_log_fugacity_coefficients(
        #    temperature=temperature, pressure=pressure
        # )
        rhs: jnp.ndarray = self._assemble_right_hand_side_values(
            temperature=temperature, pressure=pressure, constraints=constraints
        )
        logger.warning("rhs = %s", rhs)

        residual_reaction: jnp.ndarray = (
            # FIXME: Commented out for simplicity
            # coefficient_matrix.dot(log_fugacity_coefficients)
            # FIXME: Needed to add transpose for JAX
            coefficient_matrix.dot(reaction_array)
            # TODO: rhs seems to work (add *x to ensure the jacobian is non-zero)
            - rhs
        )

        return residual_reaction


# class ReactionNetworkWithCondensateStability(ReactionNetwork):
#     """A reaction network with condensate stability

#     This automatically determines condensate stability using the extended law of mass-action
#     (xLMA) equations :cite:p:`LKS17`. Also see :cite:p:`KSP24`.

#     Args:
#         species: Species

#     Attributes:
#         species: Species
#         reaction_matrix: The reaction stoichiometry matrix
#     """

#     def get_activity_modifier(self, constraints: SystemConstraints) -> npt.NDArray[np.float_]:
#         """Gets the activity modifier matrix for condensate stability

#         Args:
#             constraints: Constraints

#         Returns:
#             Activity modifier matrix
#         """
#         coefficient_matrix: npt.NDArray[np.float_] = self.get_coefficient_matrix(
#             constraints=constraints
#         )
#         activity_modifier: npt.NDArray[np.float_] = np.zeros_like(coefficient_matrix)
#         for condensed_index in self._species.condensed_species():
#             activity_modifier[:, condensed_index] = coefficient_matrix[:, condensed_index]

#         logger.debug("activity_modifier = %s", activity_modifier)

#         return activity_modifier

#     def get_equilibrium_modifier(self, constraints: SystemConstraints) -> npt.NDArray[np.float_]:
#         """Gets the equilibrium constant modifier matrix for condensate stability

#         Args:
#             constraints: Constraints

#         Returns:
#             Equilibrium constant modifier matrix
#         """
#         activity_modifier: npt.NDArray[np.float_] = self.get_activity_modifier(
#             constraints=constraints
#         )
#         equilibrium_modifier: npt.NDArray[np.float_] = copy.deepcopy(activity_modifier)
#         equilibrium_modifier[self.number_reactions :, :] = 0

#         logger.debug("equilibrium_modifier = %s", equilibrium_modifier)

#         return equilibrium_modifier

#     @override
#     def get_residual(
#         self,
#         *,
#         temperature: float,
#         pressure: float,
#         constraints: SystemConstraints,
#         coefficient_matrix: npt.NDArray[np.float_],
#         activity_modifier: npt.NDArray[np.float_],
#         equilibrium_modifier: npt.NDArray[np.float_],
#         reaction_array: npt.NDArray[np.float_],
#         stability_array: npt.NDArray[np.float_],
#     ) -> npt.NDArray[np.float_]:

#         residual_reaction: npt.NDArray[np.float_] = super().get_residual(
#             temperature=temperature,
#             pressure=pressure,
#             constraints=constraints,
#             coefficient_matrix=coefficient_matrix,
#             reaction_array=reaction_array,
#         )

#         residual_reaction += activity_modifier.dot(10**stability_array)
#         residual_reaction -= equilibrium_modifier.dot(10**stability_array)

#         return residual_reaction


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
            nonzero_row = jnp.nonzero(augmented_matrix[i:, i])[0][0] + i
            augmented_matrix = jax.numpy.index_update(
                augmented_matrix,
                jax.numpy.ix_([i, nonzero_row], slice(None)),
                augmented_matrix[[nonzero_row, i], :],
            )
        # Perform row operations to eliminate values below the pivot.
        for j in range(i + 1, nrows):
            ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix = augmented_matrix.at[j].set(
                augmented_matrix[j] - ratio * augmented_matrix[i]
            )

    logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

    # Backward substitution
    for i in range(ncols - 1, -1, -1):
        # Normalize the pivot row.
        augmented_matrix = augmented_matrix.at[i].set(augmented_matrix[i] / augmented_matrix[i, i])
        # Eliminate values above the pivot.
        for j in range(i - 1, -1, -1):
            if augmented_matrix[j, i] != 0:
                ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix = augmented_matrix.at[j].set(
                    augmented_matrix[j] - ratio * augmented_matrix[i]
                )

    logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

    # Extract the reduced matrix and component matrix
    reduced_matrix = augmented_matrix[:, :ncols]
    component_matrix = augmented_matrix[ncols:, ncols:]
    logger.debug("reduced_matrix = \n%s", reduced_matrix)
    logger.debug("component_matrix = \n%s", component_matrix)

    return component_matrix
