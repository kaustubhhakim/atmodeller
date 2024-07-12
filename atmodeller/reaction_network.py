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

# Convenient to use chemical formulas so pylint: disable=C0103

from __future__ import annotations

import copy
import logging
import pprint
import sys

import numpy as np
import numpy.typing as npt

from atmodeller import BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.core import Solution, Species
from atmodeller.utilities import partial_rref

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

TAU: float = 1e-15
"""Tau factor for the calculation of the auxilliary equations for condensate stability"""
log10_TAU: float = np.log10(TAU)
"""Log10 of Tau"""


class ReactionNetwork:
    """Determines the reactions to solve a chemical network.

    Args:
        species: Species
    """

    def __init__(self, species: Species):
        self._species: Species = species
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self._species.names)
        self.reaction_matrix: npt.NDArray | None = self.get_reaction_matrix()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        """Number of reactions"""
        if self._species.number == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def formula_matrix(self) -> npt.NDArray[np.int_]:
        """Gets the formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            The formula matrix
        """
        matrix: npt.NDArray[np.int_] = np.zeros(
            (len(self._species.elements()), len(self._species.data)), dtype=np.int_
        )
        for element_index, element in enumerate(self._species.elements()):
            for species_index, species in enumerate(self._species.data):
                try:
                    count: int = species.composition()[element].count
                except KeyError:
                    count = 0
                matrix[element_index, species_index] = count

        return matrix

    def get_reaction_matrix(self) -> npt.NDArray[np.float_] | None:
        """Gets the reaction matrix

        Returns:
            A matrix of linearly independent reactions
        """
        if self._species.number == 1:
            logger.debug("Only one species therefore no reactions")
            return None

        transpose_formula_matrix: npt.NDArray[np.int_] = self.formula_matrix().T

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
        log10Kp: float = lnKp / np.log(10)

        return log10Kp

    def _get_delta_n(self, reaction_index: int) -> float:
        """Gets the difference in the moles of products compared to the moles of reactants.

        Args:
            reaction_index: Row index of the reaction in :attr:`reaction_matrix`

        Returns:
            The difference in the moles of products compared to the moles of reactants
        """
        assert self.reaction_matrix is not None
        return np.sum(self.reaction_matrix[reaction_index, :])

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
        lnKc: float = lnKp - delta_n * (np.log(BOLTZMANN_CONSTANT_BAR) + np.log(temperature))

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
        log10Kc: float = lnKc / np.log(10)

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
        for species_index, species in enumerate(self._species.data):
            assert species.thermodata is not None
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * species.thermodata.get_formation_gibbs(temperature=temperature, pressure=pressure)

        return gibbs_energy

    def get_coefficient_matrix(self, constraints: SystemConstraints) -> npt.NDArray[np.float_]:
        """Gets the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """
        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        coeff: npt.NDArray[np.float_] = np.zeros((nrows, self._species.number))
        if self.reaction_matrix is not None:
            coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index: int = self._species.species_index(constraint.species)
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("species = %s", self._species.names)
        logger.debug("coefficient matrix = \n%s", coeff)

        return coeff

    def _assemble_right_hand_side_values(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
    ) -> npt.NDArray[np.float_]:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = self.number_reactions + constraints.number_reaction_network_constraints
        rhs: npt.NDArray[np.float_] = np.zeros(nrows, dtype=float)

        # Reactions
        for reaction_index in range(self.number_reactions):
            log10Kc: float = self._get_log10Kc(
                reaction_index,
                temperature=temperature,
                pressure=pressure,
            )
            rhs[reaction_index] = log10Kc

        # Constraints
        for index, constraint in enumerate(constraints.gas_constraints):
            row_index: int = self.number_reactions + index
            # pylint: disable=line-too-long
            # logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs[row_index] = constraint.log10_number_density(
                temperature=temperature, pressure=pressure
            )

        for index, constraint in enumerate(constraints.activity_constraints):
            row_index: int = self.number_reactions + index
            # pylint: disable=line-too-long
            # logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs[row_index] = constraint.get_log10_value(temperature=temperature, pressure=pressure)

        logger.debug("RHS vector = %s", rhs)

        return rhs

    def _assemble_log_fugacity_coefficients(
        self,
        *,
        temperature: float,
        pressure: float,
    ) -> npt.NDArray[np.float_]:
        """Assembles the fugacity coefficient vector on the left-hand side of the equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The log10(fugacity coefficient) vector
        """
        # Initialise to ideal behaviour.
        fugacity_coefficients: npt.NDArray[np.float_] = np.ones_like(self._species, dtype=float)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for gas_species in self._species.gas_species:
            fugacity_coefficient: float = gas_species.eos.fugacity_coefficient(
                temperature=temperature, pressure=pressure
            )
            index: int = self._species.species_index(gas_species)
            fugacity_coefficients[index] = fugacity_coefficient

        log_fugacity_coefficients: npt.NDArray[np.float_] = np.log10(fugacity_coefficients)
        logger.debug("Log10 fugacity coefficient vector = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray[np.float_],
        solution: Solution,
        **kwargs,
    ) -> npt.NDArray[np.float_]:
        """The residual array of the reaction network

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations
            coefficient_matrix: Coefficient matrix
            solution: Solution to compute the residual for
            **kwargs: Catches additional keyword arguments used by child classes

        Returns:
            The residual vector of the reaction network
        """
        del kwargs
        log_fugacity_coefficients: npt.NDArray[np.float_] = (
            self._assemble_log_fugacity_coefficients(temperature=temperature, pressure=pressure)
        )
        rhs: npt.NDArray[np.float_] = self._assemble_right_hand_side_values(
            temperature=temperature, pressure=pressure, constraints=constraints
        )
        residual_reaction: npt.NDArray[np.float_] = (
            coefficient_matrix.dot(log_fugacity_coefficients)
            + coefficient_matrix.dot(solution.data[: self._species.number])
            - rhs
        )

        logger.debug("residual_reaction = %s", residual_reaction)

        return residual_reaction


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

    def get_activity_modifier(self, constraints: SystemConstraints) -> npt.NDArray[np.float_]:
        """Gets the activity modifier matrix for condensate stability

        Args:
            constraints: Constraints

        Returns:
            Activity modifier matrix
        """
        coefficient_matrix: npt.NDArray[np.float_] = self.get_coefficient_matrix(
            constraints=constraints
        )
        activity_modifier: npt.NDArray[np.float_] = np.zeros_like(coefficient_matrix)
        for species in self._species.condensed_species:
            index: int = self._species.species_index(species)
            activity_modifier[:, index] = coefficient_matrix[:, index]

        logger.debug("activity_modifier = %s", activity_modifier)

        return activity_modifier

    def get_equilibrium_modifier(self, constraints: SystemConstraints) -> npt.NDArray[np.float_]:
        """Gets the equilibrium constant modifier matrix for condensate stability

        Args:
            constraints: Constraints

        Returns:
            Equilibrium constant modifier matrix
        """
        activity_modifier: npt.NDArray[np.float_] = self.get_activity_modifier(
            constraints=constraints
        )
        equilibrium_modifier: npt.NDArray[np.float_] = copy.deepcopy(activity_modifier)
        equilibrium_modifier[self.number_reactions :, :] = 0

        logger.debug("equilibrium_modifier = %s", equilibrium_modifier)

        return equilibrium_modifier

    def get_stability_residual(self, solution: Solution) -> npt.NDArray[np.float_]:
        """Returns the residual vector of condensate stability

        Args:
            solution: Solution to compute the residual for

        Returns:
            The residual vector of condensate stability
        """
        residual_stability: npt.NDArray[np.float_] = np.zeros(
            self._species.number_condensed_species, dtype=np.float_
        )
        for nn, species in enumerate(self._species.condensed_species):
            residual_stability[nn] = solution.stability.data[species] - log10_TAU
            residual_stability[nn] += solution.mass.data[species]

        logger.debug("residual_stability = %s", residual_stability)

        return residual_stability

    @override
    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray[np.float_],
        solution: Solution,
        activity_modifier: npt.NDArray[np.float_],
        equilibrium_modifier: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        # Residual of the reaction network without a stability consideration
        residual_reaction: npt.NDArray[np.float_] = super().get_residual(
            temperature=temperature,
            pressure=pressure,
            constraints=constraints,
            coefficient_matrix=coefficient_matrix,
            solution=solution,
        )

        # Reaction network correction factors for condensate stability
        residual_reaction += activity_modifier.dot(10 ** solution.stability_array())
        residual_reaction -= equilibrium_modifier.dot(10 ** solution.stability_array())

        # Residual for the auxiliary stability equations
        residual_stability: npt.NDArray[np.float_] = self.get_stability_residual(solution)

        residual: npt.NDArray[np.float_] = np.concatenate((residual_reaction, residual_stability))
        logger.debug("residual_reaction = %s", residual_reaction)

        return residual
