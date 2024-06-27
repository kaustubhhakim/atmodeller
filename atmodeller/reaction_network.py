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

from __future__ import annotations

import copy
import logging
import pprint
import sys

import numpy as np
import numpy.typing as npt

from atmodeller import GAS_CONSTANT
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

    Attributes:
        species: Species
        reaction_matrix: The reaction stoichiometry matrix
    """

    def __init__(self, species: Species):
        self.species: Species = species
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self.species.names)
        self.reaction_matrix: npt.NDArray | None = self.get_reaction_matrix()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        if self.species.number_species() == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def get_reaction_matrix(self) -> npt.NDArray | None:
        """Gets the reaction matrix

        Returns:
            A matrix of linearly independent reactions
        """
        if self.species.number_species() == 1:
            logger.debug("Only one species therefore no reactions")
            return None

        transpose_formula_matrix: npt.NDArray = self.species.formula_matrix(
            self.species.elements(), self.species.data
        ).T

        return partial_rref(transpose_formula_matrix)

    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary"""
        reactions: dict[int, str] = {}
        if self.reaction_matrix is not None:
            for reaction_index in range(self.number_reactions):
                reactants: str = ""
                products: str = ""
                for species_index, species in enumerate(self.species.data):
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

    def _get_reaction_log10_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the log10 of the reaction equilibrium constant.

        From the Gibbs free energy, we can calculate logKf as:
        logKf = - G/(ln(10)*R*T)

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature
            pressure: Pressure

        Returns:
            log10 of the reaction equilibrium constant
        """
        gibbs_energy: float = self._get_reaction_gibbs_energy_of_formation(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        equilibrium_constant: float = -gibbs_energy / (np.log(10) * GAS_CONSTANT * temperature)

        return equilibrium_constant

    def _get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`
            temperature: Temperature
            pressure: Pressure

        Returns:
            The Gibb's free energy of the reaction
        """
        gibbs_energy: float = 0
        assert self.reaction_matrix is not None
        for species_index, species in enumerate(self.species.data):
            assert species.thermodata is not None
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * species.thermodata.get_formation_gibbs(temperature=temperature, pressure=pressure)

        return gibbs_energy

    def get_coefficient_matrix(self, *, constraints: SystemConstraints) -> npt.NDArray:
        """Gets the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        coeff: npt.NDArray = np.zeros((nrows, self.species.number_species()))
        if self.reaction_matrix is not None:
            coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index = self.species.species_index(constraint.species)
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("species = %s", self.species.names)
        logger.debug("coefficient matrix = \n%s", coeff)

        return coeff

    def _assemble_right_hand_side_values(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
    ) -> npt.NDArray:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = self.number_reactions + constraints.number_reaction_network_constraints
        rhs: npt.NDArray = np.zeros(nrows, dtype=float)

        # Reactions
        for reaction_index in range(self.number_reactions):
            # logger.debug(
            #     "Row %02d: Reaction %d: %s",
            #     reaction_index,
            #     reaction_index,
            #     self.reactions()[reaction_index],
            # )
            rhs[reaction_index] = self._get_reaction_log10_equilibrium_constant(
                reaction_index=reaction_index,
                temperature=temperature,
                pressure=pressure,
            )

        # Constraints
        for index, constraint in enumerate(constraints.reaction_network_constraints):
            row_index: int = self.number_reactions + index
            # logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs[row_index] = constraint.get_log10_value(temperature=temperature, pressure=pressure)

        logger.debug("RHS vector = %s", rhs)

        return rhs

    def _assemble_log_fugacity_coefficients(
        self,
        *,
        temperature: float,
        pressure: float,
    ) -> npt.NDArray:
        """Assembles the fugacity coefficient vector on the left-hand side of the equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The log10(fugacity coefficient) vector
        """

        # Initialise to ideal behaviour.
        fugacity_coefficients: npt.NDArray = np.ones_like(self.species, dtype=float)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for gas_species in self.species.gas_species:
            fugacity_coefficient: float = gas_species.eos.fugacity_coefficient(
                temperature=temperature, pressure=pressure
            )
            index: int = self.species.species_index(gas_species)
            fugacity_coefficients[index] = fugacity_coefficient

        log_fugacity_coefficients: npt.NDArray = np.log10(fugacity_coefficients)
        logger.debug("Log10 fugacity coefficient vector = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray,
        solution: Solution,
        **kwargs,
    ) -> npt.NDArray:
        """Returns the residual vector of the reaction network.

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
        log_fugacity_coefficients: npt.NDArray = self._assemble_log_fugacity_coefficients(
            temperature=temperature, pressure=pressure
        )
        rhs: npt.NDArray = self._assemble_right_hand_side_values(
            temperature=temperature, pressure=pressure, constraints=constraints
        )
        residual_reaction: npt.NDArray = (
            coefficient_matrix.dot(log_fugacity_coefficients)
            + coefficient_matrix.dot(solution.species_values)
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

    def get_activity_modifier(
        self, *, constraints: SystemConstraints, solution: Solution
    ) -> npt.NDArray:
        """Gets the activity modifier matrix for condensate stability

        Args:
            constraints: Constraints
            solution: Solution

        Returns:
            Activity modifier matrix
        """
        coefficient_matrix: npt.NDArray = self.get_coefficient_matrix(constraints=constraints)
        activity_modifier: npt.NDArray = np.zeros_like(coefficient_matrix)
        for species in solution.condensed_species_to_solve:
            index: int = self.species.species_index(species)
            activity_modifier[:, index] = coefficient_matrix[:, index]

        logger.debug("activity_modifier = %s", activity_modifier)

        return activity_modifier

    def get_equilibrium_modifier(
        self, *, constraints: SystemConstraints, solution: Solution
    ) -> npt.NDArray:
        """Gets the equilibrium constant modifier matrix for condensate stability

        Args:
            constraints: Constraints
            solution: Solution

        Returns:
            Equilibrium constant modifier matrix
        """
        activity_modifier: npt.NDArray = self.get_activity_modifier(
            constraints=constraints, solution=solution
        )
        equilibrium_modifier: npt.NDArray = copy.deepcopy(activity_modifier)
        equilibrium_modifier[self.number_reactions :, :] = 0

        logger.debug("equilibrium_modifier = %s", equilibrium_modifier)

        return equilibrium_modifier

    def get_stability_residual(self, solution: Solution) -> npt.NDArray:
        """Returns the residual vector of condensate stability

        Args:
            solution: Solution to compute the residual for

        Returns:
            The residual vector of condensate stability
        """
        residual_stability: npt.NDArray = np.zeros(
            solution.number_condensed_species_to_solve, dtype=np.float_
        )
        for nn, species in enumerate(solution.condensed_species_to_solve):
            residual_stability[nn] = solution._stability_solution[species] - log10_TAU
            residual_stability[nn] += solution._mass_solution[species]

            # TODO: Old below, remove
            # The xLMA usually uses the condensate number density or similar, but it's simpler to
            # satisfy the auxiliary equations using the condensed mass of elements in the
            # condensate, which we have direct access to.
            # for element in species.elements:
            #    try:
            #        residual_stability += solution._beta_solution[element]
            #    except KeyError:
            #        pass

        logger.debug("residual_stability = %s", residual_stability)

        return residual_stability

    @override
    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray,
        solution: Solution,
        activity_modifier: npt.NDArray,
        equilibrium_modifier: npt.NDArray,
    ) -> npt.NDArray:

        # Residual of the reaction network without a stability consideration
        residual_reaction: npt.NDArray = super().get_residual(
            temperature=temperature,
            pressure=pressure,
            constraints=constraints,
            coefficient_matrix=coefficient_matrix,
            solution=solution,
        )

        # Reaction network correction factors for condensate stability
        residual_reaction += activity_modifier.dot(10**solution.stability_array)
        residual_reaction -= equilibrium_modifier.dot(10**solution.stability_array)

        # Residual for the auxiliary stability equations
        residual_stability: npt.NDArray = self.get_stability_residual(solution)

        residual: npt.NDArray = np.concatenate((residual_reaction, residual_stability))
        logger.debug("residual_reaction = %s", residual_reaction)

        return residual
