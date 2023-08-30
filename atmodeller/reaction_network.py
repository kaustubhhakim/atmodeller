"""Reaction network.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
import pprint
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from atmodeller import GAS_CONSTANT

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.constraints import SystemConstraints
    from atmodeller.core import InteriorAtmosphereSystem, Species
    from atmodeller.thermodynamics import StandardGibbsFreeEnergyOfFormationProtocol


@dataclass(kw_only=True)
class ReactionNetwork:
    """Determines the necessary reactions to solve a chemical network.

    Args:
        species: A list of species.
        gibbs_data: Standard Gibbs free energy of formation.

    Attributes:
        species: A list of species.
        gibbs_data: Standard Gibbs free energy of formation.
        species_matrix: The stoichiometry matrix of the species in terms of elements.
        reaction_matrix: The reaction stoichiometry matrix.
    """

    species: Species
    gibbs_data: StandardGibbsFreeEnergyOfFormationProtocol

    def __post_init__(self):
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self.species.chemical_formulas)
        self.species_matrix: np.ndarray = self.find_matrix()
        self.reaction_matrix: np.ndarray = self.partial_gaussian_elimination()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions))

    @cached_property
    def number_reactions(self) -> int:
        return self.species.number - self.number_unique_elements

    @cached_property
    def number_unique_elements(self) -> int:
        return len(self.unique_elements)

    @cached_property
    def unique_elements(self) -> list[str]:
        elements: list[str] = []
        for species in self.species:
            elements.extend(list(species.formula.composition().keys()))
        unique_elements: list[str] = list(set(elements))
        return unique_elements

    def find_matrix(self) -> np.ndarray:
        """Creates a matrix where species (rows) are split into their element counts (columns).

        Returns:
            For example, self.species = ['CO2', 'H2O'] would return:
                [[0, 1, 2],
                 [2, 0, 1]]
            if the columns represent the elements H, C, and O, respectively.
        """
        matrix: np.ndarray = np.zeros((self.species.number, self.number_unique_elements))
        for species_index, species in enumerate(self.species):
            for element_index, element in enumerate(self.unique_elements):
                try:
                    count: int = species.formula.composition()[element].count
                except KeyError:
                    count = 0
                matrix[species_index, element_index] = count
        return matrix

    def partial_gaussian_elimination(self) -> np.ndarray:
        """Performs a partial gaussian elimination to determine the required reactions.

        A copy of `self.species_matrix` is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of species - number of
        elements. These reactions are given in the last r rows of the reduced matrix.

        Returns:
            A matrix of the reaction stoichiometry.
        """
        matrix1: np.ndarray = self.species_matrix
        matrix2: np.ndarray = np.eye(self.species.number)
        augmented_matrix: np.ndarray = np.hstack((matrix1, matrix2))
        logger.debug("augmented_matrix = \n%s", augmented_matrix)

        # Forward elimination.
        for i in range(self.number_unique_elements):  # Note only over the number of elements.
            # Check if the pivot element is zero.
            if augmented_matrix[i, i] == 0:
                # Swap rows to get a non-zero pivot element.
                nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
                augmented_matrix[[i, nonzero_row]] = augmented_matrix[[nonzero_row, i]]
            # Perform row operations to eliminate values below the pivot.
            for j in range(i + 1, self.species.number):
                ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("Augmented_matrix after forward elimination = \n%s", augmented_matrix)

        # Backward substitution.
        for i in range(self.number_unique_elements - 1, -1, -1):
            # Normalize the pivot row.
            augmented_matrix[i] /= augmented_matrix[i, i]
            # Eliminate values above the pivot.
            for j in range(i - 1, -1, -1):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("Augmented_matrix after backward substitution = \n%s", augmented_matrix)

        reduced_matrix1: np.ndarray = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix: np.ndarray = augmented_matrix[
            self.number_unique_elements :, matrix1.shape[1] :
        ]
        logger.debug("Reduced_matrix1 = \n%s", reduced_matrix1)
        logger.debug("Reaction_matrix = \n%s", reaction_matrix)

        return reaction_matrix

    @cached_property
    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary."""
        reactions: dict[int, str] = {}
        for reaction_index in range(self.number_reactions):
            reactants: str = ""
            products: str = ""
            for species_index, species in enumerate(self.species):
                coeff: float = self.reaction_matrix[reaction_index, species_index]
                if coeff != 0:
                    if coeff < 0:
                        reactants += f"{abs(coeff)} {species.chemical_formula} + "
                    else:
                        products += f"{coeff} {species.chemical_formula} + "

            reactants = reactants.rstrip(" + ")  # Removes the extra + at the end.
            products = products.rstrip(" + ")  # Removes the extra + at the end.
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

        return reactions

    def get_reaction_log10_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the log10 of the reaction equilibrium constant.

        From the Gibbs free energy, we can calculate logKf as:
        logKf = - G/(ln(10)*R*T)

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            log10 of the reaction equilibrium constant.
        """
        gibbs_energy: float = self.get_reaction_gibbs_energy_of_formation(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        equilibrium_constant: float = -gibbs_energy / (np.log(10) * GAS_CONSTANT * temperature)

        return equilibrium_constant

    def get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            The Gibb's free energy of the reaction.
        """
        gibbs_energy: float = 0
        for species_index, species in enumerate(self.species):
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * self.gibbs_data.get(species, temperature=temperature, pressure=pressure)
        return gibbs_energy

    def get_reaction_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the equilibrium constant of a reaction Kf

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            The equilibrium constant of the reaction.
        """
        equilibrium_constant: float = 10 ** self.get_reaction_log10_equilibrium_constant(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        return equilibrium_constant

    def get_coefficient_matrix(self, *, constraints: SystemConstraints) -> np.ndarray:
        """Builds the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations.

        Returns:
            The coefficient matrix with the stoichiometry and constraints.
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        if nrows == self.species.number:
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.info(msg)
        else:
            num: int = self.species.number - nrows
            # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
            msg = "%d additional (mass) constraint(s) are necessary " % num
            msg += "to solve the system"
            logger.info(msg)

        coeff: np.ndarray = np.zeros((nrows, self.species.number))
        coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.info("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index: int = self.species.indices[constraint.species]
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("Species = %s", self.species.chemical_formulas)
        logger.debug("Coefficient matrix = \n%s", coeff)

        return coeff

    def get_lhs_and_rhs_vectors(
        self,
        *,
        system: InteriorAtmosphereSystem,
        constraints: SystemConstraints,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the LHS non-ideal vector and the RHS vector.

        Args:
            system: Interior atmosphere system.
            constraints: Constraints for the system of equations.

        Returns:
            The log10(fugacity coefficient) vector and the right-hand side vector.
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        if nrows == self.species.number:
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.info(msg)
        else:
            num: int = self.species.number - nrows
            # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
            msg = "%d additional (mass) constraint(s) are necessary " % num
            msg += "to solve the system"
            logger.info(msg)

        rhs: np.ndarray = np.zeros(nrows, dtype=float)
        # Initialise to ideal behaviour.
        non_ideal: np.ndarray = np.ones_like(self.species, dtype=float)

        # Reactions.
        for reaction_index in range(self.number_reactions):
            logger.info(
                "Row %02d: Reaction %d: %s",
                reaction_index,
                reaction_index,
                self.reactions[reaction_index],
            )
            rhs[reaction_index] = self.get_reaction_log10_equilibrium_constant(
                reaction_index=reaction_index,
                temperature=system.planet.surface_temperature,
                pressure=system.total_pressure,
            )

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            row_index: int = self.number_reactions + index
            logger.info("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs[row_index] = np.log10(
                constraint.get_value(
                    temperature=system.planet.surface_temperature, pressure=system.total_pressure
                )
            )
            if constraint.name == "pressure":
                rhs[row_index] += np.log10(system.fugacity_coefficients_dict[constraint.species])

        # FIXME: Should be unity for solids and calculated for gases.
        for solid in self.species.solid_species.values():
            value: float = system.fugacity_coefficients_dict[solid.chemical_formula]
            non_ideal[self.species.indices[solid.chemical_formula]] = value
        non_ideal = np.log10(non_ideal)

        logger.debug("Non-ideal vector = \n%s", non_ideal)
        logger.debug("RHS vector = \n%s", rhs)

        return rhs, non_ideal

    def get_residual(
        self,
        *,
        system: InteriorAtmosphereSystem,
        constraints: SystemConstraints,
        coefficient_matrix: np.ndarray,
    ) -> np.ndarray:
        """Returns the residual vector of the reaction network.

        Args:
            system: Interior atmosphere system.
            constraints: Constraints for the system of equations.
            coefficient_matrix: Coefficient matrix.

        Returns:
            The residual vector of the reaction network.
        """

        rhs, non_ideal = self.get_lhs_and_rhs_vectors(system=system, constraints=constraints)

        residual_reaction: np.ndarray = (
            coefficient_matrix.dot(non_ideal) + coefficient_matrix.dot(system.log_solution) - rhs
        )
        logger.debug("Residual_reaction = %s", residual_reaction)
        return residual_reaction
