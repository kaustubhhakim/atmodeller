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

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from atmodeller.jax_containers import SpeciesData
from atmodeller.utilities import partial_rref, unique_elements_in_species

logger: logging.Logger = logging.getLogger(__name__)


class ReactionNetwork:
    """Reaction network

    This processing class takes a list of species and provides output related to reactions
    """

    def formula_matrix(self, species: list[SpeciesData]) -> npt.NDArray:
        """Formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Args:
            species: A list of species

        Returns:
            The formula matrix
        """
        unique_elements: tuple[str, ...] = unique_elements_in_species(species)
        formula_matrix: npt.NDArray = np.zeros(
            (len(unique_elements), len(species)), dtype=jnp.int_
        )

        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(species):
                count: int = 0
                try:
                    count = species_.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        # logger.debug("species = %s", species)
        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def reaction_matrix(self, species: list[SpeciesData]) -> npt.NDArray:
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

        transpose_formula_matrix: npt.NDArray = self.formula_matrix(species).T
        reaction_matrix: npt.NDArray = partial_rref(transpose_formula_matrix)

        # logger.debug("species = %s", species)
        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def reactions(self, species: list[SpeciesData]) -> dict[int, str]:
        """The reactions as a dictionary

        Args:
            species: A list of species

        Returns:
            Reactions as a dictionary
        """
        reaction_matrix: npt.NDArray = self.reaction_matrix(species)
        reactions: dict[int, str] = {}
        # TODO: Would like to avoid below if possible
        # if self.reaction_matrix is not None:
        for reaction_index in range(reaction_matrix.shape[0]):
            reactants: str = ""
            products: str = ""
            for species_index, species_ in enumerate(species):
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
