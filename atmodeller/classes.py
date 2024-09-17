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
from atmodeller.utilities import partial_rref

logger: logging.Logger = logging.getLogger(__name__)


class ReactionNetwork:
    """Assembles Python objects to generate JAX-compliant arrays for numerical solution."""

    @staticmethod
    def unique_elements_in_species(species: list[SpeciesData]) -> tuple[str, ...]:
        """Unique elements in a list of species

        Args:
            species: A list of species

        Returns:
            Unique elements in the list of species
        """
        elements: list[str] = []
        for species_ in species:
            elements.extend(species_.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        logger.debug("unique_elements_in_species = %s", sorted_elements)

        return tuple(sorted_elements)

    def formula_matrix(self, species: list[SpeciesData]) -> npt.NDArray:
        """Formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Args:
            species: A list of species

        Returns:
            The formula matrix
        """
        unique_elements: tuple[str, ...] = self.unique_elements_in_species(species)
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

        logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def reaction_matrix(self, species: list[SpeciesData]) -> npt.NDArray:
        """Reaction matrix

        Returns:
            A matrix of linearly independent reactions or None
        """
        # TODO: Would prefer to always return an array even in the absence of reactions?
        # if self._species.number == 1:
        #    logger.debug("Only one species therefore no reactions")
        #    return None

        transpose_formula_matrix: npt.NDArray = self.formula_matrix(species).T
        reaction_matrix: npt.NDArray = partial_rref(transpose_formula_matrix)
        logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix
