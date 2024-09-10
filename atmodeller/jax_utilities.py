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
"""JAX functions"""

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import Array, jit, lax, vmap
from jax.tree_util import tree_map
from jax.typing import ArrayLike

from atmodeller.jax_containers import SpeciesData


@jit
def logsumexp_base10(log_values: Array, prefactors: ArrayLike = 1.0) -> Array:
    """Computes the log-sum-exp using base-10 exponentials in a numerically stable way.

    Args:
        log10_values: Array of log10 values to sum
        prefactors: Array of prefactors corresponding to each log10 value

    Returns:
        The log10 of the sum of prefactors multiplied by exponentials of the input values.
    """
    max_log: Array = jnp.max(log_values)
    value_sum: Array = jnp.sum(prefactors * jnp.power(10, log_values - max_log))

    return max_log + jnp.log10(value_sum)


# Previously
# @jit
# def dimensional_to_scaled_base10(dimensional_number_density: Array):
#     return dimensional_number_density - log10_AVOGADRO + log10_scaling


@jit
def scale_number_density(number_density: ArrayLike, scaling: ArrayLike) -> ArrayLike:
    """Scales the log10 number density

    This is in log10 space.

    Args:
        number_density: Number density in molecules per m^3
        scaling: Scaling

    Return:
        Scaled number density
    """
    return number_density - scaling


# Previously
# @jit
# def scaled_to_dimensional_base10(scaled_number_density: Array):
#     return scaled_number_density - dimensional_to_scaled_base10(0)


@jit
def unscale_number_density(number_density: ArrayLike, scaling: ArrayLike) -> ArrayLike:
    """Unscales the scaled log10 number density

    This is in log10 space.

    Args:
        number_density: Scaled number density
        scaling: Scaling

    Returns:
        Unscaled number density
    """
    return number_density + scaling


def pytrees_stack(pytrees, axis=0):
    """Stacks an iterable of pytrees along a specified axis."""
    results = tree_map(lambda *values: jnp.stack(values, axis=axis), *pytrees)
    return results


def pytrees_vmap(fn):
    """Vectorizes a function over a batch of pytrees."""

    def g(pytrees):
        stacked = pytrees_stack(pytrees)
        results = vmap(fn)(stacked)
        return results

    return g


class ReactionNetworkJAX:
    """Primary role is to assemble Python objects to generate JAX-compliants arrays for numerical
    solution. In this regard, this should only generate arrays that are required once, and not
    computed dynamically during the solve.

    Ironically, since the species list will be treated as a static argument, this will all use
    numpy since nothing here is traced.
    """

    @staticmethod
    def unique_elements_in_species(species: list[SpeciesData]) -> tuple[int, ...]:
        """Unique elements in a list of species

        Args:
            species: A list of species

        Returns:
            Unique elements in the list of species
        """
        elements: list[int] = []
        for species_ in species:
            elements.extend(species_.elements)
        unique_elements: list[int] = list(set(elements))
        sorted_elements: list[int] = sorted(unique_elements)

        return tuple(sorted_elements)

    def formula_matrix(self, species: list[SpeciesData]) -> npt.NDArray:
        """Formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            The formula matrix
        """
        unique_elements: tuple[int, ...] = self.unique_elements_in_species(species)

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

        return formula_matrix

    def reaction_matrix(self, species: list[SpeciesData]) -> Array:
        """Reaction matrix

        Returns:
            A matrix of linearly independent reactions or None
        """
        # TODO: Would prefer to always return an array
        # if self._species.number == 1:
        #    logger.debug("Only one species therefore no reactions")
        #    return None

        transpose_formula_matrix: npt.NDArray = self.formula_matrix(species).T

        return jnp.array(partial_rref(transpose_formula_matrix))


@jit
def partial_rref(matrix: Array) -> Array:
    """Computes the partial reduced row echelon form to determine linear components.

    Args:
        matrix: The matrix to compute the reduced row echelon form

    Returns:
        A matrix of linear components.
    """
    nrows, ncols = matrix.shape
    augmented_matrix: Array = jnp.hstack((matrix, jnp.eye(nrows)))

    def swap_rows(matrix: Array, row1: int, row2: int) -> Array:
        """Swaps two rows in a matrix.

        Args:
            matrix: Matrix
            row1: Row1 index
            row2: Row2 index

        Returns:
            Matrix with the rows swapped
        """
        row1_data: Array = lax.dynamic_slice(matrix, (row1, 0), (1, matrix.shape[1]))
        row2_data: Array = lax.dynamic_slice(matrix, (row2, 0), (1, matrix.shape[1]))
        matrix = lax.dynamic_update_slice(matrix, row2_data, (row1, 0))
        matrix = lax.dynamic_update_slice(matrix, row1_data, (row2, 0))
        return matrix

    def find_nonzero_row(matrix: Array, i: int) -> int:
        """Finds the first non-zero element in the column below the pivot.

        Args:
            matrix: Matrix
            i: Row index

        Returns:
            Relative row index of the first non-zero element below row i
        """

        def body_fun(j: int, nonzero_row: int) -> int:
            """Body function

            Args:
                j: Row offset from row i
                nonzero_row: Non-zero row index found

            Returns:
                The minimum row offset with a non-zero element
            """
            value: Array = lax.dynamic_slice(matrix, (i + j, i), (1, 1))[0, 0]
            # nonzero_row == -1 indicates that no non-zero element has been found yet
            return lax.cond(
                (value != 0) & (nonzero_row == -1),
                lambda _: j,
                lambda _: nonzero_row,
                operand=None,
            )

        nonzero_row: Array = lax.fori_loop(0, nrows - i, body_fun, -1)
        return lax.cond(nonzero_row == -1, lambda _: i, lambda _: nonzero_row + i, operand=None)

    def forward_step(i: int, matrix: Array) -> Array:
        """Forward step

        Args:
            i: Current row
            matrix: Matrix

        Returns:
            Matrix
        """
        # Check if the pivot element is zero and swap rows to get a non-zero pivot element.
        pivot_value: Array = lax.dynamic_slice(matrix, (i, i), (1, 1))[0, 0]
        # jax.debug.print("pivot_value = {out}", out=pivot_value)
        nonzero_row: int = lax.cond(
            pivot_value == 0, lambda _: find_nonzero_row(matrix, i), lambda _: i, operand=None
        )
        matrix = lax.cond(
            nonzero_row != i,
            lambda _: swap_rows(matrix, i, nonzero_row),
            lambda _: matrix,
            operand=None,
        )

        def eliminate_below_row(j: int, matrix: Array) -> Array:
            """Eliminates below the row

            Args:
                j: Row offset from row i
                matrix: Matrix

            Returns:
                Matrix
            """
            pivot: Array = lax.dynamic_slice(matrix, (i, i), (1, 1))[0, 0]
            ratio: Array = lax.dynamic_slice(matrix, (j, i), (1, 1))[0, 0] / pivot
            row_i: Array = lax.dynamic_slice(matrix, (i, 0), (1, ncols + nrows))
            row_j: Array = lax.dynamic_slice(matrix, (j, 0), (1, ncols + nrows))
            return lax.dynamic_update_slice(matrix, row_j - ratio * row_i, (j, 0))

        def loop_body(j: int, matrix: Array) -> Array:
            return eliminate_below_row(j, matrix)

        matrix = lax.fori_loop(i + 1, nrows, loop_body, matrix)

        return matrix

    def backward_step(i: int, matrix: Array) -> Array:
        """Backward step

        Args:
            i: Current row
            matrix: Matrix

        Returns:
            Matrix
        """
        # Normalize the pivot row.
        pivot: Array = lax.dynamic_slice(matrix, (i, i), (1, 1))[0, 0]
        normalized_row = lax.dynamic_slice(matrix, (i, 0), (1, ncols + nrows)) / pivot
        matrix = lax.dynamic_update_slice(matrix, normalized_row, (i, 0))

        def eliminate_above_row(j: int, matrix: Array) -> Array:
            """Eliminates above the row

            Args:
                j: Row offset from row i
                matrix: Matrix

            Returns:
                Matrix
            """
            is_nonzero: Array = lax.dynamic_slice(matrix, (j, i), (1, 1))[0, 0] != 0

            def eliminate_row(matrix: Array) -> Array:
                ratio: Array = lax.dynamic_slice(matrix, (j, i), (1, 1))[0, 0] / pivot
                row_i: Array = lax.dynamic_slice(matrix, (i, 0), (1, ncols + nrows))
                row_j: Array = lax.dynamic_slice(matrix, (j, 0), (1, ncols + nrows))
                return lax.dynamic_update_slice(matrix, row_j - ratio * row_i, (j, 0))

            return lax.cond(is_nonzero, eliminate_row, lambda matrix: matrix, matrix)

        def loop_body(j: int, matrix: Array) -> Array:
            return eliminate_above_row(j, matrix)

        matrix = lax.fori_loop(0, i, loop_body, matrix)

        return matrix

    def forward_elimination_body(i: int, matrix: Array) -> Array:
        return forward_step(i, matrix)

    augmented_matrix = lax.fori_loop(0, ncols, forward_elimination_body, augmented_matrix)

    def backward_elimination_body(i: int, matrix: Array) -> Array:
        return backward_step(ncols - 1 - i, matrix)

    augmented_matrix = lax.fori_loop(0, ncols, backward_elimination_body, augmented_matrix)

    # Don't need the reduced matrix, but maybe useful for debugging
    # reduced_matrix = lax.dynamic_slice(augmented_matrix, (0, 0), (nrows, ncols))
    component_matrix: Array = lax.dynamic_slice(
        augmented_matrix, (ncols, ncols), (nrows - ncols, nrows)
    )

    return component_matrix
