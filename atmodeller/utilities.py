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
"""Utilities"""

import logging
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array, lax
from jax.tree_util import tree_map
from jax.typing import ArrayLike
from scipy.constants import kilo, mega

from atmodeller import max_exp_input
from atmodeller.constants import ATMOSPHERE, BOLTZMANN_CONSTANT_BAR, OCEAN_MASS_H2

logger: logging.Logger = logging.getLogger(__name__)

OptxSolver = optx.AbstractRootFinder | optx.AbstractLeastSquaresSolver | optx.AbstractMinimiser


@eqx.filter_jit
def get_log_number_density_from_log_pressure(
    log_pressure: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets log number density from log pressure

    Args:
        log_pressure: Log pressure
        temperature: Temperature

    Returns:
        Log number density
    """
    log_number_density: Array = (
        -jnp.log(BOLTZMANN_CONSTANT_BAR) - jnp.log(temperature) + log_pressure
    )

    return log_number_density


@eqx.filter_jit
def safe_exp(x: ArrayLike) -> Array:
    return jnp.exp(jnp.clip(x, a_max=max_exp_input))


@eqx.filter_jit
def partial_rref_jax(matrix: Array) -> Array:
    """Computes the partial reduced row echelon form to determine linear components.

    This function is currently not used, since the reduction of matrices is performed prior to
    JAX operations. But since this function took some time to figure out, it is retained for
    potential future use. Note that lax.fori_loop is not compatible with reverse differentiation

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


def partial_rref(matrix: npt.NDArray) -> npt.NDArray:
    """Computes the partial reduced row echelon form to determine linear components

    Returns:
        A matrix of linear components
    """
    nrows, ncols = matrix.shape

    augmented_matrix: npt.NDArray = np.hstack((matrix, np.eye(nrows)))
    # debug("augmented_matrix = \n%s", augmented_matrix)
    # Permutation matrix
    # P: npt.NDArray = np.eye(nrows)

    # Forward elimination with partial pivoting
    for i in range(ncols):
        # Check if the pivot element is zero and swap rows to get a non-zero pivot element.
        if augmented_matrix[i, i] == 0:
            nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
            augmented_matrix[[i, nonzero_row], :] = augmented_matrix[[nonzero_row, i], :]
            # P[[i, nonzero_row], :] = P[[nonzero_row, i], :]
        # Perform row operations to eliminate values below the pivot.
        for j in range(i + 1, nrows):
            ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j] -= ratio * augmented_matrix[i]
    # logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

    # Backward substitution
    for i in range(ncols - 1, -1, -1):
        # Normalize the pivot row.
        augmented_matrix[i] /= augmented_matrix[i, i]
        # Eliminate values above the pivot.
        for j in range(i - 1, -1, -1):
            if augmented_matrix[j, i] != 0:
                ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]
    # logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

    # reduced_matrix: npt.NDArray = augmented_matrix[:, :ncols]
    component_matrix: npt.NDArray = augmented_matrix[ncols:, ncols:]
    # logger.debug("reduced_matrix = \n%s", reduced_matrix)
    # logger.debug("component_matrix = \n%s", component_matrix)
    # logger.debug("permutation_matrix = \n%s", P)

    return component_matrix


class UnitConversion(eqx.Module):
    """Unit conversions"""

    atmosphere_to_bar: float = ATMOSPHERE
    bar_to_Pa: float = 1.0e5
    bar_to_MPa: float = 1.0e-1
    bar_to_GPa: float = 1.0e-4
    Pa_to_bar: float = 1.0e-5
    MPa_to_bar: float = 1.0e1
    GPa_to_bar: float = 1.0e4
    fraction_to_ppm: float = mega
    g_to_kg: float = 1 / kilo
    ppm_to_fraction: float = 1 / mega
    ppm_to_percent: float = 100 / mega
    percent_to_ppm: float = 1.0e4
    cm3_to_m3: float = 1.0e-6
    m3_to_cm3: float = 1.0e6
    m3_bar_to_J: float = 1.0e5
    J_to_m3_bar: float = 1.0e-5
    litre_to_m3: float = 1.0e-3


unit_conversion = UnitConversion()


def bulk_silicate_earth_abundances() -> dict[str, dict[str, float]]:
    """Bulk silicate Earth element masses in kg.

    Hydrogen, carbon, and nitrogen from :cite:t:`SKG21`
    Sulfur from :cite:t:`H16`
    Chlorine from :cite:t:`KHK17`
    """
    earth_bse: dict[str, dict[str, float]] = {
        "H": {"min": 1.852e20, "max": 1.894e21},
        "C": {"min": 1.767e20, "max": 3.072e21},
        "S": {"min": 8.416e20, "max": 1.052e21},
        "N": {"min": 3.493e18, "max": 1.052e19},
        "Cl": {"min": 7.574e19, "max": 1.431e20},
    }

    for _, values in earth_bse.items():
        values["mean"] = np.mean((values["min"], values["max"]))  # type: ignore

    return earth_bse


def earth_oceans_to_hydrogen_mass(number_of_earth_oceans: ArrayLike = 1) -> ArrayLike:
    """Converts Earth oceans to hydrogen mass

    Args:
        number_of_earth_oceans: Number of Earth oceans. Defaults to 1.

    Returns:
        Hydrogen mass
    """
    h_grams: ArrayLike = number_of_earth_oceans * OCEAN_MASS_H2
    h_kg: ArrayLike = h_grams * unit_conversion.g_to_kg

    return h_kg


class ExperimentalCalibration(eqx.Module):
    """Experimental calibration

    Args:
        temperature_min: Minimum calibrated temperature. Defaults to None.
        temperature_max: Maximum calibrated temperature. Defaults to None.
        pressure_min: Minimum calibrated pressure. Defaults to None.
        pressure_max: Maximum calibrated pressure. Defaults to None.
        log10_fO2_min: Minimum calibrated log10 fO2. Defaults to None.
        log10_fO2_max: Maximum calibrated log10 fO2. Defaults to None.
    """

    temperature_min: float | None = None
    temperature_max: float | None = None
    pressure_min: float | None = None
    pressure_max: float | None = None
    log10_fO2_min: float | None = None
    log10_fO2_max: float | None = None


@eqx.filter_jit
def power_law(values: ArrayLike, constant: ArrayLike, exponent: ArrayLike) -> Array:
    """Power law

    Args:
        values: Values
        constant: Constant for the power law
        exponent: Exponent for the power law

    Returns:
        Evaluated power law
    """
    return jnp.power(values, exponent) * constant


def is_arraylike_batched(x: ArrayLike) -> Literal[0, None]:
    """Checks if x is batched.

    Args:
        x: Something arraylike

    Returns:
        0 (axis) if batched, else None (not batched)
    """
    return 0 if eqx.is_array(x) and x.ndim > 0 else None  # type: ignore


def vmap_axes_spec(x: Any) -> Any:
    """Recursively generate in_axes for vmap by checking if each leaf is batched (axis 0).

    Args:
        x: Pytree of nested containers possibly containing arrays or scalars

    Returns:
        Pytree matching the structure of x
    """
    return tree_map(is_arraylike_batched, x)


def get_batch_size(x: Any) -> int:
    """Determines the maximum batch size (i.e., length along axis 0) among all array-like leaves.

    Args:
        x: Pytree of nested containers possibly containing arrays or scalars

    Returns:
        The maximum size along axis 0 among all array-like leaves, or 0 if all leaves are scalars.
    """
    max_size: int = 0
    for leaf in jax.tree_util.tree_leaves(x):
        if eqx.is_array(leaf) and leaf.ndim > 0:
            max_size = max(max_size, leaf.shape[0])

    return max_size


def pytree_debug(pytree: Any, name: str) -> None:
    """Prints the pytree structure for debugging vmap.

    Args:
        pytree: Pytree to print
        name: Name for the debug print
    """
    arrays, static = eqx.partition(pytree, eqx.is_array)
    arrays_tree = tree_map(
        lambda x: (
            type(x),
            "True" if eqx.is_array(x) else ("False" if x is not None else "None"),
        ),
        arrays,
    )
    jax.debug.print("{name} arrays_tree = {out}", name=name, out=arrays_tree)

    static_tree = tree_map(
        lambda x: (
            type(x),
            "True" if eqx.is_array(x) else ("False" if x is not None else "None"),
        ),
        static,
    )
    jax.debug.print("{name} static_tree = {out}", name=name, out=static_tree)
