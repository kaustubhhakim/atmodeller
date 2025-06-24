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
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial, tree_map
from jaxtyping import Array, ArrayLike, Bool, Float64
from scipy.constants import kilo, mega

from atmodeller import max_exp_input
from atmodeller.constants import ATMOSPHERE, BOLTZMANN_CONSTANT_BAR, OCEAN_MASS_H2
from atmodeller.mytypes import NpArray

logger: logging.Logger = logging.getLogger(__name__)


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


def all_not_nan(x: ArrayLike) -> Bool[Array, "..."]:
    """Returns True if all entries or columns are not nan, otherwise False"""
    return ~jnp.any(jnp.isnan(jnp.atleast_1d(x)), axis=0)


def safe_exp(x: ArrayLike) -> Array:
    return jnp.exp(jnp.clip(x, max=max_exp_input))


def to_hashable(x: Any) -> Callable:
    """Converts input to a hashable type"""

    return Partial(x)


def as_j64(x: ArrayLike) -> Float64[Array, "..."]:
    """Converts input to a JAX array of dtype float64.

    This function is used to minimise the number of times jitted functions need to be compiled.

    Args:
        x: Input to convert

    Returns:
        JAX array of dtype float64
    """
    return jnp.asarray(x, dtype=jnp.float64)


def partial_rref(matrix: NpArray) -> NpArray:
    """Computes the partial reduced row echelon form to determine linear components

    Returns:
        A matrix of linear components
    """
    nrows, ncols = matrix.shape

    augmented_matrix: NpArray = np.hstack((matrix, np.eye(nrows)))
    # debug("augmented_matrix = \n%s", augmented_matrix)
    # Permutation matrix
    # P: NpArray = np.eye(nrows)

    # Forward elimination with partial pivoting
    for i in range(ncols):
        # Check if the pivot element is zero and swap rows to get a non-zero pivot element.
        if augmented_matrix[i, i] == 0:
            nonzero_row: np.int64 = np.nonzero(augmented_matrix[i:, i])[0][0] + i
            augmented_matrix[[i, nonzero_row], :] = augmented_matrix[[nonzero_row, i], :]
            # P[[i, nonzero_row], :] = P[[nonzero_row, i], :]
        # Perform row operations to eliminate values below the pivot.
        for j in range(i + 1, nrows):
            ratio: np.float64 = augmented_matrix[j, i] / augmented_matrix[i, i]
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

    # reduced_matrix: NpArray = augmented_matrix[:, :ncols]
    component_matrix: NpArray = augmented_matrix[ncols:, ncols:]
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
        temperature_min: Minimum calibrated temperature. Defaults to nan.
        temperature_max: Maximum calibrated temperature. Defaults to nan.
        pressure_min: Minimum calibrated pressure. Defaults to nan.
        pressure_max: Maximum calibrated pressure. Defaults to nan.
        log10_fO2_min: Minimum calibrated log10 fO2. Defaults to nan.
        log10_fO2_max: Maximum calibrated log10 fO2. Defaults to nan.
    """

    temperature_min: Array = eqx.field(converter=as_j64, default=jnp.nan)
    temperature_max: Array = eqx.field(converter=as_j64, default=jnp.nan)
    pressure_min: Array = eqx.field(converter=as_j64, default=jnp.nan)
    pressure_max: Array = eqx.field(converter=as_j64, default=jnp.nan)
    log10_fO2_min: Array = eqx.field(converter=as_j64, default=jnp.nan)
    log10_fO2_max: Array = eqx.field(converter=as_j64, default=jnp.nan)


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


def is_arraylike_batched(x: Any) -> Literal[0, None]:
    """Checks if x is batched.

    The logic accommodates batching for scalars, 1-D arrays, and 2-D arrays.

    Args:
        x: Something to check

    Returns:
        0 (axis) if batched, else None (not batched)
    """
    if eqx.is_array(x) and x.ndim > 0:
        return 0
    else:
        return None


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
        The maximum size along axis 0 among all array-like leaves
    """
    max_size: int = 1
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
