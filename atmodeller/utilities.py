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
from collections.abc import Callable, Iterable
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import Partial
from jaxtyping import Array, ArrayLike, Float64
from scipy.constants import kilo, mega

from atmodeller._mytypes import NpArray, NpInt, Scalar
from atmodeller.constants import ATMOSPHERE, BOLTZMANN_CONSTANT_BAR, MAX_EXP_INPUT, OCEAN_MASS_H2

logger: logging.Logger = logging.getLogger(__name__)


class ExperimentalCalibration(eqx.Module):
    r"""Experimental calibration

    Args:
        temperature_min: Minimum calibrated temperature. Defaults to ``None``.
        temperature_max: Maximum calibrated temperature. Defaults to ``None``.
        pressure_min: Minimum calibrated pressure. Defaults to ``None``.
        pressure_max: Maximum calibrated pressure. Defaults to ``None``.
        log10_fO2_min: Minimum calibrated :math:`\log_{10} f\rm{O}_2`. Defaults to ``None``.
        log10_fO2_max: Maximum calibrated :math:`\log_{10} f\rm{O}_2`. Defaults to ``None``.
    """

    temperature_min: Optional[float] = None
    """Minimum calibrated temperature"""
    temperature_max: Optional[float] = None
    """Maximum calibrated temperature"""
    pressure_min: Optional[float] = None
    """Minimum calibrated pressure"""
    pressure_max: Optional[float] = None
    """Maximum calibrated pressure"""
    log10_fO2_min: Optional[float] = None
    r"""Minimum calibrated :math:`\log_{10} f\rm{O}_2`"""
    log10_fO2_max: Optional[float] = None
    r"""Maximum calibrated :math:`\log_{10} f\rm{O}_2`"""

    def __init__(
        self,
        temperature_min: Optional[Scalar] = None,
        temperature_max: Optional[Scalar] = None,
        pressure_min: Optional[Scalar] = None,
        pressure_max: Optional[Scalar] = None,
        log10_fO2_min: Optional[Scalar] = None,
        log10_fO2_max: Optional[Scalar] = None,
    ):
        if temperature_min is not None:
            self.temperature_min = float(temperature_min)
        if temperature_max is not None:
            self.temperature_max = float(temperature_max)
        if pressure_min is not None:
            self.pressure_min = float(pressure_min)
        if pressure_max is not None:
            self.pressure_max = float(pressure_max)
        if log10_fO2_min is not None:
            self.log10_fO2_min = float(log10_fO2_min)
        if log10_fO2_max is not None:
            self.log10_fO2_max = float(log10_fO2_max)


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


unit_conversion: UnitConversion = UnitConversion()


def get_log_number_density_from_log_pressure(
    log_pressure: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets log number density from log pressure.

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


def safe_exp(x: ArrayLike) -> Array:
    """Computes the elementwise exponential of ``x`` with input clipping to prevent overflow.

    This function clips the input ``x`` to a maximum value defined by
    :data:`MAX_EXP_INPUT <atmodeller.constants.MAX_EXP_INPUT>` before applying
    :func:`jax.numpy.exp`, ensuring numerical stability for large values.

    Args:
        x: Array-like input. Can be a scalar, 1-D, or multi-dimensional array

    Returns:
        Array of the same shape as ``x``, where each element is the exponential of the clipped
        input
    """
    return jnp.exp(jnp.clip(x, max=MAX_EXP_INPUT))


def to_hashable(x: Any) -> Callable:
    """Wraps a callable in :func:`equinox.Partial` to make it hashable for JAX transformations.

    This is useful when passing callables with fixed arguments to JAX transformations
    (e.g., :func:`jax.vmap`, :func:`jax.grad`, :func:`jax.jit`) that require all static arguments
    (including function references) to be hashable.

    See discussion: https://github.com/patrick-kidger/equinox/issues/1011

    Args:
        x: A callable to wrap

    Returns:
        An :func:`equinox.Partial` object wrapping the input callable, making it hashable
    """
    return Partial(x)


def is_hashable(something: Any) -> None:
    """Checks whether an object is hashable and print the result.

    Args:
        something: Any Python object to test

    Prints:
        A message indicating whether the object is hashable
    """
    try:
        hash(something)
        print("%s is hashable" % something.__class__.__name__)

    except TypeError:
        print("%s is not hashable" % something.__class__.__name__)


def as_j64(x: ArrayLike | tuple) -> Float64[Array, "..."]:
    """Converts input to a JAX array of dtype float64.

    Args:
        x: Input to convert

    Returns:
        JAX array of dtype float64
    """
    return jnp.asarray(x, dtype=jnp.float64)


def to_native_floats(value: Any) -> Any:
    """Recursively converts any structure to nested tuples of native floats.

    Args:
        value: A scalar, list/tuple/array of floats, or nested thereof

    Returns:
        A float or nested tuple of floats
    """
    # Scalars (covers Python, NumPy, JAX scalars)
    if jnp.isscalar(value):
        return float(value)

    # Pandas DataFrame: convert to list of rows (as tuples)
    if isinstance(value, pd.DataFrame):
        iterable: Iterable = value.itertuples(index=False, name=None)
        return tuple(to_native_floats(row) for row in iterable)

    # Array-like (NumPy, JAX)
    if hasattr(value, "ndim"):
        return tuple(to_native_floats(sub) for sub in value.tolist())

    # Generic iterables (lists, tuples, etc.)
    try:
        iterable = list(value)
    except Exception:
        raise TypeError(f"Cannot convert to float or iterate over type {type(value)}")

    return tuple(to_native_floats(item) for item in iterable)


def partial_rref(matrix: NpArray) -> NpArray:
    """Computes the partial reduced row echelon form to determine linear components.

    Returns:
        A matrix of linear components
    """
    nrows, ncols = matrix.shape

    augmented_matrix: NpArray = np.hstack((matrix, np.eye(nrows)))
    logger.debug("augmented_matrix = \n%s", augmented_matrix)
    # Permutation matrix
    # P: NpArray = np.eye(nrows)

    # Forward elimination with partial pivoting
    for i in range(min(nrows, ncols)):
        # Pivot selection with check
        nonzero: NpInt = np.flatnonzero(augmented_matrix[i:, i])
        logger.debug("nonzero = %s", nonzero)
        if nonzero.size == 0:
            logger.debug("i: %d. No pivot in this column.", i)
            continue  # no pivot in this column
        # Absolute row index of first non-zero index
        pivot_row: np.int_ = nonzero[0] + i
        # Swap if pivot row is not already in place
        if pivot_row != i:
            augmented_matrix[[i, pivot_row], :] = augmented_matrix[[pivot_row, i], :]
            # P[[i, nonzero_row], :] = P[[nonzero_row, i], :]

        # Perform row operations to eliminate values below the pivot.
        pivot_value: np.float64 = augmented_matrix[i, i]
        if i + 1 < nrows:
            factors = augmented_matrix[i + 1 :, i : i + 1] / pivot_value  # shape (nrows-i-1, 1)
            augmented_matrix[i + 1 :] -= factors * augmented_matrix[i]

    logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

    # Backward substitution
    for i in range(min(nrows, ncols) - 1, -1, -1):
        pivot_value = augmented_matrix[i, i]
        if pivot_value == 0:
            logger.debug("i: %d. Pivot is zero, skipping backward elimination.", i)
            continue  # skip columns with no pivot
        # Normalize the pivot row.
        augmented_matrix[i] /= augmented_matrix[i, i]

        # Eliminate entries above the pivot
        if i > 0:
            factors = augmented_matrix[:i, i : i + 1] / pivot_value  # shape (i, 1)
            augmented_matrix[:i] -= factors * augmented_matrix[i]

    logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

    # reduced_matrix: NpArray = augmented_matrix[:, :ncols]
    component_matrix: NpArray = augmented_matrix[min(ncols, nrows) :, ncols:]
    # logger.debug("reduced_matrix = \n%s", reduced_matrix)
    logger.debug("component_matrix = \n%s", component_matrix)
    # logger.debug("permutation_matrix = \n%s", P)

    return component_matrix


def bulk_silicate_earth_abundances() -> dict[str, dict[str, float]]:
    """Bulk silicate Earth element masses in kg

    Hydrogen, carbon, and nitrogen from :cite:t:`SKG21`, sulfur from :cite:t:`H16`, and chlorine
    from :cite:t:`KHK17`

    Returns:
        A dictionary of Earth BSE element masses in kg
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
    """Converts Earth oceans to hydrogen mass.

    Args:
        number_of_earth_oceans: Number of Earth oceans. Defaults to ``1`` kg.

    Returns:
        Hydrogen mass in kg
    """
    h_kg: ArrayLike = number_of_earth_oceans * OCEAN_MASS_H2

    return h_kg


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


def get_batch_size(x: Any) -> int:
    """Determines the maximum batch size (i.e., length along axis 0) among all array-like leaves.

    Args:
        x: Pytree of nested containers possibly containing arrays or scalars

    Returns:
        The maximum size along axis 0 among all array-like leaves
    """
    max_size: int = 1
    for leaf in jax.tree_util.tree_leaves(x):
        if eqx.is_array(leaf):
            max_size = max(max_size, leaf.shape[0] if leaf.ndim else 1)

    return max_size
