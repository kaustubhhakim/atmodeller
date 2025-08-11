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
from typing import Any, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import Partial, tree_map
from jaxtyping import Array, ArrayLike, Bool, Float64
from scipy.constants import kilo, mega

from atmodeller import max_exp_input
from atmodeller._mytypes import NpArray, NpFloat, Scalar
from atmodeller.constants import ATMOSPHERE, BOLTZMANN_CONSTANT_BAR, OCEAN_MASS_H2

logger: logging.Logger = logging.getLogger(__name__)


def get_reaction_dictionary(
    reaction_matrix: NpFloat, species_names: tuple[str, ...]
) -> dict[int, str]:
    """Gets reactions as a dictionary.

    Args:
        reaction_matrix: Reaction matrix
        species_names: Names of species

    Returns:
        Reactions as a dictionary
    """
    reactions: dict[int, str] = {}
    if reaction_matrix.size != 0:
        for reaction_index in range(reaction_matrix.shape[0]):
            reactants: str = ""
            products: str = ""
            for species_index, name in enumerate(species_names):
                coeff: float = reaction_matrix[reaction_index, species_index].item()
                if coeff != 0:
                    if coeff < 0:
                        reactants += f"{abs(coeff)} {name} + "
                    else:
                        products += f"{coeff} {name} + "

            reactants = reactants.rstrip(" + ")
            products = products.rstrip(" + ")
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

    return reactions


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
    """Wraps a callable in `equinox.Partial` to make it hashable for JAX transformations.

    This is useful when passing callables with fixed arguments to JAX transformations
    (e.g., `jax.vmap`, `jax.grad`, `jax.jit`) that require all static arguments
    (including function references) to be hashable.

    See discussion: https://github.com/patrick-kidger/equinox/issues/1011

    Args:
        x: A callable to wrap

    Returns:
        An `equinox.Partial` object wrapping the input callable, making it hashable
    """
    return Partial(x)


def is_hashable(something: Any) -> None:
    """Checks whether an object is hashable and print the result.

    Args:
        something: Any Python object to test.

    Prints:
        A message indicating whether the object is hashable.
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


def as_j64_2d(x: ArrayLike | tuple) -> Float64[Array, "..."]:
    """Converts input to a 2-D JAX array of dtype float64.

    Args:
        x: Input to convert

    Returns:
        2-D JAX array of dtype float64
    """
    return jnp.atleast_2d(as_j64(x))


def to_native_floats(value: Any, force_tuple: bool = True) -> Any:
    """Recursively converts any structure to nested tuples of native floats.

    Args:
        value: A scalar, list/tuple/array of floats, or nested thereof.
        force_tuple: If True, scalars are returned as (float,). Defaults to True.

    Returns:
        A float or nested tuple of floats.
    """
    try:
        if isinstance(value, pd.Series) and len(value) == 1:
            val = float(value.iloc[0])
        else:
            val = float(value)  # type: ignore

        return (val,) if force_tuple else val
    except (TypeError, ValueError):
        pass

    # Special case for DataFrame: convert to list of rows
    if isinstance(value, pd.DataFrame):
        iterable: Iterable = value.itertuples(index=False, name=None)
    else:
        try:
            iterable = list(value)
        except Exception:
            raise TypeError(f"Cannot convert to float or iterate over type {type(value)}")

    return tuple(to_native_floats(item, force_tuple=False) for item in iterable)


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


unit_conversion: UnitConversion = UnitConversion()


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
    h_kg: ArrayLike = number_of_earth_oceans * OCEAN_MASS_H2

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

    temperature_min: Optional[float] = None
    """Minimum calibrated temperature"""
    temperature_max: Optional[float] = None
    """Maximum calibrated temperature"""
    pressure_min: Optional[float] = None
    """Minimum calibrated pressure"""
    pressure_max: Optional[float] = None
    """Maximum calibrated pressure"""
    log10_fO2_min: Optional[float] = None
    """Minimum calibrated log10 fO2"""
    log10_fO2_max: Optional[float] = None
    """Maximum calibrated log10 fO2"""

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
