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

from __future__ import annotations

import functools
import logging
from collections.abc import Collection, MutableMapping
from cProfile import Profile
from dataclasses import asdict
from functools import wraps
from pstats import SortKey, Stats
from typing import Any, Callable, Type, TypeVar

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from jax import Array
from jaxtyping import ArrayLike
from molmass import Formula
from scipy.constants import kilo, mega

from atmodeller import ATMOSPHERE, BOLTZMANN_CONSTANT_BAR, OCEAN_MASS_H2

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
MultiplyT = TypeVar("MultiplyT", float, npt.NDArray, pd.Series, pd.DataFrame, Array, ArrayLike)


def profile_decorator(func):
    """Decorator to profile a function"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with Profile() as profile:
            result = func(*args, **kwargs)
        stats = Stats(profile).strip_dirs().sort_stats(SortKey.TIME)
        stats.print_stats()
        return result

    return wrapper


def debug_decorator(logger_in: logging.Logger) -> Callable:
    """A decorator to print the result of a function to a debug logger."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # logger.info(f"Executing {func.__name__}")
            result: Any = func(*args, **kwargs)
            logger_in.debug("%s = %s", func.__name__, result)
            # logger.info(f"Finished executing {func.__name__}")
            return result

        return wrapper

    return decorator


def filter_by_type(some_collection: Collection, class_type: Type[T]) -> dict[int, T]:
    """Filters entries of a collection according to the class type

    Args:
        some_collection: A collection (e.g. a list) to filter
        class_type: Class type to filter

    Returns:
        A dictionary with indices in some_collection as keys and filtered entries as values
    """
    filtered: dict[int, T] = {
        ii: value for ii, value in enumerate(some_collection) if isinstance(value, class_type)
    }

    return filtered


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


def earth_oceans_to_hydrogen_mass(number_of_earth_oceans: float = 1) -> float:
    h_grams: float = number_of_earth_oceans * OCEAN_MASS_H2
    h_kg: float = UnitConversion().g_to_kg(h_grams)
    return h_kg


class UnitConversion:
    """Unit conversions"""

    @staticmethod
    def atmosphere_to_bar(value_atmosphere: MultiplyT = 1) -> MultiplyT:
        """atmosphere to bar"""
        return value_atmosphere * ATMOSPHERE

    @staticmethod
    def bar_to_Pa(  # Symbol name, so pylint: disable=C0103
        value_bar: MultiplyT = 1,
    ) -> MultiplyT:
        """bar to Pa"""
        return value_bar * 1e5

    @classmethod
    def Pa_to_bar(  # Symbol name, so pylint: disable=C0103
        cls,
        value_Pa: MultiplyT = 1,
    ) -> MultiplyT:
        """Pa to bar"""
        return value_Pa / cls.bar_to_Pa()

    @classmethod
    def bar_to_GPa(  # Symbol name, so pylint: disable=C0103
        cls, value_bar: MultiplyT = 1
    ) -> MultiplyT:
        """Bar to GPa"""
        return cls.bar_to_Pa(value_bar) * 1.0e-9

    @classmethod
    def GPa_to_bar(  # Symbol name, so pylint: disable=C0103
        cls, value_GPa: MultiplyT = 1
    ) -> MultiplyT:
        """GPa to bar"""
        return value_GPa / cls.bar_to_GPa()

    @staticmethod
    def fraction_to_ppm(value_fraction: MultiplyT = 1) -> MultiplyT:
        """Mole or mass fraction to parts-per-million by mole or mass, respectively."""
        return value_fraction * mega

    @staticmethod
    def g_to_kg(value_grams: MultiplyT = 1) -> MultiplyT:
        """Grams to kilograms"""
        return value_grams / kilo

    @classmethod
    def ppm_to_fraction(cls, value_ppm: MultiplyT = 1) -> MultiplyT:
        """Parts-per-million by mole or mass to mole or mass fraction, respectively."""
        return value_ppm / cls.fraction_to_ppm()

    @classmethod
    def ppm_to_percent(cls, value_ppm: MultiplyT = 1) -> MultiplyT:
        """Parts-per-million by percent"""
        return cls.ppm_to_fraction(value_ppm) * 100

    @classmethod
    def cm3_to_m3(cls, cm_cubed: MultiplyT = 1) -> MultiplyT:
        """cm\ :sup:`3` to m\ :sup:`3`"""  # type: ignore reStructuredText so pylint: disable=W1401
        return cm_cubed * 1.0e-6

    @classmethod
    def m3_bar_to_J(  # Symbol name, so pylint: disable=C0103
        cls, m3_bar: MultiplyT = 1
    ) -> MultiplyT:
        """m\ :sup:`3` bar to J"""  # type: ignore reStructuredText so pylint: disable=W1401
        return m3_bar * 1e5

    @classmethod
    def J_to_m3_bar(  # Symbol name, so pylint: disable=C0103
        cls, joules: MultiplyT = 1
    ) -> MultiplyT:
        """J to m\ :sup:`3` bar"""  # type: ignore reStructuredText so pylint: disable=W1401
        return joules / cls.m3_bar_to_J()

    @classmethod
    def litre_to_m3(cls, litre: MultiplyT = 1) -> MultiplyT:
        """litre to m\ :sup:`3`"""  # type: ignore reStructuredText so pylint: disable=W1401
        return litre * 1e-3

    @staticmethod
    def weight_percent_to_ppmw(value_weight_percent: MultiplyT = 1) -> MultiplyT:
        """Weight percent to parts-per-million by weight"""
        return value_weight_percent * 1.0e4


def flatten(
    dictionary: MutableMapping[Any, Any], parent_key: str = "", separator: str = "_"
) -> dict[Any, Any]:
    """Flattens a nested dictionary and compresses keys

    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys

    Args:
        dictionary: A MutableMapping
        parent_key: Parent key
        separator: Separator for keys

    Returns:
        A flattened dictionary
    """
    items: list = []
    for key, value in dictionary.items():
        new_key: str = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def dataclass_to_logger(data_instance, logger_in: logging.Logger, log_level=logging.INFO) -> None:
    """Logs the attributes of a dataclass.

    Args:
        data_instance: A dataclass
        logger: The logger to log to
        log_level: Log level to use. Defaults to INFO.
    """
    data: dict[Any, Any] = flatten(asdict(data_instance))

    for key, value in data.items():
        logger_in.log(log_level, "%s = %s", key, value)


def delete_entries_with_suffix(input_dict: dict[Any, Any], suffix: str) -> dict[Any, Any]:
    """Deletes entries from a dictionary for keys that have a particular suffix

    Args:
        input_dict: Input dictionary
        suffix: Suffix of the keys that defines the entries to remove

    Returns:
        A dictionary with the entries removed
    """

    return {key: value for key, value in input_dict.items() if not key.endswith(suffix)}


# TODO: Remove, now in reaction network and converted to jax
# def partial_rref(matrix: npt.NDArray) -> npt.NDArray:
#     """Computes the partial reduced row echelon form to determine linear components

#     Returns:
#         A matrix of linear components
#     """
#     nrows: int = matrix.shape[0]
#     ncols: int = matrix.shape[1]

#     augmented_matrix: npt.NDArray = np.hstack((matrix, np.eye(nrows)))
#     logger.debug("augmented_matrix = \n%s", augmented_matrix)

#     # Forward elimination
#     for i in range(ncols):
#         # Check if the pivot element is zero and swap rows to get a non-zero pivot element.
#         if augmented_matrix[i, i] == 0:
#             nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
#             augmented_matrix[[i, nonzero_row], :] = augmented_matrix[[nonzero_row, i], :]
#         # Perform row operations to eliminate values below the pivot.
#         for j in range(i + 1, nrows):
#             ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
#             augmented_matrix[j] -= ratio * augmented_matrix[i]
#     logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

#     # Backward substitution
#     for i in range(ncols - 1, -1, -1):
#         # Normalize the pivot row.
#         augmented_matrix[i] /= augmented_matrix[i, i]
#         # Eliminate values above the pivot.
#         for j in range(i - 1, -1, -1):
#             if augmented_matrix[j, i] != 0:
#                 ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
#                 augmented_matrix[j] -= ratio * augmented_matrix[i]
#     logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

#     reduced_matrix: npt.NDArray = augmented_matrix[:, :ncols]
#     component_matrix: npt.NDArray = augmented_matrix[ncols:, ncols:]
#     logger.debug("reduced_matrix = \n%s", reduced_matrix)
#     logger.debug("component_matrix = \n%s", component_matrix)

#     return component_matrix


def reorder_dict(original_dict: dict[Any, Any], key_to_move_first: Any) -> dict:
    """Reorders a dictionary to put a particular key first

    Args:
        original_dict: Original dictionary
        key_to_move_first: Key to move first in the returned dictionary

    Returns:
        The reordered dictionary and a bool, the later to indicate if reordering occurred
    """
    if key_to_move_first not in original_dict:
        return original_dict

    return {
        key_to_move_first: original_dict[key_to_move_first],
        **{k: v for k, v in original_dict.items() if k != key_to_move_first},
    }


def get_molar_mass(species: str) -> float:
    r"""Get molar mass

    Args:
        species: A species

    Returns:
        Molar mass in kg m\ :sup:`-3`
    """
    return UnitConversion.g_to_kg(Formula(species).mass)


def get_number_density(temperature: float, pressure: ArrayLike) -> ArrayLike:
    r"""Pressure to number density

    Args:
        temperature: Temperature in K
        pressure: Pressure in bar

    Returns:
        Number density in molecules m\ :sup:`-3`
    """
    return pressure / (BOLTZMANN_CONSTANT_BAR * temperature)


def get_log10_number_density(*args, **kwargs) -> Array:
    r"""Pressure to log10 number density

    Args:
        temperature: Temperature in K
        pressure: Pressure in bar

    Returns:
        Log10 number density
    """
    return jnp.log10(get_number_density(*args, **kwargs))
