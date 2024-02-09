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
from collections import OrderedDict, abc
from collections.abc import MutableMapping
from dataclasses import asdict
from typing import Any, Callable, Type, TypeVar

import numpy as np
import pandas as pd
from scipy.constants import kilo, mega

from atmodeller import OCEAN_MASS_H2

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")
MultiplyT = TypeVar("MultiplyT", float, np.ndarray, pd.Series, pd.DataFrame)


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


def filter_by_type(some_collection: abc.Collection, class_type: Type[T]) -> dict[int, T]:
    """Filters entries by the given class type and maintains order.

    Args:
        some_collection: A collection (e.g. a list) to filter.
        class_type: Class type to filter.

    Returns:
        An OrderedDict with indices in some_collection as keys and filtered entries as values.
    """

    ordered_dict = OrderedDict()
    for index, entry in enumerate(some_collection):
        if isinstance(entry, class_type):
            ordered_dict[index] = entry

    return ordered_dict


def earth_oceans_to_kg(number_of_earth_oceans: float = 1) -> float:
    h_grams: float = number_of_earth_oceans * OCEAN_MASS_H2
    h_kg: float = UnitConversion().g_to_kg(h_grams)
    return h_kg


class UnitConversion:
    """Unit conversions."""

    @staticmethod
    def bar_to_Pa(  # Symbol name, so pylint: disable=invalid-name
        value_bar: MultiplyT = 1,
    ) -> MultiplyT:
        """bar to Pa."""
        return value_bar * 1e5

    @classmethod
    def bar_to_GPa(  # Symbol name, so pylint: disable=invalid-name
        cls, value_bar: MultiplyT = 1
    ) -> MultiplyT:
        """Bar to GPa."""
        return cls.bar_to_Pa(value_bar) * 1.0e-9

    @staticmethod
    def fraction_to_ppm(value_fraction: MultiplyT = 1) -> MultiplyT:
        """Mole or mass fraction to parts-per-million by mole or mass, respectively."""
        return value_fraction * mega

    @staticmethod
    def g_to_kg(value_grams: MultiplyT = 1) -> MultiplyT:
        """Grams to kilograms."""
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
        """cm^3 to m^3"""
        return cm_cubed * 1.0e-6

    @classmethod
    def m3_bar_to_J(  # Symbol name, so pylint: disable=invalid-name
        cls, m3_bar: MultiplyT = 1
    ) -> MultiplyT:
        """m^3 bar to J"""
        return m3_bar * 1e5

    @classmethod
    def J_to_m3_bar(  # Symbol name, so pylint: disable=invalid-name
        cls, joules: MultiplyT = 1
    ) -> MultiplyT:
        """J to m^3 bar"""
        return joules / cls.m3_bar_to_J()

    @classmethod
    def litre_to_m3(cls, litre: MultiplyT = 1) -> MultiplyT:
        """litre to m^3"""
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
