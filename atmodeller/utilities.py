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
from dataclasses import asdict, dataclass, field
from functools import wraps
from pstats import SortKey, Stats
from typing import Any, Callable, Type, TypeVar

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from molmass import Formula
from scipy.constants import kilo, mega

from atmodeller import ATMOSPHERE, BOLTZMANN_CONSTANT_BAR, MACHEPS, OCEAN_MASS_H2

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    h_kg: float = h_grams * unit_conversion.g_to_kg
    return h_kg


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
    return Formula(species).mass * unit_conversion.g_to_kg


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


def logsumexp_base10(log_values: ArrayLike, prefactors: ArrayLike = 1.0) -> Array:
    """Computes the log-sum-exp using base-10 exponentials in a numerically stable way.

    Args:
        log10_values: Array of log10 values to sum
        prefactors: Array of prefactors corresponding to each log10 value

    Returns:
        The log10 of the sum of prefactors multiplied by exponentials of the input values.
    """
    max_log: Array = jnp.max(log_values)
    prefactors_: Array = jnp.asarray(prefactors)

    value_sum: Array = jnp.sum(prefactors_ * jnp.power(10, log_values - max_log))

    return max_log + jnp.log10(value_sum)


def safe_log10(x: ArrayLike) -> Array:
    """Computes log10 of x, safely adding machine epsilon to avoid log of zero."""

    return jnp.log10(x + MACHEPS)


@dataclass(frozen=True)
class ExperimentalCalibration:
    """Experimental calibration range

    Args:
        temperature_min: Minimum temperature in K. Defaults to None (i.e. not specified).
        temperature_max: Maximum temperature in K. Defaults to None (i.e. not specified).
        pressure_min: Minimum pressure in bar. Defaults to None (i.e. not specified).
        pressure_max: Maximum pressure in bar. Defaults to None (i.e. not specified).
        temperature_penalty: Penalty coefficients for temperature. Defaults to 1000.
        pressure_penalty: Penalty coefficient for pressure. Defaults to 1000.
    """

    temperature_min: float | None = None
    """Minimum temperature in K"""
    temperature_max: float | None = None
    """Maximum temperature in K"""
    pressure_min: float | None = None
    """Minimum pressure in bar"""
    pressure_max: float | None = None
    """Maximum pressure in bar"""
    temperature_penalty: float = 1e3
    """Temperature penalty"""
    pressure_penalty: float = 1e3
    """Pressure penalty"""
    _clips_to_apply: list[Callable] = field(init=False, default_factory=list, repr=False)
    """Clips to apply"""

    def __post_init__(self):
        if self.temperature_min is not None:
            logger.info(
                "Set minimum evaluation temperature (temperature > %f)", self.temperature_min
            )
            self._clips_to_apply.append(self._clip_temperature_min)
        if self.temperature_max is not None:
            logger.info(
                "Set maximum evaluation temperature (temperature < %f)", self.temperature_max
            )
            self._clips_to_apply.append(self._clip_temperature_max)
        if self.pressure_min is not None:
            logger.info("Set minimum evaluation pressure (pressure > %f)", self.pressure_min)
            self._clips_to_apply.append(self._clip_pressure_min)
        if self.pressure_max is not None:
            logger.info("Set maximum evaluation pressure (pressure < %f)", self.pressure_max)
            self._clips_to_apply.append(self._clip_pressure_max)

    def _clip_pressure_max(self, temperature: float, pressure: ArrayLike) -> tuple[float, Array]:
        """Clips maximum pressure

        Args:
            temperature: Temperature in K
            pressure: pressure in bar

        Returns:
            Temperature, and clipped pressure
        """
        assert self.pressure_max is not None

        return temperature, jnp.minimum(pressure, jnp.array(self.pressure_max))

    def _clip_pressure_min(self, temperature: float, pressure: ArrayLike) -> tuple[float, Array]:
        """Clips minimum pressure

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Temperature, and clipped pressure
        """
        assert self.pressure_min is not None

        return temperature, jnp.maximum(pressure, jnp.array(self.pressure_min))

    def _clip_temperature_max(
        self, temperature: float, pressure: ArrayLike
    ) -> tuple[float, Array]:
        """Clips maximum temperature

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Clipped temperature, and pressure
        """
        assert self.temperature_max is not None

        return min(temperature, self.temperature_max), jnp.array(pressure)

    def _clip_temperature_min(
        self, temperature: float, pressure: ArrayLike
    ) -> tuple[float, Array]:
        """Clips minimum temperature

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Clipped temperature, and pressure
        """
        assert self.temperature_min is not None

        return max(temperature, self.temperature_min), jnp.array(pressure)

    def get_within_range(self, temperature: float, pressure: ArrayLike) -> tuple[float, ArrayLike]:
        """Gets temperature and pressure conditions within the calibration range.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            temperature in K, pressure in bar, according to prescribed clips
        """
        for clip_func in self._clips_to_apply:
            temperature, pressure = clip_func(temperature, pressure)

        return temperature, pressure

    def get_penalty(self, temperature: float, pressure: ArrayLike) -> Array:
        """Gets a penalty value if temperature and pressure are outside the calibration range

        This is based on the quadratic penalty method.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            A penalty value
        """
        pressure_: Array = jnp.asarray(pressure)
        temperature_clip, pressure_clip = self.get_within_range(temperature, pressure)
        penalty = self.pressure_penalty * jnp.power(
            pressure_clip - pressure_, 2
        ) + self.temperature_penalty * jnp.power(temperature_clip - temperature, 2)

        return penalty


# Convenient to use symbol names so pylint: disable=invalid-name
@dataclass(frozen=True)
class UnitConversion:
    """Unit conversions"""

    atmosphere_to_bar: float = ATMOSPHERE
    bar_to_Pa: float = 1.0e5
    bar_to_GPa: float = 1.0e-4
    Pa_to_bar: float = 1.0e-5
    GPa_to_bar: float = 1.0e4
    fraction_to_ppm: float = mega
    g_to_kg: float = 1 / kilo
    ppm_to_fraction: float = 1 / mega
    ppm_to_percent: float = 100 / mega
    percent_to_ppm: float = 1.0e4
    cm3_to_m3: float = 1.0e-6
    m3_bar_to_J: float = 1.0e5
    J_to_m3_bar: float = 1.0e-5
    litre_to_m3: float = 1.0e-3


unit_conversion = UnitConversion()
