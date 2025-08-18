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
"""Tests general functionality"""

import logging

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from atmodeller import debug_logger
from atmodeller.type_aliases import NpFloat
from atmodeller.utilities import partial_rref, to_native_floats

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)


def test_scalar_no_tuple() -> None:
    """Tests scalar"""
    test_value: int = 1
    out = to_native_floats(test_value)
    target_value: float = float(test_value)

    assert out == target_value


def test_scalar() -> None:
    """Tests scalar that returns a single tuple"""
    test_value: tuple[int] = (1,)
    out = to_native_floats(test_value)
    target_value: tuple[float] = (1.0,)

    assert out == target_value


def test_tuple_1d() -> None:
    """Tests a 1-D tuple"""
    test_value: tuple[int, ...] = (0, 1, 2)
    out = to_native_floats(test_value)
    target_value: tuple[float, ...] = (0.0, 1.0, 2.0)

    assert out == target_value


def test_tuple_2d() -> None:
    """Tests a 2-D tuple"""
    test_value: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4, 5))
    out = to_native_floats(test_value)
    target_value: tuple[tuple[float, ...], ...] = ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))

    assert out == target_value


def test_list_1d() -> None:
    """Tests a 1-D list"""
    test_value: list[int] = [0, 1, 2]
    out = to_native_floats(test_value)
    target_value: tuple[float, ...] = (0.0, 1.0, 2.0)

    assert out == target_value


def test_list_2d() -> None:
    """Tests a 2-D list"""
    test_value: list[list[int]] = [[0, 1, 2], [3, 4, 5]]
    out = to_native_floats(test_value)
    target_value: tuple[tuple[float, ...], ...] = ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))

    assert out == target_value


def test_numpy_1d() -> None:
    """Tests a numpy 1-D array"""
    test_value = np.array([0, 1, 2])
    out = to_native_floats(test_value)
    target_value: tuple[float, ...] = (0.0, 1.0, 2.0)

    assert out == target_value


def test_numpy_2d() -> None:
    """Tests a numpy 2-D array"""
    test_value = np.array([[0, 1, 2], [3, 4, 5]])
    out = to_native_floats(test_value)
    target_value: tuple[tuple[float, ...], ...] = ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))

    assert out == target_value


def test_pandas_series() -> None:
    """Tests a pandas series"""
    test_value = pd.Series([0, 1, 2])
    out = to_native_floats(test_value)
    target_value: tuple[float, ...] = (0.0, 1.0, 2.0)

    assert out == target_value


def test_pandas_dataframe() -> None:
    """Tests a pandas dataframe"""
    test_value = pd.DataFrame([[0, 1, 2], [3, 4, 5]])
    out = to_native_floats(test_value)
    target_value: tuple[tuple[float, ...], ...] = ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0))

    assert out == target_value


def test_partial_rref() -> None:
    """Tests Gaussian elimination for a system with one reaction"""
    # E.g., H, O in rows, H2O, H2, O2 in columns
    test_value: NpFloat = np.array([[2, 2, 0], [1, 0, 2]])
    logger.debug("test_value = %s", test_value)
    out = partial_rref(test_value.T)
    target_value: NpFloat = np.array([[-2, 2, 1]])

    assert_array_equal(out, target_value)


def test_partial_rref_pivot() -> None:
    """Tests Gaussian elimination for a system with one reaction"""
    # E.g., H, O in rows, H2O, H2, O2 in columns
    test_value: NpFloat = np.array([[0, 2, 2], [2, 1, 0]])
    logger.debug("test_value = %s", test_value)
    out = partial_rref(test_value.T)
    target_value: NpFloat = np.array([[0.5, -1, 1]])

    assert_array_equal(out, target_value)


def test_partial_rref_no_reactions() -> None:
    """Tests Gaussian elimination for a system with no reactions"""
    # E.g., H, O in rows, H2O, O2 in columns
    test_value: NpFloat = np.array([[2, 0], [1, 2]])
    logger.debug("test_value = %s", test_value)
    out = partial_rref(test_value.T)
    target_shape: tuple[int, int] = (0, 2)

    assert out.shape == target_shape
