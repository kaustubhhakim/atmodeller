# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""General utilities

This module is designed to have minimal dependencies on the core Atmodeller package, as its
functionality is broadly applicable across different parts of the codebase. Keeping this module
lightweight also helps avoid circular imports.
"""

import logging
from collections.abc import Iterable
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from atmodeller.jaxhelper import NpFloat, Scalar
from atmodeller.sciencehelper import OCEAN_MASS_H2

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

    temperature_min: Optional[float]
    """Minimum calibrated temperature"""
    temperature_max: Optional[float]
    """Maximum calibrated temperature"""
    pressure_min: Optional[float]
    """Minimum calibrated pressure"""
    pressure_max: Optional[float]
    """Maximum calibrated pressure"""
    log10_fO2_min: Optional[float]
    r"""Minimum calibrated :math:`\log_{10} f\rm{O}_2`"""
    log10_fO2_max: Optional[float]
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
        self.temperature_min = float(temperature_min) if temperature_min is not None else None
        self.temperature_max = float(temperature_max) if temperature_max is not None else None
        self.pressure_min = float(pressure_min) if pressure_min is not None else None
        self.pressure_max = float(pressure_max) if pressure_max is not None else None
        self.log10_fO2_min = float(log10_fO2_min) if log10_fO2_min is not None else None
        self.log10_fO2_max = float(log10_fO2_max) if log10_fO2_max is not None else None


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


def get_reaction_dictionary(
    reaction_matrix: NpFloat, species_names: Iterable[str]
) -> dict[int, str]:
    """Gets reactions as a dictionary.

    Args:
        reaction_matrix: Reaction matrix of shape (number_reactions, number_species)
        species_names: Species names corresponding to the columns of the reaction matrix

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


def flatten_dictionary(d: dict, parent_key: str = "") -> dict:
    """Recursively flattens a nested dictionary, joining keys with "." to form column names.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used during recursion)

    Returns:
        Flat dictionary with dot-joined keys
    """
    items: dict = {}

    for k, v in d.items():
        new_key: str = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dictionary(v, new_key))
        else:
            items[new_key] = v

    return items


def recursively_merge_dictionaries(d1: dict, d2: dict) -> dict:
    """Recursively merges two dictionaries.

    Args:
        d1: The first dictionary
        d2: The second dictionary, which will overwrite values in the first dictionary if there are
            duplicate keys

    Returns:
        The merged dictionary
    """
    out: dict = dict(d1)

    for k, v in d2.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = recursively_merge_dictionaries(out[k], v)
            else:
                out[k] = v
        else:
            out[k] = v

    return out


def power_law(values: ArrayLike, constant: ArrayLike, exponent: ArrayLike) -> Array:
    return jnp.power(values, exponent) * constant
