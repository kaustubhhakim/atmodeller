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
"""Tests for core infrastructure"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.containers import (
    ConstantFugacityConstraint,
    FugacityConstraints,
    MassConstraints,
    Species,
    SpeciesCollection,
)
from atmodeller.interfaces import FugacityConstraintProtocol

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

# Simple H-C-O system with 2 species
H2O_g: Species = Species.create_gas("H2O_g")
CO2_g: Species = Species.create_gas("CO2_g")
species: SpeciesCollection = SpeciesCollection((H2O_g, CO2_g))


def test_mass_constraints_single() -> None:
    """Mass constraints with single entries"""

    mass_constraints_map: dict[str, ArrayLike] = {
        "H": 1,
        "C": 3,
    }
    mass_constraints: MassConstraints = MassConstraints.create(species, mass_constraints_map)
    # jax.debug.print("mass_constraints = {out}", out=mass_constraints.log_abundance)

    log_abundance: Array = mass_constraints.log_abundance
    target: Array = jnp.array([60.27546626107961, 61.65474558573571, jnp.nan])

    assert mass_constraints.vmap_axes().log_abundance is None
    assert jax.numpy.allclose(log_abundance, target, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_mass_constraints_broadcast1() -> None:
    """Mass constraints with one entry to broadcast"""

    mass_constraints_map: dict[str, ArrayLike] = {
        "H": np.array([1, 2]),
        "C": 3,
    }
    mass_constraints: MassConstraints = MassConstraints.create(species, mass_constraints_map)
    # jax.debug.print("mass_constraints = {out}", out=mass_constraints.log_abundance)

    log_abundance: Array = mass_constraints.log_abundance
    target: Array = jnp.array(
        [
            [60.27546626107961, 61.65474558573571, jnp.nan],
            [60.27546626107961, 62.34789276629566, jnp.nan],
        ]
    )

    assert mass_constraints.vmap_axes().log_abundance == 0
    assert jax.numpy.allclose(log_abundance, target, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_mass_constraints_broadcast2() -> None:
    """Mass constraints with two entries to broadcast"""

    mass_constraints_map: dict[str, ArrayLike] = {
        "H": np.array([1, 2]),
        "C": np.array([3, 4]),
    }
    mass_constraints: MassConstraints = MassConstraints.create(species, mass_constraints_map)
    # jax.debug.print("mass_constraints = {out}", out=mass_constraints.log_abundance)

    log_abundance: Array = mass_constraints.log_abundance
    target: Array = jnp.array(
        [
            [60.27546626107961, 61.65474558573571, jnp.nan],
            [60.56314833353139, 62.34789276629566, jnp.nan],
        ]
    )

    assert mass_constraints.vmap_axes().log_abundance == 0
    assert jax.numpy.allclose(log_abundance, target, rtol=RTOL, atol=ATOL, equal_nan=True)


def test_fugacity_constraint_single() -> None:
    """Fugacity constraints with single entries"""

    fugacity_constraints_map: dict[str, FugacityConstraintProtocol] = {
        "H2O_g": ConstantFugacityConstraint(1)
    }
    fugacity_constraints: FugacityConstraints = FugacityConstraints.create(
        species, fugacity_constraints_map
    )

    constraints: tuple[FugacityConstraintProtocol, ...] = fugacity_constraints.constraints

    print(constraints)
    print(fugacity_constraints.species)

    print(fugacity_constraints.vmap_axes())


def test_fugacity_constraint_broadcast1() -> None:
    """Fugacity constraints with one entry to broadcast"""

    fugacity_constraints_map: dict[str, FugacityConstraintProtocol] = {
        "H2O_g": ConstantFugacityConstraint(np.array([1, 2]))
    }
    fugacity_constraints: FugacityConstraints = FugacityConstraints.create(
        species, fugacity_constraints_map
    )

    constraints: tuple[FugacityConstraintProtocol, ...] = fugacity_constraints.constraints

    print(constraints)
    print(fugacity_constraints.species)

    print(fugacity_constraints.vmap_axes())
