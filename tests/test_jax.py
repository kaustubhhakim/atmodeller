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
"""Tests for ideal C-H-O interior-atmosphere systems"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import pytest

from atmodeller import __version__, debug_logger
from atmodeller.constraints import FugacityConstraint, SystemConstraints
from atmodeller.core import GasSpecies, Planet
from atmodeller.interior_atmosphere import Species
from atmodeller.reaction_network_jax import (
    ReactionNetwork,
    ReactionNetworkWithCondensateStability,
)
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()


def test_simple() -> None:

    def vector_function(x):
        return jnp.array([x[0] ** 2, jnp.sin(x[1])])

    # Compute the Jacobian
    def compute_jacobian(x):
        return jax.jacobian(vector_function)(x)

    # Test with a 2-D vector
    x = jnp.array([1.0, 2.0])
    jacobian = compute_jacobian(x)
    print("Jacobian:\n", jacobian)


def test_reaction_network() -> None:

    H2O_g: GasSpecies = GasSpecies(
        "H2O", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()
    )
    H2_g: GasSpecies = GasSpecies("H2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell())
    O2_g: GasSpecies = GasSpecies("O2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell())

    species: Species = Species([H2O_g, H2_g, O2_g])
    planet = Planet(surface_temperature=2000)
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(H2O_g, 0.2570770067190733),
            FugacityConstraint(O2_g, 8.838043080858959e-08),
        )
    )

    # reaction_network = ReactionNetwork(species=species, planet=planet)
    reaction_network = ReactionNetworkWithCondensateStability(species=species, planet=planet)

    sol, solution = reaction_network.solve_optimistix(constraints=constraints)

    print(sol.stats)
    print(sol.state)

    target_dict = {
        "H2O_g": 0.25707719341563373,
        "H2_g": 0.249646956461615,
        "O2_g": 8.838052554822744e-08,
    }

    print(solution.output_solution())

    assert solution.isclose(target_dict)


@pytest.mark.skip(reason="Cannot get vmap to work as desired")
def test_reaction_network_vmap() -> None:

    H2O_g: GasSpecies = GasSpecies(
        "H2O", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()
    )
    H2_g: GasSpecies = GasSpecies("H2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell())
    O2_g: GasSpecies = GasSpecies("O2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell())

    species: Species = Species([H2O_g, H2_g, O2_g])
    planet = Planet(surface_temperature=2000)

    reaction_network = ReactionNetwork(species=species, planet=planet)

    constraints_batch: list[SystemConstraints] = [
        SystemConstraints(
            (
                FugacityConstraint(H2O_g, 0.2570770067190733),
                FugacityConstraint(O2_g, 8.838043080858959e-08),
            )
        ),
        SystemConstraints(
            (
                FugacityConstraint(H2O_g, 0.2570770067190733),
                FugacityConstraint(O2_g, 8.838043080858959e-08),
            ),
        ),
    ]

    # Vectorize the solver method using jax.vmap with PyTree and static args
    vmap_solve = jax.vmap(reaction_network.solve_optimistix)

    # constraints_batch = jax.tree_map(lambda x: jnp.stack(x, axis=0), constraints_batch)

    vmap_solve(constraints=constraints_batch)

    # sol, solution = reaction_network.solve_optimistix(constraints=constraints)

    # target_dict = {
    #    "H2O_g": 0.25707719341563373,
    #    "H2_g": 0.249646956461615,
    #    "O2_g": 8.838052554822744e-08,
    # }

    # print(solution.output_solution())

    # assert solution.isclose(target_dict)
