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
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    FugacityConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, Planet
from atmodeller.interior_atmosphere import Species
from atmodeller.reaction_network import (
    ReactionNetwork,
    ReactionNetworkWithCondensateStability,
)
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

planet: Planet = Planet()


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


def test_H_fugacities() -> None:
    """Tests H species with imposed fugacities"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(H2O_g, 0.2570770067190733),
            FugacityConstraint(O2_g, 8.838043080858959e-08),
        )
    )
    reaction_network = ReactionNetworkWithCondensateStability(species=species, planet=planet)
    _, _, solution = reaction_network.solve_optimistix(constraints=constraints)

    target_dict = {
        "H2O_g": 0.257077006719072,
        "H2_g": 0.24964688044710262,
        "O2_g": 8.838043080858959e-08,
    }

    assert solution.isclose(target_dict, rtol=RTOL, atol=ATOL)


def test_H_with_buffer() -> None:
    """Tests H species with an imposed fO2 buffer"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(H2O_g, 0.2570770067190733),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        )
    )
    reaction_network = ReactionNetworkWithCondensateStability(species=species, planet=planet)
    _, _, solution = reaction_network.solve_optimistix(constraints=constraints)

    target_dict = {
        "H2O_g": 0.257077006719072,
        "H2_g": 0.24964688044710262,
        "O2_g": 8.838043080858887e-08,
    }

    assert solution.isclose(target_dict, rtol=RTOL, atol=ATOL)


def test_H_and_C_no_solubility() -> None:
    """Tests H2-H2O and CO-CO2 with imposed fugacities and no solubility.

    This test is based on test_C_and_H() in test_CHO.py but without solubility.
    """

    H2O_g: GasSpecies = GasSpecies("H2O")  # , solubility=H2O_peridotite_sossi())
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")  # , solubility=CO2_basalt_dixon())

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(CO_g, 26.625148913955194),
            FugacityConstraint(H2O_g, 99.19769919121012),
            FugacityConstraint(O2_g, 8.981953412412735e-08),
        ]
    )
    reaction_network = ReactionNetworkWithCondensateStability(species=species, planet=planet)
    _, _, solution = reaction_network.solve_optimistix(constraints=constraints)

    target_dict = {
        "CO2_g": 6.0622728258770024,
        "CO_g": 26.625148913955194,
        "H2O_g": 99.19769919121012,
        "H2_g": 95.55582495038334,
        "O2_g": 8.981953412412735e-08,
    }

    assert solution.isclose(target_dict, rtol=RTOL, atol=ATOL)


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
