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
from scipy.optimize import root

from atmodeller import __version__, debug_logger
from atmodeller.constraints import FugacityConstraint, SystemConstraints
from atmodeller.core import GasSpecies, Planet
from atmodeller.interior_atmosphere import Species
from atmodeller.reaction_network_jax import ReactionNetwork
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

    temperature = 1500

    planet = Planet()
    constraints: SystemConstraints = SystemConstraints(
        [FugacityConstraint(H2O_g, 10), FugacityConstraint(H2_g, 10)]
    )

    reaction_network = ReactionNetwork(species)
    reaction_network.temperature = temperature
    reaction_network.planet = planet
    reaction_network.constraints = constraints

    test_eval = jnp.array([25.0, 27.0, 24.0])

    residual = reaction_network.get_residual_jax(test_eval)
    logger.warning("residual = %s", residual)

    jacob = jax.jacobian(reaction_network.get_residual_jax)
    jacob_eval = jacob(test_eval)
    logger.warning("jacob_eval = %s", jacob_eval)

    # Define a wrapper function for scipy.optimize.root
    def residual_wrapper(x):
        # return jax.numpy.array(reaction_network.get_residual_jax(x))
        return reaction_network.get_residual_jax(x)

    sol = root(residual_wrapper, test_eval, method="hybr", jac=jacob)
    print(sol)
