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
"""Tests for new JAX"""

# Convenient to use naming convention so pylint: disable=C0103

import logging

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, debug_logger
from atmodeller.classes import ReactionNetwork
from atmodeller.jax_containers import (
    CH4_g,
    CO2_g,
    CO_g,
    Constraints,
    H2_g,
    H2O_g,
    H2O_l,
    O2_g,
    Parameters,
    Planet,
    Solution,
    SpeciesData,
)
from atmodeller.jax_engine import solve
from atmodeller.jax_utilities import (
    pytrees_stack,
    scale_number_density,
    unscale_number_density,
)
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

scaling: float = AVOGADRO
log_scaling: Array = np.log(scaling)

# Initial solution guess number density (molecules/m^3)
initial_number_density: ArrayLike = np.array([60, 60, 30, -60, 60, 30], dtype=np.float_)
initial_number_density = scale_number_density(initial_number_density, log_scaling)
logger.debug("initial_number_density = %s", initial_number_density)

# Stability is a non-dimensional quantity
initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
logger.debug("initial_stability = %s", initial_stability)


def test_CHO_low_temperature() -> None:
    """C-H-O system at 450 K"""

    species_list: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]
    reaction_network: ReactionNetwork = ReactionNetwork()
    formula_matrix: Array = jnp.array(reaction_network.formula_matrix(species_list))
    reaction_matrix: Array = jnp.array(reaction_network.reaction_matrix(species_list))

    planet: Planet = Planet(surface_temperature=450)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg
    o_kg: float = 1.02999e20

    # Unscaled total molecules constraints in alphabetical order
    mass_constraints = {
        "C": c_kg,  # 10**45.89051326565627,
        "H": h_kg,  # 10**46.96664792007732,
        "O": o_kg,  # 10**45.58848007858896,
    }
    constraints: Constraints = Constraints.create(species_list, mass_constraints)

    solution: Solution = Solution(initial_number_density, initial_stability)  # type: ignore

    parameters: Parameters = Parameters(
        formula_matrix, reaction_matrix, species_list, planet, constraints, scaling
    )

    # Pre-compile
    # solve(solution, parameters).value.block_until_ready()

    sol = solve(solution, parameters)

    if optx.RESULTS[sol.result] == "":
        out: Array = sol.value
        logger.info("Optimistix success with steps = %d", sol.stats["num_steps"])

    number_density, stability = jnp.split(out, 2)
    number_density = unscale_number_density(number_density, log_scaling)

    out = jnp.concatenate((number_density, stability))
    logger.debug("solution = %s", out)

    target: npt.NDArray = np.array(
        [
            62.05652013342668,
            60.120022576862524,
            26.01213296322353,
            -64.35958121411358,
            60.81547969099319,
            22.19204281838283,
        ]
    )

    # factsage_result: dict[str, float] = {
    #     "H2_g": 55.475,
    #     "H2O_g": 8.0,
    #     "CO2_g": 1.24e-14,
    #     "O2_g": 7.85e-54,
    #     "CH4_g": 16.037,
    #     "CO_g": 2.12e-16,
    # }

    isclose: np.bool_ = np.isclose(target, number_density, rtol=RTOL, atol=ATOL).all()

    assert isclose
