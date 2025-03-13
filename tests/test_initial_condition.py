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
"""Tests for C-H-O systems"""

import copy
import logging

import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import Planet, Species, SpeciesCollection
from atmodeller.interfaces import FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.output import Output
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()


def test_H_fO2_batch_temperature(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of surface temperatures.

    Ensures that the solution from a solve can be used to initialise a batch solve.

    The assert condition is not really testing much. It would be better to test that the number
    of solves is reduced to one or something similar. Ultimately if this test passes it at least
    means that a previous solution can be accessed and used to initialise a batch solve.
    """

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: SpeciesCollection = SpeciesCollection((H2O_g, H2_g, O2_g))
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    # Number of surface temperatures is different to number of species to test array shapes work.
    surface_temperatures: npt.NDArray[np.float_] = np.array(
        [1500, 2000, 2500, 3000], dtype=np.float_
    )
    planet: Planet = Planet(surface_temperature=surface_temperatures)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = copy.deepcopy(interior_atmosphere.output)
    # solution: dict[str, ArrayLike] = output.quick_look()
    initial_log_number_density: Array = output.log_number_density

    # We must re-solve because the previously compiled code is expecting a 1-D array for the
    # initial solution, not a batch.
    interior_atmosphere.solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
        initial_log_number_density=initial_log_number_density,
    )
    output_batch: Output = interior_atmosphere.output
    solution_batch: dict[str, ArrayLike] = output_batch.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array(
            [
                2.566653037020448e-01,
                2.570800742364757e-01,
                2.572178041535549e-01,
                2.572746043480848e-01,
            ]
        ),
        "H2_g": np.array(
            [
                3.133632393608037e-01,
                2.491511264610584e-01,
                2.265704456625875e-01,
                2.199521409043987e-01,
            ]
        ),
        "O2_g": np.array(
            [
                2.394194493859141e-12,
                8.838513516896038e-08,
                4.544970468047975e-05,
                2.739422634823809e-03,
            ]
        ),
    }

    assert helper.isclose(solution_batch, target, rtol=RTOL, atol=ATOL)
