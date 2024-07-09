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
"""Tricky condensate cases involving both condensed C and H2O"""

# Want to use chemistry symbols so pylint: disable=invalid-name

from __future__ import annotations

import logging

from atmodeller import __version__, debug_logger
from atmodeller.constraints import ElementMassConstraint, SystemConstraints
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species

logger: logging.Logger = debug_logger()

# TOLERANCE: float = 5.0e-2
# """Tolerance"""


def test_tricky1(helper) -> None:
    """Tricky test 1"""

    equilibrium_temperature = 280
    mantle_mass = 2.912e24
    planet_mass = mantle_mass / (1 - 0.295334691460966)
    trappist1e = Planet(
        surface_temperature=equilibrium_temperature,
        planet_mass=planet_mass,
        surface_radius=5.861e6,
    )

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    # N2_g = GasSpecies("N2")
    # NH3_g = GasSpecies("NH3")
    H2O_l = LiquidSpecies("H2O")
    C_cr = SolidSpecies("C")

    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, C_cr, H2O_l])  # N2_g, NH3_g

    system = InteriorAtmosphereSystem(species=species, planet=trappist1e)

    # Original values of problem case
    c_kg = 3.01336e20
    h_kg = 1.23636e20
    o_kg = 2.3257e21
    # n_kg = 4.57934e18

    # Tweak o mass to try and get convergence. Larger values don't improve the residual, but
    # reducing O content finds a solution
    o_kg = o_kg * 0.75

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("C", c_kg),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("O", o_kg),
            # ElementMassConstraint("N", n_kg),
        ]
    )

    system.solve(constraints, factor=10, attempts=10)  # , method="lm")
    system.output("test_tricky1", to_excel=True)
