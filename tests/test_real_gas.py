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
"""Tests for systems with real gases"""

# Want to use chemistry symbols so pylint: disable=invalid-name

import logging

import numpy as np
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, __version__, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import Planet, Species
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import RealGasProtocol, SolubilityProtocol
from atmodeller.solubility.library import get_solubility_models
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer, RedoxBufferProtocol
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""
SCALING: float = AVOGADRO
"""Scale the numerical problem from molecules/m^3 to moles/m^3 if SCALING is AVODAGRO"""
TAU: float = 1.0e60
"""Tau scaling factor for species stability"""

INITIAL_NUMBER_DENSITY: float = 60.0
"""Initial number density"""
INITIAL_STABILITY: float = -100.0
"""Initial stability"""

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()
eos_models: dict[str, RealGasProtocol] = get_eos_models()


def test_holley(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_beattie_holley58_bounded"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g)
    # Temperature is within the range of the Holley model
    planet: Planet = Planet(surface_temperature=900.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {
        "H": h_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = INITIAL_NUMBER_DENSITY * np.ones(
        len(species), dtype=np.float_
    )
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet,
        initial_number_density,
        initial_stability,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    target: dict[str, float] = {
        "H2O_g": 23.113789256592163,
        "H2_g": 67.28471475999238,
        "O2_g": 1.1643429977602485e-24,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    surface_temperature = 3400.0  # kelvin
    planet_mass = 4.6 * 5.972e24  # kg
    surface_radius = 1.5 * 6371000  # metre
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer(-4)}

    h_kg: float = 0.01 * planet_mass
    si_kg: float = 0.1459 * planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg}

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = INITIAL_NUMBER_DENSITY * np.ones(
        len(species), dtype=np.float_
    )
    # For this case, reducing the fO2 is required for the solver to latch onto the solution
    initial_number_density[2] *= -1
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet,
        initial_number_density,
        initial_stability,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    print(solution)

    # target: dict[str, float] = {
    #     "H2O_g": 0.25708003342259883,
    #     "H2_g": 0.2491577248312551,
    #     "O2_g": 8.83804258138063e-08,
    # }

    # assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_simple(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_beattie_holley58"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")  # , activity=eos_models["O2_beattie_holley58"])
    # SiO_g: Species = Species.create_gas("SiO_g")
    # SiH4_g: Species = Species.create_gas("SiH4_g")
    # SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g)  # , SiH4_g, SiO_g, SiO2_l)

    surface_temperature = 1000.0  # kelvin
    planet_mass = 4.6 * 5.972e24  # kg
    surface_radius = 1.5 * 6371000  # metre
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer(-4)}
    # fugacity_constraints = None

    # oceans: ArrayLike = 1
    # h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    # o_kg: ArrayLike = h_kg
    h_kg: float = 0.001 * planet_mass
    # si_kg: float = 0.1459 * planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}  # , "O": o_kg}  # , "Si": si_kg}

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = INITIAL_NUMBER_DENSITY * np.ones(
        len(species), dtype=np.float_
    )
    # For this case, reducing the fO2 is required for the solver to latch onto the solution
    initial_number_density[2] *= -1
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet,
        initial_number_density,
        initial_stability,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    print(solution)

    # target: dict[str, float] = {
    #     "H2O_g": 0.25708003342259883,
    #     "H2_g": 0.2491577248312551,
    #     "O2_g": 8.83804258138063e-08,
    # }

    # assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)
