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
import optimistix as optx
from jax.typing import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import Planet, SolverParameters, Species
from atmodeller.eos.classes import IdealGas, IdealGas2
from atmodeller.eos.core import RealGas
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer, RedoxBufferProtocol
from atmodeller.utilities import OptxSolver, earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

INITIAL_NUMBER_DENSITY: float = 60.0
"""Initial number density"""
INITIAL_STABILITY: float = -100.0
"""Initial stability"""

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()
eos_models: dict[str, RealGas] = get_eos_models()


# Test is comparable to main branch
def test_fO2_holley(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_beattie_holley58_bounded"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g)
    # Temperature is within the range of the Holley model
    planet: Planet = Planet(surface_temperature=1000.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

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
        "H2O_g": 33.04668613839122,
        "H2_g": 71.59251832343173,
        "O2_g": 1.547107185208223e-21,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21_bounded"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    planet: Planet = Planet(surface_temperature=3400.0)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = 100 * np.ones(len(species), dtype=np.float_)
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet,
        initial_number_density,
        initial_stability,
        mass_constraints=mass_constraints,
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    target: dict[str, float] = {
        "H2O_g": 7113.08023480496,
        "H2O_g_activity": 7113.08023480496,
        "H2_g": 11852.575254137557,
        "H2_g_activity": 249033.31883030405,
        "H4Si_g": 67369.04263803319,
        "H4Si_g_activity": 67369.04263803319,
        "O2Si_l": 92956.79519434669,
        "O2Si_l_activity": 1.0,
        "O2_g": 1.7600128617497757e-05,
        "O2_g_activity": 1.760012861749763e-05,
        "OSi_g": 635.9088815105993,
        "OSi_g_activity": 635.9088815105947,
    }

    # Result from the main branch is basically the same with the difference likely attributable
    # to the thermodynamic data
    # target_main_branch: dict[str, float] = {
    #     "H2O_g": 6906.686657050143,
    #     "H2_g": 12177.449403945378,
    #     "H4Si_g": 67048.93476214791,
    #     "O2Si_l": 1.0,
    #     "O2_g": 1.771347227850254e-05,
    #     "OSi_g": 634.3598055223664,
    #     "mass_O2Si_l": 1.077551466072747e24,
    # }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth_dogleg(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21_bounded"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    planet: Planet = Planet(surface_temperature=3400.0)

    solver: OptxSolver = optx.Dogleg(rtol=1.0e-8, atol=1.0e-8)
    solver_parameters: SolverParameters = SolverParameters(solver)

    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(
        species, solver_parameters=solver_parameters
    )

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = 80 * np.ones(len(species), dtype=np.float_)
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet,
        initial_number_density,
        initial_stability,
        mass_constraints=mass_constraints,
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    target: dict[str, float] = {
        "H2O_g": 7113.08023480496,
        "H2O_g_activity": 7113.08023480496,
        "H2_g": 11852.575254137557,
        "H2_g_activity": 249033.31883030405,
        "H4Si_g": 67369.04263803319,
        "H4Si_g_activity": 67369.04263803319,
        "O2Si_l": 92956.79519434669,
        "O2Si_l_activity": 1.0,
        "O2_g": 1.7600128617497757e-05,
        "O2_g_activity": 1.760012861749763e-05,
        "OSi_g": 635.9088815105993,
        "OSi_g_activity": 635.9088815105947,
    }

    # Result from the main branch is basically the same with the difference likely attributable
    # to the thermodynamic data
    # target_main_branch: dict[str, float] = {
    #     "H2O_g": 6906.686657050143,
    #     "H2_g": 12177.449403945378,
    #     "H4Si_g": 67048.93476214791,
    #     "O2Si_l": 1.0,
    #     "O2_g": 1.771347227850254e-05,
    #     "OSi_g": 634.3598055223664,
    #     "mass_O2Si_l": 1.077551466072747e24,
    # }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune(helper) -> None:

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21_bounded"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    surface_temperature = 3400.0  # kelvin
    # planet_mass = 4.6 * 5.972e24  # kg
    # surface_radius = 1.5 * 6371000  # metre
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        # planet_mass=planet_mass,
        # surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    # fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer(-4)}

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = 70 * np.ones(len(species), dtype=np.float_)
    # For this case, reducing the fO2 is required for the solver to latch onto the solution
    initial_number_density[2] = 40
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet,
        initial_number_density,
        initial_stability,
        # fugacity_constraints=fugacity_constraints,
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
