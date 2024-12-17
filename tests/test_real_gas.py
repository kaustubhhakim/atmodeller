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

import logging

import numpy as np
import optimistix as optx
import pytest
from jax.typing import ArrayLike

from atmodeller import INITIAL_LOG_STABILITY, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import ConstantFugacityConstraint, Planet, SolverParameters, Species
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import (
    FugacityConstraintProtocol,
    RealGasProtocol,
    RedoxBufferProtocol,
    SolubilityProtocol,
)
from atmodeller.output import Output
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import OptxSolver, earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()
eos_models: dict[str, RealGasProtocol] = get_eos_models()


def test_fO2_holley(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`"""

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_beattie_holley58"])
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

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 32.77037875523393,
        "H2_g": 71.50338102110962,
        "O2_g": 1.525466019972294e-21,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Superseded by the test for a sub-Neptune")
def test_chabrier_earth(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
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
    initial_log_number_density: ArrayLike = 65 * np.ones(len(species), dtype=np.float_)
    # For this case, tweaking the initial condition is necessary to latch onto the solution
    initial_log_number_density[2] = 45
    initial_log_number_density[4] = 60

    initial_log_stability: ArrayLike = INITIAL_LOG_STABILITY * np.ones_like(
        initial_log_number_density
    )

    interior_atmosphere.initialise_solve(
        planet=planet,
        initial_log_number_density=initial_log_number_density,
        initial_log_stability=initial_log_stability,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

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

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Fails. Might require improved initial condition or bounding")
def test_chabrier_earth_dogleg(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` using the dogleg solver"""

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    planet: Planet = Planet(surface_temperature=3400.0)

    solver: OptxSolver = optx.Dogleg(rtol=1.0e-8, atol=1.0e-8)
    solver_parameters: SolverParameters = SolverParameters(solver)

    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    interior_atmosphere.initialise_solve(
        planet=planet, mass_constraints=mass_constraints, solver_parameters=solver_parameters
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

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

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune

    It is recommended to impose mass constraints for all species and not to impose a buffered
    oxygen fugacity. This is because a redox buffer depends on total pressure, which is dictated by
    the atmospheric speciation and size. Hence for a given buffer choice, particularly for large
    pressure ranges, there can be multiple atmospheric structures that satisfy the fO2 constraint
    imposed by the redox buffer, with each structure possessing a different total reservoir of
    oxygen. If instead the oxygen mass is constrained then there is only one physical solution and
    root, which is preferred for robust numerical solution. Remember that the oxygen fugacity shift
    relative to the iron-wustite buffer is back-calculated for the output.

    This case effectively saturates the maximum allowable log number density at a value of 70
    based on the default hypercube that brackets the solution (see LOG_NUMBER_DENSITY_UPPER).
    This is fine for a test, but this test is not physically realistic because solubilities are
    ignored, which would greatly lower the pressure and hence the number density.
    """

    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas("H2O_g")
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    surface_temperature = 3400.0  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 429506.99705368624,
        "H2O_g_activity": 429506.99705368624,
        "H2_g": 3.0474730096539067,
        "H2_g_activity": 19509.066519228905,
        "H4Si_g": 0.0006959073713891908,
        "H4Si_g_activity": 0.0006959073713891908,
        "O2Si_l": 449791.006799111,
        "O2Si_l_activity": 1.0,
        "O2_g": 10.456415851163248,
        "O2_g_activity": 10.456415851163175,
        "OSi_g": 0.8250141324194714,
        "OSi_g_activity": 0.8250141324194655,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune_batch(helper) -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    H2O solubility is also included.

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """
    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_chabrier21"])
    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    O2_g: Species = Species.create_gas("O2_g")
    SiO_g: Species = Species.create_gas("SiO_g")
    SiH4_g: Species = Species.create_gas("SiH4_g")
    SiO2_l: Species = Species.create_condensed("SiO2_l")

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, SiH4_g, SiO_g, SiO2_l)

    surface_temperature = 3400.0  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        surface_temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    # Some pertinent output here for testing, no need to specify all the species
    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([34153.77081497173, 34647.13320000063, 34773.46401215592]),
        "H2_g": np.array([2.003779275377057, 0.170286601371338, 0.027748065184473]),
        "H2_g_activity": np.array([26.950066680710645, 14.769839969937074, 11.271846166572525]),
        "O2_g": np.array([34647.5851511399, 118713.0248545309, 205315.21994142563]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_pH2_fO2_real_gas(helper) -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """
    H2O_g: Species = Species.create_gas(
        "H2O_g",
        solubility=solubility_models["H2O_peridotite_sossi23"],
        activity=eos_models["H2O_cork_holland98"],
    )
    H2_g: Species = Species.create_gas("H2_g", activity=eos_models["H2_cork_cs_holland91"])
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        O2_g.name: ConstantFugacityConstraint(1.0453574209588085e-07),
        # Gives a H2 partial pressure of around 1000 bar
        H2_g.name: ConstantFugacityConstraint(1493.1),
    }

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 1470.2567650857518,
        "H2_g": 999.9971214963639,
        "O2_g": 1.045357420958815e-07,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Takes a bit longer to compile and run than other models")
def test_H_and_C_real_gas(helper) -> None:
    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

    H2_g: Species = Species.create_gas(
        "H2_g",
        solubility=solubility_models["H2_basalt_hirschmann12"],
        activity=eos_models["H2_cork_cs_holland91"],
    )
    H2O_g: Species = Species.create_gas(
        "H2O_g",
        solubility=solubility_models["H2O_peridotite_sossi23"],
        activity=eos_models["H2O_cork_holland98"],
    )
    O2_g: Species = Species.create_gas("O2_g")
    CO_g: Species = Species.create_gas("CO_g", activity=eos_models["CO_cork_cs_holland91"])
    CO2_g: Species = Species.create_gas(
        "CO2_g",
        solubility=solubility_models["CO2_basalt_dixon95"],
        activity=eos_models["CO2_cork_holland98"],
    )
    CH4_g: Species = Species.create_gas("CH4_g", activity=eos_models["CH4_cork_cs_holland91"])

    species: tuple[Species, ...] = (H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        H2_g.name: ConstantFugacityConstraint(958.0),
        O2_g.name: ConstantFugacityConstraint(1.0132255325169718e-07),
    }

    oceans: ArrayLike = 10.0
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "CH4_g": 11.612216513852113,
        "CO2_g": 67.19430723726805,
        "CO_g": 276.8796027243849,
        "H2O_g": 955.2659883622448,
        "H2_g": 694.2030982090682,
        "O2_g": 1.0132255325169676e-07,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)
