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
"""Tests for the initial solution"""

# Convenient to use naming convention so pylint: disable=C0103

import logging

import numpy as np

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ElementMassConstraint,
    FugacityConstraint,
    PressureConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies, Planet, SolidSpecies, Species
from atmodeller.initial_solution import (
    InitialSolutionDict,
    InitialSolutionLast,
    InitialSolutionRegressor,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
REGRESSORTOL: float = 8e-2
"""Tolerance for testing the regressor output"""

dummy_variable: float = 1
"""Dummy variable used for pressure arguments when they are not used internally"""

planet = Planet()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_default_dict():
    """Tests a dict with no arguments and no constraints"""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("CO")
    species = Species([H2O_g, H2_g, O2_g])

    constraints: SystemConstraints = SystemConstraints([])

    initial_solution = InitialSolutionDict(species=species)
    result = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )

    assert np.all(result == 26)


def test_constraint_fill_dict():
    """Tests a dict with no arguments, but with constraints and a number density fill value."""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    CO_g = GasSpecies("CO")
    species = Species([H2O_g, H2_g, CO_g])

    constraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution = InitialSolutionDict(species=species, fill_log10_number_density=20)
    result = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target = np.array([25.25785673, 20, 20])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_args_constraint_fill_dict():
    """Tests a dict with arguments, constraints, and a number density fill value."""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    species = Species([H2O_g, H2_g, CO_g, CO2_g])

    constraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution = InitialSolutionDict(
        {CO_g: 1e24, H2_g: 1e25}, species=species, fill_log10_number_density=26
    )
    result = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target = np.array([25.25785673, 25, 24, 26])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_args_fill_stability_dict():
    """Tests a dict with arguments and an activity fill value for condensed phases."""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    H2O_l = LiquidSpecies("H2O")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    C_cr = SolidSpecies("C")
    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])

    constraints = SystemConstraints([])

    initial_solution = InitialSolutionDict(
        {CO_g: 1e24, H2_g: 1e25, "stability_C_cr": 2, H2O_l: 1e22},
        species=species,
        fill_log10_activity=np.log10(0.8),
    )

    result = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target = np.array([26, 25, 26, 24, 26, 26, 0, 22, -15, 0, 26, 0.30103])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_last_solution():
    """Tests an initial solution based on the last solution"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    species: Species = Species([H2_g, H2O_g, O2_g])

    constraints = SystemConstraints([])

    initial_solution = InitialSolutionLast(species=species)

    # The first initial condition will return the fill value for the pressure
    result = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target = np.array([26, 26, 26])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)

    # This is the same as test_H_O in test_benchmark.py
    oceans = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 6.25774e20

    constraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("O", o_kg),
        ]
    )

    system = InteriorAtmosphereSystem(species=species, planet=planet)

    # Following the solve we test that the initial condition returns the previous solution
    system.solve(constraints=constraints, initial_solution=initial_solution)

    result = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target = np.array([26.42725976, 26.44234405, 17.50993722])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_regressor(generate_regressor_data):
    """Tests the basic functionality of the initial solution regressor"""

    interior_atmosphere, filename = generate_regressor_data

    species = interior_atmosphere.species
    O2_g = species.get_species_from_name("O2_g")
    H2O_g = species.get_species_from_name("H2O_g")

    dataframes = interior_atmosphere.output(to_dataframes=True)
    raw_solution = dataframes["raw_solution"]
    solution = dataframes["solution"]

    initial_solution = InitialSolutionRegressor.from_pickle(filename, species=species)

    for index in [0, 5, 10, 15, 20, 25, 30, 35]:

        test_constraints = SystemConstraints(
            [
                FugacityConstraint(O2_g, solution.iloc[index]["O2_g"]),
                PressureConstraint(H2O_g, solution.iloc[index]["H2O_g"]),
            ]
        )

        result = initial_solution.get_log10_value(
            test_constraints, temperature=planet.surface_temperature, pressure=dummy_variable
        )

        assert np.isclose(
            raw_solution.iloc[index]["H2O_g"], result[0], atol=REGRESSORTOL, rtol=REGRESSORTOL
        )
        assert np.isclose(
            raw_solution.iloc[index]["H2_g"], result[1], atol=REGRESSORTOL, rtol=REGRESSORTOL
        )
        assert np.isclose(
            raw_solution.iloc[index]["O2_g"], result[2], atol=REGRESSORTOL, rtol=REGRESSORTOL
        )


def test_regressor_override(generate_regressor_data):
    """Tests the regressor with an override option"""

    interior_atmosphere, filename = generate_regressor_data

    species = interior_atmosphere.species
    O2_g = species.get_species_from_name("O2_g")
    H2O_g = species.get_species_from_name("H2O_g")
    H2_g = species.get_species_from_name("H2_g")

    dataframes = interior_atmosphere.output(to_dataframes=True)
    raw_solution = dataframes["raw_solution"]
    solution = dataframes["solution"]

    # For the override we must restrict the species to the specific species we want to override
    number_density_H2: float = 1e30
    species_H2 = species.get_species_from_name("H2_g")
    species_only_H2 = Species([species_H2])
    solution_override = InitialSolutionDict({H2_g: number_density_H2}, species=species_only_H2)

    initial_solution = InitialSolutionRegressor.from_pickle(
        filename, species=species, solution_override=solution_override
    )

    test_constraints = SystemConstraints(
        [
            FugacityConstraint(O2_g, solution.iloc[0]["O2_g"]),
            PressureConstraint(H2O_g, solution.iloc[0]["H2O_g"]),
        ]
    )

    result = initial_solution.get_log10_value(
        test_constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )

    assert np.isclose(
        raw_solution.iloc[0]["H2O_g"], result[0], atol=REGRESSORTOL, rtol=REGRESSORTOL
    )
    assert np.isclose(np.log10(number_density_H2), result[1], atol=REGRESSORTOL, rtol=REGRESSORTOL)
    assert np.isclose(
        raw_solution.iloc[0]["O2_g"], result[2], atol=REGRESSORTOL, rtol=REGRESSORTOL
    )
