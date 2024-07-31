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
    ActivityConstraint,
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
from atmodeller.utilities import earth_oceans_to_kg

logger: logging.Logger = debug_logger()

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
REGRESSORTOL: float = 8e-2
"""Tolerance for testing the regressor output"""

dummy_variable: float = 1
"""Dummy variable used for temperature and pressure arguments when they are not used internally"""

planet = Planet()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_no_args_no_constraints_dict():
    """Tests a dict with no arguments and no constraints"""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("CO")
    species = Species([H2O_g, H2_g, O2_g])

    constraints: SystemConstraints = SystemConstraints([])

    initial_solution = InitialSolutionDict(species=species, planet=planet)
    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )

    assert np.all(result == 26)


def test_no_args_with_constraints_dict():
    """Tests a dict with no arguments, but with constraints and a pressure fill value."""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    CO_g = GasSpecies("CO")
    species = Species([H2O_g, H2_g, CO_g])

    constraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution = InitialSolutionDict(
        species=species, planet=planet, fill_log10_number_density=20
    )
    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target = np.array([28.55888672, 20, 20])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_with_args_with_constraints_dict():
    """Tests a dict with arguments, constraints, and a pressure fill value."""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    species = Species([H2O_g, H2_g, CO_g, CO2_g])

    constraints = SystemConstraints([PressureConstraint(H2O_g, 5)])

    initial_solution = InitialSolutionDict(
        {CO_g: 1e24, H2_g: 1e25}, species=species, planet=planet, fill_log10_number_density=26
    )
    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target = np.array([28.55888672, 25, 24, 26])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_with_args_cond_dict():
    """Tests a dict with arguments, no constraints, for condensed phases

    In particular, this tests that activity arguments (and fill values) are applied correctly.
    """

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
        {CO_g: 1e24, H2_g: 1e25, C_cr: 0.9},
        species=species,
        planet=planet,
        fill_log10_activity=np.log10(0.8),
    )

    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target = np.array([26, 25, 26, 24, 26, 26, -0.09691001, 26, -10, -0.04575749, 22, -37])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_with_args_constraints_cond_dict():
    """Tests a dict with arguments, constraints, for condensed phases

    In particular, this tests that activity constraints are applied correctly.
    """

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    H2O_l = LiquidSpecies("H2O")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    C_cr = SolidSpecies("C")
    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])

    constraints = SystemConstraints([ActivityConstraint(C_cr, 0.7)])

    initial_solution = InitialSolutionDict(
        {CO_g: 1e24, H2_g: 1e25}, species=species, planet=planet, fill_log10_activity=np.log10(0.8)
    )

    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target = np.array([22, 25, 22, 24, 22, 22, -0.09691001, 22, -37, -0.15490196, 22, -37])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_with_stability_constraints_cond_dict():
    """Tests a dict with arguments, constraints, for condensed phases

    In particular, this tests that activity constraints are applied correctly.
    """

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    H2O_l = LiquidSpecies("H2O")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    C_cr = SolidSpecies("C")
    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])

    constraints = SystemConstraints([ActivityConstraint(C_cr, 0.7)])

    initial_solution = InitialSolutionDict(
        {CO_g: 1e24, H2_g: 1e25, "stability_C_cr": 2, "mass_H2O_l": 1e22},
        species=species,
        planet=planet,
        fill_log10_activity=np.log10(0.8),
    )

    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target = np.array([22, 25, 22, 24, 22, 22, -0.09691001, 22, -37, -0.15490196, 22, 0.30103])

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

    initial_solution = InitialSolutionLast(species=species, planet=planet)

    # The first initial condition will return the fill value for the pressure
    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
    )
    target = np.array([22, 22, 22])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)

    # This is the same as test_H_O in test_benchmark.py
    oceans = 1
    h_kg: float = earth_oceans_to_kg(oceans)
    o_kg: float = 6.25774e20

    constraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("O", o_kg),
        ]
    )

    system = InteriorAtmosphereSystem(species=species, planet=planet)

    # Following the solve we test that the initial condition returns the previous solution
    system.solve(constraints, initial_solution=initial_solution)

    result = initial_solution.get_log10_value(
        constraints, temperature=dummy_variable, pressure=dummy_variable
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

    for index in [0]:  # , 5, 10, 15, 20, 25, 30, 35]:

        test_constraints = SystemConstraints(
            [
                FugacityConstraint(O2_g, solution.iloc[index]["O2_g"]),
                PressureConstraint(H2O_g, solution.iloc[index]["H2O_g"]),
            ]
        )

        result = initial_solution.get_log10_value(
            test_constraints, temperature=dummy_variable, pressure=dummy_variable
        )

        print(raw_solution.iloc[0]["H2O_g"])
        print(result[0])

        assert np.isclose(
            raw_solution.iloc[index]["H2O_g"], result[0], atol=REGRESSORTOL, rtol=REGRESSORTOL
        )
        # assert np.isclose(
        #    raw_solution.iloc[index]["H2_g"], result[1], atol=REGRESSORTOL, rtol=REGRESSORTOL
        # )
        # assert np.isclose(
        #    raw_solution.iloc[index]["O2_g"], result[2], atol=REGRESSORTOL, rtol=REGRESSORTOL
        # )


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

    solution_override = InitialSolutionDict({H2_g: 100000}, species=species, planet=planet)
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
        test_constraints, temperature=dummy_variable, pressure=dummy_variable
    )

    assert np.isclose(
        raw_solution.iloc[0]["H2O_g"], result[0], atol=REGRESSORTOL, rtol=REGRESSORTOL
    )
    assert np.isclose(5, result[1], atol=REGRESSORTOL, rtol=REGRESSORTOL)
    assert np.isclose(
        raw_solution.iloc[0]["O2_g"], result[2], atol=REGRESSORTOL, rtol=REGRESSORTOL
    )
