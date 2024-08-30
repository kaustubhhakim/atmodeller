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
import numpy.typing as npt
import pandas as pd
from jax import Array

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ElementMassConstraint,
    FugacityConstraint,
    PressureConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies, Planet, SolidSpecies, Species
from atmodeller.initial_solution import (
    FILL_LOG10_NUMBER_DENSITY,
    FILL_LOG10_STABILITY,
    InitialSolutionDict,
    InitialSolutionLast,
    InitialSolutionProtocol,
    InitialSolutionRegressor,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
from atmodeller.solver import SolverOptimistix
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
    """Tests a dict with no arguments"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("CO")
    species: Species = Species([H2O_g, H2_g, O2_g])

    constraints: SystemConstraints = SystemConstraints([])

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(species=species, planet=planet)
    result: Array = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )

    assert np.all(result == FILL_LOG10_NUMBER_DENSITY)


def test_constraint_fill_dict():
    """Tests a dict with no arguments with a number density fill value."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    species: Species = Species([H2O_g, H2_g, CO_g])

    constraints: SystemConstraints = SystemConstraints([])
    fill_log10_number_density: float = 20

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        species=species, planet=planet, fill_log10_number_density=fill_log10_number_density
    )
    result: Array = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.array([20, 20, 20])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.all(result == fill_log10_number_density)


def test_args_constraint_fill_dict():
    """Tests a dict with arguments and a number density fill value."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    species: Species = Species([H2O_g, H2_g, CO_g, CO2_g])

    constraints: SystemConstraints = SystemConstraints([])
    CO2_g_value: float = 1e24
    H2_g_value: float = 1e25
    fill_log10_number_density: float = 26

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        {CO_g: CO2_g_value, H2_g: H2_g_value},
        species=species,
        planet=planet,
        fill_log10_number_density=fill_log10_number_density,
    )
    result: Array = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.array(
        [
            fill_log10_number_density,
            np.log10(H2_g_value),
            np.log10(CO2_g_value),
            fill_log10_number_density,
        ]
    )

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_args_fill_stability_dict():
    """Tests a dict with arguments and an activity fill value for condensed phases."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")
    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])

    constraints: SystemConstraints = SystemConstraints([])
    CO_g_value: float = 1e24
    H2_g_value: float = 1e25
    stability_C_cr_value: float = 2
    H2O_l_value: float = 1e22
    fill_log10_activity: float = np.log10(0.8)

    initial_solution: InitialSolutionProtocol = InitialSolutionDict(
        {
            CO_g: CO_g_value,
            H2_g: H2_g_value,
            "stability_C_cr": stability_C_cr_value,
            H2O_l: H2O_l_value,
        },
        species=species,
        planet=planet,
        fill_log10_activity=fill_log10_activity,
    )

    result: Array = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.array(
        [
            FILL_LOG10_NUMBER_DENSITY,
            np.log10(H2_g_value),
            FILL_LOG10_NUMBER_DENSITY,
            np.log10(CO_g_value),
            FILL_LOG10_NUMBER_DENSITY,
            FILL_LOG10_NUMBER_DENSITY,
            fill_log10_activity,
            np.log10(H2O_l_value),
            FILL_LOG10_STABILITY,
            fill_log10_activity,
            FILL_LOG10_NUMBER_DENSITY,
            np.log10(stability_C_cr_value),
        ]
    )

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_last_solution():
    """Tests an initial solution based on the last solution"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    species: Species = Species([H2_g, H2O_g, O2_g])

    constraints: SystemConstraints = SystemConstraints([])

    initial_solution: InitialSolutionProtocol = InitialSolutionLast(species=species, planet=planet)

    # The first initial condition will return the fill value for the pressure
    result: Array = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.full(3, FILL_LOG10_NUMBER_DENSITY, dtype=np.float_)

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)

    # This is the same as test_H_O in test_benchmark.py
    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 6.25774e20

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("O", o_kg),
        ]
    )

    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )

    # Following the solve we test that the initial condition returns the previous solution
    _ = interior_atmosphere.solve(
        solver=SolverOptimistix(), constraints=constraints, initial_solution=initial_solution
    )

    result: Array = initial_solution.get_log10_value(
        constraints, temperature=planet.surface_temperature, pressure=dummy_variable
    )
    target: npt.NDArray[np.float_] = np.array([26.42725976, 26.44234405, 17.50993722])

    logger.debug("result = %s", result)
    logger.debug("target = %s", target)

    assert np.allclose(result, target, rtol=RTOL, atol=ATOL)


def test_regressor(generate_regressor_data):
    """Tests the basic functionality of the initial solution regressor"""

    interior_atmosphere, filename = generate_regressor_data

    species: Species = interior_atmosphere.species
    O2_g: GasSpecies = species.get_species_from_name("O2_g")  # type: ignore (is GasSpecies)
    H2O_g: GasSpecies = species.get_species_from_name("H2O_g")  # type: ignore (is GasSpecies)

    dataframes: dict[str, pd.DataFrame] = interior_atmosphere.output(to_dataframes=True)
    raw_solution: pd.DataFrame = dataframes["raw_solution"]
    solution: pd.DataFrame = dataframes["solution"]

    initial_solution: InitialSolutionProtocol = InitialSolutionRegressor.from_pickle(
        filename, species=species, planet=interior_atmosphere.planet
    )

    for index in [0, 5, 10, 15, 20, 25, 30, 35]:

        test_constraints: SystemConstraints = SystemConstraints(
            [
                FugacityConstraint(O2_g, solution.iloc[index]["O2_g_bar"]),
                PressureConstraint(H2O_g, solution.iloc[index]["H2O_g_bar"]),
            ]
        )

        result: Array = initial_solution.get_log10_value(
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

    species: Species = interior_atmosphere.species
    O2_g: GasSpecies = species.get_species_from_name("O2_g")  # type: ignore (is GasSpecies)
    H2O_g: GasSpecies = species.get_species_from_name("H2O_g")  # type: ignore (is GasSpecies)
    H2_g: GasSpecies = species.get_species_from_name("H2_g")  # type: ignore (is GasSpecies)

    dataframes: dict[str, pd.DataFrame] = interior_atmosphere.output(to_dataframes=True)
    raw_solution: pd.DataFrame = dataframes["raw_solution"]
    solution: pd.DataFrame = dataframes["solution"]

    # For the override we must restrict the species to the specific species we want to override
    number_density_H2: float = 1e30
    species_H2 = species.get_species_from_name("H2_g")
    species_only_H2 = Species([species_H2])
    solution_override = InitialSolutionDict(
        {H2_g: number_density_H2}, species=species_only_H2, planet=planet
    )

    initial_solution = InitialSolutionRegressor.from_pickle(
        filename, species=species, planet=planet, solution_override=solution_override
    )

    test_constraints = SystemConstraints(
        [
            FugacityConstraint(O2_g, solution.iloc[0]["O2_g_bar"]),
            PressureConstraint(H2O_g, solution.iloc[0]["H2O_g_bar"]),
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
