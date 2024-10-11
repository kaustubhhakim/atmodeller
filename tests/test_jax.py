# #
# # Copyright 2024 Dan J. Bower
# #
# # This file is part of Atmodeller.
# #
# # Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# # General Public License as published by the Free Software Foundation, either version 3 of the
# # License, or (at your option) any later version.
# #
# # Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# # even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# # General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# # see <https://www.gnu.org/licenses/>.
# #
# """Tests for JAX"""

# # Convenient to use naming convention so pylint: disable=C0103

# from __future__ import annotations

# import logging
# from typing import Callable

# import jax
# import jax.numpy as jnp
# from jax import Array

# from atmodeller import __version__, debug_logger
# from atmodeller.constraints import (
#     BufferedFugacityConstraint,
#     ElementMassConstraint,
#     FugacityConstraint,
#     SystemConstraints,
#     TotalPressureConstraint,
# )
# from atmodeller.containers import Planet
# from atmodeller.core import GasSpecies, Species
# from atmodeller.eos.holland import CO_CORK_HP91, H2_CORK_HP91, CO2_CORK_simple_HP91
# from atmodeller.eos.saxena import H2_SF87
# from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
# from atmodeller.solution import Solution
# from atmodeller.solver import SolverOptimistix
# from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
# from atmodeller.utilities import earth_oceans_to_hydrogen_mass

# RTOL: float = 1.0e-8
# """Relative tolerance"""
# ATOL: float = 1.0e-8
# """Absolute tolerance"""

# logger: logging.Logger = debug_logger()
# # logger.setLevel(logging.INFO)

# planet: Planet = Planet()


# def test_H_fO2_buffer() -> None:
#     """Tests H2-H2O at the IW buffer."""

#     O2_g: Species = GasSpecies(O2_g_data)
#     H2_g: Species = GasSpecies(H2_g_data)
#     H2O_g: Species = GasSpecies(H2O_g_data, solubility=H2O_peridotite_sossi)

#     species: list[Species] = [H2O_g, H2_g, O2_g]
#     planet: Planet = Planet()
#     interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

#     oceans: float = 1
#     h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
#     # o_kg: float = 1.22814e21

#     mass_constraints = {
#         "H": h_kg,
#     }

#     # Original argument was:
#     # BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#     fO2: RedoxBufferProtocol = IronWustiteBuffer(0.0)
#     fugacity_constraints = {O2_g.name: fO2}

#     # Initial solution guess number density (molecules/m^3)
#     initial_number_density: ArrayLike = np.array([30, 30, 30], dtype=np.float_)
#     initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
#     interior_atmosphere.initialise_solve(
#         planet,
#         initial_number_density,
#         initial_stability,
#         fugacity_constraints=fugacity_constraints,
#         mass_constraints=mass_constraints,
#     )
#     log_number_density, _ = interior_atmosphere.solve()
#     log_pressure: Array = log_pressure_from_log_number_density(
#         log_number_density, planet.surface_temperature
#     )
#     logger.debug("log_pressure = %s", log_pressure)

#     target: dict[str, float] = {
#         "H2O_g": 0.2570770067190733,
#         "H2_g": 0.24964688044710354,
#         "O2_g": 8.838043080858959e-08,
#     }

#     # FIXME: Setup test comparison
#     assert True


# def test_H_fO2_buffer_batch() -> None:
#     """Tests H2-H2O at the IW buffer."""

#     O2_g: Species = GasSpecies(O2_g_data)
#     H2_g: Species = GasSpecies(H2_g_data)
#     H2O_g: Species = GasSpecies(H2O_g_data, solubility=H2O_peridotite_sossi)

#     species: list[Species] = [H2O_g, H2_g, O2_g]
#     interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

#     num: int = 1000

#     surface_temperature: npt.NDArray = 2000.0 + np.arange(1, num + 1)
#     planet: Planet = Planet(surface_temperature=surface_temperature)
#     logger.info("planet = %s", planet)

#     oceans: float = 1
#     h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
#     # o_kg: float = 1.22814e21

#     h_kg_np: npt.NDArray = h_kg * np.arange(1, num + 1) / 100

#     mass_constraints: Mapping[str, ArrayLike] = {"H": h_kg_np}

#     fO2_batch: RedoxBufferProtocol = IronWustiteBuffer(np.linspace(-5, 5, num))
#     fugacity_constraints = {O2_g.name: fO2_batch}

#     # Initial solution guess number density (molecules/m^3)
#     initial_number_density: ArrayLike = np.array([30, 30, 30], dtype=np.float_)
#     initial_stability: ArrayLike = -100.0 * np.ones_like(initial_number_density)
#     interior_atmosphere.initialise_solve(
#         planet,
#         initial_number_density,
#         initial_stability,
#         fugacity_constraints=fugacity_constraints,
#         mass_constraints=mass_constraints,
#     )
#     log_number_density, _ = interior_atmosphere.solve()
#     # TODO: This needs to be made to work with batched output, including planet
#     # log_pressure: Array = log_pressure_from_log_number_density(
#     #     log_number_density, planet.surface_temperature
#     # )
#     # logger.debug("log_pressure = %s", log_pressure)

#     target: dict[str, float] = {
#         "H2O_g": 0.2570770067190733,
#         "H2_g": 0.24964688044710354,
#         "O2_g": 8.838043080858959e-08,
#     }

#     # FIXME: Setup test comparison
#     assert True


# def test_H_and_C_holland(helper) -> None:
#     """Tests H2-H2O and CO-CO2 with real gas EOS from Holland and Powell."""

#     H2O_g: GasSpecies = GasSpecies("H2O")
#     H2_g: GasSpecies = GasSpecies("H2", eos=H2_CORK_HP91)
#     O2_g: GasSpecies = GasSpecies("O2")
#     CO_g: GasSpecies = GasSpecies("CO", eos=CO_CORK_HP91)
#     CO2_g: GasSpecies = GasSpecies("CO2", eos=CO2_CORK_simple_HP91)

#     species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])
#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(CO_g, 26.625148913955194),
#             FugacityConstraint(H2O_g, 10000),
#             FugacityConstraint(O2_g, 8.981953412412735e-08),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     target: dict[str, float] = {
#         "CO2_g": 0.6283663007874475,
#         "CO_g": 2.5425375910504195,
#         "H2O_g": 10000.0,
#         "H2_g": 1693.4983324561576,
#         "O2_g": 8.981953412412754e-08,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


# def test_H_and_C_saxena(helper) -> None:
#     """Tests H2-H2O and real gas EOS from Saxena

#     The fugacity is large to check that the volume integral is performed correctly.
#     """

#     H2O_g: GasSpecies = GasSpecies("H2O")
#     H2_g: GasSpecies = GasSpecies("H2", eos=H2_SF87)
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2O_g, H2_g, O2_g])
#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(H2O_g, 10000),
#             FugacityConstraint(O2_g, 8.981953412412735e-08),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     target: dict[str, float] = {
#         "H2O_g": 10000.0,
#         "H2_g": 9539.109221925035,
#         "O2_g": 8.981953412412754e-08,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


# def test_H_fO2_no_solubility(helper) -> None:
#     """Tests H2-H2O at the IW buffer."""

#     H2O_g: GasSpecies = GasSpecies("H2O")
#     H2_g: GasSpecies = GasSpecies("H2")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2O_g, H2_g, O2_g])

#     oceans: float = 1
#     h_kg: float = earth_oceans_to_hydrogen_mass(oceans)

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             ElementMassConstraint("H", h_kg),
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     target: dict[str, float] = {
#         "H2O_g": 76.46402689279567,
#         "H2_g": 73.85383684279368,
#         "O2_g": 8.934086206704404e-08,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)
