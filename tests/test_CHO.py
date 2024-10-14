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
"""Tests for C-H-O systems to test some of the main functionality of atmodeller"""

# Want to use chemistry symbols so pylint: disable=invalid-name

import logging

import numpy as np
import numpy.typing as npt
import pytest
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, __version__, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import Planet, Species
from atmodeller.solubility.carbon_species import CO2_basalt_dixon
from atmodeller.solubility.hydrogen_species import H2O_peridotite_sossi
from atmodeller.thermodata.core import IronWustiteBuffer, RedoxBufferProtocol
from atmodeller.thermodata.species_data import (
    CO2_g_data,
    CO_g_data,
    H2_g_data,
    H2O_g_data,
    O2_g_data,
)
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

# from atmodeller.constraints import (
#     BufferedFugacityConstraint,
#     ElementMassConstraint,
#     FugacityConstraint,
#     PressureConstraint,
#     SystemConstraints,
#     TotalPressureConstraint,
# )
# from atmodeller.containers import Planet
# from atmodeller.core import GasSpecies, Species
# from atmodeller.eos.holland import get_holland_eos_models
# from atmodeller.eos.interfaces import RealGas
# from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
# from atmodeller.solubility.carbon_species import CO2_basalt_dixon
# from atmodeller.solubility.hydrogen_species import (
#     H2_basalt_hirschmann,
#     H2O_peridotite_sossi,
# )
# from atmodeller.solution import Solution
# from atmodeller.solver import SolverOptimistix
# from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
# from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
# from atmodeller.utilities import earth_oceans_to_hydrogen_mass

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

INITIAL_NUMBER_DENSITY: float = 30.0
"""Initial number density"""
INITIAL_STABILITY: float = -100.0
"""Initial stability"""


# eos_holland: dict[str, RealGas] = get_holland_eos_models()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_H2O(helper) -> None:
    """Tests a single species (H2O)."""

    H2O_g: Species = Species.create_gas(H2O_g_data, solubility=H2O_peridotite_sossi)

    species: tuple[Species, ...] = (H2O_g,)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    oceans: ArrayLike = 2
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = INITIAL_NUMBER_DENSITY * np.ones(
        len(species), dtype=np.float_
    )
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet, initial_number_density, initial_stability, mass_constraints=mass_constraints
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    target: dict[str, float] = {
        "H2O_g": 1.0312913336898137,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility."""

    H2O_g: Species = Species.create_gas(H2O_g_data, solubility=H2O_peridotite_sossi)
    H2_g: Species = Species.create_gas(H2_g_data)
    O2_g: Species = Species.create_gas(O2_g_data)

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

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
        "H2O_g": 0.25708003342259883,
        "H2_g": 0.2491577248312551,
        "O2_g": 8.83804258138063e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of surface temperatures."""

    H2O_g: Species = Species.create_gas(H2O_g_data, solubility=H2O_peridotite_sossi)
    H2_g: Species = Species.create_gas(H2_g_data)
    O2_g: Species = Species.create_gas(O2_g_data)

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    # Set up a range of surface temperatures to solve from 2000 K to 3000 K
    surface_temperatures: npt.NDArray[np.float_] = np.array([2000, 2500, 3000], dtype=np.float_)
    planet: Planet = Planet(surface_temperature=surface_temperatures)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

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

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([0.257080033422599, 0.25721776694131, 0.257274566532843]),
        "H2_g": np.array([0.249157724831255, 0.226576661188286, 0.219958433575132]),
        "O2_g": np.array([8.838042581380630e-08, 4.544719798221670e-05, 2.739265090516618e-03]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_and_C(helper) -> None:
    """Tests H2-H2O and CO-CO2 with H2O and CO2 solubility."""

    H2O_g: Species = Species.create_gas(H2O_g_data, solubility=H2O_peridotite_sossi)
    H2_g: Species = Species.create_gas(H2_g_data)
    O2_g: Species = Species.create_gas(O2_g_data)
    CO_g: Species = Species.create_gas(CO_g_data)
    CO2_g: Species = Species.create_gas(CO2_g_data, solubility=CO2_basalt_dixon)

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    ch_ratio: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: float = ch_ratio * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

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
        "CO2_g": 13.468919648305025,
        "CO_g": 59.63530829377201,
        "H2O_g": 0.25824676047561873,
        "H2_g": 0.24960964905685357,
        "O2_g": 8.886180538941781e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


# TODO: Kept to repurpose with a batch solve to test surface temperature
# def test_H_1500K(helper) -> None:
#     """Tests H2-H2O at a different temperature."""

#     H2O_g: GasSpecies = GasSpecies("H2O", solubility=H2O_peridotite_sossi())
#     H2_g: GasSpecies = GasSpecies("H2")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2O_g, H2_g, O2_g])

#     oceans: float = 1
#     warm_planet: Planet = Planet(surface_temperature=1500)
#     h_kg: float = earth_oceans_to_hydrogen_mass(oceans)

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             ElementMassConstraint("H", h_kg),
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=warm_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     target: dict[str, float] = {
#         "H2O_g": 0.25666635568842355,
#         "H2_g": 0.31320683835217444,
#         "O2_g": 2.3940728554564946e-12,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)

# TODO: Kept for real gas testing when the EOSs have been setup for JAX
# @pytest.mark.skip(reason="Holland H2O model is not configured for JAX")
# def test_pH2_fO2_real_gas(helper) -> None:
#     """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

#     Applies a constraint to the partial pressure of H2.
#     """

#     H2O_g: GasSpecies = GasSpecies(
#         "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
#     )
#     H2_g: GasSpecies = GasSpecies("H2", eos=eos_holland["H2"])
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2O_g, H2_g, O2_g])

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             PressureConstraint(H2_g, value=1000),
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=planet
#     )
#     solution: Solution = interior_atmosphere.solve(constraints=constraints)

#     target: dict[str, float] = {
#         "H2O_g": 1466.9613852210507,
#         "H2_g": 1000.0,
#         "O2_g": 1.0453574209588085e-07,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


# @pytest.mark.skip(reason="Holland H2O model is not configured for JAX")
# def test_fH2_fO2_real_gas(helper) -> None:
#     """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

#     Applies a constraint to the fugacity of H2.
#     """

#     H2O_g: GasSpecies = GasSpecies(
#         "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
#     )
#     H2_g: GasSpecies = GasSpecies("H2", eos=eos_holland["H2"])
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2O_g, H2_g, O2_g])

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(H2_g, value=1000),
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=planet
#     )
#     solution: Solution = interior_atmosphere.solve(constraints=constraints)

#     target: dict[str, float] = {
#         "H2O_g": 1001.131462103614,
#         "H2_g": 755.5960144468955,
#         "O2_g": 9.96495147231471e-08,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


# @pytest.mark.skip(reason="Holland H2O model is not configured for JAX")
# def test_H_and_C_real_gas(helper) -> None:
#     """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

#     H2_g: GasSpecies = GasSpecies("H2", solubility=H2_basalt_hirschmann(), eos=eos_holland["H2"])
#     H2O_g: GasSpecies = GasSpecies(
#         "H2O", solubility=H2O_peridotite_sossi(), eos=eos_holland["H2O"]
#     )
#     O2_g: GasSpecies = GasSpecies("O2")
#     CO_g: GasSpecies = GasSpecies(formula="CO", eos=eos_holland["CO"])
#     CO2_g: GasSpecies = GasSpecies(
#         formula="CO2", solubility=CO2_basalt_dixon(), eos=eos_holland["CO2"]
#     )
#     CH4_g: GasSpecies = GasSpecies(formula="CH4", eos=eos_holland["CH4"])

#     species: Species = Species([H2_g, H2O_g, O2_g, CO_g, CO2_g, CH4_g])

#     oceans: float = 10
#     h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
#     c_kg: float = h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(H2_g, value=958),
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#             ElementMassConstraint("C", value=c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=planet
#     )
#     solution: Solution = interior_atmosphere.solve(constraints=constraints)

#     target: dict[str, float] = {
#         "CH4_g": 10.300421855316944,
#         "CO2_g": 67.62567996145735,
#         "CO_g": 277.9018480265112,
#         "H2O_g": 953.4324527154502,
#         "H2_g": 694.3036008172556,
#         "O2_g": 1.0132255325169718e-07,
#     }

#     assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)

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
