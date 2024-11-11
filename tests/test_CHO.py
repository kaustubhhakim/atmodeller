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

import logging

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.typing import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.containers import ConstantFugacityConstraint, Planet, Species
from atmodeller.interfaces import FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.output import Output
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer, RedoxBufferProtocol
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
logger.setLevel(logging.INFO)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: dict[str, SolubilityProtocol] = get_solubility_models()


def test_version():
    """Test version."""
    assert __version__ == "0.2.0"


def test_H2O(helper) -> None:
    """Tests a single species (H2O)."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )

    species: tuple[Species, ...] = (H2O_g,)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    oceans: ArrayLike = 2
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 1.0312913336898137,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "H2O_g": 0.25708003342259883,
        "H2_g": 0.2491577248312551,
        "O2_g": 8.83804258138063e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_fH2(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility and mixed fugacity constraints."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        O2_g.name: IronWustiteBuffer(jnp.array([-1, 0, 1])),
        H2_g.name: ConstantFugacityConstraint(jnp.array([1.0e-8, 1.0e-7, 1.0e-6])),
    }

    # oceans: float = 1
    # h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    # mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        # mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([3.262689650880299e-09, 1.031753073747030e-07, 3.262689895293855e-06]),
        "H2_g": np.array([1.000000000000005e-08, 9.999999999999959e-08, 1.000000000000000e-06]),
        "O2_g": np.array([8.837300808668651e-09, 8.837301052658971e-08, 8.837302132702910e-07]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_temperature(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of surface temperatures."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    # Number of surface temperatures is different to number of species to test array shapes work.
    surface_temperatures: npt.NDArray[np.float_] = np.array(
        [1500, 2000, 2500, 3000], dtype=np.float_
    )
    # Additionally test batching planet radius
    # earth_radius: float = 6371000.0
    # surface_radius: npt.NDArray[np.float_] = earth_radius * np.array([0.5, 1, 1.5, 2])
    planet: Planet = Planet(
        surface_temperature=surface_temperatures
    )  # , surface_radius=surface_radius
    # )

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array(
            [0.25666525060921, 0.257080033422599, 0.25721776694131, 0.257274566532845]
        ),
        "H2_g": np.array(
            [0.313371131998901, 0.249157724831257, 0.226576661188286, 0.219958433575134]
        ),
        "O2_g": np.array(
            [
                2.394072903442518e-12,
                8.838042581380567e-08,
                4.544719798221670e-05,
                2.739265090516618e-03,
            ]
        ),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_fO2_shift(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of fO2 shifts."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    # Set up a range of fO2 shifts
    num: int = 4
    fO2_shifts: npt.NDArray[np.float_] = np.linspace(-10, 10, num, dtype=np.float_)
    fugacity_constraints: dict[str, RedoxBufferProtocol] = {
        O2_g.name: IronWustiteBuffer(fO2_shifts)
    }

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array(
            [0.00029780832006, 0.160943544377939, 0.258302012780794, 0.258540269218237]
        ),
        "H2_g": np.array(
            [
                2.883109764678719e01,
                7.237950850551736e00,
                5.393537121196381e-03,
                2.425598009366862e-06,
            ]
        ),
        "O2_g": np.array(
            [
                8.857668173953893e-18,
                4.104724093203387e-11,
                1.904041209642506e-04,
                9.431631962648368e02,
            ]
        ),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_H_mass(helper) -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of H budgets."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    # Set up a range of H masses
    mass_constraints: dict[str, ArrayLike] = {"H": np.array([h_kg, 10 * h_kg, 100 * h_kg])}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, ArrayLike] = {
        "H2O_g": np.array([2.570800334225988e-01, 2.426344578243460e01, 1.661069629540155e03]),
        "H2_g": np.array([2.491577248312551e-01, 2.347358252845090e01, 1.448607479781265e03]),
        "O2_g": np.array([8.838042581380630e-08, 8.869809838835624e-08, 1.091546738162749e-07]),
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_and_C(helper) -> None:
    """Tests H2-H2O and CO-CO2 with H2O and CO2 solubility."""

    H2O_g: Species = Species.create_gas(
        "H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    H2_g: Species = Species.create_gas("H2_g")
    O2_g: Species = Species.create_gas("O2_g")
    CO_g: Species = Species.create_gas("CO_g")
    CO2_g: Species = Species.create_gas(
        "CO2_g", solubility=solubility_models["CO2_basalt_dixon95"]
    )

    species: tuple[Species, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g)
    planet: Planet = Planet()
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species)

    fugacity_constraints: dict[str, RedoxBufferProtocol] = {O2_g.name: IronWustiteBuffer()}

    oceans: float = 1
    ch_ratio: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = ch_ratio * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    interior_atmosphere.initialise_solve(
        planet=planet,
        fugacity_constraints=fugacity_constraints,
        mass_constraints=mass_constraints,
    )
    output: Output = interior_atmosphere.solve()
    solution: dict[str, ArrayLike] = output.quick_look()

    target: dict[str, float] = {
        "CO2_g": 13.468919648305025,
        "CO_g": 59.63530829377201,
        "H2O_g": 0.25824676047561873,
        "H2_g": 0.24960964905685357,
        "O2_g": 8.886180538941781e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


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
#    """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`."""

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
