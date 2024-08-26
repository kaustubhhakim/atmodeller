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
"""Tests for JAX"""

# Convenient to use naming convention so pylint: disable=C0103

from __future__ import annotations

import logging
from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ActivityConstraint,
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies, LiquidSpecies, Planet, SolidSpecies, Species
from atmodeller.eos.holland import CO_CORK_HP91, H2_CORK_HP91, CO2_CORK_simple_HP91
from atmodeller.eos.saxena import H2_SF87
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
from atmodeller.solver import SolverOptimistix, SolverScipy
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

planet: Planet = Planet()


def test_simple() -> None:

    def vector_function(x):
        return jnp.array([x[0] ** 2, jnp.sin(x[1])])

    # Compute the Jacobian
    def compute_jacobian(x):
        return jax.jacobian(vector_function)(x)

    # Test with a 2-D vector
    x: Array = jnp.array([1.0, 2.0])
    jacobian: Callable = compute_jacobian(x)
    print("Jacobian:\n", jacobian)


def test_H_fugacities(helper) -> None:
    """Tests H species with imposed fugacities"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(H2O_g, 0.2570770067190733),
            FugacityConstraint(O2_g, 8.838043080858959e-08),
        )
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(solver=SolverOptimistix(), constraints=constraints)

    target: dict[str, float] = {
        "H2O_g": 0.257077006719072,
        "H2_g": 0.24964688044710262,
        "O2_g": 8.838043080858959e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fugacities_system(helper) -> None:
    """Tests H species with imposed fugacities"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(H2O_g, 0.2570770067190733),
            FugacityConstraint(O2_g, 8.838043080858959e-08),
        )
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target: dict[str, float] = {
        "H2O_g": 0.257077006719072,
        "H2_g": 0.24964688044710262,
        "O2_g": 8.838043080858959e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_total_pressure(helper) -> None:
    """Tests H species with a total pressure constraint."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(O2_g, 8.838043080858959e-08),
            TotalPressureConstraint(0.5067239755466055),
        )
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target: dict[str, float] = {
        "H2O_g": 0.257077006719072,
        "H2_g": 0.24964688044710262,
        "O2_g": 8.838043080858959e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_with_buffer(helper) -> None:
    """Tests H species with an imposed fO2 buffer"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        (
            FugacityConstraint(H2O_g, 0.2570770067190733),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        )
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target: dict[str, float] = {
        "H2O_g": 0.257077006719072,
        "H2_g": 0.24964688044710262,
        "O2_g": 8.838043080858887e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_and_C_no_solubility(helper) -> None:
    """Tests H2-H2O and CO-CO2 with imposed fugacities and no solubility.

    This test is based on test_H_and_C() in test_CHO.py but without solubility.
    """

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(CO_g, 26.625148913955194),
            FugacityConstraint(H2O_g, 99.19769919121012),
            FugacityConstraint(O2_g, 8.981953412412735e-08),
        ]
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    # Interestingly, this test passes with the Optimistix newton solver but the Scipy newton solver
    # struggles even with the jacobian specified. lm works well though.
    solution, _ = interior_atmosphere.solve(
        solver=SolverScipy(method="lm", jac=True), constraints=constraints
    )

    target: dict[str, float] = {
        "CO2_g": 6.0622728258770024,
        "CO_g": 26.625148913955194,
        "H2O_g": 99.19769919121012,
        "H2_g": 95.55582495038334,
        "O2_g": 8.981953412412735e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_and_C_holland(helper) -> None:
    """Tests H2-H2O and CO-CO2 with real gas EOS from Holland and Powell."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2", eos=H2_CORK_HP91)
    O2_g: GasSpecies = GasSpecies("O2")
    CO_g: GasSpecies = GasSpecies("CO", eos=CO_CORK_HP91)
    CO2_g: GasSpecies = GasSpecies("CO2", eos=CO2_CORK_simple_HP91)

    species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g])
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(CO_g, 26.625148913955194),
            FugacityConstraint(H2O_g, 10000),
            FugacityConstraint(O2_g, 8.981953412412735e-08),
        ]
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target: dict[str, float] = {
        "CO2_g": 0.6283663007874475,
        "CO_g": 2.5425375910504195,
        "H2O_g": 10000.0,
        "H2_g": 1693.4983324561576,
        "O2_g": 8.981953412412754e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_and_C_saxena(helper) -> None:
    """Tests H2-H2O and real gas EOS from Saxena

    The fugacity is large to check that the volume integral is performed correctly.
    """

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2", eos=H2_SF87)
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2O_g, 10000),
            FugacityConstraint(O2_g, 8.981953412412735e-08),
        ]
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target: dict[str, float] = {
        "H2O_g": 10000.0,
        "H2_g": 9539.109221925035,
        "O2_g": 8.981953412412754e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


def test_H_fO2_no_solubility(helper) -> None:
    """Tests H2-H2O at the IW buffer."""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
        ]
    )
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target: dict[str, float] = {
        "H2O_g": 76.46402689279567,
        "H2_g": 73.85383684279368,
        "O2_g": 8.934086206704404e-08,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Need to make SystemConstraints compatible with JAX's array capabilities")
def test_H_fugacities_batched(helper) -> None:
    """Tests H species with imposed fugacities for multiple constraints"""

    H2O_g: GasSpecies = GasSpecies("H2O")
    H2_g: GasSpecies = GasSpecies("H2")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2O_g, H2_g, O2_g])
    constraints_list: list[SystemConstraints] = [
        SystemConstraints(
            (
                FugacityConstraint(H2O_g, 0.2570770067190733),
                FugacityConstraint(O2_g, 8.838043080858959e-08),
            )
        ),
        SystemConstraints(
            (
                FugacityConstraint(H2O_g, 1.0),
                FugacityConstraint(O2_g, 8.838043080858959e-08),
            )
        ),
        # Add more constraints as needed
    ]
    # interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
    #    species=species, planet=planet
    # )
    # solution: Solution = interior_atmosphere.solve(constraints=constraints)


@pytest.mark.skip(reason="Probably don't make sense without a mass constraint")
def test_graphite_condensed(helper) -> None:
    """Graphite stable with around 50% condensed C mass fraction"""

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr])
    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(CH4_g, 96.86234030526238),
            FugacityConstraint(H2O_g, 4.532525784559842),
            ActivityConstraint(C_cr, 1),
        ]
    )

    warm_planet: Planet = Planet(surface_temperature=873)
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=warm_planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target = {
        "CH4_g": 96.86234030526238,
        "CO2_g": 0.06051299046298187,
        "CO_g": 0.07273637701072179,
        "activity_C_cr": 1.0,
        "H2O_g": 4.532525784559842,
        "H2_g": 14.593415393593949,
        "O2_g": 1.2762699614653323e-25,
        "mass_C_cr": 3.5416383046342194e20,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Probably don't make sense without a mass constraint")
def test_graphite_water_condensed(helper) -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    H2O_g = GasSpecies("H2O")
    H2_g = GasSpecies("H2")
    O2_g = GasSpecies("O2")
    # TODO: Using the 10 bar thermo data pushes the atmodeller result away from FactSage. Why?
    H2O_l = LiquidSpecies("H2O")  # , thermodata_name="Water, 10 Bar")
    CO_g = GasSpecies("CO")
    CO2_g = GasSpecies("CO2")
    CH4_g = GasSpecies("CH4")
    C_cr = SolidSpecies("C")

    species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g, H2O_l, C_cr])
    constraints = SystemConstraints(
        [
            FugacityConstraint(CO2_g, 4.286974741683041),
            FugacityConstraint(H2O_g, 5.3837944619697),
            ActivityConstraint(H2O_l, 1),
            ActivityConstraint(C_cr, 1),
        ]
    )

    cool_planet = Planet(surface_temperature=430)
    interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species, planet=cool_planet
    )
    solution, _ = interior_atmosphere.solve(constraints=constraints)

    target = {
        "CH4_g": 0.32688481623407045,
        "CO2_g": 4.286974741683041,
        "CO_g": 2.7984895865705653e-06,
        "C_cr": 1.0,
        "H2O_g": 5.3837944619697,
        "H2O_l": 1.0,
        "H2_g": 0.0023431770406624397,
        "O2_g": 4.7858901816146536e-48,
        "mass_C_cr": 9.810548377514692e19,
        "mass_H2O_l": 2.7509418192551134e21,
    }

    assert helper.isclose(solution, target, rtol=RTOL, atol=ATOL)


@pytest.mark.skip(reason="Cannot get vmap to work as desired")
def test_reaction_network_vmap(helper) -> None:

    H2O_g: GasSpecies = GasSpecies(
        "H2O", thermodata_dataset=ThermodynamicDatasetHollandAndPowell()
    )
    H2_g: GasSpecies = GasSpecies("H2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell())
    O2_g: GasSpecies = GasSpecies("O2", thermodata_dataset=ThermodynamicDatasetHollandAndPowell())

    species: Species = Species([H2O_g, H2_g, O2_g])

    # reaction_network = ReactionNetwork(species=species, planet=planet)

    constraints_batch: list[SystemConstraints] = [
        SystemConstraints(
            (
                FugacityConstraint(H2O_g, 0.2570770067190733),
                FugacityConstraint(O2_g, 8.838043080858959e-08),
            )
        ),
        SystemConstraints(
            (
                FugacityConstraint(H2O_g, 0.2570770067190733),
                FugacityConstraint(O2_g, 8.838043080858959e-08),
            ),
        ),
    ]

    # Vectorize th method using jax.vmap with PyTree and static args
    # vmap_solve = jax.vmap(reaction_network.solve_optimistix)

    # constraints_batch = jax.tree_map(lambda x: jnp.stack(x, axis=0), constraints_batch)

    # vmap_solve(constraints=constraints_batch)

    # sol, solution = reaction_network.solve_optimistix(constraints=constraints)

    # target = {
    #    "H2O_g": 0.25707719341563373,
    #    "H2_g": 0.249646956461615,
    #    "O2_g": 8.838052554822744e-08,
    # }

    # print(solution.output_solution())

    # assert solution.isclose(target)
