# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for melt systems"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from jaxtyping import ArrayLike

from atmodeller import __version__, debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, Planet, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import ActivityProtocol, SolubilityProtocol
from atmodeller.phases import GasPhase, MeltPhase
from atmodeller.solubility import get_solubility_models
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()
eos_models: Mapping[str, ActivityProtocol] = get_eos_models()


def test_version():
    """Test version."""
    assert __version__ == "0.11.0"


def test_H2O_no_dilute_limit() -> None:
    """Tests a single species (H2O) without the dilute limit.

    This can be compared with test_H2O, which tests the same system but with the dilute limit.
    """

    gas: GasPhase = GasPhase.create(("H2O",))
    H2O_di: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    melt: MeltPhase = MeltPhase((H2O_di,))
    planet: Planet = Planet()
    model: EquilibriumModel = EquilibriumModel(gas, melt=melt)

    oceans: ArrayLike = 2
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    model.solve(state=planet, mass_constraints=mass_constraints)

    target: dict[str, Any] = {
        "gas": {
            "partial_pressure_bar": {"H2O_g": 1.0299426742644398},
            "number_moles": {"H2O_g": 2.969522562007359e20},
        }
    }

    # output.to_excel("test_H2O_dilute_limit")

    assert model.output.compare(target, rtol=RTOL, atol=ATOL)


def test_subNeptune() -> None:
    """Tests a subNeptune

    Similar to test_chabrier_subNeptune

    This is a simplified test based on the model setup in Hakim et al. (2026), MRNAS
    """

    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_models["H2_chabrier21"])
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
    SiO_g: ChemicalSpecies = ChemicalSpecies.create_gas("OSi")
    H4Si_g: ChemicalSpecies = ChemicalSpecies.create_gas("H4Si")
    gas: GasPhase = GasPhase((H2_g, H2O_g, O2_g, SiO_g, H4Si_g))

    O2Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed("O2Si", state="l")
    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"]
    )
    melt: MeltPhase = MeltPhase((H2O_d, O2Si_l))

    subneptune_model: EquilibriumModel = EquilibriumModel(gas, melt=melt)

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        temperature=surface_temperature, planet_mass=planet_mass, surface_radius=surface_radius
    )
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {
        "H": np.array([h_kg, h_kg * 2, h_kg * 3]),
        "Si": si_kg,
        "O": o_kg,
    }

    subneptune_model.solve(state=planet, mass_constraints=mass_constraints)

    subneptune_model.output.quick_look()

    subneptune_model.output.group_by_species

    subneptune_model.output.to_excel("test_subNeptune")
