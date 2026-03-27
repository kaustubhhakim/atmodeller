# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for H-O systems"""

import logging
from collections.abc import Mapping
from typing import Any

import jax
import numpy as np
from jaxmod.type_aliases import NpFloat
from jaxtyping import ArrayLike, PRNGKeyArray

from atmodeller import __version__, debug_logger
from atmodeller.containers import ChemicalSpecies, FixedFugacityConstraint, ReservoirSpecies
from atmodeller.interfaces import FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.solubility import get_solubility_models
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import Planet
from atmodeller.thermodata import IronWustiteBuffer
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

# Gas species
H2O_g = ChemicalSpecies.create_gas("H2O")
H2_g = ChemicalSpecies.create_gas("H2")
O2_g = ChemicalSpecies.create_gas("O2")
CO_g = ChemicalSpecies.create_gas("CO")
CO2_g = ChemicalSpecies.create_gas("CO2")
CH4_g = ChemicalSpecies.create_gas("CH4")

# Melt species
H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
)

key: PRNGKeyArray = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)  # Split the key for use in this function


def test_version() -> None:
    """Test version."""
    assert __version__ == "0.11.0"


def test_H2O() -> None:
    """Tests a single species (H2O)."""

    gas_species = (H2O_g,)
    melt_species = (H2O_d,)

    planet: Planet = Planet.create(gas_species, melt_species=melt_species)

    oceans: ArrayLike = 2
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {"species": {"partial_pressure": {"H2O_g": 1.0312913336898137}}}
    }

    # output.to_excel("test_H2O")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_O() -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint."""

    gas_species = (H2_g, H2O_g, O2_g)

    planet: Planet = Planet.create(gas_species)

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: ArrayLike = 6.25774e20
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    fastchem_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 76.45861543,
                    "H2_g": 73.84378192,
                    "O2_g": 8.91399329e-08,
                }
            }
        }
    }

    # output.to_excel("test_H_O")

    assert output.compare(fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_H_fO2() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility."""

    gas_species = (H2_g, H2O_g, O2_g)
    melt_species = (H2O_d,)

    planet: Planet = Planet.create(gas_species, melt_species=melt_species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 0.2570800742364775,
                    "H2_g": 0.2491511264610601,
                    "O2_g": 8.838513516896038e-08,
                }
            }
        }
    }

    # output.to_excel("test_H_fO2")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_fH2() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility and mixed fugacity constraints."""

    gas_species = (H2_g, H2O_g, O2_g)
    melt_species = (H2O_d,)

    planet: Planet = Planet.create(gas_species, melt_species=melt_species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "H2_g": FixedFugacityConstraint(np.array([1.0e-8, 1.0e-7, 1.0e-6])),
        "O2_g": IronWustiteBuffer(np.array([-1, 0, 1])),
    }

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [3.262913506271090e-09, 1.031823848794260e-07, 3.262913506271089e-06]
                    ),
                    "H2_g": np.array(
                        [1.000000000000005e-08, 9.999999999999959e-08, 1.000000000000000e-06]
                    ),
                    "O2_g": np.array(
                        [8.838513516896060e-09, 8.838513516896038e-08, 8.838513516896018e-07]
                    ),
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_temperature() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of surface temperatures."""

    gas_species = (H2_g, H2O_g, O2_g)
    melt_species = (H2O_d,)

    # Number of surface temperatures is different to number of species to test array shapes work.
    temperature = np.array([1500, 2000, 2500, 3000])

    planet: Planet = Planet.create(gas_species, melt_species=melt_species, temperature=temperature)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(),
        "H2_g": FixedFugacityConstraint(np.nan),
    }

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [
                            2.566653037020448e-01,
                            2.570800742364757e-01,
                            2.572178041535549e-01,
                            2.572746043480848e-01,
                        ]
                    ),
                    "H2_g": np.array(
                        [
                            3.133632393608037e-01,
                            2.491511264610584e-01,
                            2.265704456625875e-01,
                            2.199521409043987e-01,
                        ]
                    ),
                    "O2_g": np.array(
                        [
                            2.394194493859141e-12,
                            8.838513516896038e-08,
                            4.544970468047975e-05,
                            2.739422634823809e-03,
                        ]
                    ),
                }
            }
        }
    }

    # output.to_excel("test_H_fO2_batch_temperature")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_fO2_shift() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of fO2 shifts."""

    gas_species = (H2_g, H2O_g, O2_g)
    melt_species = (H2O_d,)

    planet: Planet = Planet.create(gas_species, melt_species=melt_species)

    # Set up a range of fO2 shifts
    num: int = 4
    fO2_shifts: NpFloat = np.linspace(-10, 10, num, dtype=np.float64)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(fO2_shifts),
    }

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [
                            2.974916728388850e-04,
                            1.609262626310640e-01,
                            2.583020136413676e-01,
                            2.585402668639946e-01,
                        ]
                    ),
                    "H2_g": np.array(
                        [
                            2.883163373152490e01,
                            7.239157580397396e00,
                            5.393312233908341e-03,
                            2.505662833497315e-06,
                        ]
                    ),
                    "O2_g": np.array(
                        [
                            8.838513516896005e-18,
                            4.102474564576031e-11,
                            1.904200012911665e-04,
                            8.838513516896137e02,
                        ]
                    ),
                }
            }
        }
    }

    # output.to_excel("test_H_fO2_batch_fO2_shift")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_H_fO2_batch_H_mass() -> None:
    """Tests H2-H2O at the IW buffer with H2O solubility for a range of H budgets."""

    gas_species = (H2_g, H2O_g, O2_g)
    melt_species = (H2O_d,)

    planet: Planet = Planet.create(gas_species, melt_species=melt_species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    # Set up a range of H masses
    mass_constraints: dict[str, ArrayLike] = {"H": np.array([h_kg, 10 * h_kg, 100 * h_kg])}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [2.570800742364757e-01, 2.426110356931991e01, 1.610286613431932e03]
                    ),
                    "H2_g": np.array(
                        [2.491511264610584e-01, 2.351283467393216e01, 1.560621626756960e03]
                    ),
                    "O2_g": np.array(
                        [8.838513516896038e-08, 8.838513516896038e-08, 8.838513516896102e-08]
                    ),
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)
