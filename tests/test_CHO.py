# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C-H-O systems"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np
import pytest
from jaxtyping import ArrayLike, PRNGKeyArray

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.interfaces import FugacityConstraintProtocol, SolubilityProtocol, SpeciesProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.sci_utils import earth_oceans_to_hydrogen_mass
from atmodeller.solubility import get_solubility_models
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.DEBUG)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

# Gas species
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
CO_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO")
CO2_g: ChemicalSpecies = ChemicalSpecies.create_gas("CO2")
CH4_g: ChemicalSpecies = ChemicalSpecies.create_gas("CH4")

# Melt species
H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
)
CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
    "CO2", solubility=solubility_models["CO2_basalt_dixon95"], include_in_phase_mass=False
)

key: PRNGKeyArray = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)  # Split the key for use in this function


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2 with H2O and CO2 solubility."""

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g)
    melt_species: tuple[SpeciesProtocol, ...] = (H2O_d, CO2_d)

    planet: BaseThermodynamicState = Planet.create(gas_species, melt_species=melt_species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
    ch_ratio: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = ch_ratio * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 0.2582458752325180,
                    "H2_g": 0.2502809714412906,
                    "O2_g": 8.838513516896038e-08,
                    "CO_g": 59.65835224848439,
                    "CO2_g": 13.43793686555727,
                }
            }
        }
    }

    assert output.compare(target, rtol=TOLERANCE, atol=TOLERANCE, log=False)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_reduced() -> None:
    """Tests C-H-O system at IW-2

    Similar to :cite:p:`BHS22{Table E, row 1}`.
    """

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    planet: BaseThermodynamicState = Planet.create(gas_species, temperature=1400)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(-2)}

    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2_g": 175.5,
                    "H2O_g": 13.8,
                    "CO_g": 6.21,
                    "CO2_g": 0.228,
                    "CH4_g": 38.07,
                    "O2_g": 1.25e-15,
                }
            }
        }
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_IW() -> None:
    """Tests C-H-O system at IW+0.5

    Similar to :cite:p:`BHS22{Table E, row 2}`.
    """

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    planet: BaseThermodynamicState = Planet.create(gas_species, temperature=1400)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.5)}

    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 28.66,
                    "CO2_g": 30.88,
                    "CO_g": 46.42,
                    "H2O_g": 337.16,
                    "H2_g": 236.98,
                    "O2_g": 4.11e-13,
                }
            }
        }
    }

    fastchem_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 29.61919788,
                    "CO2_g": 29.82548282,
                    "CO_g": 45.94958264,
                    "H2O_g": 332.03616807,
                    "H2_g": 236.73845646,
                    "O2_g": 3.96475584e-13,
                }
            }
        }
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert output.compare(fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_oxidised() -> None:
    """Tests C-H-O system at IW+2

    Similar to :cite:p:`BHS22{Table E, row 3}`.
    """

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    planet: BaseThermodynamicState = Planet.create(gas_species, temperature=1400)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(2)}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 0.1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 0.00129,
                    "CO2_g": 3.25,
                    "CO_g": 0.873,
                    "H2O_g": 218.48,
                    "H2_g": 27.40,
                    "O2_g": 1.29e-11,
                }
            }
        }
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_highly_oxidised() -> None:
    """Tests C-H-O system at IW+4

    Similar to :cite:p:`BHS22{Table E, row 4}`.
    """

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    planet: BaseThermodynamicState = Planet.create(gas_species, temperature=1400)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(4)}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 5 * h_kg
    # Mass of O that gives the same solution as applying the buffer at IW+4
    # o_kg: ArrayLike = 3.25196e21
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 7.13e-05,
                    "CO2_g": 357.23,
                    "CO_g": 10.21,
                    "H2O_g": 432.08,
                    "H2_g": 5.78,
                    "O2_g": 1.14e-09,
                }
            }
        }
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_middle_temperature() -> None:
    """Tests C-H-O system at 873 K"""

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    planet = Planet.create(gas_species, temperature=873)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2_g": 59.066,
                    "H2O_g": 18.320,
                    "CO_g": 8.91e-4,
                    "CO2_g": 7.48e-4,
                    "CH4_g": 19.548,
                    "O2_g": 1.27e-25,
                }
            }
        }
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_low_temperature() -> None:
    """Tests C-H-O system at 450 K"""

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)

    planet: BaseThermodynamicState = Planet.create(gas_species, temperature=450)

    # TODO: to revisit this below, since currently this test recreates the solver.
    # This is a trick to keep the same argument structure and avoid JAX recompilation, even though
    # for this case we want to turn off the O2_g constraint.
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(np.nan)
    }
    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    o_kg: ArrayLike = 1.02999e20
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2_g": 55.475,
                    "H2O_g": 8.0,
                    "CO2_g": 1.24e-14,
                    "O2_g": 7.85e-54,
                    "CH4_g": 16.037,
                    "CO_g": 2.12e-16,
                }
            }
        }
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
