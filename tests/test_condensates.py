# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C-H-O systems with stable or unstable condensates"""

import logging
from typing import Any

import jax
import numpy as np
from jaxtyping import ArrayLike, PRNGKeyArray
from molmass import Formula

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import Planet, ThermodynamicState
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

# Gas species
H2O_g = ChemicalSpecies.create_gas("H2O")
H2_g = ChemicalSpecies.create_gas("H2")
O2_g = ChemicalSpecies.create_gas("O2")
CO_g = ChemicalSpecies.create_gas("CO")
CO2_g = ChemicalSpecies.create_gas("CO2")
CH4_g = ChemicalSpecies.create_gas("CH4")
N2_g = ChemicalSpecies.create_gas("N2")
CHN_g = ChemicalSpecies.create_gas("CHN")
H_g = ChemicalSpecies.create_gas("H")

# Condensates
graphite: PurePhase = PurePhase.from_species("C", state="s")
water: PurePhase = PurePhase.from_species("H2O", state="l")

key: PRNGKeyArray = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)  # Split the key for use in this function


def test_graphite_stable() -> None:
    """Tests graphite stable with around 50% condensed C mass fraction"""

    gas_species = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)
    condensates = (graphite,)

    planet: Planet = Planet.create(gas_species, temperature=873, condensates=condensates)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(np.nan)
    }

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 5 * h_kg
    o_kg: ArrayLike = 2.73159e19
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "O2_g": 1.27e-25,
                    "H2_g": 14.564,
                    "CO_g": 0.07276,
                    "H2O_g": 4.527,
                    "CO2_g": 0.061195,
                    "CH4_g": 96.74,
                },
            },
        },
        "condensates": {
            "C_s": {"species": {"activity": {"C_s": 1.0}, "mass": {"C_s": 3.54162e20}}},
        },
    }

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_unstable() -> None:
    """Tests C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    gas_species = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)
    condensates = (graphite,)

    planet: Planet = Planet.create(gas_species, temperature=1400, condensates=condensates)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.5)}
    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "O2_g": 4.11e-13,
                    "H2_g": 236.98,
                    "CO_g": 46.42,
                    "H2O_g": 337.16,
                    "CO2_g": 30.88,
                    "CH4_g": 28.66,
                }
            }
        },
        "condensates": {"C_s": {"species": {"activity": {"C_s": 0.12202}}}},
    }

    # output.to_excel("test_graphite_unstable")

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_water_stable() -> None:
    """Condensed water at 10 bar"""

    gas_species = (H2_g, H2O_g, O2_g)
    condensates = (water,)

    planet: Planet = Planet.create(gas_species, temperature=411.75, condensates=condensates)

    oceans: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 1.14375e21
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {"partial_pressure": {"H2O_g": 3.3596, "H2_g": 6.5604, "O2_g": 5.6433e-58}},
        "condensates": {
            "H2O_l": {"species": {"activity": {"H2O_l": 1.0}, "mass": {"H2O_l": 1.247201e21}}}
        },
    }

    # output.to_excel("test_water_stable")

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_water_stable() -> None:
    """Tests C and water in equilibrium at 430 K and 10 bar"""

    gas_species = (H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g)
    condensates = (water, graphite)

    planet: Planet = Planet.create(gas_species, temperature=430, condensates=condensates)

    h_kg: float = 3.10e20
    c_kg: float = 1.08e20
    o_kg: float = 2.48298883581636e21
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 0.3241,
                    "CO2_g": 4.3064,
                    "CO_g": 2.77e-6,
                    "H2_g": 0.0023,
                    "O2_g": 4.74e-48,
                    "H2O_g": 5.3672,
                }
            }
        },
        "condensates": {
            "C_s": {"species": {"activity": {"C_s": 1.0}, "mass": {"C_s": 8.75101e19}}},
            "H2O_l": {"species": {"activity": {"H2O_l": 1.0}, "mass": {"H2O_l": 2.74821e21}}},
        },
    }

    # output.to_excel("test_graphite_water_stable")

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_impose_stable() -> None:
    """Tests a user-imposed stable condensate"""

    gas_species = (H2_g, N2_g, CH4_g, CHN_g, H_g)

    # Since in this example we do not provide carbon in the injected gas stream, we cannot solve
    # for the stability of any carbon-bearing products because in order to do so requires
    # specification of the mass of carbon in the system.
    graphite = PurePhase.from_species("C", state="s", solve_for_stability=False)
    condensates = (graphite,)

    # Set the temperature and pressure
    state: ThermodynamicState = ThermodynamicState.create(
        gas_species, pressure=1, temperature=1773.15, condensates=condensates
    )

    # Define the mole fractions of input gases
    mole_fractions: dict[str, ArrayLike] = {"H2": 0.5, "N2": 0.5}

    # Define the composition of the input gas mixture by mass
    mass_constraints: dict[str, ArrayLike] = {
        key: value * Formula(key).mass for key, value in mole_fractions.items()
    }

    parameters: Parameters = Parameters.create(state, mass_constraints=mass_constraints)

    solver = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "CH4_g": 0.000194708,
                    "H_g": 0.000201266,
                    "H2_g": 0.49807992,
                    "N2_g": 0.49866269,
                }
            }
        },
        "condensates": {
            "C_s": {"species": {"activity": {"C_s": 1.0}}},
        },
    }

    # output.to_excel("test_impose_stable")

    assert output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
