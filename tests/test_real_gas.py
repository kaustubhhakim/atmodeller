# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for systems with real gases"""

import logging
from collections.abc import Callable, Mapping
from typing import Any

import jax
import numpy as np
from jaxtyping import ArrayLike, PRNGKeyArray

from atmodeller import debug_logger
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import (
    ActivityProtocol,
    FugacityConstraintProtocol,
    SolubilityProtocol,
    SpeciesProtocol,
)
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import PurePhase
from atmodeller.sci_utils import earth_oceans_to_hydrogen_mass
from atmodeller.solubility import get_solubility_models
from atmodeller.solvers import make_solver_with_jit
from atmodeller.state import BaseThermodynamicState, Planet
from atmodeller.thermodata import IronWustiteBuffer

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()
eos_models: Mapping[str, ActivityProtocol] = get_eos_models()

# Gas Species
H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_models["H2_chabrier21"])
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
SiO_g: ChemicalSpecies = ChemicalSpecies.create_gas("OSi")
H4Si_g: ChemicalSpecies = ChemicalSpecies.create_gas("H4Si")
gas_species_subneptune: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g, SiO_g, H4Si_g)

# Force SiO2 to have unity activity, as per previous tests. Eventually can be relaxed to allow for
# a self-consistent activity, but this is fine for testing the real gas EOS.
O2Si_l: PurePhase = PurePhase.from_species("O2Si", state="l")
condensates_subneptune: tuple[PurePhase, ...] = (O2Si_l,)

key: PRNGKeyArray = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)  # Split the key for use in this function


def test_fO2_holley() -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`"""

    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_beattie_holley58"]
    )
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    gas_species: tuple[ChemicalSpecies, ...] = (H2_g, H2O_g, O2_g)

    # Temperature is within the range of the Holley model
    planet: BaseThermodynamicState = Planet.create(gas_species, temperature=1000)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 32.77037875523393,
                    "H2_g": 71.50338102110962,
                    "O2_g": 1.525466019972294e-21,
                }
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    planet: Planet = Planet.create(
        gas_species_subneptune, temperature=3400, condensates=condensates_subneptune
    )
    h_kg: ArrayLike = 0.01 * planet.default_planet_mass
    si_kg: ArrayLike = 0.1459 * planet.default_planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 7.253556287801738e03,
                    "H2_g": 1.162520652380062e04,
                    "H4Si_g": 6.759146395057408e04,
                    "O2_g": 1.791815879185495e-05,
                    "OSi_g": 6.302402285027329e02,
                },
                "activity": {
                    "H2O_g": 7.253556287801635e03,
                    "H2_g": 2.516876841308367e05,
                    "H4Si_g": 6.759146395057408e04,
                    "O2_g": 1.791815879185482e-05,
                    "OSi_g": 6.302402285027240e02,
                },
            }
        },
        "condensates": {"activity": {"O2Si_l": 1.0}},
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune

    This case effectively saturates the maximum allowable log number density at a value of 70
    based on the default hypercube that brackets the solution (see LOG_NUMBER_MOLES_UPPER).
    This is fine for a test, but this test is not physically realistic because solubilities are
    ignored, which would greatly lower the pressure and hence the number density.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet.create(
        gas_species_subneptune,
        condensates=condensates_subneptune,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    h_kg: ArrayLike = 0.01 * planet.default_planet_mass
    si_kg: ArrayLike = 0.1459 * planet.default_planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 4.295071823974879e05,
                    "H2_g": 2.926773356736283e00,
                    "H4Si_g": 7.038499826508187e-04,
                    "O2_g": 1.039725511931324e01,
                    "OSi_g": 8.273579821046055e-01,
                },
                "activity": {
                    "H2O_g": 4.295071823974879e05,
                    "H2_g": 1.956449985411128e04,
                    "H4Si_g": 7.038499826508187e-04,
                    "O2_g": 1.039725511931324e01,
                    "OSi_g": 8.273579821046055e-01,
                },
            }
        },
        "condensates": {"activity": {"O2Si_l": 1.0}},
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune_batch() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet.create(
        gas_species_subneptune,
        condensates=condensates_subneptune,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )
    h_kg: ArrayLike = 0.01 * planet.default_planet_mass
    si_kg: ArrayLike = 0.1459 * planet.default_planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array(
                        [4.477789711513712e05, 4.785890592398898e05, 5.039107471956282e05]
                    ),
                    "H2_g": np.array(
                        [3.463824822645956e-02, 7.208115634579626e-03, 2.129125602157067e-03]
                    ),
                    "O2_g": np.array(
                        [2.597033179470946e04, 8.263153509596182e04, 1.447811285078976e05]
                    ),
                },
                "activity": {
                    "H2_g": np.array(
                        [4.081150539627139e02, 2.445386584856476e02, 1.945159917637966e02]
                    )
                },
            }
        }
    }

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_pH2_fO2_real_gas() -> None:
    """Tests H2-H2O at the IW buffer using real gas EOS from :cite:t:`HP91,HP98`.

    Applies a constraint to the fugacity of H2.
    """
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2O", activity=eos_models["H2O_cork_holland98"]
    )
    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_cork_cs_holland91"]
    )
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )

    gas_species: tuple[ChemicalSpecies, ...] = (H2O_g, H2_g, O2_g)
    melt_species: tuple[ReservoirSpecies, ...] = (H2O_d,)

    planet: Planet = Planet.create(gas_species, melt_species=melt_species)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(0.072885576196744)
    }

    mass_constraints: dict[str, ArrayLike] = {"H": 1.47126255324872e22}

    parameters: Parameters = Parameters.create(
        planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": 1470.2567650857518,
                    "H2_g": 999.9971214963639,
                    "O2_g": 1.045357420958815e-07,
                }
            }
        }
    }

    # output.to_excel("pH2_fO2_real_gas")

    assert output.compare(target, rtol=RTOL, atol=ATOL)


def test_subNeptune_melt_phase() -> None:
    """Tests a more realistic sub-Neptune with computed melt-phase activities.

    This is an extension of the model setup in Hakim et al. (2026), MRNAS

    The melt phase consists of a chemically-reactive component SiO2(l) and a dissolved component
    H2O(l). Here, the activities of both species in the melt phase are calculated
    self-consistently.
    """

    # The species we specify in the melt should be considered as already included in the
    # "background" melt mass, so we set include_in_phase_mass=False for both species
    O2Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed(
        "O2Si", state="l", include_in_phase_mass=False
    )
    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )
    melt_species: tuple[SpeciesProtocol, ...] = (O2Si_l, H2O_d)

    # Temperature must be compatible with the choice of species, i.e. chemically-reactive species
    # must have thermodynamic data available at the specified temperature. Here, we are limited by
    # O2Si(l).
    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m

    planet: Planet = Planet.create(
        gas_species_subneptune,
        melt_species=melt_species,
        temperature=surface_temperature,
        planet_mass=planet_mass,
        surface_radius=surface_radius,
    )

    # The previous mass constraints are still OK, because we are not allowing the melt species to
    # contribute additionally to the planet mass. So these calculations are still exact.
    h_kg: ArrayLike = 0.01 * planet.default_planet_mass
    si_kg: ArrayLike = 0.1459 * planet.default_planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    # o_kg: ArrayLike = 6.74717e24
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    parameters: Parameters = Parameters.create(planet, mass_constraints=mass_constraints)

    solver: Callable = make_solver_with_jit(parameters)

    output: Output = solver(parameters, subkey)

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": {
                    "H2O_g": np.array([34153.77806762769, 34647.133760130455, 34773.46409320044]),
                    "H2_g": np.array([1.953888592704683, 0.164931377760951, 0.026752106127028]),
                    "O2_g": np.array([34647.573618670154, 118713.0216717798, 205315.21707732175]),
                },
                "activity": {
                    "H2_g": np.array([26.950076888813005, 14.769840406710053, 11.271846271463033])
                },
            }
        },
        "melt": {
            "species": {
                "activity": {
                    "H2O_d": np.array([0.398229768986479, 0.401095696396106, 0.401826268347003]),
                    "O2Si_l": np.array([0.442325000552759, 0.442325002931253, 0.442325003495892]),
                }
            }
        },
    }

    # output.to_excel(file_prefix="test_subNeptune_melt_phase")

    assert output.compare(target, rtol=RTOL, atol=ATOL)
