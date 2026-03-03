# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for systems with real gases"""

import logging
from typing import Any, Mapping

import numpy as np
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ChemicalSpecies, Planet, ReservoirSpecies
from atmodeller.eos.library import get_eos_models
from atmodeller.interfaces import ActivityProtocol, FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.phases import GasPhase, MeltPhase, PurePhase
from atmodeller.solubility import get_solubility_models
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.WARNING)

RTOL: float = 1.0e-6
"""Relative tolerance"""
ATOL: float = 1.0e-6
"""Absolute tolerance"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()
eos_models: Mapping[str, ActivityProtocol] = get_eos_models()

H2_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2", activity=eos_models["H2_chabrier21"])
H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")
SiO_g: ChemicalSpecies = ChemicalSpecies.create_gas("OSi")
H4Si_g: ChemicalSpecies = ChemicalSpecies.create_gas("H4Si")
O2Si_l: ChemicalSpecies = ChemicalSpecies.create_condensed("O2Si", state="l")

gas: GasPhase = GasPhase((H2_g, H2O_g, O2_g, SiO_g, H4Si_g))

# To force SiO2 to have unity activity, as per previous tests. Eventually can be relaxed to allow
# for a more realistic activity, but this is fine for testing the real gas EOS.
condensates: PurePhase = PurePhase((O2Si_l,))

subneptune_model: EquilibriumModel = EquilibriumModel(gas, condensates=(condensates,))


def test_fO2_holley() -> None:
    """Tests a system with the H2 EOS from :cite:t:`HWZ58`"""

    H2_g: ChemicalSpecies = ChemicalSpecies.create_gas(
        "H2", activity=eos_models["H2_beattie_holley58"]
    )
    H2O_g: ChemicalSpecies = ChemicalSpecies.create_gas("H2O")
    O2_g: ChemicalSpecies = ChemicalSpecies.create_gas("O2")

    gas: GasPhase = GasPhase((H2_g, H2O_g, O2_g))

    # Temperature is within the range of the Holley model
    planet: Planet = Planet(temperature=1000)
    model: EquilibriumModel = EquilibriumModel(gas)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg}

    model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    target: dict[str, Any] = {
        "gas": {
            "partial_pressure_bar": {
                "H2O_g": 32.77037875523393,
                "H2_g": 71.50338102110962,
                "O2_g": 1.525466019972294e-21,
            }
        }
    }

    assert model.output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_earth() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21`"""

    planet: Planet = Planet(temperature=3400)
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = h_kg * 10
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    subneptune_model.solve(state=planet, mass_constraints=mass_constraints)

    target: dict[str, Any] = {
        "gas": {
            "partial_pressure_bar": {
                "H2O_g": 7.253556287801738e03,
                "H2_g": 1.162520652380062e04,
                "H4Si_g": 6.759146395057408e04,
                "O2_g": 1.791815879185495e-05,
                "OSi_g": 6.302402285027329e02,
            },
            "fugacity": {
                "H2O_g": 7.253556287801635e03,
                "H2_g": 2.516876841308367e05,
                "H4Si_g": 6.759146395057408e04,
                "O2_g": 1.791815879185482e-05,
                "OSi_g": 6.302402285027240e02,
            },
        },
        "condensates": {"activity": {"O2Si_l": 1.0}},
    }

    assert subneptune_model.output.compare(target, rtol=RTOL, atol=ATOL)


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
    planet: Planet = Planet(
        temperature=surface_temperature, planet_mass=planet_mass, surface_radius=surface_radius
    )
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    o_kg: ArrayLike = 6.74717e24

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    subneptune_model.solve(state=planet, mass_constraints=mass_constraints)

    target: dict[str, Any] = {
        "gas": {
            "partial_pressure_bar": {
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
        },
        "condensates": {"activity": {"O2Si_l": 1.0}},
    }

    assert subneptune_model.output.compare(target, rtol=RTOL, atol=ATOL)


def test_chabrier_subNeptune_batch() -> None:
    """Tests a system with the H2 EOS from :cite:t:`CD21` for a sub-Neptune for several O masses

    As above, this test has questionable physical relevance without the inclusion of more species'
    solubility, but it serves its purpose as a test.
    """

    surface_temperature = 3400  # K
    planet_mass = 4.6 * 5.97224e24  # kg
    surface_radius = 1.5 * 6371000  # m
    planet: Planet = Planet(
        temperature=surface_temperature, planet_mass=planet_mass, surface_radius=surface_radius
    )
    h_kg: ArrayLike = 0.01 * planet.planet_mass
    si_kg: ArrayLike = 0.1459 * planet.planet_mass  # Si = 14.59 wt% Kargel & Lewis (1993)
    # Batch solve for three oxygen masses
    o_kg: ArrayLike = 1e24 * np.array([7.0, 7.5, 8.0])

    logger.info("h_kg = %s", h_kg)
    logger.info("si_kg = %s", si_kg)
    logger.info("o_kg = %s", o_kg)

    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "Si": si_kg, "O": o_kg}

    subneptune_model.solve(state=planet, mass_constraints=mass_constraints)

    target: dict[str, Any] = {
        "gas": {
            "partial_pressure_bar": {
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

    assert subneptune_model.output.compare(target, rtol=RTOL, atol=ATOL)


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
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"]
    )

    gas: GasPhase = GasPhase((H2O_g, H2_g, O2_g))
    melt: MeltPhase = MeltPhase((H2O_d,))
    model: EquilibriumModel = EquilibriumModel(gas, melt=melt)

    planet: Planet = Planet()

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {
        "O2_g": IronWustiteBuffer(0.072885576196744)
    }

    mass_constraints: dict[str, ArrayLike] = {"H": 1.47126255324872e22}

    model.solve(
        state=planet,
        mass_constraints=mass_constraints,
        fugacity_constraints=fugacity_constraints,
        # Guide the solver with an improved initial guess
        initial_log_number_moles=np.array([55, 55, 30, 55]),
    )

    # output.to_excel("pH2_fO2_real_gas")

    target: dict[str, Any] = {
        "gas": {
            "partial_pressure_bar": {
                "H2O_g": 1470.2567650857518,
                "H2_g": 999.9971214963639,
                "O2_g": 1.045357420958815e-07,
            }
        }
    }

    assert model.output.compare(target, rtol=RTOL, atol=ATOL)
