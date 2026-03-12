# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for C-H-O systems"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from jaxtyping import ArrayLike

from atmodeller import debug_logger
from atmodeller.classes import EquilibriumModel
from atmodeller.containers import ReservoirSpecies
from atmodeller.interfaces import FugacityConstraintProtocol, SolubilityProtocol
from atmodeller.phases import GasPhase, MeltPhase
from atmodeller.solubility import get_solubility_models
from atmodeller.state import Planet
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""

solubility_models: Mapping[str, SolubilityProtocol] = get_solubility_models()

gas: GasPhase = GasPhase.create(("H2", "H2O", "CO", "CO2", "CH4", "O2"))
gas_CHO_model: EquilibriumModel = EquilibriumModel(gas)


def test_H_and_C() -> None:
    """Tests H2-H2O and CO-CO2 with H2O and CO2 solubility."""

    gas: GasPhase = GasPhase.create(("H2O", "H2", "O2", "CO", "CO2"))

    H2O_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "H2O", solubility=solubility_models["H2O_peridotite_sossi23"], include_in_phase_mass=False
    )
    CO2_d: ReservoirSpecies = ReservoirSpecies.create_dissolved(
        "CO2", solubility=solubility_models["CO2_basalt_dixon95"], include_in_phase_mass=False
    )
    melt: MeltPhase = MeltPhase((H2O_d, CO2_d))

    planet: Planet = Planet()
    model: EquilibriumModel = EquilibriumModel(gas, melt=melt)

    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}

    oceans: float = 1
    ch_ratio: float = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = ch_ratio * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    target: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array(
                    [
                        0.2582458752325180,
                        0.2502809714412906,
                        8.838513516896038e-08,
                        59.65835224848439,
                        13.43793686555727,
                    ]
                )
            }
        }
    }

    assert model.output.compare(target, rtol=TOLERANCE, atol=TOLERANCE, log=False)


# @pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_reduced() -> None:
    """Tests C-H-O system at IW-2

    Similar to :cite:p:`BHS22{Table E, row 1}`.
    """

    planet: Planet = Planet(temperature=1400)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(-2)}
    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    gas_CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {"partial_pressure": np.array([175.5, 13.8, 6.21, 0.228, 38.07, 1.25e-15])}
        }
    }

    assert gas_CHO_model.output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_IW() -> None:
    """Tests C-H-O system at IW+0.5

    Similar to :cite:p:`BHS22{Table E, row 2}`.
    """

    planet: Planet = Planet(temperature=1400)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(0.5)}
    oceans: ArrayLike = 3
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    gas_CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array([236.98, 337.16, 46.42, 30.88, 28.66, 4.11e-13])
            }
        }
    }

    fastchem_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array(
                    [
                        236.73845646,
                        332.03616807,
                        45.94958264,
                        29.82548282,
                        29.61919788,
                        3.96475584e-13,
                    ]
                )
            }
        }
    }

    assert gas_CHO_model.output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert gas_CHO_model.output.compare(fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_oxidised() -> None:
    """Tests C-H-O system at IW+2

    Similar to :cite:p:`BHS22{Table E, row 3}`.
    """

    planet: Planet = Planet(temperature=1400)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(2)}
    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 0.1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    gas_CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array([27.40, 218.48, 0.873, 3.25, 0.00129, 1.29e-11])
            }
        }
    }

    assert gas_CHO_model.output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="Checks result against previous work but not different functionality")
def test_CHO_highly_oxidised() -> None:
    """Tests C-H-O system at IW+4

    Similar to :cite:p:`BHS22{Table E, row 4}`.
    """

    planet: Planet = Planet(temperature=1400)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer(4)}
    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 5 * h_kg
    # Mass of O that gives the same solution as applying the buffer at IW+4
    # o_kg: ArrayLike = 3.25196e21
    mass_constraints: dict[str, ArrayLike] = {"H": h_kg, "C": c_kg}

    gas_CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array([5.78, 432.08, 10.21, 357.23, 7.13e-05, 1.14e-09])
            }
        }
    }

    assert gas_CHO_model.output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_middle_temperature() -> None:
    """Tests C-H-O system at 873 K"""

    planet: Planet = Planet(temperature=873)
    fugacity_constraints: dict[str, FugacityConstraintProtocol] = {"O2_g": IronWustiteBuffer()}
    oceans: ArrayLike = 1
    h_kg: ArrayLike = earth_oceans_to_hydrogen_mass(oceans)
    c_kg: ArrayLike = 1 * h_kg
    mass_constraints: dict[str, ArrayLike] = {"C": c_kg, "H": h_kg}

    gas_CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array([59.066, 18.320, 8.91e-4, 7.48e-4, 19.548, 1.27e-25])
            }
        }
    }

    assert gas_CHO_model.output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_low_temperature() -> None:
    """Tests C-H-O system at 450 K"""

    planet: Planet = Planet(temperature=450)
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

    gas_CHO_model.solve(
        state=planet, fugacity_constraints=fugacity_constraints, mass_constraints=mass_constraints
    )

    factsage_result: dict[str, Any] = {
        "gas": {
            "species": {
                "partial_pressure": np.array([55.475, 8.0, 2.12e-16, 1.24e-14, 16.037, 7.85e-54])
            }
        }
    }

    assert gas_CHO_model.output.compare(factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
