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
"""Comparisons with FactSage 8.2"""

# Want to use chemistry symbols so pylint: disable=invalid-name

from __future__ import annotations

import logging

import pytest

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.utilities import earth_oceans_to_kg

logger: logging.Logger = debug_logger()

# 3% tolerance of log values to satisfy comparison with FactSage
TOLERANCE: float = 3.0e-2
FACTSAGE_COMPARISON: str = "Comparing with FactSage result"


def test_CHO_reduced(helper) -> None:
    """C-H-O system at IW-2

    Similar to :cite:p:`BHS22{Table E, row 1}`
    """

    species: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintHirschmann(log10_shift=-2),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 175.5,
        "H2O_g": 13.8,
        "CO_g": 6.21,
        "CO2_g": 0.228,
        "CH4_g": 38.07,
        "O2_g": 1.25e-15,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_IW(helper) -> None:
    """C-H-O system at IW

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    species: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintHirschmann(log10_shift=0.5),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 28.66,
        "CO2_g": 30.88,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "H2_g": 236.98,
        "O2_g": 4.11e-13,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_oxidised(helper) -> None:
    """C-H-O system at IW+2

    Similar to :cite:p:`BHS22{Table E, row 3}`
    """

    species: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 0.1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintHirschmann(log10_shift=2),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 0.00129,
        "CO2_g": 3.25,
        "CO_g": 0.873,
        "H2O_g": 218.48,
        "H2_g": 27.40,
        "O2_g": 1.29e-11,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_highly_oxidised(helper) -> None:
    """C-H-O system at IW+4

    Similar to :cite:p:`BHS22{Table E, row 4}`
    """

    species: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 5 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintHirschmann(log10_shift=4),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "CH4_g": 7.13e-05,
        "CO2_g": 357.23,
        "CO_g": 10.21,
        "H2O_g": 432.08,
        "H2_g": 5.78,
        "O2_g": 1.14e-09,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_CHO_low_temperature(helper) -> None:
    """C-H-O system at 873 K"""

    species: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintHirschmann(),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 59.066,
        "H2O_g": 18.320,
        "CO_g": 8.91e-4,
        "CO2_g": 7.48e-4,
        "CH4_g": 19.548,
        "O2_g": 1.27e-25,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_half_condensed(helper) -> None:
    """Graphite with 50% condensed C mass fraction"""

    species: Species = Species(
        [
            GasSpecies(formula="H2"),
            GasSpecies(formula="H2O"),
            GasSpecies(formula="CO"),
            GasSpecies(formula="CO2"),
            GasSpecies(formula="CH4"),
            GasSpecies(formula="O2"),
            SolidSpecies(formula="C"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            IronWustiteBufferConstraintHirschmann(),
            FugacityConstraint(species="H2", value=5.8),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 5.802,
        "H2O_g": 1.789,
        "CO_g": 0.07185,
        "CO2_g": 0.06,
        "CH4_g": 15.30,
        "O2_g": 1.2525e-25,
        "C_cr": 1.0,
        "degree_of_condensation_C": 0.51348,
    }

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.skip(reason="debugging")
def test_water_condensed_10bar(helper) -> None:
    """Condensed water at 10 bar"""

    species_with_water: Species = Species(
        [
            GasSpecies(formula="H2O"),
            GasSpecies(formula="H2"),
            GasSpecies(formula="O2"),
            LiquidSpecies(formula="H2O", thermodata_name="Water, 10 Bar"),
        ]
    )
    planet: Planet = Planet()
    planet.surface_temperature = 411.75
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
        species=species_with_water, planet=planet
    )

    h_kg: float = earth_oceans_to_kg(1)

    constraints: SystemConstraints = SystemConstraints(
        [
            # TotalPressureConstraint(value=10),
            # FugacityConstraint(species="O2", value=5.3267e-58),
            FugacityConstraint(species="H2", value=6.6205),
            # FugacityConstraint(species="H2O", value=3.0157),
            MassConstraint(species="H", value=h_kg),
        ]
    )

    # TODO: Update below when recomputed
    # Calculated by Paolo 19/02/2024
    # factsage_comparison: dict[str, float] = {
    #     "H2": 5.766,
    #     "H2O": 1.790,
    #     "O2": 1.268e-25,
    #     "degree_of_condensation_H": 0.513,
    # }

    factsage_result: dict[str, float] = {
        "H2O_g": 5.178544339893213,
        "H2_g": 4.821455660106786,
        "O2_g": 4.431323828352432e-52,
        "H2O_l": 1.0,
        "degree_of_condensation_H": 0.9344220206230801,
    }

    # TODO: Update
    # msg: str = "Compatible with FactSage result"
    # system.isclose_tolerance(factsage_comparison, msg)

    system.solve(constraints)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
