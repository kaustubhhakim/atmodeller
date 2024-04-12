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
"""Comparison tests with FactSage"""

# Want to use chemistry symbols so pylint: disable=invalid-name

from __future__ import annotations

import logging

import pytest

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (  # IronWustiteBufferConstraintOneill,; PressureConstraint,
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    MassConstraint,
    SystemConstraints,
    TotalPressureConstraint,
)
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.utilities import earth_oceans_to_kg

logger: logging.Logger = debug_logger()

# Tolerance to satisfy comparison with FactSage
TOLERANCE: float = 1.0e-3
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
            # IronWustiteBufferConstraintOneill(log10_shift=-2),
            # PressureConstraint(species="O2", value=2.774505894510357e-15),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    system.solve(constraints)

    # FIXME: To recompute
    factsage_result: dict[str, float] = {
        "H2_g": 175.4554515669669,
        "H2O_g": 13.776152804526566,
        "CO_g": 6.221337247436662,
        "CO2_g": 0.22549239036959995,
        "CH4_g": 38.10481325867202,
        "O2_g": 1.2530641173017539e-15,
    }

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
            # IronWustiteBufferConstraintOneill(log10_shift=0.5),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    system.solve(constraints)

    # FIXME: To recompute
    factsage_result: dict[str, float] = {
        "CH4_g": 28.673181986100104,
        "CO2_g": 30.67480325405014,
        "CO_g": 46.64081376634055,
        "H2O_g": 337.3541612052904,
        "H2_g": 236.78678795816822,
        "O2_g": 4.1258015513565896e-13,
    }

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
            # IronWustiteBufferConstraintOneill(log10_shift=2),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    system.solve(constraints)

    # FIXME: To recompute
    factsage_result: dict[str, float] = {
        "CH4_g": 0.0013660718550200582,
        "CO2_g": 3.234021086939191,
        "CO_g": 0.8916119856511471,
        "H2O_g": 218.12574742133162,
        "H2_g": 27.760484694381987,
        "O2_g": 1.2549051460563437e-11,
    }

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
            # IronWustiteBufferConstraintOneill(log10_shift=4),
            MassConstraint(species="H", value=h_kg),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    system.solve(constraints)

    # FIXME: To recompute
    factsage_result: dict[str, float] = {
        "CH4_g": 5.373946431827908e-05,
        "CO2_g": 357.8233109893475,
        "CO_g": 9.620427735331576,
        "H2O_g": 432.48249470696953,
        "H2_g": 5.3676148627962545,
        "O2_g": 1.3195489273596867e-09,
    }

    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_no_condensed(helper) -> None:
    """Graphite with near 0% condensed C mass fraction."""

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
            FugacityConstraint(species="H2", value=8.55),
            MassConstraint(species="C", value=c_kg),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2_g": 8.497,
        "H2O_g": 2.644,
        "CO_g": 0.0721,
        "CO2_g": 0.0607,
        "CH4_g": 33.29,
        "O2_g": 1.274e-25,
        "C_cr": 1.0,
        "degree_of_condensation_C": 0.0057,
    }

    system.solve(constraints)

    assert helper.isclose(system, factsage_result, log=True, rtol=5e-2, atol=5e-2)


def test_graphite_half_condensed(helper) -> None:
    """Graphite with around 50% condensed C mass fraction."""

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
        "H2_g": 5.766,
        "H2O_g": 1.790,
        "CO_g": 0.072,
        "CO2_g": 0.060,
        "CH4_G": 15.33,
        "O2_G": 1.268e-25,
        "C_cr": 1.0,
        "degree_of_condensation_C": 0.513,
    }

    system.solve(constraints)

    assert helper.isclose(system, factsage_result, log=True, rtol=5e-3, atol=5e-3)


@pytest.mark.skip(reason="debugging")
def test_water_condensed_10bar(helper) -> None:
    """Condensed water at 10 bar"""

    species_with_water: Species = Species(
        [
            GasSpecies(formula="H2O"),
            GasSpecies(formula="H2"),
            GasSpecies(formula="O2"),
            LiquidSpecies(formula="H2O", name="Water, 10 Bar"),
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

    system.solve(constraints)

    system.output(to_excel=True)

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

    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
