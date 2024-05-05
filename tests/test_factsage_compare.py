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

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (  # TotalPressureConstraint,
    ActivityConstraint,
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies, SolidSpecies
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_kg

logger: logging.Logger = debug_logger()

# 3% tolerance of log values to satisfy comparison with FactSage
TOLERANCE: float = 5.0e-2
FACTSAGE_COMPARISON: str = "Comparing with FactSage result"


def test_CHO_reduced(helper) -> None:
    """C-H-O system at IW-2

    Similar to :cite:p:`BHS22{Table E, row 1}`
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
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

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
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

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 0.1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
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

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 1400
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 5 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(4)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
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

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
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

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g, C_cr])

    planet: Planet = Planet()
    planet.surface_temperature = 873
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            FugacityConstraint(H2_g, 5.8),
            ElementMassConstraint("C", c_kg),
            ActivityConstraint(C_cr, 1),
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


def test_water_condensed_10bar(helper) -> None:
    """Condensed water at 10 bar"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O")

    species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

    planet: Planet = Planet()
    planet.surface_temperature = 411.75
    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    h_kg: float = earth_oceans_to_kg(1)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, value=7),
            ElementMassConstraint("H", h_kg),
            ActivityConstraint(H2O_l, 1),
        ]
    )

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5647,
        "O2_g": 5.63582e-58,
        "H2O_l": 1.0,
        "degree_of_condensation_H": 0.9040,
        # degree_of_condensation_O = 0.9628
    }

    system.solve(constraints)
    system.output(to_excel=True)
    assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# @pytest.mark.skip(reason="Unphysical since temperature is too high for stable liquid H2O")
# def test_graphite_water_condensed_10bar(helper) -> None:
#     """Graphite and condensed water at 10 bar"""

#     H2O_g: GasSpecies = GasSpecies("H2O")
#     H2_g: GasSpecies = GasSpecies("H2")
#     O2_g: GasSpecies = GasSpecies("O2")
#     # H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     # C_cr: SolidSpecies = SolidSpecies("C")

#     species: Species = Species([H2O_g, H2_g, O2_g, CO_g, CO2_g, CH4_g])  # , C_cr, H2O_l])

#     planet: Planet = Planet()
#     planet.surface_temperature = 450
#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

#     h_kg: float = earth_oceans_to_kg(1) * 1
#     c_kg = h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(H2O_g, 8),
#             # TotalPressureConstraint(10),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#             # ActivityConstraint(H2O_l, 1),
#             # ActivityConstraint(C_cr, 1),
#         ]
#     )

#     # This is the solution when the CH4 fugacity is fixed at 3.428 bar
#     factsage_result_CH4: dict[str, float] = {
#         "CH4_g": 3.4287054461076254,
#         "CO2_g": 0.6361664824724061,
#         "CO_g": 5.249106319198347e-06,
#         "C_cr": 1.0,
#         "H2O_g": 5.919505502971372,
#         "H2O_l": 1.0,
#         "H2_g": 0.01561731936431075,
#         "O2_g": 9.326226158104995e-46,
#         # "degree_of_condensation_C": 0.5687210428967688,
#         # "degree_of_condensation_H": 0.7721952621991293,
#     }

#     # This is the solution when instead, the total pressure is fixed at 10 bar
#     # pylint: disable=unused-variable
#     factsage_result_total_pressure: dict[str, float] = {
#         "CH4_g": 0.6241604132666166,
#         "CO2_g": 3.494658483175357,
#         "CO_g": 1.230275421512879e-05,
#         "C_cr": 1.0,
#         "H2O_g": 5.919505502971371,
#         "H2O_l": 1.0,
#         "H2_g": 0.00666330224360131,
#         "O2_g": 5.123183358036684e-45,
#         "degree_of_condensation_C": 0.6921548202806902,
#         "degree_of_condensation_H": 0.9099995423718071,
#     }

#     # Start solution
#     data_to_start = factsage_result_CH4
#     # Compare solution
#     data_to_compare = factsage_result_CH4

#     # FIXME: Need to refresh based on new object approach for initialising the dict
#     initial_solution = InitialSolutionDict(value=data_to_start, species=species)

#     system.solve(constraints, initial_solution=initial_solution)
#     system.output(to_excel=True)
#     assert helper.isclose(system, data_to_compare, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# @pytest.mark.skip(reason="debugging")
# def test_graphite_water_condensed_10bar_paolo(helper) -> None:
#     """Condensed water at 10 bar"""

#     species_with_water: Species = Species(
#         [
#             GasSpecies(formula="H2O"),
#             GasSpecies(formula="H2"),  # Very low abundance
#             GasSpecies(formula="O2"),
#             LiquidSpecies(formula="H2O", thermodata_name="Water, 10 Bar"),
#             GasSpecies(formula="CO"),
#             GasSpecies(formula="CO2"),
#             GasSpecies(formula="CH4"),
#             SolidSpecies(formula="C"),
#         ]
#     )
#     planet: Planet = Planet()
#     planet.surface_temperature = 430
#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species_with_water, planet=planet
#     )

#     # h_kg: float = earth_oceans_to_kg(1)

#     h_kg: float = 2e20 / 1000  # kg
#     c_kg: float = 1.1e21 / 1000  # kg
#     # Oxygen
#     o_kg: float = 1.144e21 / 1000  # kg

#     # Total pressure of 10 bar. FIXME: Need to manually tweak to get 10 bar
#     constraints: SystemConstraints = SystemConstraints(
#         [
#             # TotalPressureConstraint(value=10),
#             # FugacityConstraint(species="O2", value=5.3267e-58),
#             FugacityConstraint(species="H2O", value=5),  # 6.6205),
#             # FugacityConstraint(species="H2O", value=3.0157),
#             MassConstraint(species="H", value=h_kg),
#             MassConstraint(species="C", value=c_kg),
#         ]
#     )

#     factsage_result: dict[str, float] = {
#         "H2O_g": 5.4383,
#         "H2_g": 0.00839,
#         "O2_g": 3.75e-49,
#         "H2O_l": 1.0,
#         "C_cr": 1.0,
#         "CH4_g": 4.21,
#         "CO2_g": 0.34,
#         "CO_g": 7.82e-7,
#         "degree_of_condensation_H": 0.5,
#         "degree_of_condensation_C": 0.82,
#     }

#     # msg: str = "Compatible with FactSage result"
#     # system.isclose_tolerance(factsage_comparison, msg)

#     system.solve(constraints)
#     system.output(to_excel=True)
#     assert helper.isclose(system, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
