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
"""Comparisons with FactSage 8.2 and FastChem 3.1.1"""

# Want to use chemistry symbols so pylint: disable=invalid-name

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from atmodeller import AVOGADRO, __version__, debug_logger
from atmodeller.classes import InteriorAtmosphere
from atmodeller.jax_containers import CondensedSpecies, GasSpecies, Planet, Species
from atmodeller.jax_utilities import log_pressure_from_log_number_density
from atmodeller.solubility.jax_hydrogen_species import H2O_peridotite_sossi
from atmodeller.thermodata.jax_species_data import (
    C_cr_data,
    CH4_g_data,
    CO2_g_data,
    CO_g_data,
    H2_g_data,
    H2O_g_data,
    H2O_l_data,
    O2_g_data,
)
from atmodeller.thermodata.jax_thermo import IronWustiteBuffer, RedoxBufferProtocol
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
logger.setLevel(logging.INFO)

RTOL: float = 1.0e-8
"""Relative tolerance"""
ATOL: float = 1.0e-8
"""Absolute tolerance"""
TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage and FastChem"""
SCALING: float = AVOGADRO
"""Scale the numerical problem from molecules/m^3 to moles/m^3 if SCALING is AVODAGRO"""
TAU: float = 1.0e60
"""Tau scaling factor for species stability"""

INITIAL_NUMBER_DENSITY: float = 30.0
INITIAL_STABILITY: float = -100.0
planet: Planet = Planet()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_H_O(helper) -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint."""

    H2_g: Species = GasSpecies(H2_g_data)
    H2O_g: Species = GasSpecies(H2O_g_data)
    O2_g: Species = GasSpecies(O2_g_data)

    species: list[Species] = [H2_g, H2O_g, O2_g]
    interior_atmosphere: InteriorAtmosphere = InteriorAtmosphere(species, SCALING)

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 6.25774e20
    mass_constraints: dict[str, float] = {
        "H": h_kg,
        "O": o_kg,
    }

    # Initial solution guess number density (molecules/m^3)
    initial_number_density: ArrayLike = INITIAL_NUMBER_DENSITY * np.ones(
        len(species), dtype=np.float_
    )
    initial_stability: ArrayLike = INITIAL_STABILITY * np.ones_like(initial_number_density)

    interior_atmosphere.initialise_solve(
        planet, initial_number_density, initial_stability, mass_constraints=mass_constraints
    )
    solution: dict[str, ArrayLike] = interior_atmosphere.solve()

    fastchem_result: dict[str, float] = {
        "H2O_g": 76.45861543,
        "H2_g": 73.84378192,
        "O2_g": 8.91399329e-08,
    }

    assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_CHO_reduced(helper) -> None:
#     """C-H-O system at IW-2

#     Similar to :cite:p:`BHS22{Table E, row 1}`
#     """

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

#     warm_planet: Planet = Planet(surface_temperature=1400)

#     h_kg: float = earth_oceans_to_hydrogen_mass(3)
#     c_kg: float = 1 * h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
#             # FugacityConstraint(O2_g, 1.25e-15),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=warm_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "H2_g": 175.5,
#         "H2O_g": 13.8,
#         "CO_g": 6.21,
#         "CO2_g": 0.228,
#         "CH4_g": 38.07,
#         "O2_g": 1.25e-15,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_CHO_IW(helper) -> None:
#     """C-H-O system at IW+0.5

#     Similar to :cite:p:`BHS22{Table E, row 2}`.

#     The FastChem element abundance file is:

#     # test_CHO_IW from atmodeller
#     e-  0.0
#     H   12.00
#     O   11.54211516
#     C   10.92386535
#     """

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

#     warm_planet: Planet = Planet(surface_temperature=1400)

#     h_kg: float = earth_oceans_to_hydrogen_mass(3)
#     c_kg: float = 1 * h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
#             # FugacityConstraint(O2_g, 1.01633868e-14),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=warm_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "CH4_g": 28.66,
#         "CO2_g": 30.88,
#         "CO_g": 46.42,
#         "H2O_g": 337.16,
#         "H2_g": 236.98,
#         "O2_g": 4.11e-13,
#     }

#     fastchem_result: dict[str, float] = {
#         "CH4_g": 29.61919788,
#         "CO2_g": 29.82548282,
#         "CO_g": 45.94958264,
#         "H2O_g": 332.03616807,
#         "H2_g": 236.73845646,
#         "O2_g": 3.96475584e-13,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
#     assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_CHO_oxidised(helper) -> None:
#     """C-H-O system at IW+2

#     Similar to :cite:p:`BHS22{Table E, row 3}`
#     """

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

#     warm_planet: Planet = Planet(surface_temperature=1400)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)
#     c_kg: float = 0.1 * h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer(2)),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=warm_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "CH4_g": 0.00129,
#         "CO2_g": 3.25,
#         "CO_g": 0.873,
#         "H2O_g": 218.48,
#         "H2_g": 27.40,
#         "O2_g": 1.29e-11,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_CHO_highly_oxidised(helper) -> None:
#     """C-H-O system at IW+4

#     Similar to :cite:p:`BHS22{Table E, row 4}`
#     """

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

#     warm_planet: Planet = Planet(surface_temperature=1400)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)
#     c_kg: float = 5 * h_kg
#     o_kg: float = 3.25196e21

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             # BufferedFugacityConstraint(O2_g, IronWustiteBuffer(4)),
#             ElementMassConstraint("O", o_kg),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=warm_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(method="lm"), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "CH4_g": 7.13e-05,
#         "CO2_g": 357.23,
#         "CO_g": 10.21,
#         "H2O_g": 432.08,
#         "H2_g": 5.78,
#         "O2_g": 1.14e-09,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_CHO_middle_temperature(helper) -> None:
#     """C-H-O system at 873 K"""

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO_g: GasSpecies = GasSpecies("CO")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     O2_g: GasSpecies = GasSpecies("O2")

#     species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

#     cool_planet: Planet = Planet(surface_temperature=873)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)
#     c_kg: float = 1 * h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=cool_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(method="lm"), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "H2_g": 59.066,
#         "H2O_g": 18.320,
#         "CO_g": 8.91e-4,
#         "CO2_g": 7.48e-4,
#         "CH4_g": 19.548,
#         "O2_g": 1.27e-25,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_CHO_low_temperature(helper) -> None:
#     """C-H-O system at 450 K"""

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     O2_g: GasSpecies = GasSpecies("O2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     CO_g: GasSpecies = GasSpecies("CO")

#     species: Species = Species([H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g])

#     cool_planet: Planet = Planet(surface_temperature=450)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)
#     c_kg: float = 1 * h_kg
#     o_kg: float = 1.02999e20

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             # PressureConstraint(H2O_g, 8),
#             ElementMassConstraint("O", o_kg),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=cool_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverScipy(jac=True), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "H2_g": 55.475,
#         "H2O_g": 8.0,
#         "CO2_g": 1.24e-14,
#         "O2_g": 7.85e-54,
#         "CH4_g": 16.037,
#         "CO_g": 2.12e-16,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_graphite_condensed(helper) -> None:
#     """Graphite stable with around 50% condensed C mass fraction"""

#     O2_g: GasSpecies = GasSpecies("O2")
#     H2_g: GasSpecies = GasSpecies("H2")
#     CO_g: GasSpecies = GasSpecies("CO")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     C_cr: SolidSpecies = SolidSpecies("C")

#     species: Species = Species([O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr])

#     cool_planet: Planet = Planet(surface_temperature=873)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)
#     c_kg: float = 5 * h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#             ActivityConstraint(C_cr, 1),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=cool_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "O2_g": 1.27e-25,
#         "H2_g": 14.564,
#         "CO_g": 0.07276,
#         "H2O_g": 4.527,
#         "CO2_g": 0.061195,
#         "CH4_g": 96.74,
#         "activity_C_cr": 1.0,
#         "mass_C_cr": 3.54162e20,
#     }

#     interior_atmosphere.output(to_excel=True)

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_graphite_unstable(helper) -> None:
#     """C-H-O system at IW+0.5 with graphite unstable

#     Similar to :cite:p:`BHS22{Table E, row 2}`
#     """

#     O2_g: GasSpecies = GasSpecies("O2")
#     H2_g: GasSpecies = GasSpecies("H2")
#     CO_g: GasSpecies = GasSpecies("CO")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     CO2_g: GasSpecies = GasSpecies("CO2")
#     CH4_g: GasSpecies = GasSpecies("CH4")
#     C_cr: SolidSpecies = SolidSpecies("C")

#     species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g, C_cr])

#     warm_planet: Planet = Planet(surface_temperature=1400)

#     h_kg: float = earth_oceans_to_hydrogen_mass(3)
#     c_kg: float = 1 * h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
#             ElementMassConstraint("H", h_kg),
#             ElementMassConstraint("C", c_kg),
#             ActivityConstraint(C_cr, 1),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=warm_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(method="dogleg"), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "O2_g": 4.11e-13,
#         "H2_g": 236.98,
#         "CO_g": 46.42,
#         "H2O_g": 337.16,
#         "CO2_g": 30.88,
#         "CH4_g": 28.66,
#         "activity_C_cr": 0.12202,
#         # FactSage also predicts no C, so these values are set close to the atmodeller output so
#         # the test knows to pass.
#         "mass_C_cr": 941506.7454759097,
#     }

#     interior_atmosphere.output(to_excel=True)

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_water_condensed(helper) -> None:
#     """Condensed water at 10 bar"""

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     O2_g: GasSpecies = GasSpecies("O2")
#     H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")

#     species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

#     cool_planet: Planet = Planet(surface_temperature=411.75)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(H2_g, 6.5604),
#             ElementMassConstraint("H", h_kg),
#             ActivityConstraint(H2O_l, 1),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=cool_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverOptimistix(method="lm"), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "H2O_g": 3.3596,
#         "H2_g": 6.5604,
#         "O2_g": 5.6433e-58,
#         "activity_H2O_l": 1.0,
#         "mass_H2O_l": 1.23802e21,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_water_condensed_O_abundance(helper) -> None:
#     """Condensed water at 10 bar

#     This is the same test as above, but this time constraining the total pressure and oxygen
#     abundance.
#     """

#     H2_g: GasSpecies = GasSpecies("H2")
#     H2O_g: GasSpecies = GasSpecies("H2O")
#     O2_g: GasSpecies = GasSpecies("O2")
#     H2O_l: LiquidSpecies = LiquidSpecies("H2O")  # , thermodata_name="Water, 10 Bar")

#     species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

#     cool_planet: Planet = Planet(surface_temperature=411.75)

#     h_kg: float = earth_oceans_to_hydrogen_mass(1)
#     o_kg: float = 1.14375e21

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             ElementMassConstraint("O", o_kg),
#             ElementMassConstraint("H", h_kg),
#             ActivityConstraint(H2O_l, 1),
#         ]
#     )
#     interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
#         species=species, planet=cool_planet
#     )
#     solution: Solution = interior_atmosphere.solve(
#         solver=SolverScipy(method="lm", jac=True), constraints=constraints
#     )

#     factsage_result: dict[str, float] = {
#         "H2O_g": 3.3596,
#         "H2_g": 6.5604,
#         "O2_g": 5.6433e-58,
#         "activity_H2O_l": 1.0,
#         "mass_H2O_l": 1.247201e21,
#     }

#     interior_atmosphere.output(to_excel=True)

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# def test_graphite_water_condensed(helper, graphite_water_condensed) -> None:
#     """C and water in equilibrium at 430 K and 10 bar"""

#     solution: Solution = graphite_water_condensed

#     factsage_result: dict[str, float] = {
#         "CH4_g": 0.3241,
#         "CO2_g": 4.3064,
#         "CO_g": 2.77e-6,
#         "activity_C_cr": 1.0,
#         "H2O_g": 5.3672,
#         "activity_H2O_l": 1.0,
#         "H2_g": 0.0023,
#         "O2_g": 4.74e-48,
#         "mass_C_cr": 8.75101e19,
#         "mass_H2O_l": 2.74821e21,
#     }

#     assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
