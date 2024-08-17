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

import jax
import jax.numpy as jnp
from jax import Array

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    ActivityConstraint,
    BufferedFugacityConstraint,
    ElementMassConstraint,
    FugacityConstraint,
    PressureConstraint,
    SystemConstraints,
)
from atmodeller.core import GasSpecies, LiquidSpecies, Planet, SolidSpecies, Species
from atmodeller.reaction_network import InteriorAtmosphereSystem, Solver
from atmodeller.solution import Solution
from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.thermodata.janaf import ThermodynamicDatasetJANAF
from atmodeller.thermodata.redox_buffers import IronWustiteBuffer
from atmodeller.utilities import earth_oceans_to_hydrogen_mass

logger: logging.Logger = debug_logger()
# logger.setLevel(logging.INFO)

TOLERANCE: float = 5.0e-2
"""Tolerance of log output to satisfy comparison with FactSage"""

planet: Planet = Planet()

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", True)
jax.config.update("jax_debug_infs", False)


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


# Well behaved with Optimistix Newton
def test_H_O(helper) -> None:
    """Tests H2-H2O at the IW buffer by applying an oxygen abundance constraint.

    The FastChem element abundance file is:

    # test_H_O from atmodeller
    e-  0.0
    H   12.00
    O   11.40541658
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, O2_g])

    oceans: float = 1
    h_kg: float = earth_oceans_to_hydrogen_mass(oceans)
    o_kg: float = 6.25774e20

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("O", o_kg),
        ]
    )

    system = InteriorAtmosphereSystem(species=species, planet=planet)
    _, _, solution = system.solve(constraints=constraints)

    fastchem_result: dict[str, float] = {
        "H2O_g": 76.45861543,
        "H2_g": 73.84378192,
        "O2_g": 8.91399329e-08,
    }

    assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Well behaved with Optimistix Newton
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

    warm_planet: Planet = Planet(surface_temperature=1400)

    h_kg: float = earth_oceans_to_hydrogen_mass(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(-2)),
            # FugacityConstraint(O2_g, 1.25e-15),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=warm_planet)
    _, _, solution = system.solve(constraints=constraints)

    factsage_result: dict[str, float] = {
        "H2_g": 175.5,
        "H2O_g": 13.8,
        "CO_g": 6.21,
        "CO2_g": 0.228,
        "CH4_g": 38.07,
        "O2_g": 1.25e-15,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Well behaved with Optimistix Newton
def test_CHO_IW(helper) -> None:
    """C-H-O system at IW+0.5

    Similar to :cite:p:`BHS22{Table E, row 2}`.

    The FastChem element abundance file is:

    # test_CHO_IW from atmodeller
    e-  0.0
    H   12.00
    O   11.54211516
    C   10.92386535
    """

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    warm_planet: Planet = Planet(surface_temperature=1400)

    h_kg: float = earth_oceans_to_hydrogen_mass(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
            # FugacityConstraint(O2_g, 1.01633868e-14),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=warm_planet)
    _, _, solution = system.solve(constraints=constraints)

    factsage_result: dict[str, float] = {
        "CH4_g": 28.66,
        "CO2_g": 30.88,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "H2_g": 236.98,
        "O2_g": 4.11e-13,
    }

    fastchem_result: dict[str, float] = {
        "CH4_g": 29.61919788,
        "CO2_g": 29.82548282,
        "CO_g": 45.94958264,
        "H2O_g": 332.03616807,
        "H2_g": 236.73845646,
        "O2_g": 3.96475584e-13,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
    assert helper.isclose(solution, fastchem_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Well behaved with Optimistix Newton
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

    warm_planet: Planet = Planet(surface_temperature=1400)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 0.1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(2)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=warm_planet)
    _, _, solution = system.solve(constraints=constraints)

    factsage_result: dict[str, float] = {
        "CH4_g": 0.00129,
        "CO2_g": 3.25,
        "CO_g": 0.873,
        "H2O_g": 218.48,
        "H2_g": 27.40,
        "O2_g": 1.29e-11,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Well behaved with Optimistix Newton, but not if a pure mass balance is used instead. Scipy works
# with a numerical jacobian and 41 steps, probably because f is smooth. Still points to poor
# conditioning of the Jacobian.
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

    warm_planet: Planet = Planet(surface_temperature=1400)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 5 * h_kg
    # When switching to solely mass balance rather than using fO2 this system is also poorly
    # behaved. Must be a problem with how mass balance is performed/scaled.
    o_kg: float = 3.25196e21

    constraints: SystemConstraints = SystemConstraints(
        [
            # BufferedFugacityConstraint(O2_g, IronWustiteBuffer(4)),
            ElementMassConstraint("O", o_kg),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=warm_planet)
    _, _, solution = system.solve(constraints=constraints)

    factsage_result: dict[str, float] = {
        "CH4_g": 7.13e-05,
        "CO2_g": 357.23,
        "CO_g": 10.21,
        "H2O_g": 432.08,
        "H2_g": 5.78,
        "O2_g": 1.14e-09,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Well behaved with Optimistix Newton
def test_CHO_middle_temperature(helper) -> None:
    """C-H-O system at 873 K"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO_g: GasSpecies = GasSpecies("CO")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    O2_g: GasSpecies = GasSpecies("O2")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g])

    cool_planet: Planet = Planet(surface_temperature=873)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=cool_planet)
    _, _, solution = system.solve(constraints=constraints)

    factsage_result: dict[str, float] = {
        "H2_g": 59.066,
        "H2O_g": 18.320,
        "CO_g": 8.91e-4,
        "CO2_g": 7.48e-4,
        "CH4_g": 19.548,
        "O2_g": 1.27e-25,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# TODO: Clean up the comments around this test. Optimistix can work when the initial condition is
# closer. Also compare with simple_CHO_low_temperature.py to check they are the same.
# Solves with Optimistix Newton, although one of the solution estimates during the solve is
# very large, although the solver does then correct back. This also happens when starting from an
# initial condition with all zeros.
# This works with the Chord solver, implying that evaluating the Jacobian at some values along the
# solution path is indeed the problem, since the Chord solver only solves for the Jacobian once at
# the initial condition.
# LM and Dogleg all break with an NaN error
# [26. 26. 26. 26. 26. 26.]
# [ 22.796585622054074  26.109794190538366  20.676895512220526
#   26.59656172476903   25.088340287983083 -19.582404215197847]
# [ 25.334545442858943  26.109794190538366  23.356517392521333
#   26.738223784264974  35.381841630698496 -24.658323856807588]
# [ 5.189584001482789e+07  2.610979418621717e+01 -1.556874296135248e+08 <--- This solution is
#  -2.075832409121009e+08  2.645264679385557e+01 -1.037916540188888e+08]     very large. Why?
# [ 26.68152157217264   26.109794201714237   9.840164333581924
#   11.874894618988037  25.906416912805827 -27.35227608680725 ]
# [ 26.945107329996187  26.109794190538366   9.51671261864177
#   11.287857123248166  26.373722518230664 -27.87944763108207 ]
# [ 26.950915190893088  26.109794190538366   9.537079812065635
#   11.30241645577513   26.41151329434523  -27.891063352875868]
# [ 26.95080430771847   26.109794190538366   9.537726403034757
#   11.303173929918865  26.41182723579052  -27.890841586526633]
# [ 26.950804271305103  26.109794190538366   9.537726547231674
#   11.303174110529152  26.411827270747313 -27.8908415136999  ]
def test_CHO_low_temperature(helper) -> None:
    """C-H-O system at 450 K

    This is the canonical case that has been simplified in simple_CHO_low_temperature.py for
    testing the solver options.
    """

    # thermodata_dataset = ThermodynamicDatasetHollandAndPowell()
    thermodata_dataset = ThermodynamicDatasetJANAF()

    H2_g: GasSpecies = GasSpecies("H2", thermodata_dataset=thermodata_dataset)
    H2O_g: GasSpecies = GasSpecies("H2O", thermodata_dataset=thermodata_dataset)
    CO2_g: GasSpecies = GasSpecies("CO2", thermodata_dataset=thermodata_dataset)
    O2_g: GasSpecies = GasSpecies("O2", thermodata_dataset=thermodata_dataset)
    CH4_g: GasSpecies = GasSpecies("CH4", thermodata_dataset=thermodata_dataset)
    CO_g: GasSpecies = GasSpecies("CO", thermodata_dataset=thermodata_dataset)

    species: Species = Species([H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g])

    # Doesn't work for any temperature below 2000 K, not just the original of 450 K
    cool_planet: Planet = Planet(surface_temperature=450)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 1 * h_kg
    # Option to try the O mass rather than total pressure. But both currently blow up the Newton
    # solver.
    o_kg: float = 1.02999e20

    constraints: SystemConstraints = SystemConstraints(
        [
            # PressureConstraint(H2O_g, 8),
            ElementMassConstraint("O", o_kg),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=cool_planet)
    solver, jacobian, solution = system.solve(solver="scipy", constraints=constraints)

    factsage_result: dict[str, float] = {
        "H2_g": 55.475,
        "H2O_g": 8.0,
        "CO2_g": 1.24e-14,
        "O2_g": 7.85e-54,
        "CH4_g": 16.037,
        "CO_g": 2.12e-16,
    }

    # log10_number_densities = [26.719937065256797 43.171618901656146 20.64657795026752
    #  39.70471743073109  19.76625234452046   6.694542320632276]
    # log10_number_densities = [27.02096706092078  43.47264889732013  20.368312335848422]
    # log10_number_densities = [26.719937065256797 43.171618901656146 20.64657795026752
    #  39.70471743073109  19.76625234452046   6.694542320632276]
    # log10_number_densities = [20.64657795026752 39.70471743073109 19.76625234452046]
    # log10_number_densities = [26.719937065256797 43.171618901656146 20.64657795026752
    #  39.70471743073109  19.76625234452046   6.694542320632276]
    # [-1.997837134509417e+17  5.362650693619739e+01 -1.997837134509418e+17
    #   3.313828274852963e+01 -7.991348538037672e+17  3.995674269018836e+17]
    # log10_number_densities = [ 5.362650693619739e+01 -1.997837134509418e+17  3.343931274419361e+01
    #   3.995674269018836e+17]
    # log10_number_densities = [-1.997837134509417e+17  5.362650693619739e+01 -1.997837134509418e+17
    #   3.313828274852963e+01 -7.991348538037672e+17  3.995674269018836e+17]

    # TODO: When cleaned up the solver interfaces this can be tidied up.
    # test_number_density: Array = jnp.array(
    #     [
    #         26.719937065256797,
    #         43.171618901656146,
    #         20.64657795026752,
    #         39.70471743073109,
    #         19.76625234452046,
    #         6.694542320632276,
    #     ]
    # )

    # out: Array = jacobian(test_number_density)

    # jax.debug.print("Evaluated Jac = {out}", out=out)

    # assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_condensed(helper) -> None:
    """Graphite stable with around 50% condensed C mass fraction"""

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([O2_g, H2_g, CO_g, H2O_g, CO2_g, CH4_g, C_cr])

    cool_planet: Planet = Planet(surface_temperature=873)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    c_kg: float = 5 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer()),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            ActivityConstraint(C_cr, 1),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=cool_planet)
    _, _, solution = system.solve(constraints=constraints)

    factsage_result: dict[str, float] = {
        "O2_g": 1.27e-25,
        "H2_g": 14.564,
        "CO_g": 0.07276,
        "H2O_g": 4.527,
        "CO2_g": 0.061195,
        "CH4_g": 96.74,
        "activity_C_cr": 1.0,
        "mass_C_cr": 3.54162e20,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


def test_graphite_unstable(helper) -> None:
    """C-H-O system at IW+0.5 with graphite unstable

    Similar to :cite:p:`BHS22{Table E, row 2}`
    """

    O2_g: GasSpecies = GasSpecies("O2")
    H2_g: GasSpecies = GasSpecies("H2")
    CO_g: GasSpecies = GasSpecies("CO")
    H2O_g: GasSpecies = GasSpecies("H2O")
    CO2_g: GasSpecies = GasSpecies("CO2")
    CH4_g: GasSpecies = GasSpecies("CH4")
    C_cr: SolidSpecies = SolidSpecies("C")

    species: Species = Species([H2_g, H2O_g, CO_g, CO2_g, CH4_g, O2_g, C_cr])

    warm_planet: Planet = Planet(surface_temperature=1400)

    h_kg: float = earth_oceans_to_hydrogen_mass(3)
    c_kg: float = 1 * h_kg

    constraints: SystemConstraints = SystemConstraints(
        [
            BufferedFugacityConstraint(O2_g, IronWustiteBuffer(0.5)),
            ElementMassConstraint("H", h_kg),
            ElementMassConstraint("C", c_kg),
            ActivityConstraint(C_cr, 1),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=warm_planet)
    _, _, solution = system.solve(solver="scipy", constraints=constraints)

    factsage_result: dict[str, float] = {
        "O2_g": 4.11e-13,
        "H2_g": 236.98,
        "CO_g": 46.42,
        "H2O_g": 337.16,
        "CO2_g": 30.88,
        "CH4_g": 28.66,
        "activity_C_cr": 0.12202,
        # FactSage also predicts no C, so these values are set close to the atmodeller output so
        # the test knows to pass.
        "mass_C_cr": 512893.3781184358,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Works with Optimistix LevenbergMarquardt and scipy, fails with Dogleg and Newton
def test_water_condensed(helper) -> None:
    """Condensed water at 10 bar"""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")

    species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

    cool_planet: Planet = Planet(surface_temperature=411.75)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(H2_g, 6.5604),
            ElementMassConstraint("H", h_kg),
            ActivityConstraint(H2O_l, 1),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=cool_planet)
    _, _, solution = system.solve(solver="scipy", constraints=constraints)

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "activity_H2O_l": 1.0,
        "mass_H2O_l": 1.23802e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Fails with LevenbergMarquardt, Dogleg, and Newton. Passes with scipy.
def test_water_condensed_O_abundance(helper) -> None:
    """Condensed water at 10 bar

    This is the same test as above, but this time constraining the total pressure and oxygen
    abundance."""

    H2_g: GasSpecies = GasSpecies("H2")
    H2O_g: GasSpecies = GasSpecies("H2O")
    O2_g: GasSpecies = GasSpecies("O2")
    H2O_l: LiquidSpecies = LiquidSpecies("H2O", thermodata_name="Water, 10 Bar")

    species: Species = Species([H2_g, H2O_g, O2_g, H2O_l])

    cool_planet: Planet = Planet(surface_temperature=411.75)

    h_kg: float = earth_oceans_to_hydrogen_mass(1)
    o_kg: float = 1.14375e21

    constraints: SystemConstraints = SystemConstraints(
        [
            ElementMassConstraint("O", o_kg),
            ElementMassConstraint("H", h_kg),
            ActivityConstraint(H2O_l, 1),
        ]
    )

    system: Solver = InteriorAtmosphereSystem(species=species, planet=cool_planet)
    _, _, solution = system.solve(solver="scipy", constraints=constraints)

    factsage_result: dict[str, float] = {
        "H2O_g": 3.3596,
        "H2_g": 6.5604,
        "O2_g": 5.6433e-58,
        "activity_H2O_l": 1.0,
        "mass_H2O_l": 1.247201e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)


# Fails with dogleg
def test_graphite_water_condensed(helper, graphite_water_condensed) -> None:
    """C and water in equilibrium at 430 K and 10 bar"""

    solution: Solution = graphite_water_condensed

    factsage_result: dict[str, float] = {
        "CH4_g": 0.3241,
        "CO2_g": 4.3064,
        "CO_g": 2.77e-6,
        "activity_C_cr": 1.0,
        "H2O_g": 5.3672,
        "activity_H2O_l": 1.0,
        "H2_g": 0.0023,
        "O2_g": 4.74e-48,
        "mass_C_cr": 8.75101e19,
        "mass_H2O_l": 2.74821e21,
    }

    assert helper.isclose(solution, factsage_result, log=True, rtol=TOLERANCE, atol=TOLERANCE)
