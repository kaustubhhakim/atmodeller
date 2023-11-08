"""Tests for the Holland and Powell EOS models

See the LICENSE file for licensing information.
"""

from typing import Type

from atmodeller import __version__, debug_logger
from atmodeller.constraints import (
    FugacityConstraint,
    IronWustiteBufferConstraintHirschmann,
    SystemConstraints,
)
from atmodeller.eos.holland import (
    CORKCH4HP91,
    CORKCO2HP98,
    CORKCOHP91,
    CORKH2HP91,
    CORKH2OHP98,
    CORKSimpleCO2HP91,
    MRKSimpleCO2HP91,
)
from atmodeller.interfaces import (
    GasSpecies,
    IdealGas,
    NoSolubility,
    ThermodynamicData,
    ThermodynamicDataBase,
)
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet, Species
from atmodeller.solubilities import PeridotiteH2O

# Tolerances to compare the test results with target output.
# rtol: float = 1.0e-8
# atol: float = 1.0e-8

thermodynamic_data: Type[ThermodynamicDataBase] = ThermodynamicData

debug_logger()


def test_version():
    """Test version."""
    assert __version__ == "0.1.0"


def test_MRK(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, MRKSimpleCO2HP91, 9.80535714428564)


def test_CorkH2(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, CORKH2HP91, 4.672042007568433)


def test_CorkCO(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, CORKCOHP91, 7.737070657107842)


def test_CorkCH4(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, CORKCH4HP91, 8.013532244610671)


def test_simple_CorkCO2(check_values) -> None:
    check_values.fugacity_coefficient(2000, 10e3, CORKSimpleCO2HP91, 7.120242298956865)


def test_CorkCO2_at_P0(check_values) -> None:
    """Below P0 so virial contribution excluded."""
    check_values.fugacity_coefficient(2000, 2e3, CORKCO2HP98, 1.6063624424808558)


def test_CorkCO2_above_P0(check_values) -> None:
    """Above P0 so virial contribution included."""
    check_values.fugacity_coefficient(2000, 10e3, CORKCO2HP98, 7.4492345831832525)


def test_CorkH2O_above_Tc_below_P0(check_values) -> None:
    """Above Tc and below P0."""
    check_values.fugacity_coefficient(2000, 1e3, CORKH2OHP98, 1.048278616058322)


def test_CorkH2O_above_Tc_above_P0(check_values) -> None:
    """Above Tc and above P0."""
    check_values.fugacity_coefficient(2000, 5e3, CORKH2OHP98, 1.3444013638026706)


def test_CorkH2O_below_Tc_below_Psat(check_values) -> None:
    """Below Tc and below Psat."""
    # Psat = 0.118224 kbar at T = 600 K.
    check_values.fugacity_coefficient(600, 0.1e3, CORKH2OHP98, 0.7910907770688191)


def test_CorkH2O_below_Tc_above_Psat(check_values) -> None:
    """Below Tc and above Psat."""
    # Psat = 0.118224 kbar at T = 600 K.
    check_values.fugacity_coefficient(600, 1e3, CORKH2OHP98, 0.14052644311851598)


def test_CorkH2O_below_Tc_above_P0(check_values) -> None:
    """Below Tc and above P0."""
    check_values.fugacity_coefficient(600, 10e3, CORKH2OHP98, 0.40066985009753664)


# TODO: Test is just for an ideal gas so not particularly useful
# def test_H_fO2() -> None:
#     """Tests H2-H2O at the IW buffer."""

#     species: Species = Species(
#         [
#             GasSpecies(
#                 chemical_formula="H2O",
#                 solubility=PeridotiteH2O(),
#                 thermodynamic_class=thermodynamic_data,
#                 eos=IdealGas(),
#             ),
#             GasSpecies(
#                 chemical_formula="H2",
#                 solubility=NoSolubility(),
#                 thermodynamic_class=thermodynamic_data,
#                 eos=IdealGas(),
#             ),
#             GasSpecies(
#                 chemical_formula="O2",
#                 solubility=NoSolubility(),
#                 thermodynamic_class=thermodynamic_data,
#             ),
#         ]
#     )

#     oceans: float = 1
#     planet: Planet = Planet()
#     h_kg: float = earth_oceans_to_kg(oceans)

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             MassConstraint(species="H", value=h_kg),
#             IronWustiteBufferConstraintHirschmann(),
#         ]
#     )

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

#     target_pressures: dict[str, float] = dict(
#         [("H2O", 0.19626421729663665), ("H2", 0.19386112601058758), ("O2", 8.69970008669977e-08)]
#     )

#     system.solve(SystemConstraints(constraints))
#     assert system.isclose(target_pressures)


def test_H2_with_cork() -> None:
    """Tests H2-H2O at the IW buffer."""

    species: Species = Species(
        [
            GasSpecies(
                chemical_formula="H2O",
                solubility=PeridotiteH2O(),
                thermodynamic_class=thermodynamic_data,
                eos=IdealGas(),  # This is the default if nothing specified
            ),
            GasSpecies(
                chemical_formula="H2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                eos=CORKH2HP91,
            ),
            GasSpecies(
                chemical_formula="O2",
                solubility=NoSolubility(),
                thermodynamic_class=thermodynamic_data,
                eos=IdealGas(),  # This is the default if nothing specified
            ),
        ]
    )

    # oceans: float = 1
    planet: Planet = Planet(surface_temperature=2000)
    # h_kg: float = earth_oceans_to_kg(oceans)

    constraints: SystemConstraints = SystemConstraints(
        [
            FugacityConstraint(species="H2", value=1e3),
            IronWustiteBufferConstraintHirschmann(),
        ]
    )

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

    target_pressures: dict[str, float] = {
        "H2": 747.5737656770727,
        "H2O": 1072.4328856736947,
        "O2": 9.76211086495026e-08,
    }

    system.solve(SystemConstraints(constraints))
    assert system.isclose(target_pressures)


# More complicated test to possibly reinstate at some point.
# def test_CORK() -> None:
#     """Tests H2-H2O-O2-CO-CO2-CH4 at the IW buffer."""

#     fugacity_models: dict[str, RealGasABC] = get_holland_fugacity_models()

#     species: Species = Species(
#         [
#             GasSpecies(
#                 chemical_formula="H2",
#                 solubility=BasaltH2(),
#                 thermodynamic_class=thermodynamic_data,
#                 fugacity_coefficient=fugacity_models["H2"],
#             ),
#             GasSpecies(
#                 chemical_formula="H2O",
#                 solubility=PeridotiteH2O(),
#                 thermodynamic_class=thermodynamic_data,
#                 fugacity_coefficient=fugacity_models["H2O"],
#             ),
#             GasSpecies(
#                 chemical_formula="O2",
#                 solubility=NoSolubility(),
#                 thermodynamic_class=thermodynamic_data,
#             ),
#             GasSpecies(
#                 chemical_formula="CO",
#                 solubility=NoSolubility(),
#                 thermodynamic_class=thermodynamic_data,
#                 fugacity_coefficient=fugacity_models["CO"],
#             ),
#             GasSpecies(
#                 chemical_formula="CO2",
#                 solubility=BasaltDixonCO2(),
#                 thermodynamic_class=thermodynamic_data,
#                 fugacity_coefficient=fugacity_models["CO2"],
#             ),
#             GasSpecies(
#                 chemical_formula="CH4",
#                 solubility=NoSolubility(),
#                 thermodynamic_class=thermodynamic_data,
#                 fugacity_coefficient=fugacity_models["CH4"],
#             ),
#             # SolidSpecies(
#             #     chemical_formula="C",
#             #     name_in_thermodynamic_data="graphite",
#             #     thermodynamic_class=thermodynamic_data,
#             # ),  # Ideal activity by default.
#         ]
#     )

#     oceans: float = 10
#     planet: Planet = Planet()
#     planet.surface_temperature = 2000  # K 600 + 273  # K
#     h_kg: float = earth_oceans_to_kg(oceans)
#     c_kg: float = h_kg

#     constraints: SystemConstraints = SystemConstraints(
#         [
#             FugacityConstraint(species="H2", value=958),
#             # MassConstraint(species="H", value=h_kg),
#             # PressureConstraint(species="H2", value=734),
#             IronWustiteBufferConstraintHirschmann(),
#             # FugacityConstraint(species="CO", value=1e3),
#             MassConstraint(species="C", value=c_kg),
#             # FugacityConstraint(species="CH4", value=1e2),
#         ]
#     )

#     system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)

#     target_pressures: dict[str, float] = {
#         "CH4": 10.362368367415021,
#         "CO": 276.0229490743045,
#         "CO2": 64.1797264653132,
#         "H2": 696.9735901947636,
#         "H2O": 933.3318808785544,
#         "O2": 9.862056623578625e-08,
#     }

#     system.solve(SystemConstraints(constraints))
#     assert system.isclose(target_pressures)
