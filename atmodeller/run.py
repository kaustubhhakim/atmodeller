#!/usr/bin/env python3

"""Driver script to provide a command line option to compute atmospheres."""

import argparse
import logging
import time

from atmodeller import OCEAN_MOLES, logger
from atmodeller.core import (
    BufferedFugacityConstraint,
    InteriorAtmosphereSystem,
    MassConstraint,
    Planet,
    SystemConstraint,
)
from atmodeller.thermodynamics import (
    BasaltDixonCO2,
    BasaltLibourelN2,
    GasPhase,
    NoSolubility,
    PeridotiteH2O,
    PhaseProtocol,
)
from atmodeller.utilities import MolarMasses


def main():
    """Main driver."""

    logger.setLevel(logging.INFO)

    logger.info("Started")
    start: float = time.time()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Atmosphere modeller")
    parser.add_argument(
        "-f",
        "--fo2_shift",
        help="fo2 shift in log10 units relative to IW",
        action="store",
        type=float,
        default=0,
    )
    parser.add_argument(
        "-c",
        "--ch_ratio",
        help="C/H ratio by wt %",
        action="store",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-o",
        "--oceans",
        help="Number of Earth oceans",
        action="store",
        type=float,
        default=1,
    )
    parser.add_argument(
        "-n",
        "--nitrogen",
        help="Nitrogen abundance in ppmw",
        action="store",
        type=float,
        default=2.8,  # 2.8 is the mantle value of N in ppmw.
    )

    args = parser.parse_args()

    molecules: list[PhaseProtocol] = [
        GasPhase(name="H2O", solubility=PeridotiteH2O()),
        GasPhase(name="H2", solubility=NoSolubility()),
        GasPhase(name="CO", solubility=NoSolubility()),
        GasPhase(name="CO2", solubility=BasaltDixonCO2()),
        GasPhase(name="CH4", solubility=NoSolubility()),
        GasPhase(name="O2", solubility=NoSolubility()),
    ]

    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = args.oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = args.ch_ratio * h_kg
    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(log10_shift=args.fO2_shift),
    ]

    # Include nitrogen if desired.
    if args.nitrogen != 0:
        molecules.append(GasPhase(name="N2", solubility=BasaltLibourelN2()))
        n_kg: float = args.nitrogen * 1.0e-6 * planet.mantle_mass
        constraints.append(MassConstraint(species="N", value=n_kg))

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    system.solve(constraints)
    logger.info(system.fugacities_dict)

    end: float = time.time()
    runtime: float = round(end - start, 1)
    logger.info("Execution time (seconds) = %0.1f", runtime)
    logger.info("Finished")


if __name__ == "__main__":
    main()
