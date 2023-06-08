#!/usr/bin/env python3

"""Driver script to provide a command line option to compute atmospheres."""

import argparse
import logging
import time

from atmodeller import OCEAN_MOLES, logger
from atmodeller.core import InteriorAtmosphereSystem, Molecule, Planet, SystemConstraint
from atmodeller.thermodynamics import (
    BasaltDixonCO2,
    LibourelN2,
    MolarMasses,
    NoSolubility,
    PeridotiteH2O,
)


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

    molecules: list[Molecule] = [
        Molecule(name="H2O", solubility=PeridotiteH2O()),
        Molecule(name="H2", solubility=NoSolubility()),
        Molecule(name="CO", solubility=NoSolubility()),
        Molecule(name="CO2", solubility=BasaltDixonCO2()),
        Molecule(name="CH4", solubility=NoSolubility()),
        Molecule(name="O2", solubility=NoSolubility()),
    ]

    planet: Planet = Planet()
    molar_masses: MolarMasses = MolarMasses()
    h_kg: float = args.oceans * OCEAN_MOLES * molar_masses.H2
    c_kg: float = args.ch_ratio * h_kg
    planet.fo2_shift = args.fo2_shift
    constraints: list[SystemConstraint] = [
        SystemConstraint(species="H", value=h_kg, field="mass"),
        SystemConstraint(species="C", value=c_kg, field="mass"),
    ]

    # Include nitrogen if desired.
    if args.nitrogen != 0:
        molecules.append(Molecule(name="N2", solubility=LibourelN2()))
        n_kg: float = args.nitrogen * 1.0e-6 * planet.mantle_mass
        constraints.append(SystemConstraint(species="N", value=n_kg, field="mass"))

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(molecules=molecules, planet=planet)
    system.solve(constraints, fo2_constraint=True)
    logger.info(system.pressures_dict)

    end: float = time.time()
    runtime: float = round(end - start, 1)
    logger.info("Execution time (seconds) = %0.1f", runtime)
    logger.info("Finished")


if __name__ == "__main__":
    main()
