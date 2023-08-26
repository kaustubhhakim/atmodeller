#!/usr/bin/env python3

"""Driver script to provide a command line option to compute atmospheres.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import argparse
import logging
import time

from atmodeller import logger
from atmodeller.constraints import (
    BufferedFugacityConstraint,
    MassConstraint,
    SystemConstraint,
    SystemConstraints,
)
from atmodeller.core import InteriorAtmosphereSystem, Planet, Species
from atmodeller.interfaces import NoSolubility
from atmodeller.solubilities import BasaltDixonCO2, BasaltLibourelN2, PeridotiteH2O
from atmodeller.thermodynamics import GasSpecies
from atmodeller.utilities import earth_oceans_to_kg


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

    species: Species = Species(
        [
            GasSpecies(chemical_formula="H2O", solubility=PeridotiteH2O()),
            GasSpecies(chemical_formula="H2", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO", solubility=NoSolubility()),
            GasSpecies(chemical_formula="CO2", solubility=BasaltDixonCO2()),
            GasSpecies(chemical_formula="CH4", solubility=NoSolubility()),
            GasSpecies(chemical_formula="O2", solubility=NoSolubility()),
        ]
    )

    planet: Planet = Planet()
    h_kg: float = earth_oceans_to_kg(args.oceans)
    c_kg: float = args.ch_ratio * h_kg
    constraints: list[SystemConstraint] = [
        MassConstraint(species="H", value=h_kg),
        MassConstraint(species="C", value=c_kg),
        BufferedFugacityConstraint(log10_shift=args.fO2_shift),
    ]

    # Include nitrogen if desired.
    if args.nitrogen != 0:
        species.append(GasSpecies(chemical_formula="N2", solubility=BasaltLibourelN2()))
        n_kg: float = args.nitrogen * 1.0e-6 * planet.mantle_mass
        constraints.append(MassConstraint(species="N", value=n_kg))

    system: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species, planet=planet)
    system_constraints: SystemConstraints = SystemConstraints(constraints)
    system.solve(system_constraints)
    logger.info(system.pressures_dict)

    end: float = time.time()
    runtime: float = round(end - start, 1)
    logger.info("Execution time (seconds) = %0.1f", runtime)
    logger.info("Finished")


if __name__ == "__main__":
    main()
