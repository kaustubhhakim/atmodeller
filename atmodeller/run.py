#!/usr/bin/env python3

"""Driver script."""

import argparse
import logging
import pprint
import time
from typing import Any

from atmodeller.core import InteriorAtmosphereSystem

logger: logging.Logger = logging.getLogger("atmodeller")


def main():
    """Main driver."""

    logger.info("Started")
    start: float = time.time()

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Atmosphere modeller"
    )
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
        "-m", "--monte_carlo", help="Run Monte Carlo simulation", action="store_true"
    )
    parser.add_argument(
        "-n",
        "--nitrogen",
        help="Nitrogen abundance in ppmw",
        action="store",
        type=float,
        default=2.8,
    )  # 2.8 is the mantle value of N in ppmw
    parser.add_argument("-t", "--test", help="Test", action="store_true")
    args = parser.parse_args()
    kwargs: dict[str, Any] = vars(args)

    if args.monte_carlo:
        logger.info("Running a Monte Carlo simulation")
        # equilibrium_atmosphere_monte_carlo(kwargs["nitrogen"])
    else:
        logger.info("Running a single solve")
        n_ocean_moles: float = kwargs["oceans"]
        ch_ratio: float = kwargs["ch_ratio"]
        fo2_shift: float = kwargs["fo2_shift"]
        nitrogen_ppmw: float = kwargs["nitrogen"]
        interior_atmos = InteriorAtmosphereSystem()
        p_d: dict[str, float] = interior_atmos.equilibrium_atmosphere(
            n_ocean_moles, ch_ratio, fo2_shift, nitrogen_ppmw)
        logger.info(pprint.pformat(p_d))

    if args.test:
        test()

    end: float = time.time()
    runtime: float = round(end - start, 1)
    logger.info("Execution time (seconds) = %0.1f", runtime)
    logger.info("Finished")


def test():
    logger.info("Running test")
    interior_atmos = InteriorAtmosphereSystem()
    print(interior_atmos.surface_gravity)


if __name__ == "__main__":
    main()
