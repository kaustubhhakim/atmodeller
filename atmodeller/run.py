#!/usr/bin/env python3

"""Driver script for the package."""

import argparse
import logging
import pprint
import time

from atmodeller.core import (
    equilibrium_atmosphere,
    equilibrium_atmosphere_MC,
    get_global_parameters,
)

logger: logging.Logger = logging.getLogger("atmodeller")


def main():
    """Main driver."""

    logger.info("Started")
    start: float = time.time()

    parser = argparse.ArgumentParser(description="Atmosphere modeller")
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

    args = parser.parse_args()
    kwargs = vars(args)

    if args.monte_carlo:
        logger.info("Running a Monte Carlo simulation")
        equilibrium_atmosphere_MC(kwargs["nitrogen"])
    else:
        logger.info("Running a single solve")
        global_d = get_global_parameters()
        n_ocean_moles = kwargs["oceans"]
        ch_ratio = kwargs["ch_ratio"]
        fo2_shift = kwargs["fo2_shift"]
        nitrogen = kwargs["nitrogen"]
        p_d = equilibrium_atmosphere(
            n_ocean_moles, ch_ratio, fo2_shift, global_d, nitrogen
        )
        logger.info(pprint.pformat(p_d))

    end: float = time.time()
    runtime: int = int(round(end - start, 0))
    logger.info("Execution time (seconds) = %d", runtime)
    logger.info("Finished")


if __name__ == "__main__":
    main()
