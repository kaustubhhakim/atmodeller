#!/usr/bin/env python3

"""Driver script."""

import argparse
import csv
import logging
import pprint
import time
from typing import Any

import numpy as np

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
        "--n_ocean_moles",
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
        default=2.8,  # 2.8 is the mantle value of N in ppmw
    )
    parser.add_argument(
        "-r",
        "--realisations",
        help="Number of realisations for the Monte Carlo simulation",
        action="store",
        type=int,
        default=10,
    )
    parser.add_argument("-t", "--test", help="Test", action="store_true")
    args = parser.parse_args()
    kwargs: dict[str, Any] = vars(args)

    interior_atmos_system = InteriorAtmosphereSystem()

    if args.monte_carlo:
        logger.info("Running a Monte Carlo simulation")
        monte_carlo_simulation(kwargs, interior_atmos_system)
    else:
        single_solve(kwargs, interior_atmos_system)

    end: float = time.time()
    runtime: float = round(end - start, 1)
    logger.info("Execution time (seconds) = %0.1f", runtime)
    logger.info("Finished")


def single_solve(
    kwargs: dict[str, Any], interior_atmos_system: InteriorAtmosphereSystem
) -> dict[str, float]:
    """Solve an atmosphere-interior system.

    Args:
        kwargs: A dictionary of kwargs.
        interior_atmos_system: The interior atmosphere system.

    Returns:
        A dictionary of the solution and some input parameters.
    """
    n_ocean_moles: float = kwargs["n_ocean_moles"]
    ch_ratio: float = kwargs["ch_ratio"]
    fo2_shift: float = kwargs["fo2_shift"]
    nitrogen_ppmw: float = kwargs["nitrogen"]
    output: dict[str, float] = interior_atmos_system.solve(
        n_ocean_moles=n_ocean_moles,
        ch_ratio=ch_ratio,
        fo2_shift=fo2_shift,
        nitrogen_ppmw=nitrogen_ppmw,
    )
    logger.debug(pprint.pformat(output))

    return output


def monte_carlo_simulation(
    kwargs: dict[str, Any], interior_atmos_system: InteriorAtmosphereSystem
):
    """Monte Carlo simulation to produce realisations of atmospheric conditions.

    Args:
        kwargs: A dictionary of kwargs.
        interior_atmos_system: The interior atmosphere system.
    """
    number: int = kwargs["realisations"]
    # Parameters are normally distributed between bounds as given below.
    n_ocean_moles_a: np.ndarray = np.random.uniform(1, 10, number)
    ch_ratio_a: np.ndarray = np.random.uniform(0.1, 1, number)
    fo2_shift_a: np.ndarray = np.random.uniform(-4, 4, number)

    out: list[dict[str, float]] = []
    for realisation in range(number):
        logger.info("Realisation number = %d", realisation)
        kwargs["n_ocean_moles"] = n_ocean_moles_a[realisation]
        kwargs["ch_ratio"] = ch_ratio_a[realisation]
        kwargs["fo2_shift"] = fo2_shift_a[realisation]
        output: dict[str, float] = single_solve(kwargs, interior_atmos_system)
        out.append(output)

    filename: str = "equilibrium_atmosphere_monte_carlo.csv"
    logger.info("Writing output to: %s", filename)
    fieldnames: list[str] = list(out[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)


if __name__ == "__main__":
    main()
