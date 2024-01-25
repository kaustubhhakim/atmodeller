#!/usr/bin/env python3

"""Plotting

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from atmodeller import debug_logger
from atmodeller.output import Output

# Reinstate below when preliminary development and testing is complete
# logger: logging.Logger = logging.getLogger(__name__)
logger: logging.Logger = debug_logger()

# TODO: From Maggie. Use constants already in atmodeller.
OCEAN_MOLES: float = 7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.
OCEAN_MASS: float = OCEAN_MOLES * (2.016 / 1000)


@dataclass
class Plotter:
    """Plotter

    Args:
        output: Output
    """

    output: Output
    dataframes: dict[str, pd.DataFrame] = field(init=False)

    def __post_init__(self):
        self.dataframes = self.output.to_dataframes()
        logger.info("Found data for %s", self.dataframes.keys())

    @classmethod
    def read_pickle(cls, pickle_file: Path | str) -> Plotter:
        output: Output = Output.read_pickle(pickle_file)
        return cls(output)

    def get_element_ratio_in_reservoir(
        self, element1: str, element2: str, reservoir: str = "melt"
    ) -> pd.Series:
        """Gets the mass ratio of two elements in a reservoir

        Args:
            element1: Element in the numerator
            element2: Element in the denominator
            reservoir: Can be 'atmosphere', 'solid', 'melt', or 'total'

        Returns:
            A series of the ratio
        """
        column_name: str = f"{reservoir}_mass"

        required_elements: list = [element1, element2]
        for elements in required_elements:
            assert f"{elements}_totals" in self.dataframes

        element1_mass: pd.Series = self.dataframes[f"{element1}_totals"][column_name]
        element2_mass: pd.Series = self.dataframes[f"{element2}_totals"][column_name]

        return element1_mass / element2_mass

    def plot_element_ratio_grid(self, reservoir: str = "melt") -> None:
        """Plots a grid of the data in element ratio space.

        Args:
            reservoir: Can be 'atmosphere', 'solid', 'melt', or 'total'
        """
        # TODO: Generalise, but to get up and running just work with C/H and C/O
        CH_ratio: pd.Series = self.get_element_ratio_in_reservoir("C", "H", reservoir)
        CO_ratio: pd.Series = self.get_element_ratio_in_reservoir("C", "O", reservoir)

        X, Y = np.meshgrid(CO_ratio, CH_ratio)
        logger.info("number of data points = %d", CO_ratio.size)
        _, ax = plt.subplots()
        ax.scatter(X.flatten(), Y.flatten(), s=0.1)
        ax.set_xlabel("C/O")
        ax.set_ylabel("C/H")
        ax.set_title("Data points")
        # plt.savefig("Ratio_Grid.jpg", dpi=500)
        plt.show()


def main():
    """Test area"""

    # Let's say you have some model output in a directory. The path must be relative to this
    # file location (i.e. plot.py)
    filename: Path = Path("../notebooks/simpleHCsystem.pkl")
    plotter: Plotter = Plotter.read_pickle(filename)
    plotter.plot_element_ratio_grid("total")


if __name__ == "__main__":
    main()
