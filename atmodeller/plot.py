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
import seaborn as sns

from atmodeller import debug_logger
from atmodeller.output import Output
from atmodeller.utilities import UnitConversion

# Reinstate below when preliminary development and testing is complete
# logger: logging.Logger = logging.getLogger(__name__)
logger: logging.Logger = debug_logger()


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
    ) -> np.ndarray:
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
        mass_ratio: pd.Series = element1_mass / element2_mass

        return mass_ratio.to_numpy(copy=True)

    @staticmethod
    def fO2_categorise(fo2_shift: pd.Series) -> str:
        if fo2_shift < -1:
            return "reduced (IW<1)"
        elif fo2_shift < 1:
            return "IW"
        else:
            return "oxidised (IW>1)"

    def species_pairplot(self, species: tuple[str, ...] = ("C", "O", "H", "N")) -> None:
        """Pair plot of species"""

        # Categorise by oxygen fugacity
        fO2_shift: pd.Series = self.dataframes["extra"]["fO2_shift"]
        fO2_categorise = fO2_shift.apply(self.fO2_categorise)
        fO2_categorise.name = "Oxygen fugacity"
        threshold: float = 1e-5

        to_weight_percent: float = UnitConversion.ppm_to_wt_percent()
        output: list[pd.Series] = [fO2_categorise]

        for entry in species:
            # First, try to find species totals (assuming elemental)
            try:
                totals: pd.DataFrame = self.dataframes[f"{entry}_totals"]
            # Otherwise, get the species directly
            except KeyError:
                totals = self.dataframes[entry]
            atmos: pd.Series = totals["atmosphere_ppmw"] * to_weight_percent
            atmos.name = f"{entry} atmos (wt %)"
            output.append(atmos)
            melt: pd.Series = totals["melt_ppmw"]
            all_close_to_zero = np.all(np.isclose(melt, 0, atol=threshold))
            if not all_close_to_zero:
                melt.name = f"{entry} melt (ppmw)"
                output.append(melt)

        data = pd.concat(output, axis=1)

        ax = sns.pairplot(data, hue="Oxygen fugacity", corner=True)
        sns.move_legend(ax, "center left", bbox_to_anchor=(0.6, 0.6))

        # plt.show()


def main():
    """Test area for development"""

    filename: Path = Path("../notebooks/simpleHCsystem.pkl")
    plotter: Plotter = Plotter.read_pickle(filename)

    # By elements
    plotter.species_pairplot()

    # By species
    plotter.species_pairplot(("H2O", "H2", "CO2", "CO"))

    plt.show()


if __name__ == "__main__":
    main()


# Below is previous code, but I'm not sure we'd ever use this approach compared to plotting corner
# plots

# def plot_element_ratio_grid(self, reservoir: str = "melt") -> None:
#     """Plots a grid of the data in element ratio space.

#     Args:
#         reservoir: Can be 'atmosphere', 'solid', 'melt', or 'total'
#     """
#     # TODO: Generalise, but to get up and running just work with C/H and C/O
#     CH_ratio: np.ndarray = self.get_element_ratio_in_reservoir("C", "H", reservoir)
#     CO_ratio: np.ndarray = self.get_element_ratio_in_reservoir("C", "O", reservoir)

#     H_total: np.ndarray = self.dataframes["H_totals"]["total_mass"].to_numpy(copy=True)
#     H_total /= earth_oceans_to_kg()

#     fO2_shift: np.ndarray = self.dataframes["extra"]["fO2_shift"].to_numpy(copy=True)

#     # X, Y = np.meshgrid(CO_ratio, CH_ratio)
#     # logger.info("Number of data points = %d", CO_ratio.size)
#     # _, ax = plt.subplots()
#     # ax.scatter(X.flatten(), Y.flatten(), s=0.1)
#     # ax.set_xlabel("C/O")
#     # ax.set_ylabel("C/H")
#     # ax.set_title("Data points")
#     # plt.savefig("Ratio_Grid.jpg", dpi=500)
#     # plt.show()

#     # Some test quantity to plot
#     values: np.ndarray = self.dataframes["H2O"]["atmosphere_pressure"].values
#     # triang = mtri.Triangulation(CO_ratio, CH_ratio)

#     # Interpolate to regular grid
#     # x: np.ndarray = np.linspace(0.01, 0.15, 100)
#     y: np.ndarray = np.linspace(1, 10, 100)
#     x: np.ndarray = np.linspace(-4, 4, 100)
#     # y: np.ndarray = np.linspace(0.1, 1, 50)
#     xi, yi = np.meshgrid(x, y)
#     # xf = xi.flatten()
#     # yf = yi.flatten()
#     # interpolate_points = np.hstack((xf, yf))
#     # print(xi)
#     # print(yi)

#     points = np.column_stack((fO2_shift, H_total))

#     print(points.shape)
#     print(values.size)

#     z_cubic = griddata(points, values, (xi, yi), method="cubic")

#     print(z_cubic.shape)

#     print(z_cubic)

#     # interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind="geom")
#     # zi_cubic_geom = interp_cubic_geom(xi, yi)

#     _, axs = plt.subplots()
#     # axs = axs.flatten()

#     # axs[0].tricontourf(triang, z)
#     # axs[0].triplot(triang, "ko-")
#     # axs[0].set_title("Triangular grid")

#     # z_cubic = z_cubic.reshape(100, 100)

#     axs.scatter(xi, yi, z_cubic)
#     axs.contour(xi, yi, z_cubic, levels=[1, 5, 10])
#     axs.set_xlabel("fO2_shift")
#     axs.set_ylabel("H_total")

#     # plt.tricontourf(triang, z, cmap="viridis")
#     # plt.colorbar(label="H in Melt (Earth Oceans)")
#     # plt.xlabel("C/O")
#     # plt.ylabel("C/H")
#     # plt.savefig("T1e_CHONSCl_CHSols_500Its_TotalMassRatios_HMeltContour.jpg", dpi=500)
#     plt.show()

# Class probably originally wrote by Aaron for SPIDER datatable interpolation?
# class TwoDFunc_irreg(object):

#     """For 2-D lookup with irregularly spaced input data"""

#     def __init__(self, *args):
#         x_a = args[0].astype(float)
#         y_a = args[1].astype(float)
#         z_a = args[2].astype(float)

#         # exit the code if NaNs since user should ensure that the
#         # input arrays do not contain NaNs prior to calling this
#         # function.
#         for val_a in (x_a, y_a, z_a):
#             if np.isnan(np.sum(val_a)):
#                 logging.critical("TwoDFunc_irreg:")
#                 logging.critical("input arrays cannot contain NaNs")
#                 sys.exit(1)

#         self.xbnds = (np.min(x_a), np.max(x_a))
#         self.ybnds = (np.min(y_a), np.max(y_a))
#         self.zbnds = (np.min(z_a), np.max(z_a))

#         triang = tri.Triangulation(
#             self._relval(x_a, self.xbnds), self._relval(y_a, self.ybnds)
#         )
#         self.tci = tri.CubicTriInterpolator(triang, self._relval(z_a, self.zbnds), kind="geom")

#     def __call__(self, *args):
#         x_a = args[0]
#         y_a = args[1]
#         relz = self.tci(self._relval(x_a, self.xbnds), self._relval(y_a, self.ybnds))
#         z = (self.zbnds[1] - self.zbnds[0]) * relz + self.zbnds[0]
#         return z

#     def _relval(self, v, bnds):
#         return (v - bnds[0]) / (bnds[1] - bnds[0])

#     def gradient(self, *args):
#         x_a = args[0]
#         y_a = args[1]
#         relzgrad = self.tci.gradient(
#             self._relval(x_a, self.xbnds), self._relval(y_a, self.ybnds)
#         )
#         zgrad = [
#             (self.zbnds[1] - self.zbnds[0]) / (self.xbnds[1] - self.xbnds[0]) * relzgrad[0],
#             (self.zbnds[1] - self.zbnds[0]) / (self.ybnds[1] - self.ybnds[0]) * relzgrad[1],
#         ]
#         return zgrad
