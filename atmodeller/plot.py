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
from typing import Any, Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from scipy.constants import kilo

from atmodeller.output import Output
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


class Category:
    """Defines a category based on a series in a dataframe in :class:`output.Output`.

    Args:
        dataframe_name: Name of the dataframe
        column_name: Name of the column in the dataframe
        categories: Categories and their maximum values
        column_rename: New name of the column
    """

    def __init__(
        self,
        *,
        dataframe_name: str,
        column_name: str,
        categories: dict[str, float],
        column_rename: str | None = None,
    ):
        self.categories: dict[str, float] = categories
        self._dataframe_name: str = dataframe_name
        self._column_name: str = column_name
        self._column_rename: str = column_rename if column_rename is not None else column_name
        self._palette: dict = self.get_custom_palette()

    @property
    def name(self) -> str:
        return self._column_rename

    @property
    def hue_order(self) -> list[str]:
        return list(self._palette.keys())

    @property
    def palette(self) -> dict:
        return self._palette

    def _get_category_name_for_value(self, value: float) -> str:
        """Gets the category name for a value

        Args:
            value: Value to get the category

        Returns:
            The category
        """
        for category_name, category_max_value in self.categories.items():
            if value < category_max_value:
                return category_name
        msg: str = "value = %f exceeds the maximum value of the category list"
        logger.warning(msg)
        logger.warning("categories = %s", self.categories)
        raise ValueError(msg)

    def get_category(self, output: Output) -> pd.Series:
        """Gets the category.

        Args:
            output: Output

        Returns:
            A series of the category
        """
        raw_data: pd.Series = output.to_dataframes()[self._dataframe_name][self._column_name]
        categorised_data: pd.Series = raw_data.apply(self._get_category_name_for_value)
        categorised_data.name = self._column_rename

        return categorised_data

    def get_custom_palette(self) -> dict:
        """Gets a custom palette

        Returns:
            A custom palette
        """
        colormap: Colormap = cm.batlowS  # type: ignore
        colormap_values: list[tuple[float, ...]] = [colormap(4), colormap(2), colormap(3)]
        custom_palette: dict[str, tuple[float, ...]] = {
            name: value for name, value in zip(self.categories.keys(), colormap_values)
        }

        return custom_palette


# Define categories for grouping and colouring data

oxygen_fugacity_categories: dict[str, float] = {"Reduced": -1, "IW": 1, "Oxidised": 5}
oxygen_fugacity: Category = Category(
    dataframe_name="extra",
    column_name="fO2_shift",
    categories=oxygen_fugacity_categories,
    column_rename="Oxygen fugacity",
)

C_H_ratio_categories: dict[str, float] = {"Low C/H": 1, "Medium C/H": 5, "High C/H": 10}
C_H_ratio: Category = Category(
    dataframe_name="extra", column_name="C/H ratio", categories=C_H_ratio_categories
)

H_oceans_categories: dict[str, float] = {"Low H": 3, "Medium H": 5, "Large H": 10}
H_oceans: Category = Category(
    dataframe_name="extra",
    column_name="Number of ocean moles",
    categories=H_oceans_categories,
    column_rename="H budget",
)

categories: dict[str, Category] = {
    "Oxygen fugacity": oxygen_fugacity,
    "C/H ratio": C_H_ratio,
    "H budget": H_oceans,
}

# Standard kws for bivariate pairplot
pairplot_kws: dict[str, Any] = {
    "fill": True,
    "alpha": 0.7,
    # "thresh": 0.1,
    # "levels": 4,
    "levels": [0.1, 0.25, 0.5, 0.75, 1],
    "common_norm": False,
}

# Standard kws for univariate pairplot
# Other parameters seem to be consistently carry across from pairplot_kws, which operates on the
# bivariate plots, but the common_norm apparently needs to be applied again.
diag_kws: dict[str, Any] = {"common_norm": False}


def get_axis(
    grid: sns.PairGrid,
    *,
    data: pd.DataFrame | None = None,
    column_name: str | None = None,
    column_index: int | None = None,
) -> Axes:
    """Gets an axis from a grid.

    The order of the axes in the list is top left to bottom right for the bivariate plots,
    with an empty axes for the univariate plots. The univariate axes are then appended to
    the end of list, also ordered from top left to bottom right.

    Args:
        grid: Grid to get the axes
        data: The data. Defaults to None.
        column_name: Name of the column. Defaults to None.
        column_index: Index of the column in the plot. Defaults to None.

    Returns:
        The axes
    """
    axes: list[Axes] = grid.figure.axes
    if column_name is not None:
        assert data is not None
        logger.info("Getting column index for %s", column_name)
        # Recall that the first column is the category hence minus 1
        column_index = data.columns.get_loc(column_name) - 1
        logger.info("column_index = %s", column_index)

    try:
        assert column_index is not None
    except AssertionError as e:
        msg: str = "Both column_name and column_index cannot be None"
        logger.error(msg)
        raise ValueError(msg) from e

    axis_index: int = sum(range(column_index + 1)) + column_index

    return axes[axis_index]


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
        """Reads output data from a pickle file and creates a Plotter instance.

        Args:
            pickle_file: Pickle file of the output from a model run.

        Returns:
            Plotter
        """
        output: Output = Output.read_pickle(pickle_file)
        return cls(output)

    def get_element_ratio_in_reservoir(
        self,
        element1: str,
        element2: str,
        *,
        reservoir: str = "total",
        mass_or_moles: str = "mass",
    ) -> pd.Series:
        """Gets the mass or mole ratio of two elements in a reservoir

        Args:
            element1: Element in the numerator
            element2: Element in the denominator
            reservoir: Can be atmosphere, solid, melt, or total. Defaults to total.
            mass_or_moles: Can be mass or moles. Defaults to mass.

        Returns:
            The ratio of the two elements
        """
        column_name: str = f"{reservoir}_{mass_or_moles}"

        required_elements: list = [element1, element2]
        for elements in required_elements:
            assert f"{elements}_totals" in self.dataframes

        element1_mass: pd.Series = self.dataframes[f"{element1}_totals"][column_name]
        element2_mass: pd.Series = self.dataframes[f"{element2}_totals"][column_name]
        mass_ratio: pd.Series = element1_mass / element2_mass
        mass_ratio.name = f"{element1}/{element2} (by {mass_or_moles}) {reservoir}"

        return mass_ratio

    def species_pairplot(
        self,
        *,
        species: tuple[str, ...] = ("C", "H", "O", "N"),
        mass_or_moles: str = "moles",
        category: str = "Oxygen fugacity",
        plot_atmosphere: bool = True,
        minor_species: bool = False,
    ) -> sns.PairGrid:
        """Plots a pair plot of species and/or atmospheric properties.

        Args:
            species: A tuple of species or elements to plot, which can be empty. Defaults to
                (C, H, O, N).
            mass_or_moles: Plot the species by mass or moles. Defaults to moles.
            category: Category to group and colour the data by. Defaults to oxygen fugacity.
            plot_atmosphere: Plots atmosphere quantities. Defaults to True.
            minor_species: Do not set axes parameters.

        Returns:
            The grid
        """
        if mass_or_moles == "mass":
            suffix: str = "ppmw"
            units: str = "wt.%"
        elif mass_or_moles == "moles":
            suffix = "ppm"
            units = "mol.%"
        else:
            msg: str = "%s is unknown" % mass_or_moles
            logger.error(msg)
            raise ValueError(msg)

        try:
            colour_category: Category = categories[category]
        except KeyError:
            msg: str = "%s not in %s" % (category, categories)
            logger.error(msg)
            raise KeyError(msg)

        output: list[pd.Series] = [colour_category.get_category(self.output)]

        for entry in species:
            # Try to find species totals (i.e. assume species is an elemental total)
            try:
                totals: pd.DataFrame = self.dataframes[f"{entry}_totals"]
            # Otherwise, get the species directly
            except KeyError:
                totals = self.dataframes[entry]
            atmos: pd.Series = UnitConversion.ppm_to_percent(totals[f"atmosphere_{suffix}"])
            atmos.name = f"{entry} atmos ({units})"
            output.append(atmos)

        if plot_atmosphere:
            atmosphere: pd.DataFrame = self.dataframes["atmosphere"]
            pressure: pd.Series = atmosphere["pressure"] / kilo  # to kbar
            pressure.name = "Pressure (kbar)"
            mean_molar_mass: pd.Series = atmosphere["mean_molar_mass"] * kilo  # to g/mol
            mean_molar_mass.name = "Molar mass (g/mol)"
            atmosphere_series: list[pd.Series] = [pressure, mean_molar_mass]
            output.extend(atmosphere_series)

        data: pd.DataFrame = pd.concat(output, axis=1)

        sns.set_theme(font_scale=1.3)

        grid: sns.PairGrid = sns.pairplot(
            data,
            hue=colour_category.name,
            corner=True,
            plot_kws=pairplot_kws,
            diag_kws=diag_kws,
            palette=colour_category.palette,
            kind="kde",
            hue_order=colour_category.hue_order,
        )

        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        sns.move_legend(grid, "center left", bbox_to_anchor=(0.6, 0.6))

        for nn, species_name in enumerate(species):
            logger.info("Setting axis properties for %s", species_name)
            axis: Axes = get_axis(grid, column_index=nn)

            # N is an exception because its abundance is so low.
            if species_name == "N":
                N_min: float = -0.01
                N_max: float = 0.31
                N_start: float = 0
                N_step: float = 0.1
                ticksa: np.ndarray = np.arange(N_start, N_max, N_step)
                axis.set_xlim(N_min, N_max)
                axis.set_ylim(N_min, N_max)
                axis.set_xticks(ticksa)
                axis.set_yticks(ticksa)

            # Let S and Cl plot as they wish for the moment.
            elif species_name == "S" or species_name == "Cl":
                continue

            else:
                if not minor_species:
                    ticks: range = range(0, 101, 25)
                    axis.set_xlim(-5, 105)
                    axis.set_ylim(-5, 105)
                    axis.set_xticks(ticks)
                    axis.set_yticks(ticks)

        if plot_atmosphere:
            column_name: str = "Pressure (kbar)"
            logger.info("Setting axis properties for %s" % column_name)
            axis: Axes = get_axis(grid, data=data, column_name=column_name)
            ticks = range(0, 16, 5)
            axis.set_xlim(-1, 15)
            axis.set_ylim(-1, 15)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            column_name = "Molar mass (g/mol)"
            logger.info("Setting axis properties for %s", column_name)
            axis: Axes = get_axis(grid, data=data, column_name=column_name)
            ticks = range(0, 41, 10)
            axis.set_xlim(-1, 45)
            axis.set_ylim(-1, 45)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)

        return grid

    def ratios_pairplot(
        self,
        reservoirs: Iterable[str] = ("atmosphere", "melt", "total"),
        mass_or_moles: str = "mass",
    ) -> sns.PairGrid:
        """Pair plots of C/H and C/O ratios in the reservoirs

        Args:
            reservoirs: Reservoirs to plot. Defaults to all reservoirs.
            mass_or_moles: Compute ratios by mass or moles. Defaults to mass.
        """
        colour_category, category_order = self.fO2_categorise()
        output: list[pd.Series] = [colour_category]

        for element in ("H", "O"):
            for reservoir in reservoirs:
                series_data: pd.Series = self.get_element_ratio_in_reservoir(
                    "C", element, reservoir=reservoir, mass_or_moles=mass_or_moles
                )
                output.append(series_data)

        data: pd.DataFrame = pd.concat(output, axis=1)

        colormap = cm.batlowS  # type: ignore
        colormap_values: list = [colormap(4), colormap(2), colormap(3)]
        custom_palette = {name: value for name, value in zip(category_order, colormap_values)}
        plot_kws: dict = {
            "fill": True,
            "alpha": 0.7,
            "thresh": 0.1,
            "levels": 4,
            "common_norm": False,
        }
        ax: sns.PairGrid = sns.pairplot(
            data,
            hue=str(colour_category.name),
            corner=True,
            plot_kws=plot_kws,
            palette=custom_palette,
            kind="kde",
            hue_order=custom_palette.keys(),
        )

        sns.move_legend(ax, "center left", bbox_to_anchor=(0.6, 0.6))

        return ax


# Temporary store for potentially useful plotting code

# Threshold to determine whether to plot melt ppmw, since if all values are basically
# zero then no solubility was applied and we don't need to plot these data.
# threshold: float = 1e-5

# Plotting the melt as well is overwhelming for one figure.
# melt: pd.Series = totals["melt_ppmw"]
# all_close_to_zero = np.all(np.isclose(melt, 0, atol=threshold))
# if not all_close_to_zero:
#     melt.name = f"{entry} melt (ppmw)"
#     output.append(melt)
