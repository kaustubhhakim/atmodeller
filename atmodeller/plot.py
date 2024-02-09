#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Plotting"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

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


@dataclass(frozen=True)
class AxesSpec:
    """Parameters for configuring the axes.

    Args:
        xylim: Tuple to set_xlim and set_ylim of axes
        ticks: Tick marks

    Attributes:
        xylim: Tuple to set_xlim and set_ylim of axes
        ticks: Tick marks
    """

    xylim: tuple[float, float]
    ticks: list[float]


class Category:
    """Defines a category based on a column in a dataframe in :class:`atmodeller.output.Output`.

    Args:
        dataframe_name: Name of the dataframe
        column_name: Name of the column in the dataframe
        categories: Categories and their maximum values
        category_name: Name of the category. Defaults to `column_name`.
    """

    def __init__(
        self,
        *,
        dataframe_name: str,
        column_name: str,
        categories: dict[str, float],
        category_name: str | None = None,
    ):
        self.categories: dict[str, float] = categories
        self._dataframe_name: str = dataframe_name
        self._column_name: str = column_name
        self._category_name: str = category_name if category_name is not None else column_name
        self._palette: dict = self.get_custom_palette()

    @property
    def name(self) -> str:
        return self._category_name

    @property
    def hue_order(self) -> list[str]:
        return list(self._palette.keys())

    @property
    def palette(self) -> dict:
        return self._palette

    def _get_category_name_for_value(self, value: float) -> str:
        """Gets the category name for a value

        Args:
            value: Value to get the category from

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
            The category data
        """
        raw_data: pd.Series = output.to_dataframes()[self._dataframe_name][self._column_name]
        categorised_data: pd.Series = raw_data.apply(self._get_category_name_for_value)
        categorised_data.name = self._category_name

        return categorised_data

    def get_custom_palette(self) -> dict:
        """Gets a custom palette

        https://www.fabiocrameri.ch/colourmaps/

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

# Oxygen fugacity
oxygen_fugacity_categories: dict[str, float] = {"Reduced": -1, "IW": 1, "Oxidised": 5}
oxygen_fugacity: Category = Category(
    dataframe_name="extra",
    column_name="fO2_shift",
    categories=oxygen_fugacity_categories,
    category_name="Oxygen fugacity",
)

# C/H ratio
C_H_ratio_categories: dict[str, float] = {"Low C/H": 1, "Medium C/H": 5, "High C/H": 10}
C_H_ratio: Category = Category(
    dataframe_name="extra", column_name="C/H ratio", categories=C_H_ratio_categories
)

# H budget
H_oceans_categories: dict[str, float] = {"Low H": 3, "Medium H": 5, "Large H": 10}
H_oceans: Category = Category(
    dataframe_name="extra",
    column_name="Number of ocean moles",
    categories=H_oceans_categories,
    category_name="H budget",
)

categories: dict[str, Category] = {
    "Oxygen fugacity": oxygen_fugacity,
    "C/H ratio": C_H_ratio,
    "H budget": H_oceans,
}


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

    @property
    def plot_kws(self) -> dict[str, Any]:
        """Keyword arguments for the bivariate plotting function"""
        kws: dict[str, Any] = {
            "fill": True,
            "alpha": 0.7,
            # "thresh": 0.1,
            # "levels": 4,
            # Ideally keep the levels the same between plots to allow direct comparison.
            "levels": [0.1, 0.25, 0.5, 0.75, 1],
            "common_norm": False,
            # "cut": 0,
        }

        return kws

    @property
    def diag_kws(self) -> dict[str, Any]:
        """Keyword arguments for the univariate plotting function"""
        kws: dict[str, Any] = {"common_norm": False}  # "cut": 0

        return kws

    def get_units(self, mass_or_moles: str = "moles") -> tuple[str, str]:
        """Gets the units.

        Args:
            mass_or_moles: mass or moles

        Returns:
            A tuple with the name and the units
        """
        if mass_or_moles == "moles":
            return ("ppm", "mol.%")
        elif mass_or_moles == "mass":
            return ("ppmw", "wt.%")
        else:
            msg: str = "%s is unknown" % mass_or_moles
            logger.error(msg)
            raise ValueError(msg)

    @classmethod
    def read_pickle(cls, pickle_file: Path | str) -> Plotter:
        """Reads output data from a pickle file and creates a Plotter instance.

        Args:
            pickle_file: Pickle file of the output.

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
        mass_ratio.name = f"{element1}/{element2} {reservoir[:5]} ({mass_or_moles}) "

        return mass_ratio

    def get_axes(
        self,
        grid: sns.PairGrid,
        *,
        plot_data: pd.DataFrame | None = None,
        column_name: str | None = None,
        column_index: int | None = None,
    ) -> Axes:
        """Gets axes from a grid.

        The order of the axes in the list is top left to bottom right for the bivariate plots,
        with empty axes for the univariate plots. The univariate axes are then appended to the end
        of the list, also ordered from top left to bottom right.

        Args:
            grid: Grid to get the axes from
            plot_data: The data used by the plot. Defaults to None.
            column_name: Name of the column. Defaults to None.
            column_index: Index of the column in the plot. Defaults to None.

        Returns:
            The axes
        """
        axes: list[Axes] = grid.figure.axes

        if column_name is not None:
            assert plot_data is not None
            logger.info("Getting column index for %s", column_name)
            # Recall that the first column is the category, hence must minus one
            column_index = plot_data.columns.get_loc(column_name) - 1
            logger.info("column_index = %s", column_index)

        try:
            assert column_index is not None
        except AssertionError as e:
            msg: str = "Both column_name and column_index cannot be None"
            logger.error(msg)
            raise ValueError(msg) from e

        axis_index: int = sum(range(column_index + 1)) + column_index

        return axes[axis_index]

    def species_pairplot(
        self,
        *,
        species: dict[str, AxesSpec | None],
        mass_or_moles: str = "moles",
        category: str = "Oxygen fugacity",
        plot_atmosphere: bool = True,
    ) -> sns.PairGrid:
        """Plots a pair plot of species and/or atmospheric properties.

        Args:
            species: A dict of species or elements to plot, which can be empty.
            mass_or_moles: Plot the species by mass or moles. Defaults to moles.
            category: Category to group and colour the data by. Defaults to oxygen fugacity.
            plot_atmosphere: Plots atmosphere quantities. Defaults to True.

        Returns:
            The grid
        """
        suffix, units = self.get_units(mass_or_moles)

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
            plot_kws=self.plot_kws,
            diag_kws=self.diag_kws,
            palette=colour_category.palette,
            kind="kde",
            hue_order=colour_category.hue_order,
        )

        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        sns.move_legend(grid, "center left", bbox_to_anchor=(0.6, 0.6))

        for nn, (species_name, species_axes_spec) in enumerate(species.items()):
            if species_axes_spec is not None:
                logger.info("Setting axis properties for %s", species_name)
                axis: Axes = self.get_axes(grid, column_index=nn)
                axis.set_xlim(*species_axes_spec.xylim)
                axis.set_ylim(*species_axes_spec.xylim)
                axis.set_xticks(species_axes_spec.ticks)
                axis.set_yticks(species_axes_spec.ticks)
            else:
                logger.info("Using default axis properties for %s", species_name)

        if plot_atmosphere:
            # Axes for atmosphere quantites are currently set once here.
            column_name: str = "Pressure (kbar)"
            logger.info("Setting axis properties for %s" % column_name)
            axis: Axes = self.get_axes(grid, plot_data=data, column_name=column_name)
            ticks = range(0, 16, 5)
            axis.set_xlim(0, 15)
            axis.set_ylim(0, 15)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            column_name = "Molar mass (g/mol)"
            logger.info("Setting axis properties for %s", column_name)
            axis: Axes = self.get_axes(grid, plot_data=data, column_name=column_name)
            ticks = range(0, 41, 10)
            axis.set_xlim(0, 45)
            axis.set_ylim(0, 45)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)

        return grid

    def ratios_pairplot(
        self,
        reservoirs: Iterable[str] = ("atmosphere", "melt", "total"),
        mass_or_moles: str = "moles",
        category: str = "Oxygen fugacity",
    ) -> sns.PairGrid:
        """Pair plots of C/H and C/O ratios in the reservoirs

        Args:
            reservoirs: Reservoirs to plot. Defaults to all reservoirs.
            mass_or_moles: Compute ratios by mass or moles. Defaults to moles.
            category: Category to group and colour the data by. Defaults to oxygen fugacity.
        """
        try:
            colour_category: Category = categories[category]
        except KeyError:
            msg: str = "%s not in %s" % (category, categories)
            logger.error(msg)
            raise KeyError(msg)

        output: list[pd.Series] = [colour_category.get_category(self.output)]

        for element in ("H", "O"):
            for reservoir in reservoirs:
                series_data: pd.Series = self.get_element_ratio_in_reservoir(
                    "C", element, reservoir=reservoir, mass_or_moles=mass_or_moles
                )
                output.append(series_data)

        for element in ("C", "H", "O"):
            element_data: pd.DataFrame = self.dataframes[f"{element}_totals"]
            interior: pd.Series = element_data["melt_mass"] * 100 / element_data["total_mass"]
            interior.name = f"{element} melt (wt.%)"
            output.append(interior)

        data: pd.DataFrame = pd.concat(output, axis=1)

        sns.set_theme(font_scale=1.3)

        grid: sns.PairGrid = sns.pairplot(
            data,
            hue=colour_category.name,
            corner=True,
            plot_kws=self.plot_kws,
            diag_kws=self.diag_kws,
            palette=colour_category.palette,
            kind="kde",
            hue_order=colour_category.hue_order,
        )

        # These axes are all tuned for ratios in moles and interior storage in wt.%
        axes = self.get_axes(grid, column_index=0)
        specs: AxesSpec = AxesSpec(xylim=(0, 1500), ticks=[0, 500, 1000, 1500])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=1)
        specs: AxesSpec = AxesSpec(xylim=(0, 1), ticks=[0, 0.5, 1])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=2)
        specs: AxesSpec = AxesSpec(xylim=(0, 2), ticks=[0, 0.5, 1, 1.5, 2])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=3)
        specs: AxesSpec = AxesSpec(xylim=(0, 1.5), ticks=[0, 0.5, 1, 1.5])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=4)
        specs: AxesSpec = AxesSpec(xylim=(0, 25), ticks=[0, 5, 10, 15, 20, 25])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=5)
        specs: AxesSpec = AxesSpec(xylim=(25, 100), ticks=[25, 50, 75, 100])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=6)
        specs: AxesSpec = AxesSpec(xylim=(0, 100), ticks=[0, 25, 50, 75, 100])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        sns.move_legend(grid, "center left", bbox_to_anchor=(0.6, 0.6))

        return grid
