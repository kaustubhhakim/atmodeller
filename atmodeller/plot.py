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

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.constants import kilo
from scipy.ndimage import gaussian_filter1d

from atmodeller.output import Output
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AxesSpec:
    """Parameters for configuring the axes.

    Args:
        xylim: Tuple to set_xlim and set_ylim of axes
        ticks: Tick marks
    """

    xylim: tuple[float, float]
    """Tuple to `set_xlim` and `set_ylim` of axes"""
    ticks: list[float]
    """Tick marks"""


class Category:
    """A category based on a column in a dataframe in :class:`atmodeller.output.Output`.

    Args:
        dataframe_name: Name of the dataframe
        column_name: Name of the column in the dataframe
        categories: Categories and their maximum values
        category_name: Name of the category. Defaults to ``column_name``.
    """

    def __init__(
        self,
        *,
        dataframe_name: str,
        column_name: str,
        categories_dict: dict[str, float],
        category_name: str | None = None,
    ):
        self.categories_dict: dict[str, float] = categories_dict
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
        for category_name, category_max_value in self.categories_dict.items():
            if value < category_max_value:
                return category_name

        msg: str = "value = %f exceeds the maximum value of the category list"
        logger.warning(msg)
        logger.warning("categories_dict = %s", self.categories_dict)
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
            name: value for name, value in zip(self.categories_dict.keys(), colormap_values)
        }

        return custom_palette


# Define categories for grouping and colouring data

# Oxygen fugacity
oxygen_fugacity_categories: dict[str, float] = {"Reduced": -1, "IW": 1, "Oxidised": 5}
oxygen_fugacity: Category = Category(
    dataframe_name="extra",
    column_name="fO2 (delta IW)",
    categories_dict=oxygen_fugacity_categories,
    category_name="Oxygen fugacity",
)

# C/H ratio
C_H_ratio_categories: dict[str, float] = {"Low C/H": 1, "Medium C/H": 5, "High C/H": 10}
C_H_ratio: Category = Category(
    dataframe_name="extra", column_name="C/H ratio", categories_dict=C_H_ratio_categories
)

# H budget
H_oceans_categories: dict[str, float] = {"Low H": 3, "Medium H": 5, "Large H": 10}
H_oceans: Category = Category(
    dataframe_name="extra",
    column_name="Number of ocean moles",
    categories_dict=H_oceans_categories,
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

    def get_units(
        self, mass_or_moles: str = "moles", log10_transform: bool = False
    ) -> tuple[str, ...]:
        """Gets the units.

        Args:
            mass_or_moles: mass, moles, or pressure
            log10_transform: Transform all quantities to log10. Defaults to False.

        Returns:
            A tuple with the name and the units
        """
        if mass_or_moles == "moles":
            out: list[str] = ["ppm", "mol.%"]
        elif mass_or_moles == "mass":
            out = ["ppmw", "wt.%"]
        elif mass_or_moles == "pressure":
            out = ["pressure", "bar"]
        else:
            raise ValueError(f"{mass_or_moles} is unknown")

        if log10_transform:
            out[1] = f"log10({out[1]})"

        return tuple(out)

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
        index_shift: int = 1,
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
            column_index = cast(int, plot_data.columns.get_loc(column_name))
            column_index -= index_shift
            logger.info("column_index = %s", column_index)

        try:
            assert column_index is not None
        except AssertionError as e:
            msg: str = "Both column_name and column_index cannot be None"
            logger.error(msg)
            raise ValueError(msg) from e

        axis_index: int = sum(range(column_index + 1)) + column_index

        return axes[axis_index]

    def _sort(self, sort_sheet_name: str, sort_column_name: str) -> dict[str, pd.DataFrame]:
        """Sorts the dataframes in ascending order according to a column.

        Args:
            sort_sheet_name: Name of the sheet containing the column to sort by.
            sort_column_name: Name of the column to sort by.

        Returns:
            The sorted dataframes
        """
        output: dict[str, pd.DataFrame] = copy.deepcopy(self.dataframes)
        for sheet_name in self.dataframes:
            output[sheet_name][sort_column_name] = output[sort_sheet_name][sort_column_name]
            output[sheet_name].sort_values(by=[sort_column_name], inplace=True)

        return output

    def bin_data(
        self, sort_sheet_name: str, sort_column_name: str, bin_size: int
    ) -> dict[str, pd.DataFrame]:
        """Bins the data.

        Args:
            sort_sheet_name: Name of the sheet containing the column to sort by.
            sort_column_name: Name of the column to sort by.
            bin_size: Size of the bin.

        Returns:
            Binned data
        """
        data_size: int = self.dataframes["solution"].shape[0]
        try:
            assert not data_size % bin_size
        except AssertionError as exc:
            raise ValueError(
                f"Data size = {data_size} and bin_size = {bin_size} must be exactly divisible"
            ) from exc

        sorted_data: dict[str, pd.DataFrame] = self._sort(sort_sheet_name, sort_column_name)

        output: dict[str, pd.DataFrame] = {}
        for sheet_name in sorted_data:
            out: pd.DataFrame = pd.DataFrame()
            for column in sorted_data[sheet_name].columns:
                data_reshape: npt.NDArray = np.array(sorted_data[sheet_name][column]).reshape(
                    -1, bin_size
                )
                try:
                    data_median: npt.NDArray = np.median(data_reshape, axis=1)
                    data_std: npt.NDArray = cast(npt.NDArray, np.std(data_reshape, axis=1))
                    out[f"{column}"] = data_median
                    out[f"{column}_std"] = data_std
                except TypeError:
                    logger.warning(
                        "Cannot compute a median for column = %s due to invalid type", column
                    )
            output[sheet_name] = out

        return output

    # Convenient to use fO2 so pylint: disable=C0103
    def plot_binned_data_by_fO2(
        self,
        bin_size: int,
        y_axis: str,
        species_set: list[str],
        colors_set: list[str],
        *,
        scale_factor: float = 1,
        sigma: float = 2,
        smooth: bool = True,
        xmin: float | None = None,
        xmax: float | None = None,
        xlabel: str | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        ylabel: str | None = None,
        yscale: str = "linear",
        fill_between: bool = True,
    ) -> Figure:
        """Plots binned data by fO2

        Args:
            bin_size: Size of the bin
            y_axis: Field to plot for the y axis for each species in ``species_set``
            species_set: Species
            colors_set: Colors
            scale_factor: Factor to scale the y data
            sigma: Sigma for the Gaussian filter. Defaults to 2.
            smooth: Apply the Gaussian filter to smooth the data. Defaults to True.
            xmin: Minimum for x axis. Defaults to None.
            xmax: Maximum for x axis. Defaults to None.
            xlabel: Label for x axis. Defaults to None.
            ymin: Minimum for y axis. Defaults to None.
            ymax: Maximum for y axis. Defaults to None.
            ylabel: Label for y axis. Defaults to None.
            yscale: Scale for y axis. Defaults to linear.
            fill_between: True to plot one std either side of the mean. Defaults to True.
        """
        fig, ax = plt.subplots()
        x_axis: str = "fO2 (delta IW)"
        binned_data: dict[str, pd.DataFrame] = self.bin_data("extra", x_axis, bin_size)

        for species, color in zip(species_set, colors_set):
            label: str = species.rstrip("_g")
            x_data: npt.NDArray | pd.Series = binned_data[species][x_axis]
            y_data: npt.NDArray | pd.Series = binned_data[species][y_axis] / scale_factor
            if smooth is not None:
                y_data = gaussian_filter1d(y_data, sigma=sigma)
                y_minus = gaussian_filter1d(
                    y_data - binned_data[species][f"{y_axis}_std"] / scale_factor, sigma=sigma
                )
                y_plus = gaussian_filter1d(
                    y_data + binned_data[species][f"{y_axis}_std"] / scale_factor, sigma=sigma
                )
            ax.plot(x_data, y_data, color=color, label=label)

            if fill_between:
                ax.fill_between(
                    binned_data[species][x_axis],
                    y_minus,
                    y_plus,
                    color=color,
                    alpha=0.2,
                    # label=label,
                )

        if xlabel is None:
            xlabel = x_axis
        if ylabel is None:
            ylabel = y_axis

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale(yscale)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend()

        return fig

    def plot_volume_mixing_ratio(
        self,
        bin_size: int,
        species_set: list[str],
        colors_set: list[str],
        *,
        sigma: float = 2,
        smooth: bool = True,
        xmin: float | None = None,
        xmax: float | None = None,
        xlabel: str | None = None,
        ymin: float | None = 0,
        ymax: float | None = 1,
        ylabel: str | None = "Volume mixing ratio",
        yscale: str = "linear",
        fill_between: bool = True,
    ) -> Figure:
        """Plots volume mixing ratios by fO2

        Args:
            bin_size: Size of the bin
            species_set: Species
            colors_set: Colors
            sigma: Sigma for the Gaussian filter. Defaults to 2.
            smooth: Apply the Gaussian filter to smooth the data. Defaults to True.
            xmin: Minimum for x axis. Defaults to None.
            xmax: Maximum for x axis. Defaults to None.
            xlabel: Label for x axis. Defaults to None.
            ymin: Minimum for y axis. Defaults to 0.
            ymax: Maximum for y axis. Defaults to 1.
            ylabel: Label for y axis. Defaults to `Volume mixing ratio`.
            yscale: Scale for y axis. Defaults to linear.
            fill_between: True to plot one std either side of the mean. Defaults to True.
        """
        return self.plot_binned_data_by_fO2(
            bin_size,
            "atmosphere_ppm",
            species_set,
            colors_set,
            scale_factor=1e6,
            sigma=sigma,
            smooth=smooth,
            xmin=xmin,
            xmax=xmax,
            xlabel=xlabel,
            ymin=ymin,
            ymax=ymax,
            ylabel=ylabel,
            yscale=yscale,
            fill_between=fill_between,
        )

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
        except KeyError as exc:
            raise KeyError(f"{category} is not in {categories}") from exc

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
            logger.info("Setting axis properties for %s", column_name)
            axis: Axes = self.get_axes(grid, plot_data=data, column_name=column_name)
            ticks = range(0, 16, 5)
            axis.set_xlim(0, 15)
            axis.set_ylim(0, 15)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            column_name = "Molar mass (g/mol)"
            logger.info("Setting axis properties for %s", column_name)
            axis: Axes = self.get_axes(grid, plot_data=data, column_name=column_name)
            ticks = range(0, 51, 10)
            axis.set_xlim(0, 55)
            axis.set_ylim(0, 55)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)

        return grid

    def species_pairplot_scatter(
        self,
        *,
        species: dict[str, AxesSpec | None],
        mass_or_moles: str = "moles",
        plot_atmosphere: bool = True,
        sns_bbox_to_anchor: tuple[float, float] = (0.6, 0.6),
        bbox_to_anchor: tuple[float, float] = (0, 0),
        log10_transform: bool = False,
        name: str = "",
    ) -> sns.PairGrid:
        """Plots a pair plot of species and/or atmospheric properties.

        Plots a scatter plot with a kde overlay.

        Args:
            species: A dict of species or elements to plot, which can be empty.
            mass_or_moles: Plot the species by mass or moles or bar. Defaults to moles.
            plot_atmosphere: Plots atmosphere quantities. Defaults to True.
            sns: Seaborn positioning of the legend. Defaults to (0.6, 0.6).
            bbox_to_anchor: Positioning of the data density legend. Defaults to (0, 0).
            log10_transform: Transform all quantities to log10. Defaults to False.
            name: Extra name for label. Defaults to empty string.

        Returns:
            The grid
        """
        suffix, units = self.get_units(mass_or_moles, log10_transform)

        output: list[pd.Series] = []

        for entry in species:
            # Try to find species totals (i.e. assume species is an elemental total)
            try:
                totals: pd.DataFrame = self.dataframes[f"{entry}_totals"]
            # Otherwise, get the species directly
            except KeyError:
                totals = self.dataframes[entry]
            if mass_or_moles == "pressure":
                atmos: pd.Series = totals[f"atmosphere_{suffix}"]
            else:
                atmos = UnitConversion.ppm_to_percent(totals[f"atmosphere_{suffix}"])
            if log10_transform:
                atmos = cast(pd.Series, np.log10(atmos))
            if name:
                atmos.name = f"{entry.removesuffix('_g')} {name} ({units})"
            else:
                atmos.name = f"{entry.removesuffix('_g')} ({units})"
            output.append(atmos)

        if plot_atmosphere:
            atmosphere: pd.DataFrame = self.dataframes["atmosphere"]
            pressure: pd.Series = atmosphere["pressure"] / kilo  # to kbar
            pressure.name = "Pressure (kbar)"
            mean_molar_mass: pd.Series = atmosphere["mean_molar_mass"] * kilo  # to g/mol
            mean_molar_mass.name = "Molar mass (g/mol)"
            atmosphere_series: list[pd.Series] = [pressure, mean_molar_mass]
            output.extend(atmosphere_series)

        # Categorise C/H, which is cleaner and also better behaved with the seaborn legend
        # This assumes that C/H is log10 distributed between -1 and 1
        CH_sizes: dict[str, float] = {"0.5 > C/H": 5, "0.5 < C/H < 2.2": 10, "2.2 < C/H": 15}
        CH_ratio = pd.cut(
            self.dataframes["extra"]["C/H ratio"],
            [0.1, 0.464158883717533, 2.154434688378294, 10],
            labels=list(CH_sizes.keys()),
        )

        data: pd.DataFrame = pd.concat(output, axis=1)

        sns.set_theme(style="white", font_scale=1.3)

        grid: sns.PairGrid = sns.pairplot(
            data,
            corner=True,
            diag_kind="kde",
            plot_kws={
                "size": CH_ratio,
                "legend": "auto",
                "alpha": 0.3,
                "hue": self.dataframes["extra"]["fO2 (delta IW)"],
                "hue_norm": (-5, 5),
                "sizes": CH_sizes,
                "size_order": reversed(list(CH_sizes.keys())),
                "palette": "crest",
            },
        )

        grid.map_lower(
            sns.kdeplot,
            # Must match with the legend specification below
            levels=[0.33, 0.66],
            # clip=(0, None),
            alpha=1.0,
            legend=True,
            # Keyword arguments passed to matplotlib contour
            # These must match with the legend specification below
            colors=["black"],
            linestyles=["dashed", "solid"],
        )

        grid.add_legend()

        # Easier to create a second legend than try to amend the seaborn legend
        # Entries must match the contour specifications above
        extra_legend_elements = [
            Line2D([0], [0], color="k", ls="--", label=r"0.33"),
            Line2D([0], [0], color="k", label=r"0.66"),
        ]
        plt.legend(
            handles=extra_legend_elements,
            title="Density contours",
            frameon=False,
            bbox_to_anchor=bbox_to_anchor,
        )

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        sns.move_legend(grid, "center left", bbox_to_anchor=sns_bbox_to_anchor)

        for nn, (species_name, species_axes_spec) in enumerate(species.items()):
            if species_axes_spec is not None:
                logger.info("Setting axis properties for %s", species_name)
                axis: Axes = self.get_axes(grid, column_index=nn, index_shift=0)
                axis.set_xlim(*species_axes_spec.xylim)
                axis.set_ylim(*species_axes_spec.xylim)
                axis.set_xticks(species_axes_spec.ticks)
                axis.set_yticks(species_axes_spec.ticks)
            else:
                logger.info("Using default axis properties for %s", species_name)

        if plot_atmosphere:
            # Axes for atmosphere quantites are currently set once here.
            column_name: str = "Pressure (kbar)"
            logger.info("Setting axis properties for %s", column_name)
            axis: Axes = self.get_axes(
                grid, plot_data=data, column_name=column_name, index_shift=0
            )
            ticks = range(0, 16, 5)
            axis.set_xlim(0, 15)
            axis.set_ylim(0, 15)
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            column_name = "Molar mass (g/mol)"
            logger.info("Setting axis properties for %s", column_name)
            axis: Axes = self.get_axes(
                grid, plot_data=data, column_name=column_name, index_shift=0
            )
            ticks = range(0, 51, 10)
            axis.set_xlim(0, 55)
            axis.set_ylim(0, 55)
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
        except KeyError as exc:
            raise KeyError(f"{category} is not in {categories}") from exc

        output: list[pd.Series] = [colour_category.get_category(self.output)]

        for element in "H":
            for reservoir in reservoirs:
                series_data: pd.Series = self.get_element_ratio_in_reservoir(
                    "C", element, reservoir=reservoir, mass_or_moles=mass_or_moles
                )
                output.append(series_data)

        for element in ("C", "H"):
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
        specs: AxesSpec = AxesSpec(xylim=(0, 25), ticks=[0, 5, 10, 15, 20, 25])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        axes = self.get_axes(grid, column_index=3)
        specs: AxesSpec = AxesSpec(xylim=(25, 100), ticks=[25, 50, 75, 100])
        axes.set_xlim(specs.xylim)
        axes.set_ylim(specs.xylim)
        axes.set_xticks(specs.ticks)
        axes.set_yticks(specs.ticks)

        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        sns.move_legend(grid, "center left", bbox_to_anchor=(0.6, 0.6))

        return grid
