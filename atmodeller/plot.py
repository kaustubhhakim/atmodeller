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
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from matplotlib import pyplot as plt

from atmodeller.output import Output
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


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
            reservoir: Can be 'atmosphere', 'solid', 'melt', or 'total'. Defaults to 'total'
            mass_or_moles: Can be 'mass' or 'moles'. Defaults to 'mass'

        Returns:
            A series of the ratio
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

    @staticmethod
    def _get_fO2_category(fo2_shift: float) -> str:
        """Gets the atmosphere category based on fO2.

        The bounds are somewhat arbitrary, but centered around IW.

        Args:
            fO2_shift: fO2_shift relative to the IW buffer

        Returns:
            Category of the atmosphere
        """
        if fo2_shift < -1:
            return "Reduced"
        elif fo2_shift < 1:
            return "IW"
        else:
            return "Oxidised"

    def fO2_categorise(self) -> tuple[pd.Series, tuple[str, ...]]:
        """Gets a series of the atmosphere category based on fO2.

        Returns:
            Category of the atmosphere as a series and the order of the categories
        """
        fO2_shift: pd.Series = self.dataframes["extra"]["fO2_shift"]
        fO2_categorise = fO2_shift.apply(self._get_fO2_category)
        fO2_categorise.name = "Oxygen fugacity"
        category_order: tuple[str, ...] = ("Reduced", "IW", "Oxidised")

        return fO2_categorise, category_order

    @staticmethod
    def _get_CH_category(CH_ratio: float) -> str:
        """Gets the atmosphere category based on C/H.

        The bounds are somewhat arbitrary.

        Args:
            CH_ratio: C/H ratio

        Returns:
            Category of the atmosphere
        """
        if CH_ratio < 1:
            return "Low C/H"
        elif CH_ratio < 5:
            return "Medium C/H"
        else:
            return "High C/H"

    def CH_categorise(self) -> tuple[pd.Series, tuple[str, ...]]:
        """Gets a series of the atmosphere category based on C/H.

        Returns:
            Category of the atmosphere as a series and the order of the categories
        """
        CH_ratio: pd.Series = self.dataframes["extra"]["C/H ratio"]
        CH_categorise = CH_ratio.apply(self._get_CH_category)
        CH_categorise.name = "C/H ratio"
        category_order: tuple[str, ...] = ("Low C/H", "Medium C/H", "High C/H")

        return CH_categorise, category_order

    @staticmethod
    def _get_H_category(H: float) -> str:
        """Gets the atmosphere category based on H.

        The bounds are somewhat arbitrary.

        Args:
            H: Ocean moles of H

        Returns:
            Category of the atmosphere
        """
        if H < 3:
            return "Low oceans"
        elif H < 5:
            return "Medium oceans"
        else:
            return "High oceans"

    def H_oceans(self) -> tuple[pd.Series, tuple[str, ...]]:
        """Gets a series of the atmosphere category based on total H

        Returns:
            Category of the atmosphere as a series and the order of the categories
        """
        H_oceans: pd.Series = self.dataframes["extra"]["Number of ocean moles"]
        H_oceans = H_oceans.apply(self._get_H_category)
        H_oceans.name = "Oceans"
        category_order: tuple[str, ...] = ("Low oceans", "Medium oceans", "High oceans")

        return H_oceans, category_order

    def species_pairplot(
        self,
        species: tuple[str, ...] = ("C", "H", "O", "N"),
        *,
        mass_or_moles: str = "mass",
        category: str = "fO2",
    ) -> sns.PairGrid:
        """Pair plot of species

        Args:
            species: A tuple of species to plot. Defaults to (C, H, O, N).
            mass_or_moles: Can be 'mass' or 'moles'. Defaults to 'mass'.
            category: Can be 'fO2', 'CH', or 'H'. Defaults to 'fO2'.
        """
        # Threshold to determine whether to plot melt ppmw, since if all values are basically
        # zero then no solubility was applied and we don't need to plot these data.
        # threshold: float = 1e-5

        output: list[pd.Series] = []

        if category == "fO2":
            colour_category, category_order = self.fO2_categorise()
        elif category == "CH":
            colour_category, category_order = self.CH_categorise()
        elif category == "H":
            colour_category, category_order = self.H_oceans()
        else:
            msg: str = "{category} is unknown"
            logger.error(msg)
            raise ValueError(msg)
        output.append(colour_category)

        to_percent: float = UnitConversion.ppm_to_percent()

        if mass_or_moles == "mass":
            suffix: str = "ppmw"
            units: str = "wt.%"
        elif mass_or_moles == "moles":
            suffix = "ppm"
            units = "mol.%"
        else:
            msg: str = "{mass_or_moles} is unknown (expecting 'mass' or 'moles')"
            logger.error(msg)
            raise ValueError(msg)

        for entry in species:
            # Try to find species totals (i.e. assume species is an elemental total)
            try:
                totals: pd.DataFrame = self.dataframes[f"{entry}_totals"]
            # Otherwise, get the species directly
            except KeyError:
                totals = self.dataframes[entry]
            atmos: pd.Series = totals[f"atmosphere_{suffix}"] * to_percent
            atmos.name = f"{entry} atmos ({units})"
            output.append(atmos)
            # Plotting the melt as well is overwhelming for one figure.
            # melt: pd.Series = totals["melt_ppmw"]
            # all_close_to_zero = np.all(np.isclose(melt, 0, atol=threshold))
            # if not all_close_to_zero:
            #     melt.name = f"{entry} melt (ppmw)"
            #     output.append(melt)

        data: pd.DataFrame = pd.concat(output, axis=1)

        colormap = cm.batlowS  # type: ignore
        colormap_values: list = [colormap(4), colormap(2), colormap(3)]
        custom_palette = {name: value for name, value in zip(category_order, colormap_values)}
        plot_kws: dict = {
            "fill": True,
            "alpha": 0.7,
            "thresh": 0.1,
            "levels": 4,
            "clip": (0, 100),
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
        ticks: Iterable[float] = range(0, 101, 25)
        ax.set(xlim=(-5, 100), ylim=(-5, 105), xticks=ticks, yticks=ticks)
        plt.subplots_adjust(hspace=0.15, wspace=0.15)

        return ax

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
