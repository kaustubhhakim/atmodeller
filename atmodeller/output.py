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
"""Output"""

from __future__ import annotations

import copy
import logging
import pickle
from collections import UserDict
from pathlib import Path
from typing import Hashable

import pandas as pd

from atmodeller.solution import Solution

logger: logging.Logger = logging.getLogger(__name__)


class Output(UserDict):
    """Stores inputs and outputs of the models.

    Changing the dictionary keys or entries may require downstream changes to the Plotter class,
    which uses Output to source data to plot.
    """

    @property
    def size(self) -> int:
        """Number of rows"""
        try:
            return len(self.data["solution"])
        except KeyError:
            return 0

    @classmethod
    def read_pickle(cls, pickle_file: Path | str) -> Output:
        """Reads output data from a pickle file and creates an Output instance.

        Args:
            pickle_file: Pickle file of the output from a previous (or similar) model run.
                Importantly, the reaction network must be the same (same number of species in the
                same order) and the constraints must be the same (also in the same order).

        Returns:
            Output
        """
        with open(pickle_file, "rb") as handle:
            output_data: dict[str, list[dict[str, float]]] = pickle.load(handle)

        logger.info("%s: Reading data from %s", cls.__name__, pickle_file)

        return cls(output_data)

    @classmethod
    def from_dataframes(cls, dataframes: dict[str, pd.DataFrame]) -> Output:
        """Reads a dictionary of dataframes and creates an Output instance.

        Args:
            dataframes: A dictionary of dataframes.

        Returns:
            Output
        """
        output_data: dict[str, list[dict[Hashable, float]]] = {}
        for key, dataframe in dataframes.items():
            output_data[key] = dataframe.to_dict(orient="records")

        return cls(output_data)

    def add(self, solution: Solution, extra_output: dict[str, float] | None = None) -> None:
        """Adds all outputs.

        Args:
            solution: Solution
            extra_output: Extra data to write to the output. Defaults to None.
        """
        for key, value in solution.output_full().items():
            data_list: list[dict[str, float]] = self.data.setdefault(key, [])
            data_list.append(value)

        # self._add_constraints(interior_atmosphere)
        # self._add_residual(interior_atmosphere)

        if extra_output is not None:
            data_list: list[dict[str, float]] = self.data.setdefault("extra", [])
            data_list.append(extra_output)

    # TODO: To reinstate
    # def _add_constraints(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
    #     """Adds constraints.

    #     Args:
    #         interior_atmosphere: Interior atmosphere system
    #     """
    #     temperature: float = interior_atmosphere.solution.atmosphere.temperature()
    #     pressure: float = interior_atmosphere.solution.atmosphere.pressure()
    #     evaluate_dict: dict[str, float] = interior_atmosphere.constraints.evaluate(
    #         temperature, pressure
    #     )
    #     data_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
    #     data_list.append(evaluate_dict)

    # def _add_residual(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
    #     """Adds the residual.

    #     Args:
    #         interior_atmosphere: Interior atmosphere system
    #     """
    #     data_list: list[dict[str, float]] = self.data.setdefault("residual", [])
    #     data_list.append(interior_atmosphere.residual_dict())

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Output as a dictionary of dataframes

        Returns:
            The output as a dictionary of dataframes
        """
        out: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(value) for key, value in self.data.items()
        }
        return out

    def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        out: dict[str, pd.DataFrame] = self.to_dataframes()
        output_file: Path = Path(f"{file_prefix}.xlsx")

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:  # pylint: disable=E0110
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output written to %s", output_file)

    def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)

    def _check_keys_the_same(self, other: Output) -> None:
        """Checks if the keys are the same in 'other' before combining output.

        Args:
            other: Other output to potentially combine (if keys are the same)
        """
        if not self.keys() == other.keys():
            msg: str = "Keys for 'other' are not the same as 'self' so cannot combine them"
            logger.error(msg)
            raise KeyError(msg)

    def __add__(self, other: Output) -> Output:
        """Addition

        Args:
            other: Other output to combine with self

        Returns:
            Combined output
        """
        self._check_keys_the_same(other)
        output: Output = copy.deepcopy(self)
        for key in self.keys():
            output[key].extend(other[key])

        return output

    def __iadd__(self, other: Output) -> Output:
        """In-place addition

        Args:
            other: Other output to combine with self in-place

        Returns:
            self
        """
        self._check_keys_the_same(other)
        for key in self:
            self[key].extend(other[key])

        return self

    def filter_by_index_notin(self, other: Output, index_key: str, index_name: str) -> Output:
        """Filters out the entries in `self` that are not present in the index of `other`

        Args:
            other: Other output with the filtering index
            index_key: Key of the index
            index_name: Name of the index

        Returns:
            The filtered output
        """
        self_dataframes: dict[str, pd.DataFrame] = self.to_dataframes()
        other_dataframes: dict[str, pd.DataFrame] = other.to_dataframes()
        index: pd.Index = pd.Index(other_dataframes[index_key][index_name])

        for key, dataframe in self_dataframes.items():
            self_dataframes[key] = dataframe[~dataframe.index.isin(index)]

        return self.from_dataframes(self_dataframes)

    def reorder(self, other: Output, index_key: str, index_name: str) -> Output:
        """Reorders all the entries according to an index in `other`

        Args:
            other: Other output with the reordering index
            index_key: Key of the index
            index_name: Name of the index

        Returns:
            The reordered output
        """
        self_dataframes: dict[str, pd.DataFrame] = self.to_dataframes()
        other_dataframes: dict[str, pd.DataFrame] = other.to_dataframes()
        index: pd.Index = pd.Index(other_dataframes[index_key][index_name])

        for key, dataframe in self_dataframes.items():
            self_dataframes[key] = dataframe.reindex(index)

        return self.from_dataframes(self_dataframes)

    def __call__(
        self,
        file_prefix: Path | str = "atmodeller_out",
        to_dict: bool = True,
        to_dataframes: bool = False,
        to_pickle: bool = False,
        to_excel: bool = False,
    ) -> dict:
        """Gets the output and/or optionally write it to a pickle or Excel file.

        Args:
            file_prefix: Prefix of the output file if writing to a pickle or Excel. Defaults to
                atmodeller_out
            to_dict: Returns the output data in a dictionary. Defaults to True.
            to_dataframes: Returns the output data in a dictionary of dataframes. Defaults to
                False.
            to_pickle: Writes a pickle file. Defaults to False.
            to_excel: Writes an Excel file. Defaults to False.

        Returns:
            A dictionary of the output if `to_dict = True`, otherwise None.
        """
        if to_pickle:
            self.to_pickle(file_prefix)

        if to_excel:
            self.to_excel(file_prefix)

        # Acts as an override if to_dict is also set.
        if to_dataframes:
            return self.to_dataframes()

        if to_dict:
            return self.data

        raise ValueError("No output option(s) specified")
