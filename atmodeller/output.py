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
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Hashable

import numpy as np
import pandas as pd
from molmass import Formula

from atmodeller.utilities import UnitConversion, reorder_dict

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem

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

    def add(
        self, interior_atmosphere: InteriorAtmosphereSystem, extra_output: dict[str, float] | None
    ) -> None:
        """Adds all outputs.

        Args:
            interior_atmosphere: Interior atmosphere system
            extra_output: Extra data to write to the output. Defaults to None.
        """
        self._add_gas_species(interior_atmosphere)
        self._add_condensed_species(interior_atmosphere)
        self._add_elements(interior_atmosphere)
        self._add_atmosphere(interior_atmosphere)
        self._add_constraints(interior_atmosphere)
        self._add_planet(interior_atmosphere)
        self._add_residual(interior_atmosphere)
        self._add_solution(interior_atmosphere)
        self._add_raw_solution(interior_atmosphere)

        if extra_output is not None:
            data_list: list[dict[str, float]] = self.data.setdefault("extra", [])
            data_list.append(extra_output)

    def _add_atmosphere(
        self,
        interior_atmosphere: InteriorAtmosphereSystem,
    ) -> None:
        """Adds atmosphere.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        atmosphere: dict[str, float] = {}
        atmosphere["pressure"] = interior_atmosphere.atmosphere_pressure
        atmosphere["mean_molar_mass"] = interior_atmosphere.atmosphere_molar_mass
        atmosphere["mass"] = interior_atmosphere.atmosphere_mass
        atmosphere["element_moles"] = interior_atmosphere.atmosphere_element_moles
        atmosphere["species_moles"] = interior_atmosphere.atmosphere_species_moles

        data_list: list[dict[str, float]] = self.data.setdefault("atmosphere", [])
        data_list.append(atmosphere)

    def _add_condensed_species(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds condensed species.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        for species in interior_atmosphere.species.condensed_species:
            output: dict[str, float] = {}
            output["activity"] = interior_atmosphere.solution.activity.physical[species]
            output["mass"] = interior_atmosphere.solution.mass.physical[species]
            output["moles"] = output["mass"] / species.molar_mass
            output["molar_mass"] = species.molar_mass

            data_list: list[dict[str, float]] = self.data.setdefault(species.name, [])
            data_list.append(output)

    def _add_constraints(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds constraints.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        evaluate_dict: dict[str, float] = interior_atmosphere.constraints.evaluate(
            interior_atmosphere.planet.surface_temperature, interior_atmosphere.atmosphere_pressure
        )
        data_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
        data_list.append(evaluate_dict)

    def _add_elements(
        self,
        interior_atmosphere: InteriorAtmosphereSystem,
    ) -> None:
        """Adds elements.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        mass: dict[str, Any] = {}
        condensed_element_masses: dict[str, float] = interior_atmosphere.condensed_element_masses()

        for element in interior_atmosphere.species.elements():
            mass[element] = interior_atmosphere.element_gas_mass(element)

        # To compute astronomical logarithmic abundances (dex) we need to store H abundance, which
        # is used to normalise all other elemental abundances
        mass = reorder_dict(mass, "H")
        H_total_moles: float | None = None  # pylint: disable=invalid-name

        # Create and add the output
        for nn, (element, element_mass) in enumerate(mass.items()):
            output: dict[str, float] = {}
            logger.debug("Adding %s to output", element)
            output["molar_mass"] = UnitConversion.g_to_kg(Formula(element).mass)
            output["atmosphere_mass"] = element_mass["atmosphere_mass"]
            output["melt_mass"] = element_mass["melt_mass"]
            output["solid_mass"] = element_mass["solid_mass"]
            try:
                output["condensed_mass"] = condensed_element_masses[element]
            except KeyError:
                output["condensed_mass"] = 0
            output["total_mass"] = (
                output["atmosphere_mass"]
                + output["melt_mass"]
                + output["solid_mass"]
                + output["condensed_mass"]
            )
            output["degree_of_condensation"] = output["condensed_mass"] / output["total_mass"]
            output["atmosphere_moles"] = output["atmosphere_mass"] / output["molar_mass"]
            output["volume_mixing_ratio"] = (
                output["atmosphere_moles"] / interior_atmosphere.atmosphere_element_moles
            )
            output["melt_moles"] = output["melt_mass"] / output["molar_mass"]
            output["solid_moles"] = output["solid_mass"] / output["molar_mass"]
            output["condensed_moles"] = output["condensed_mass"] / output["molar_mass"]
            output["total_moles"] = output["total_mass"] / output["molar_mass"]
            if interior_atmosphere.planet.mantle_melt_mass:
                output["melt_ppmw"] = (
                    output["melt_mass"] / interior_atmosphere.planet.mantle_melt_mass
                )
            else:
                output["melt_ppmw"] = 0
            if interior_atmosphere.planet.mantle_solid_mass:
                output["solid_ppmw"] = (
                    output["solid_mass"] / interior_atmosphere.planet.mantle_solid_mass
                )
            else:
                output["solid_ppmw"] = 0

            # Create a unique key name
            key_name: str = f"{element}_total"
            data_list: list[dict[str, float]] = self.data.setdefault(key_name, [])
            data_list.append(output)

            # H, if present, is the first in the dictionary, so set as the normalising abundance
            if nn == 0 and element == "H":
                H_total_moles = output["total_moles"]  # pylint: disable=invalid-name

            if H_total_moles is not None:
                # Astronomical logarithmic abundance (dex), e.g. used by FastChem
                output["logarithmic_abundance"] = (
                    np.log10(output["total_moles"] / H_total_moles) + 12
                )

    def _add_planet(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the planetary properties.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        planet_dict: dict[str, float] = asdict(interior_atmosphere.planet)
        data_list: list[dict[str, float]] = self.data.setdefault("planet", [])
        data_list.append(planet_dict)

    def _add_gas_species(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds gas species.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        for species in interior_atmosphere.solution.gas.data:
            output: dict[str, float] = interior_atmosphere.gas_species_reservoir_masses(species)
            output["fugacity_coefficient"] = (
                interior_atmosphere.solution.gas.fugacity_coefficients(
                    interior_atmosphere.planet.surface_temperature
                )[species]
            )
            output["volume_mixing_ratio"] = interior_atmosphere.solution.gas.volume_mixing_ratios(
                interior_atmosphere.planet.surface_temperature
            )[species]
            output["molar_mass"] = interior_atmosphere.species.get_species(species).molar_mass
            output["total_mass"] = (
                output["atmosphere_mass"] + output["melt_mass"] + output["solid_mass"]
            )
            output["atmosphere_moles"] = output["atmosphere_mass"] / output["molar_mass"]
            output["melt_moles"] = output["melt_mass"] / output["molar_mass"]
            output["solid_moles"] = output["solid_mass"] / output["molar_mass"]
            output["total_moles"] = output["total_mass"] / output["molar_mass"]

            data_list: list[dict[str, float]] = self.data.setdefault(species.name, [])
            data_list.append(output)

    def _add_residual(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the residual.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("residual", [])
        data_list.append(interior_atmosphere.residual_dict())

    def _add_raw_solution(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the raw solution.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("raw_solution", [])
        data_list.append(interior_atmosphere.solution.raw_solution_dict())

    def _add_solution(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the solution.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("solution", [])
        data_list.append(interior_atmosphere.solution_dict())

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
