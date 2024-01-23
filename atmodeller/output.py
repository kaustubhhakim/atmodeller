"""Output

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
import pickle
from collections import UserDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from atmodeller.interfaces import GasSpecies
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ReservoirOutput:
    """Mass and moles of a species or element in a reservoir

    Args:
        species: GasSpecies
        reservoir: Reservoir name (e.g., atmosphere, (silicate) melt, (silicate) solid)
        mass: Mass in kg

    Attributes:
        See Args.
        moles: Moles
    """

    species: GasSpecies
    reservoir: str
    mass: float

    @property
    def moles(self) -> float:
        return self.mass / self.species.molar_mass


@dataclass(kw_only=True)
class MantleReservoirOutput(ReservoirOutput):
    """A species or element in a mantle reservoir

    Args:
        name: Species or element name
        reservoir: Reservoir name (e.g., atmosphere, (silicate) melt, (silicate) solid)
        mass: Mass in kg
        moles: Moles
        ppmw: Part-per-million by weight

    Attributes:
        See Args.
    """

    ppmw: float


@dataclass(kw_only=True)
class SpeciesAtmosphereOutput(ReservoirOutput):
    """A species in the atmosphere

    Args:
        name: Species name
        mass: Mass in kg
        moles: Moles
        fugacity: Fugacity in bar
        fugacity_coefficient: Fugacity coefficient
        pressure: Pressure in bar
        volume_mixing_ratio: Volume mixing ratio

    Attributes:
        See Args.
    """

    fugacity: float
    fugacity_coefficient: float
    pressure: float
    volume_mixing_ratio: float
    reservoir: str = field(init=False, default="atmosphere")


@dataclass(kw_only=True)
class CondensedSpeciesOutput:
    """Output for a condensed species

    These data are not currently output because all condensed phases have an activity of unity
    """

    activity: float


@dataclass(kw_only=True)
class GasSpeciesOutput:
    """Output for a gas species"""

    atmosphere: SpeciesAtmosphereOutput
    melt: MantleReservoirOutput
    solid: MantleReservoirOutput

    @property
    def mass_total(self) -> float:
        return self.atmosphere.mass + self.melt.mass + self.solid.mass

    @property
    def moles_total(self) -> float:
        return self.atmosphere.moles + self.melt.moles + self.solid.moles

    # TODO: Compute elemental breakdowns.


class Output(UserDict):
    """Stores inputs and outputs of the models."""

    def __init__(self, dict=None, /, **kwargs):
        """Init definition from the base class provided for clarity."""
        self.data = {}
        if dict is not None:
            self.update(dict)
        if kwargs:
            self.update(kwargs)

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

    def add(
        self, interior_atmosphere: InteriorAtmosphereSystem, extra_output: dict[str, float] | None
    ) -> None:
        """Adds all outputs.

        Args:
            interior_atmosphere: Interior atmosphere system
            extra_output: Extra data to write to the output
        """
        self._add_atmosphere(interior_atmosphere)
        self._add_constraints(interior_atmosphere)
        self._add_planet(interior_atmosphere)
        self._add_gas_species(interior_atmosphere)
        self._add_residual(interior_atmosphere)
        self._add_solution(interior_atmosphere)
        if extra_output is not None:
            data_list: list[dict[str, float]] = self.data.setdefault("extra", [])
            data_list.append(extra_output)

    def _add_atmosphere(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds atmosphere.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        atmosphere_dict: dict[str, float] = {}
        atmosphere_dict["total_pressure"] = interior_atmosphere.total_pressure
        atmosphere_dict["mean_molar_mass"] = interior_atmosphere.atmospheric_mean_molar_mass

        data_list: list[dict[str, float]] = self.data.setdefault("atmosphere", [])
        data_list.append(atmosphere_dict)

    def _add_constraints(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds constraints.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        evaluate_dict: dict[str, float] = interior_atmosphere.constraints.evaluate(
            interior_atmosphere
        )

        data_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
        data_list.append(evaluate_dict)

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
        for species in interior_atmosphere.species.gas_species.values():
            assert species.output is not None
            data_list: list[dict[str, float]] = self.data.setdefault(species.formula, [])
            data_list.append(asdict(species.output))

    def _add_residual(self, interior_atmosphere: InteriorAtmosphereSystem):
        """Adds the residual.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("residual", [])
        data_list.append(interior_atmosphere.residual_dict)

    def _add_solution(self, interior_atmosphere: InteriorAtmosphereSystem):
        """Adds the solution.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("solution", [])
        data_list.append(interior_atmosphere.solution_dict)

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Output as a dictionary of dataframes"""
        out: dict[str, pd.DataFrame] = {
            key: pd.DataFrame(value) for key, value in self.data.items()
        }
        return out

    def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)

    def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        out: dict[str, pd.DataFrame] = self.to_dataframes()
        output_file: Path = Path(f"{file_prefix}.xlsx")

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output written to %s", output_file)

    def __call__(
        self,
        file_prefix: Path | str = "atmodeller_out",
        to_dict: bool = True,
        to_dataframes: bool = False,
        to_pickle: bool = False,
        to_excel: bool = False,
    ) -> dict | None:
        """Gets the output and/or optionally write it to a pickle or Excel file.

        Args:
            file_prefix: Prefix of the output file if writing to a pickle or Excel. Defaults to
                'atmodeller_out'
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
