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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from molmass import Formula

from atmodeller.interfaces import CondensedSpecies
from atmodeller.utilities import UnitConversion, delete_entries_with_suffix, flatten

if TYPE_CHECKING:
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ReservoirOutput:
    r"""Mass and moles of a species (or element) in a reservoir

    Args:
        mass: Mass of the species in the reservoir in kg
        molar_mass: Molar mass of the species in :math:`\mathrm{kg}\mathrm{mol}^{-1}`
        reservoir_mass: Mass of the reservoir in kg. For the atmosphere reservoir this is the total
            mass of the atmosphere. For a mantle reservoir this is the total mass of the silicate
            component only (i.e. not including the volatile mass).
    """

    mass: float
    """Mass of the species in the reservoir in kg"""
    molar_mass: float
    r"""Molar mass of the species in :math:`\mathrm{kg}\mathrm{mol}^{-1}`"""
    reservoir_mass: float
    """Mass of the mantle reservoir in kg"""
    moles: float = field(init=False)
    """Number of moles"""
    ppmw: float = field(init=False, default=0)
    """Parts-per-million by weight of the species relative to the reservoir"""

    def __post_init__(self):
        self.moles = self.mass / self.molar_mass
        if self.reservoir_mass > 0:
            self.ppmw = UnitConversion.fraction_to_ppm(self.mass / self.reservoir_mass)


@dataclass(kw_only=True)
class ReservoirOutputMoleFraction(ReservoirOutput):
    r"""Additionally computes the ppm by moles

    Args:
        mass: Mass of the species in the reservoir in kg
        molar_mass: Molar mass of the species in :math:`\mathrm{kg}\mathrm{mol}^{-1}`
        reservoir_mass: Mass of the reservoir in kg
        reservoir_moles: Total moles of the reservoir
    """

    reservoir_moles: float
    """Total moles of the reservoir"""
    ppm: float = field(init=False, default=0)
    """Parts-per-million by moles of the species relative to the reservoir"""

    def __post_init__(self):
        super().__post_init__()
        if self.reservoir_moles > 0:
            self.ppm = UnitConversion.fraction_to_ppm(self.moles / self.reservoir_moles)


@dataclass(kw_only=True)
class AtmosphereReservoirOutput(ReservoirOutputMoleFraction):
    r"""A species in the atmosphere

    Args:
        mass: Mass of the species in the reservoir in kg
        molar_mass: Molar mass of the species in :math:`\mathrm{kg}\mathrm{mol}^{-1}`
        reservoir_mass: Total mass of the atmosphere
        reservoir_moles: Total moles of the reservoir
        fugacity: Fugacity in bar
        fugacity_coefficient: Fugacity coefficient
        pressure: Pressure in bar
        volume_mixing_ratio: Volume mixing ratio
    """

    fugacity: float
    """Fugacity"""
    fugacity_coefficient: float
    """Fugacity coefficient"""
    pressure: float
    """Pressure"""
    volume_mixing_ratio: float
    """Volume mixing ratio"""


@dataclass(kw_only=True)
class CondensedSpeciesOutput:
    """Output for a condensed species

    Args:
        mass: Mass of the condensed species in kg
        activity: Activity
    """

    activity: float
    """Activity"""
    mass: float
    """Mass in kg"""

    def asdict(self) -> dict[str, float]:
        """Data as a dictionary"""
        output_dict: dict[str, Any] = asdict(self)

        return output_dict


@dataclass(kw_only=True)
class SpeciesOutput:
    """Output for a species"""

    atmosphere: ReservoirOutput
    melt: ReservoirOutput
    solid: ReservoirOutput
    condensed_mass: float = 0
    total_mass: float = field(init=False)
    degree_of_condensation: float = field(init=False)
    condensed_moles: float = field(init=False)
    total_moles: float = field(init=False)

    def __post_init__(self):
        self.total_mass = (
            self.atmosphere.mass + self.melt.mass + self.solid.mass + self.condensed_mass
        )
        self.degree_of_condensation = self.condensed_mass / self.total_mass
        doc_factor: float = self.degree_of_condensation / (1 - self.degree_of_condensation)
        self.condensed_moles = doc_factor * (
            self.atmosphere.moles + self.melt.moles + self.solid.moles
        )
        self.total_moles = (
            self.atmosphere.moles + self.melt.moles + self.solid.moles + self.condensed_moles
        )

    def asdict(self) -> dict[str, float]:
        """Data as a dictionary

        Deletes some entries to avoid duplication of output quantities. For example, the reservoir
        masses also appear in the 'planet' output.
        """
        output_dict: dict[str, float] = flatten(asdict(self))
        # It's useful to see the exact value of the molar mass for calculations, but we only need
        # to output it once.
        molar_mass: float = output_dict["atmosphere_molar_mass"]
        output_dict = delete_entries_with_suffix(output_dict, "molar_mass")
        output_dict["molar_mass"] = molar_mass
        output_dict = delete_entries_with_suffix(output_dict, "reservoir_mass")
        output_dict = delete_entries_with_suffix(output_dict, "reservoir_moles")
        # TODO: A better approach would be to not calculate these for gas species at all
        if self.degree_of_condensation == 0:
            del output_dict["degree_of_condensation"]
            del output_dict["condensed_mass"]
            del output_dict["condensed_moles"]

        return output_dict


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

    def add(
        self, interior_atmosphere: InteriorAtmosphereSystem, extra_output: dict[str, float] | None
    ) -> None:
        """Adds all outputs.

        Args:
            interior_atmosphere: Interior atmosphere system
            extra_output: Extra data to write to the output. Defaults to None.
        """
        self._add_gas_species(interior_atmosphere)
        condensed_element_mass: dict[str, float] = self._add_condensed_species(interior_atmosphere)
        self._add_elements(interior_atmosphere, condensed_element_mass)
        self._add_atmosphere(interior_atmosphere)
        self._add_constraints(interior_atmosphere)
        self._add_planet(interior_atmosphere)
        self._add_residual(interior_atmosphere)
        self._add_solution(interior_atmosphere)
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
            element_moles: Total number of moles of elements
        """
        atmosphere: dict[str, float] = {}
        atmosphere["pressure"] = interior_atmosphere.total_pressure
        atmosphere["mean_molar_mass"] = interior_atmosphere.atmospheric_mean_molar_mass
        atmosphere["mass"] = interior_atmosphere.total_mass

        data_list: list[dict[str, float]] = self.data.setdefault("atmosphere", [])
        data_list.append(atmosphere)

    def _add_condensed_species(
        self, interior_atmosphere: InteriorAtmosphereSystem
    ) -> dict[str, float]:
        """Adds condensed species.

        Args:
            interior_atmosphere: Interior atmosphere system

        Returns:
            Condensed element masses
        """
        condensed_species_mass: dict[CondensedSpecies, dict[str, float]] = (
            interior_atmosphere.condensed_species_masses()
        )
        for species in interior_atmosphere.species.condensed_species:
            activity: float = interior_atmosphere.solution.activities[species]
            mass: float = sum(condensed_species_mass[species].values())
            output = CondensedSpeciesOutput(mass=mass, activity=activity)
            data_list: list[dict[str, float]] = self.data.setdefault(species.name, [])
            data_list.append(output.asdict())

        # TODO: Probably move elsewhere, but need to compute the condensed elements for the
        # later elemental output
        condensed_element_mass: dict[str, float] = {}
        for species, element_masses in condensed_species_mass.items():
            for element, value in element_masses.items():
                if element in condensed_element_mass:
                    condensed_element_mass[element] += value
                else:
                    condensed_element_mass[element] = value

        logger.debug("condensed_element_mass = %s", condensed_element_mass)

        return condensed_element_mass

    def _add_constraints(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds constraints.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        evaluate_dict: dict[str, float] = interior_atmosphere.constraints.evaluate(
            interior_atmosphere.planet.surface_temperature, interior_atmosphere.total_pressure
        )
        data_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
        data_list.append(evaluate_dict)

    # def condensed_species_mass(
    #     self, interior_atmosphere: InteriorAtmosphereSystem
    # ) -> dict[CondensedSpecies, dict[str, float]]:
    #     """Computes the element condensate mapping matrix

    #     Args:
    #         interior_atmosphere: Interior atmosphere system
    #     """
    #     condensed_elements: list[str] = interior_atmosphere._solution.condensed_elements_to_solve
    #     condensed_species: list[CondensedSpecies] = (
    #         interior_atmosphere._solution.condensed_species_to_solve
    #     )

    #     mapping: npt.NDArray = np.zeros((len(condensed_elements), len(condensed_species)))

    #     logger.warning("Testing element-condensate mapping")
    #     logger.debug("mapping empty = %s", mapping)
    #     logger.debug("condensed_elements = %s", condensed_elements)
    #     logger.debug("condensed_species = %s", condensed_species)

    #     condensed_mass: dict[str, float] = {}
    #     for ii, condensed_element in enumerate(condensed_elements):
    #         condensed_mass[condensed_element] = interior_atmosphere.element_condensed_mass(
    #             condensed_element
    #         )
    #         for jj, species in enumerate(condensed_species):
    #             logger.debug(species.composition())
    #             if condensed_element in species.composition():
    #                 mapping[ii, jj] = species.composition()[condensed_element].count

    #     logger.debug("mapping = %s", mapping)
    #     logger.debug("condensed_mass = %s", condensed_mass)

    #     # Required for iteration, but not when can be calculated in one
    #     # sorted_indices = np.argsort(associations)
    #     # logger.debug("sorted_indices = %s", sorted_indices)
    #     # sorted_mapping = mapping[sorted_indices]
    #     # logger.debug("sorted_mapping = %s", sorted_mapping)

    #     # condensed_moles: dict[str, float] = {
    #     #    element: value / UnitConversion.g_to_kg(Formula(element).mass)
    #     #    for element, value in condensed_mass.items()
    #     # }
    #     # logger.debug("condensed_moles = %s", condensed_moles)
    #     # condensed_moles_array: npt.NDArray = np.array(list(condensed_moles.values())).reshape(
    #     #    len(condensed_elements), -1
    #     # )
    #     # logger.debug("condensed_moles_array = %s", condensed_moles_array)

    #     condensed_mass_array: npt.NDArray = np.array(list(condensed_mass.values())).reshape(
    #         len(condensed_elements), -1
    #     )

    #     # This solves for the number of moles of the single element in each species
    #     x: npt.NDArray = np.linalg.solve(mapping, condensed_mass_array)
    #     logger.debug("x = %s", x)

    #     condensed_moles: dict[str, float] = {
    #         element: mass / UnitConversion.g_to_kg(Formula(element).mass)
    #         for element, mass in zip(condensed_elements, x)
    #     }
    #     logger.debug("condensed_moles = %s", condensed_moles)
    #     condensed_moles_array: npt.NDArray = np.array(list(condensed_moles.values()))

    #     # Now need to back-compute other elements in the species based on stoichiometry
    #     condensed_species_mass: dict[CondensedSpecies, dict[str, float]] = {}
    #     for ii, species in enumerate(condensed_species):
    #         moles: float = condensed_moles_array[ii]
    #         dataframe: pd.DataFrame = species.composition().dataframe()
    #         dataframe["Moles"] = dataframe["Count"] * moles
    #         dataframe["Mass"] = (
    #             dataframe["Moles"]
    #             * UnitConversion.g_to_kg(dataframe["Relative mass"])
    #             / dataframe["Count"]
    #         )
    #         logger.debug("dataframe = %s", dataframe)
    #         condensed_species_mass[species] = dataframe.to_dict()["Mass"]

    #     logger.debug("condensed_species_mass = %s", condensed_species_mass)

    #     return condensed_species_mass

    def _add_elements(
        self,
        interior_atmosphere: InteriorAtmosphereSystem,
        condensed_element_mass: dict[str, float],
    ) -> float:
        """Adds elements.

        Args:
            interior_atmosphere: Interior atmosphere system
            condensed_element_mass: Condensed element masses

        Returns:
            Total number of moles of elements
        """
        mass: dict[str, Any] = {}

        for element in interior_atmosphere.species.elements():
            mass[element] = interior_atmosphere.element_gas_mass(element)

        # Preprocess to get total number of moles of elements in the atmosphere
        atmosphere_total_element_moles: float = 0
        for element, element_mass in mass.items():
            formula: Formula = Formula(element)
            molar_mass: float = UnitConversion.g_to_kg(formula.mass)
            atmosphere_total_element_moles += element_mass["atmosphere_mass"] / molar_mass

        # Create and add the output
        for element, element_mass in mass.items():
            logger.info("Adding %s to output", element)
            formula: Formula = Formula(element)
            molar_mass: float = UnitConversion.g_to_kg(formula.mass)
            atmosphere: ReservoirOutput = ReservoirOutputMoleFraction(
                molar_mass=molar_mass,
                mass=element_mass["atmosphere_mass"],
                reservoir_mass=interior_atmosphere.total_mass,
                reservoir_moles=atmosphere_total_element_moles,
            )
            melt: ReservoirOutput = ReservoirOutput(
                molar_mass=molar_mass,
                mass=element_mass["melt_mass"],
                reservoir_mass=interior_atmosphere.planet.mantle_melt_mass,
            )
            # Trapped in the solid mantle
            solid: ReservoirOutput = ReservoirOutput(
                molar_mass=molar_mass,
                mass=element_mass["solid_mass"],
                reservoir_mass=interior_atmosphere.planet.mantle_solid_mass,
            )
            try:
                condensed_element_mass_element = condensed_element_mass[element]
            except KeyError:
                condensed_element_mass_element = 0
            output = SpeciesOutput(
                atmosphere=atmosphere,
                melt=melt,
                solid=solid,
                condensed_mass=condensed_element_mass_element,
            )
            # Create a unique key name to avoid a potential name conflict with atomic species
            key_name: str = f"{element}_totals"
            data_list: list[dict[str, float]] = self.data.setdefault(key_name, [])
            data_list.append(output.asdict())

        return atmosphere_total_element_moles

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
        for species in interior_atmosphere.species.gas_species:
            output: dict[str, float] = interior_atmosphere.gas_species_reservoir_masses(species)
            output["fugacity_coefficient"] = interior_atmosphere.solution.fugacity_coefficients[
                species
            ]
            output["volume_mixing_ratio"] = interior_atmosphere.solution.volume_mixing_ratios[
                species
            ]
            output["molar_mass"] = interior_atmosphere.species.get_species(species).molar_mass
            output["total_mass"] = (
                output["atmosphere_mass"] + output["melt_mass"] + output["solid_mass"]
            )
            data_list: list[dict[str, float]] = self.data.setdefault(species.name, [])
            data_list.append(output)

    def _add_residual(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the residual.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("residual", [])
        data_list.append(interior_atmosphere.residual_dict)

    def _add_solution(self, interior_atmosphere: InteriorAtmosphereSystem) -> None:
        """Adds the solution.

        Args:
            interior_atmosphere: Interior atmosphere system
        """
        data_list: list[dict[str, float]] = self.data.setdefault("solution", [])
        data_list.append(interior_atmosphere.solution.solution_dict())

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

        raise ValueError("No output option(s) specified")
