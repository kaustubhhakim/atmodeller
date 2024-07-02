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
"""Initial solution"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from atmodeller.constraints import SystemConstraints
from atmodeller.core import GasSpecies, Solution, Species
from atmodeller.interfaces import (
    ChemicalSpecies,
    CondensedSpecies,
    TypeChemicalSpecies_co,
)
from atmodeller.output import Output

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

MIN_LOG10_PRESSURE: float = -12
"""Minimum log10 (bar) of the initial gas species pressures

Motivated by typical values of oxygen fugacity at the iron-wustite buffer
"""
MAX_LOG10_PRESSURE: float = 8
"""Maximum log10 (bar) of the initial gas species pressures"""


class InitialSolutionData(Solution):
    """TODO"""

    def apply_log10_gas_constraints(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> None:
        """Applies constraints to the log10 gas pressures.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
        """
        for constraint in constraints.gas_constraints:
            self.gas.data[constraint.species] = constraint.get_log10_value(
                temperature=temperature, pressure=pressure
            )

    def apply_log10_activity_constraints(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> None:
        """Applies constraints to the log10 activities.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
        """
        for constraint in constraints.activity_constraints:
            self.activity.data[constraint.species] = constraint.get_log10_value(
                temperature=temperature, pressure=pressure
            )


class InitialSolution(ABC, Generic[T]):
    """Initial solution

    Args:
        value: An object used to compute the initial solution
        species: Species
        min_log10_pressure: Minimum log10 gas pressure. Defaults to :data:`MIN_LOG10_PRESSURE`.
        max_log10_pressure: Maximum log10 gas pressure. Defaults to :data:`MAX_LOG10_PRESSURE`.
        fill_log10_pressure: Fill value for pressure in bar. Defaults to 1.
        fill_log10_activity: Fill value for activity. Defaults to 1.
        fill_log10_mass: Fill value for mass. Defaults to 10.
        fill_log10_stability: Fill value for stability. Defaults to -15.

    Attributes:
        value: An object used to compute the initial solution
    """

    def __init__(
        self,
        value: T,
        *,
        species: Species,
        min_log10_pressure: float = MIN_LOG10_PRESSURE,
        max_log10_pressure: float = MAX_LOG10_PRESSURE,
        fill_log10_pressure: float = 1,
        fill_log10_activity: float = 1,
        fill_log10_mass: float = 20,
        fill_log10_stability: float = -15,
    ):
        logger.info("Creating %s", self.__class__.__name__)
        self.value: T = value
        self._species: Species = species
        self.solution: InitialSolutionData = InitialSolutionData(species)
        self._min_log10_pressure: float = min_log10_pressure
        self._max_log10_pressure: float = max_log10_pressure
        self._fill_log10_pressure: float = fill_log10_pressure
        self._fill_log10_activity: float = fill_log10_activity
        self._fill_log10_mass: float = fill_log10_mass
        self._fill_log10_stability: float = fill_log10_stability

    @abstractmethod
    def _get_log10_gas_pressures(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> dict[GasSpecies, float]:
        """Initial solution for log10 gas pressures

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log10 gas pressures
        """

    @abstractmethod
    def _get_log10_activities(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> dict[CondensedSpecies, float]:
        """Initial solution for log10 activities

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log10 activities
        """

    @abstractmethod
    def _get_log10_masses(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> dict[CondensedSpecies, float]:
        """Initial solution for log10 condensed masses

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log10 condensed masses
        """

    @abstractmethod
    def _get_log10_stabilities(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> dict[CondensedSpecies, float]:
        """Initial solution for log10 stabilities

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log10 stabilities
        """

    def _set_log10_gas_pressures(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_log10: float = 0,
        apply_constraints: bool = True,
    ) -> None:
        """Sets the log10 gas pressures.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_log10: Maximum log10 value to perturb the initial solution of the gas
                pressures. Defaults to 0, i.e. not used.
            apply_constraints: Apply pressure constraints, if any. Defaults to True.
        """
        log10_gas_pressures: dict[GasSpecies, float] = self._get_log10_gas_pressures(
            constraints, temperature=temperature, pressure=pressure
        )
        self.solution.gas.data = log10_gas_pressures

        if perturb_log10:
            self.solution.gas.perturb_values(perturb_log10)

        self.solution.gas.clip_values(self._min_log10_pressure, self._max_log10_pressure)

        if apply_constraints:
            self.solution.apply_log10_gas_constraints(
                constraints, temperature=temperature, pressure=pressure
            )

    def _set_log10_activities(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        apply_constraints: bool = True,
    ) -> None:
        """Sets the log10 activities.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            apply_constraints: Apply activity constraints, if any. Defaults to True.
        """
        log10_activities: dict[CondensedSpecies, float] = self._get_log10_activities(
            constraints, temperature=temperature, pressure=pressure
        )
        self.solution.activity.data = log10_activities
        self.solution.activity.clip_values(maximum_value=1)

        if apply_constraints:
            self.solution.apply_log10_activity_constraints(
                constraints, temperature=temperature, pressure=pressure
            )

    def set_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_log10: float = 0,
        apply_gas_constraints: bool = True,
        apply_activity_constraints: bool = True,
    ) -> None:
        """Sets the log10 value of the initial solution.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_log10: Maximum log10 value to perturb the initial solution of the gas
                pressures. Defaults to 0, i.e. not used.
            apply_gas_constraints: Apply gas constraints, if any. Defaults to True.
            apply_constraints: Apply activity constraints, if any. Defaults to True.
        """
        self._set_log10_gas_pressures(
            constraints,
            temperature=temperature,
            pressure=pressure,
            perturb_log10=perturb_log10,
            apply_constraints=apply_gas_constraints,
        )

        self._set_log10_activities(
            constraints,
            temperature=temperature,
            pressure=pressure,
            apply_constraints=apply_activity_constraints,
        )

        self.solution.mass.data = self._get_log10_masses(
            constraints, temperature=temperature, pressure=pressure
        )

        self.solution.stability.data = self._get_log10_stabilities(
            constraints, temperature=temperature, pressure=pressure
        )

    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_log10: float = 0,
        apply_gas_constraints: bool = True,
        apply_activity_constraints: bool = True,
    ) -> npt.NDArray[np.float_]:
        """Gets the log10 value of the initial solution

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_log10: Maximum log10 value to perturb the initial solution of the gas
                pressures. Defaults to 0, i.e. not used.
            apply_gas_constraints: Apply gas constraints, if any. Defaults to True.
            apply_constraints: Apply activity constraints, if any. Defaults to True.

        Returns:
            The initial solution
        """
        self.set_log10_value(
            constraints,
            temperature=temperature,
            pressure=pressure,
            perturb_log10=perturb_log10,
            apply_gas_constraints=apply_gas_constraints,
            apply_activity_constraints=apply_activity_constraints,
        )

        logger.debug("initial_solution = %s", self.solution.raw_solution_dict())

        return self.solution.data

    def update(self, output: Output) -> None:
        """Updates the initial solution.

        This base class does nothing.

        Args;
            output: output
        """
        del output


class InitialSolutionDict(InitialSolution[dict]):
    """A dictionary of species and their values for the initial solution"""

    @override
    def __init__(self, value=None, **kwargs):
        if value is None:
            value_dict = {}
        else:
            value_dict = value
        super().__init__(value_dict, **kwargs)

    @override
    def _get_log10_gas_pressures(self, *args, **kwargs) -> dict[GasSpecies, float]:
        """Initial solution for log10 gas pressures

        Returns:
            Log10 gas pressures
        """
        del args
        del kwargs

        output: dict[GasSpecies, float] = {}
        for species in self._species.gas_species:
            try:
                output[species] = np.log10(self.value[species])
            except KeyError:
                output[species] = self._fill_log10_pressure

        return output

    @override
    def _get_log10_activities(self, *args, **kwargs) -> dict[CondensedSpecies, float]:
        """Initial solution for log10 activities

        Returns:
            Log10 activities
        """
        del args
        del kwargs

        output: dict[CondensedSpecies, float] = {}
        for species in self._species.condensed_species:
            try:
                output[species] = np.log10(self.value[species])
            except KeyError:
                output[species] = self._fill_log10_activity

        return output

    @override
    def _get_log10_masses(self, *args, **kwargs) -> dict[CondensedSpecies, float]:
        """Initial solution for log10 condensed masses

        Returns:
            Log10 condensed masses
        """
        del args
        del kwargs

        output: dict[CondensedSpecies, float] = {}
        for species in self._species.condensed_species:
            try:
                output[species] = np.log10(self.value[f"mass_{species}"])
            except KeyError:
                output[species] = self._fill_log10_mass

        return output

    @override
    def _get_log10_stabilities(self, *args, **kwargs) -> dict[CondensedSpecies, float]:
        """Initial solution for log10 stabilities

        Returns:
            Log10 stabilities
        """
        del args
        del kwargs

        output: dict[CondensedSpecies, float] = {}
        for species in self._species.condensed_species:
            try:
                output[species] = np.log10(self.value[f"stability_{species}"])
            except KeyError:
                output[species] = self._fill_log10_stability

        return output

    # @override
    # def __init__(
    #     self,
    #     value: Mapping[TypeChemicalSpecies_co, float],
    #     *,
    #     species: Species,
    #     min_log10_pressure: float = MIN_LOG10_PRESSURE,
    #     max_log10_pressure: float = MAX_LOG10_PRESSURE,
    #     fill_pressure: float = 1,
    #     fill_activity: float = 1,
    # ):
    #     pressures_dict: Mapping[TypeChemicalSpecies_co, float] = {
    #         unique_species: fill_pressure for unique_species in species.gas_species
    #     }
    #     activities_dict: Mapping[TypeChemicalSpecies_co, float] = {
    #         unique_species: fill_activity for unique_species in species.condensed_species
    #     }
    #     species_dict |= value
    #     species_ic: npt.NDArray[np.float_] = np.array(list(species_dict.values()))
    #     super().__init__(
    #         species_ic,
    #         species=species,
    #         min_log10_pressure=min_log10_pressure,
    #         max_log10_pressure=max_log10_pressure,
    #     )
    #     logger.debug("initial_solution = %s", self.asdict())

    # def asdict(self) -> dict[str, float]:
    #     """Dictionary of the initial solution"""
    #     return dict(zip(self.species.names, self.value))

    # @override
    # def get_value(
    #     self, constraints: SystemConstraints, *, temperature: float, pressure: float
    # ) -> npt.NDArray:
    #     del args
    #     del kwargs
    #     logger.debug("%s: value = %s", self.__class__.__name__, self.value)

    #     return self.value


# class InitialSolutionRegressor(InitialSolution[Output]):
#     """A regressor to compute the initial solution

#     Args:
#         value: Output for constructing the regressor
#         species: Species
#         min_log10_pressure: Minimum log10 value. Defaults to :data:`MIN_LOG10_PRESSURE`.
#         max_log10_pressure: Maximum log10 value. Defaults to :data:`MAX_LOG10_PRESSURE`.
#         species_fill: Dictionary of missing species and their initial values. Defaults to None.
#         fill_value: Initial value for species that are not specified in `species_fill`. Defaults to
#             1.
#         fit: Fit the regressor during the model run. This will replace the original regressor by a
#             regressor trained only on the data from the current model. Defaults to True.
#         fit_batch_size: Number of solutions to calculate before fitting model data if fit is True.
#             Defaults to 100.
#         partial_fit: Partial fit the regressor during the model run. Defaults to True.
#         partial_fit_batch_size: Number of solutions to calculate before partial refit of the
#             regressor. Defaults to 500.

#     Attributes:
#         value: Output for constructing the regressor
#         fit: Fit the regressor during the model run, which replaces the data used to initialise
#             the regressor.
#         fit_batch_size: Number of solutions to calculate before fitting model data if fit is True
#         partial_fit: Partial fit the regressor during the model run
#         partial_fit_batch_size: Number of solutions to calculate before partial refit of the
#             regressor
#     """

#     # For typing
#     _reg: MultiOutputRegressor
#     _solution_scalar: StandardScaler
#     _constraints_scalar: StandardScaler

#     @override
#     def __init__(
#         self,
#         value: Output,
#         *,
#         species: Species,
#         min_log10_pressure: float = MIN_LOG10_PRESSURE,
#         max_log10_pressure: float = MAX_LOG10_PRESSURE,
#         species_fill: dict[TypeChemicalSpecies_co, float] | None = None,
#         fill_value: float = 1,
#         fit: bool = True,
#         fit_batch_size: int = 100,
#         partial_fit: bool = True,
#         partial_fit_batch_size: int = 500,
#     ):
#         self.fit: bool = fit
#         # Ensure consistency of arguments and correct handling of fit versus partial refit.
#         self.fit_batch_size: int = fit_batch_size if self.fit else 0
#         self.partial_fit: bool = partial_fit
#         self.partial_fit_batch_size: int = partial_fit_batch_size

#         self._conform_solution(
#             value, species=species, species_fill=species_fill, fill_value=fill_value
#         )
#         super().__init__(
#             value,
#             species=species,
#             min_log10_pressure=min_log10_pressure,
#             max_log10_pressure=max_log10_pressure,
#         )
#         self._fit(self.value)

#     @classmethod
#     def from_pickle(cls, pickle_file: Path | str, **kwargs) -> InitialSolutionRegressor:
#         """Creates a regressor from output read from a pickle file.

#         Args:
#             pickle_file: Pickle file of the output from a previous (or similar) model run. The
#                 constraints must be the same as the new model and in the same order.
#             **kwargs: Arbitrary keyword arguments to pass through to the constructor

#         Returns:
#             A regressor
#         """
#         output: Output = Output.read_pickle(pickle_file)

#         return cls(output, **kwargs)

#     def _conform_solution(
#         self,
#         output: Output,
#         *,
#         species: Species,
#         species_fill: dict[TypeChemicalSpecies_co, float] | None,
#         fill_value: float,
#     ) -> None:
#         """Conforms the solution in output to the species and their fill values

#         Args:
#             output: Output
#             species: Species
#             species_fill: Dictionary of missing species and their initial values. Defaults to None.
#             fill_value: Initial value for species that are not specified in `species_fill`.
#         """
#         solution: pd.DataFrame = output.to_dataframes()["solution"].copy()
#         logger.debug("solution = %s", solution)

#         species_fill_: dict[TypeChemicalSpecies_co, float] = (
#             species_fill if species_fill is not None else {}
#         )
#         initial_solution_dict: InitialSolutionDict = InitialSolutionDict(
#             species_fill_,
#             species=species,
#             fill_value=fill_value,
#         )
#         fill_df: pd.DataFrame = pd.DataFrame(initial_solution_dict.asdict(), index=[0])
#         fill_df = fill_df.loc[fill_df.index.repeat(len(solution))].reset_index(drop=True)
#         logger.debug("fill_df = %s", fill_df)

#         # Preference the values in the solution and fill missing species
#         conformed_solution: pd.DataFrame = solution.combine_first(fill_df)[species.names]
#         logger.debug("conformed_solution = %s", conformed_solution)
#         output["solution"] = conformed_solution.to_dict(orient="records")

#     def _get_log10_values(
#         self,
#         output: Output,
#         name: str,
#         start_index: int | None,
#         end_index: int | None,
#     ) -> npt.NDArray[np.float_]:
#         """Gets log10 values of either the constraints or the solution from `output`

#         Args:
#             output: Output
#             name: solution or constraints
#             start_index: Start index for fit. Defaults to None, meaning use all available data.
#             end_index: End index for fit. Defaults to None, meaning use all available data.

#         Returns:
#             Log10 values of either the solution or constraints depending on `name`
#         """
#         output_dataframes: dict[str, pd.DataFrame] = output.to_dataframes()
#         data: pd.DataFrame = output_dataframes[name]
#         if start_index is not None and end_index is not None:
#             data = data.iloc[start_index:end_index]
#         data_log10_values: npt.NDArray = np.log10(data.values)

#         return data_log10_values

#     def _fit(
#         self,
#         output: Output,
#         start_index: int | None = None,
#         end_index: int | None = None,
#     ) -> None:
#         """Fits and sets the regressor.

#         Args:
#             output: Output
#             start_index: Start index for fit. Defaults to None, meaning use all available data.
#             end_index: End index for fit. Defaults to None, meaning use all available data.
#         """
#         logger.info("%s: Fit (%s, %s)", self.__class__.__name__, start_index, end_index)

#         constraints_log10_values: npt.NDArray = self._get_log10_values(
#             output, "constraints", start_index, end_index
#         )
#         self._constraints_scalar = StandardScaler().fit(constraints_log10_values)
#         constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
#             constraints_log10_values
#         )

#         solution_log10_values: npt.NDArray = self._get_log10_values(
#             output, "solution", start_index, end_index
#         )
#         self._solution_scalar = StandardScaler().fit(solution_log10_values)
#         solution_scaled: npt.NDArray | spmatrix = self._solution_scalar.transform(
#             solution_log10_values
#         )

#         base_regressor: SGDRegressor = SGDRegressor()
#         multi_output_regressor: MultiOutputRegressor = MultiOutputRegressor(base_regressor)
#         multi_output_regressor.fit(constraints_scaled, solution_scaled)

#         self._reg = multi_output_regressor

#     def _partial_fit(
#         self,
#         output: Output,
#         start_index: int,
#         end_index: int,
#     ) -> None:
#         """Partial fits the regressor.

#         Args:
#             output: Output
#             start_index: Start index for partial fit
#             end_index: End index for partial fit
#         """
#         logger.info("%s: Partial fit (%d, %d)", self.__class__.__name__, start_index, end_index)

#         constraints_log10_values: npt.NDArray = self._get_log10_values(
#             output, "constraints", start_index, end_index
#         )
#         constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
#             constraints_log10_values
#         )
#         solution_log10_values: npt.NDArray = self._get_log10_values(
#             output, "solution", start_index, end_index
#         )
#         solution_scaled: npt.NDArray | spmatrix = self._solution_scalar.transform(
#             solution_log10_values
#         )

#         self._reg.partial_fit(constraints_scaled, solution_scaled)

#     @override
#     def get_value(
#         self, constraints: SystemConstraints, temperature: float, pressure: float
#     ) -> npt.NDArray:
#         evaluated_constraints_log10: dict[str, float] = constraints.evaluate_log10(
#             temperature=temperature, pressure=pressure
#         )
#         values_constraints_log10: npt.NDArray = np.array(
#             list(evaluated_constraints_log10.values())
#         )
#         values_constraints_log10 = values_constraints_log10.reshape(1, -1)
#         constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
#             values_constraints_log10
#         )
#         solution_scaled: npt.NDArray | spmatrix = self._reg.predict(constraints_scaled)
#         solution_original: npt.NDArray = cast(
#             npt.NDArray, self._solution_scalar.inverse_transform(solution_scaled)
#         )

#         value: npt.NDArray = 10 ** solution_original.flatten()
#         logger.debug("%s: value = %s", self.__class__.__name__, value)

#         return value

#     # TODO: Dan testing improving the regressor when condensates are present
#     @override
#     def get_log10_value(
#         self,
#         constraints: SystemConstraints,
#         *,
#         temperature: float,
#         pressure: float,
#         perturb: bool = False,
#         perturb_log10: float = 2,
#     ) -> npt.NDArray[np.float_]:
#         """Computes the log10 value of the initial solution with additional processing.

#         Args:
#             constraints: Constraints
#             temperature: Temperature in K
#             pressure: Pressure in bar
#             perturb: Randomly perturb the log10 value by `perturb_log10`. Defaults to False.
#             perturb_log10: Maximum absolute log10 value to perturb the initial solution. Defaults
#                 to 2.

#         Returns
#             The log10 initial solution adhering to bounds and the constraints
#         """
#         value: npt.NDArray[np.float_] = self.get_value(constraints, temperature, pressure)
#         log10_value: npt.NDArray[np.float_] = np.log10(value)

#         if perturb:
#             logger.info(
#                 "Randomly perturbing the initial solution by a maximum of %f log10 units",
#                 perturb_log10,
#             )
#             log10_value += perturb_log10 * (2 * np.random.rand(log10_value.size) - 1)

#         if np.any((log10_value < self.min_log10) | (log10_value > self.max_log10)):
#             logger.warning("Initial solution has values outside the min and max thresholds")
#             logger.warning(
#                 "Clipping the initial solution between %f and %f", self.min_log10, self.max_log10
#             )
#             log10_value = np.clip(log10_value, self.min_log10, self.max_log10)

#         # Apply constraints from the reaction network (activities and fugacities)
#         # for constraint in constraints.reaction_network_constraints:
#         #    index: int = self.species.species_index(constraint.species)
#         #    logger.debug("Setting %s %d", constraint.species, index)
#         #    log10_value[index] = constraint.get_log10_value(
#         #        temperature=temperature, pressure=pressure
#         #    )
#         logger.debug("Conform initial solution to constraints = %s", log10_value)

#         # This assumes an initial condensed mass of 10^20 kg, but this selection is quite
#         # arbitrary and a smarter initial guess could probably be made.
#         log_condensed_mass: npt.NDArray = 20 * np.ones(self._species.number_condensed_species)
#         log10_value = np.append(log10_value, log_condensed_mass)

#         # Small lambda factors assume the condensates are stable, which is probably a reasonable
#         # assumption given that the user has chosen to include them as species.
#         log_lambda: npt.NDArray = -12 * np.ones(self._species.number_condensed_species)
#         log10_value = np.append(log10_value, log_lambda)

#         return log10_value

#     def action_fit(self, output: Output) -> tuple[int, int] | None:
#         """Checks if a fit is necessary.

#         Args:
#             output: Output

#         Returns:
#             The start and end index of the data to fit, or None (meaning no fit necessary)
#         """
#         trigger_fit: bool = self.fit_batch_size == output.size

#         if self.fit and trigger_fit:
#             return (0, output.size)

#     def action_partial_fit(self, output: Output) -> tuple[int, int] | None:
#         """Checks if a partial refit is necessary.

#         Args:
#             output: Output

#         Returns:
#             The start and end index of the data to fit, or None (meaning no fit necessary)
#         """
#         trigger_partial_fit: bool = (
#             not (output.size - self.fit_batch_size) % self.partial_fit_batch_size
#             and (output.size - self.fit_batch_size) > 0
#         )
#         if self.partial_fit and trigger_partial_fit:
#             batch_number: int = (output.size - self.fit_batch_size) // self.partial_fit_batch_size
#             start_index: int = (
#                 self.fit_batch_size + (batch_number - 1) * self.partial_fit_batch_size
#             )
#             end_index: int = start_index + self.partial_fit_batch_size

#             return (start_index, end_index)

#     @override
#     def update(self, output: Output) -> None:
#         action_fit: tuple[int, int] | None = self.action_fit(output)
#         action_partial_fit: tuple[int, int] | None = self.action_partial_fit(output)

#         if action_fit is not None:
#             self._fit(output, start_index=action_fit[0], end_index=action_fit[1])

#         if action_partial_fit is not None:
#             self._partial_fit(
#                 output, start_index=action_partial_fit[0], end_index=action_partial_fit[1]
#             )


# class InitialSolutionSwitchRegressor(InitialSolution[InitialSolution]):
#     """An initial solution that uses an initial solution before switching to a regressor.

#     Args:
#         value: An initial solution
#         species: Species
#         min_log10_pressure: Minimum log10 value. Defaults to :data:`MIN_LOG10_PRESSURE`.
#         max_log10_pressure: Maximum log10 value. Defaults to :data:`MAX_LOG10_PRESSURE`.
#         fit_batch_size: Number of simulations to generate before fitting the regressor. Defaults
#             to 100.
#         **kwargs: Keyword arguments that are specific to :class:`InitialSolutionRegressor`

#     Attributes:
#         value: An initial solution
#         fit_batch_size: Number of simulations to generate before fitting the regressor
#     """

#     @override
#     def __init__(
#         self,
#         value: InitialSolution,
#         *,
#         species: Species,
#         min_log10_pressure: float = MIN_LOG10_PRESSURE,
#         max_log10_pressure: float = MAX_LOG10_PRESSURE,
#         fit_batch_size: int = 100,
#         **kwargs,
#     ):
#         super().__init__(
#             value,
#             species=species,
#             min_log10_pressure=min_log10_pressure,
#             max_log10_pressure=max_log10_pressure,
#         )
#         self._fit_batch_size: int = fit_batch_size
#         # Store to instantiate regressor once the switch occurs.
#         self._kwargs: dict[str, Any] = kwargs

#     @override
#     def get_value(self, *args, **kwargs) -> npt.NDArray:
#         return self.value.get_value(*args, **kwargs)

#     @override
#     def update(self, output: Output, *args, **kwargs) -> None:
#         if output.size == self._fit_batch_size:
#             # The fit keyword argument of InitialSolutionRegressor is effectively ignored because
#             # the fit is done once when InitialSolutionRegressor is instantiated and action_fit
#             # cannot be triggered regardless of the value of fit.
#             self.value = InitialSolutionRegressor(
#                 output,
#                 species=self.species,
#                 min_log10_pressure=self.min_log10_pressure,
#                 max_log10_pressure=self.max_log10_pressure,
#                 **self._kwargs,
#             )
#         else:
#             self.value.update(output, *args, **kwargs)


# class InitialSolutionLast(InitialSolution[InitialSolution]):
#     """An initial solution that uses the previous output value as the next initial solution.

#     This is useful if you are incrementing sequentially through a grid of parameters.

#     Args:
#         value: An initial solution for the first solution only
#         species: Species
#         min_log10_pressure: Minimum log10 value. Defaults to :data:`MIN_LOG10_PRESSURE`.
#         max_log10_pressure: Maximum log10 value. Defaults to :data:`MAX_LOG10_PRESSURE`.

#     Attributes:
#         value: An object or value used to compute the initial solution
#     """

#     @override
#     def get_value(self, *args, **kwargs) -> npt.NDArray:
#         return self.value.get_value(*args, **kwargs)

#     @override
#     def update(self, output: Output, *args, **kwargs) -> None:
#         del args
#         del kwargs
#         last_output: dict[str, float] = output["solution"][-1]
#         next_values: dict[ChemicalSpecies, float] = {}
#         for species in self.species.data:
#             next_values[species] = last_output[species.name]

#         self.value = InitialSolutionDict(
#             next_values,
#             species=self.species,
#             min_log10_pressure=self.min_log10_pressure,
#             max_log10_pressure=self.max_log10_pressure,
#         )
