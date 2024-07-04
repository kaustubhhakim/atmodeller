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
from typing import Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from atmodeller.constraints import SystemConstraints
from atmodeller.core import MASS_PREFIX, STABILITY_PREFIX, Solution, Species
from atmodeller.interfaces import ChemicalSpecies, TypeChemicalSpecies_co
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


class InitialSolution(ABC, Generic[T]):
    """Initial solution

    Args:
        value: An object used to compute the initial solution
        species: Species
        min_log10_pressure: Minimum log10 gas pressure. Defaults to :data:`MIN_LOG10_PRESSURE`.
        max_log10_pressure: Maximum log10 gas pressure. Defaults to :data:`MAX_LOG10_PRESSURE`.
        fill_log10_pressure: Fill value for pressure in bar. Defaults to 1.
        fill_log10_activity: Fill value for activity. Defaults to 0.
        fill_log10_mass: Fill value for mass. Defaults to 20.
        fill_log10_stability: Fill value for stability. Defaults to -35.
        solution_override: Dictionary to override the initial solution values. Defaults to None.

    Attributes:
        value: An object used to compute the initial solution
        solution: The initial solution
    """

    def __init__(
        self,
        value: T,
        *,
        species: Species,
        min_log10_pressure: float = MIN_LOG10_PRESSURE,
        max_log10_pressure: float = MAX_LOG10_PRESSURE,
        fill_log10_pressure: float = 1,
        fill_log10_activity: float = 0,
        fill_log10_mass: float = 20,
        fill_log10_stability: float = -35,
        solution_override: InitialSolutionDict | None = None,
    ):
        logger.info("Creating %s", self.__class__.__name__)
        self.value: T = value
        self.solution: Solution = Solution(species)
        self._species: Species = species
        self._min_log10_pressure: float = min_log10_pressure
        self._max_log10_pressure: float = max_log10_pressure
        self._fill_log10_pressure: float = fill_log10_pressure
        self._fill_log10_activity: float = fill_log10_activity
        self._fill_log10_mass: float = fill_log10_mass
        self._fill_log10_stability: float = fill_log10_stability
        if solution_override is None:
            self._solution_override: InitialSolutionDict | None = None
        else:
            self._solution_override = solution_override

    @abstractmethod
    def set_data_preprocessing(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> None:
        """Sets the raw data for the initial solution.

        This sets the raw data without additional processing such as clipping or perturbing.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
        """

    def set_data_postprocessing(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
    ) -> None:
        """Applies a user-specified override to the initial solution

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
        """
        if self._solution_override is not None:
            self._solution_override.set_data(
                constraints,
                temperature=temperature,
                pressure=pressure,
                fill_missing_values=False,
                apply_gas_constraints=False,
                apply_activity_constraints=False,
            )
            self.solution.merge(self._solution_override.solution)

    def set_data(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_gas_log10: float = 0,
        fill_missing_values: bool = True,
        apply_gas_constraints: bool = True,
        apply_activity_constraints: bool = True,
    ) -> None:
        """Sets the initial solution data.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_gas_log10: Maximum log10 value to perturb the gas pressures. Defaults to 0.
            apply_gas_constraints: Apply gas constraints, if any. Defaults to True.
            apply_activity_constraints: Apply activity constraints, if any. Defaults to True.
        """

        self.set_data_preprocessing(constraints, temperature=temperature, pressure=pressure)

        if fill_missing_values:
            self.solution.gas.fill_missing_values(self._fill_log10_pressure)
            self.solution.activity.fill_missing_values(self._fill_log10_activity)
            self.solution.mass.fill_missing_values(self._fill_log10_mass)
            self.solution.stability.fill_missing_values(self._fill_log10_stability)

        # Gas pressures
        if perturb_gas_log10:
            self.solution.gas.perturb_values(perturb_gas_log10)

        self.solution.gas.clip_values(self._min_log10_pressure, self._max_log10_pressure)

        if apply_gas_constraints:
            for constraint in constraints.gas_constraints:
                self.solution.gas.data[constraint.species] = constraint.get_log10_value(
                    temperature=temperature, pressure=pressure
                )

        # Activities
        self.solution.activity.clip_values(maximum_value=0)
        if apply_activity_constraints:
            for constraint in constraints.activity_constraints:
                self.solution.activity.data[constraint.species] = constraint.get_log10_value(
                    temperature=temperature, pressure=pressure
                )

        self.set_data_postprocessing(constraints, temperature=temperature, pressure=pressure)

    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_gas_log10: float = 0,
    ) -> npt.NDArray[np.float_]:
        """Gets the log10 value of the initial solution.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_gas_log10: Maximum log10 value to perturb the gas pressures. Defaults to 0.

        Returns:
            The initial solution
        """
        self.set_data(
            constraints,
            temperature=temperature,
            pressure=pressure,
            perturb_gas_log10=perturb_gas_log10,
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
    """A dictionary for the initial solution

    Args:
        value: Dictionary of the initial solution. Defaults to None, meaning to use default values.
        **kwargs: Keyword arguments to pass through to the base class.

    Attributes:
        value: An object used to compute the initial solution
    """

    @override
    def __init__(self, value: dict | None = None, **kwargs):
        if value is None:
            value_dict: dict = {}
        else:
            value_dict = value
        super().__init__(value_dict, **kwargs)

    def _get_log10_values(
        self,
        species_list: list[TypeChemicalSpecies_co],
        prefix: str,
    ) -> dict[TypeChemicalSpecies_co, float]:
        """Gets log10 values.

        Args:
            species_list: List of species
            prefix: Key prefix
            fill_value: Fill value

        Returns:
            Log10 values
        """
        output: dict[TypeChemicalSpecies_co, float] = {}
        for species in species_list:
            key: TypeChemicalSpecies_co | str = f"{prefix}{species.name}" if prefix else species
            try:
                output[species] = np.log10(self.value[key])
            # TODO: Clean up or add explaining comment
            except KeyError:
                continue

        return output

    @override
    def set_data_preprocessing(self, *args, **kwargs) -> None:
        del args
        del kwargs
        self.solution.gas.data = self._get_log10_values(self._species.gas_species, "")
        self.solution.activity.data = self._get_log10_values(self._species.condensed_species, "")
        self.solution.mass.data = self._get_log10_values(
            self._species.condensed_species, MASS_PREFIX
        )
        self.solution.stability.data = self._get_log10_values(
            self._species.condensed_species, STABILITY_PREFIX
        )


class InitialSolutionLast(InitialSolution[InitialSolution]):
    """An initial solution that uses the previous output value as the current solution guess.

    This is useful if you are incrementing through a grid of parameters, such that the previous
    solution is a reasonable initial estimate for the current solution.

    Args:
        value: An initial solution for the first solution only
        **kwargs: Keyword arguments to pass through to the base class.

    Attributes:
        value: An initial solution for the first solution only
    """

    @override
    def __init__(self, value: InitialSolution | None = None, **kwargs):
        if value is None:
            value_initial: InitialSolution = InitialSolutionDict(**kwargs)
        else:
            value_initial = value
        super().__init__(value_initial, **kwargs)
        # Must re-route solution
        self.solution = self.value.solution

    @override
    def set_data_preprocessing(self, *args, **kwargs) -> None:
        return self.value.set_data_preprocessing(*args, **kwargs)

    @override
    def update(self, output: Output, *args, **kwargs) -> None:
        del args
        del kwargs
        value_dict: dict[ChemicalSpecies | str, float] = output["raw_solution"][-1]
        # InitialSolutionDict takes the log10, so we must raise 10 to the values.
        value_dict = {key: 10**value for key, value in value_dict.items()}
        # Convert species from strings to objects
        for species in self._species.data:
            value_dict[species] = value_dict.pop(species.name)

        self.value = InitialSolutionDict(value_dict, species=self._species)
        # Must re-route solution
        self.solution = self.value.solution


class InitialSolutionRegressor(InitialSolution[Output]):
    """A regressor to compute the initial solution

    Importantly, the type and order of constraints must be the same in the new model as the
    previous model, but the values of the constraints can be different.

    Args:
        value: Output for constructing the regressor
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model. Defaults to True.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True.
            Defaults to 100.
        partial_fit: Partial fit the regressor during the model run. Defaults to True.
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor. Defaults to 500.
        **kwargs: Keyword arguments to pass through to the base class.

    Attributes:
        value: Output for constructing the regressor
        fit: Fit the regressor during the model run, which replaces the data used to initialise
            the regressor.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True
        partial_fit: Partial fit the regressor during the model run
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor
    """

    # For typing
    _reg: MultiOutputRegressor
    _solution_scalar: StandardScaler
    _constraints_scalar: StandardScaler

    @override
    def __init__(
        self,
        value: Output,
        *,
        fit: bool = True,
        fit_batch_size: int = 100,
        partial_fit: bool = True,
        partial_fit_batch_size: int = 500,
        **kwargs,
    ):
        self.fit: bool = fit
        # Ensure consistency of arguments and correct handling of fit versus partial refit.
        self.fit_batch_size: int = fit_batch_size if self.fit else 0
        self.partial_fit: bool = partial_fit
        self.partial_fit_batch_size: int = partial_fit_batch_size

        super().__init__(value, **kwargs)
        self._fit(self.value)

    @classmethod
    def from_pickle(cls, pickle_file: Path | str, **kwargs) -> InitialSolutionRegressor:
        """Creates a regressor from output read from a pickle file.

        Args:
            pickle_file: Pickle file of the output from a previous (or similar) model run. The
                constraints must be the same as the new model and in the same order.
            **kwargs: Arbitrary keyword arguments to pass through to the constructor

        Returns:
            A regressor
        """
        try:
            output: Output = Output.read_pickle(pickle_file)
        except FileNotFoundError:
            output = Output.read_pickle(Path(pickle_file).with_suffix(".pkl"))

        return cls(output, **kwargs)

    def _get_select_values(
        self,
        data: pd.DataFrame,
        start_index: int | None,
        end_index: int | None,
    ) -> npt.NDArray[np.float_]:
        """Gets select values from a dataframe

        Args:
            data: A dataframe
            start_index: Start index. Defaults to None, meaning use all available data.
            end_index: End index. Defaults to None, meaning use all available data.

        Returns:
            Select values from the dataframe
        """
        if start_index is not None and end_index is not None:
            data = data.iloc[start_index:end_index]

        return data.values

    def _fit(
        self,
        output: Output,
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> None:
        """Fits and sets the regressor.

        Args:
            output: Output
            start_index: Start index for fit. Defaults to None, meaning use all available data.
            end_index: End index for fit. Defaults to None, meaning use all available data.
        """
        logger.info("%s: Fit (%s, %s)", self.__class__.__name__, start_index, end_index)

        dataframes: dict[str, pd.DataFrame] = output.to_dataframes()

        constraints_log10_values: npt.NDArray = np.log10(
            self._get_select_values(dataframes["constraints"], start_index, end_index)
        )
        self._constraints_scalar = StandardScaler().fit(constraints_log10_values)
        constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )

        raw_solution_log10_values: npt.NDArray = self._get_select_values(
            dataframes["raw_solution"], start_index, end_index
        )
        self._solution_scalar = StandardScaler().fit(raw_solution_log10_values)
        solution_scaled: npt.NDArray | spmatrix = self._solution_scalar.transform(
            raw_solution_log10_values
        )

        base_regressor: SGDRegressor = SGDRegressor(tol=1e-6)
        multi_output_regressor: MultiOutputRegressor = MultiOutputRegressor(base_regressor)
        multi_output_regressor.fit(constraints_scaled, solution_scaled)

        self._reg = multi_output_regressor

    def _partial_fit(
        self,
        output: Output,
        start_index: int,
        end_index: int,
    ) -> None:
        """Partial fits the regressor.

        Args:
            output: Output
            start_index: Start index for partial fit
            end_index: End index for partial fit
        """
        logger.info("%s: Partial fit (%d, %d)", self.__class__.__name__, start_index, end_index)

        dataframes: dict[str, pd.DataFrame] = output.to_dataframes()

        constraints_log10_values: npt.NDArray = np.log10(
            self._get_select_values(dataframes["constraints"], start_index, end_index)
        )
        constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )
        solution_log10_values: npt.NDArray = self._get_select_values(
            dataframes["raw_solution"], start_index, end_index
        )
        solution_scaled: npt.NDArray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

        self._reg.partial_fit(constraints_scaled, solution_scaled)

    @override
    def set_data_preprocessing(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> None:
        evaluated_constraints_log10: dict[str, float] = constraints.evaluate_log10(
            temperature=temperature, pressure=pressure
        )
        values_constraints_log10: npt.NDArray = np.array(
            list(evaluated_constraints_log10.values())
        )
        values_constraints_log10 = values_constraints_log10.reshape(1, -1)
        constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
            values_constraints_log10
        )
        solution_scaled: npt.NDArray | spmatrix = self._reg.predict(constraints_scaled)
        solution_original: npt.NDArray = cast(
            npt.NDArray, self._solution_scalar.inverse_transform(solution_scaled)
        )

        self.solution.data = solution_original.flatten()

    def action_fit(self, output: Output) -> tuple[int, int] | None:
        """Checks if a fit is necessary.

        Args:
            output: Output

        Returns:
            The start and end index of the data to fit, or None (meaning no fit necessary)
        """
        trigger_fit: bool = self.fit_batch_size == output.size

        if self.fit and trigger_fit:
            return (0, output.size)

    def action_partial_fit(self, output: Output) -> tuple[int, int] | None:
        """Checks if a partial refit is necessary.

        Args:
            output: Output

        Returns:
            The start and end index of the data to fit, or None (meaning no fit necessary)
        """
        trigger_partial_fit: bool = (
            not (output.size - self.fit_batch_size) % self.partial_fit_batch_size
            and (output.size - self.fit_batch_size) > 0
        )
        if self.partial_fit and trigger_partial_fit:
            batch_number: int = (output.size - self.fit_batch_size) // self.partial_fit_batch_size
            start_index: int = (
                self.fit_batch_size + (batch_number - 1) * self.partial_fit_batch_size
            )
            end_index: int = start_index + self.partial_fit_batch_size

            return (start_index, end_index)

    @override
    def update(self, output: Output) -> None:
        action_fit: tuple[int, int] | None = self.action_fit(output)
        action_partial_fit: tuple[int, int] | None = self.action_partial_fit(output)

        if action_fit is not None:
            self._fit(output, start_index=action_fit[0], end_index=action_fit[1])

        if action_partial_fit is not None:
            self._partial_fit(
                output, start_index=action_partial_fit[0], end_index=action_partial_fit[1]
            )


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
