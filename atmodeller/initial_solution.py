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

# import random
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from atmodeller.constraints import SystemConstraints
from atmodeller.core import Species
from atmodeller.interfaces import ChemicalSpecies
from atmodeller.output import Output
from atmodeller.reaction_network import log10_TAU
from atmodeller.solution import (
    ACTIVITY_PREFIX,
    CONDENSED_MASS_PREFIX,
    STABILITY_PREFIX,
    Solution,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

MIN_LOG10_NUMBER_DENSITY: float = -40
"""Minimum log10 of the initial number density"""
MAX_LOG10_NUMBER_DENSITY: float = 40
"""Maximum log10 of the initial number density"""


class InitialSolutionProtocol(Protocol):
    """Initial solution protocol"""

    @property
    def species(self) -> Species: ...

    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_gas_log10: float = 0,
        attempt: int = 0,
    ) -> npt.NDArray[np.float_]: ...

    def update(self, output: Output) -> None: ...


class InitialSolution(ABC, Generic[T]):
    """Initial solution

    Args:
        value: An object used to compute the initial solution
        species: Species
        min_log10_number_density: Minimum log10 number density. Defaults to
            :data:`MIN_LOG10_NUMBER_DENSITY`.
        max_log10_number_density: Maximum log10 number density. Defaults to
            :data:`MAX_LOG10_NUMBER_DENSITY`.
        fill_log10_number_density: Fill value for number density. Defaults to 22.
        fill_log10_activity: Fill value for activity. Defaults to 0.
        fill_log10_stability: Fill value for stability. Defaults to -35.

    Attributes:
        value: An object used to compute the initial solution
        solution: The initial solution
    """

    def __init__(
        self,
        value: T,
        *,
        species: Species,
        min_log10_number_density: float = MIN_LOG10_NUMBER_DENSITY,
        max_log10_number_density: float = MAX_LOG10_NUMBER_DENSITY,
        fill_log10_number_density: float = 26,
        fill_log10_activity: float = 0,
        fill_log10_stability: float = -20,
    ):
        logger.info("Creating %s", self.__class__.__name__)
        self.value: T = value
        self.solution: Solution = Solution(species)
        self._species: Species = species
        self._min_log10_number_density: float = min_log10_number_density
        self._max_log10_number_density: float = max_log10_number_density
        self._fill_log10_number_density: float = fill_log10_number_density
        self._fill_log10_activity: float = fill_log10_activity
        self._fill_log10_stability: float = fill_log10_stability

    @property
    def species(self) -> Species:
        """Species"""
        return self._species

    @abstractmethod
    def set_data(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float
    ) -> None:
        """Sets the raw data for the initial solution.

        This sets the raw data without additional processing such as clipping or perturbing.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
        """

    def process_data(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_gas_log10: float = 0,
    ) -> None:
        """Processes the initial solution data.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_gas_log10: Maximum log10 value to perturb the gas number densities. Defaults to
                0.
        """
        self.set_data(constraints, temperature=temperature, pressure=pressure)

        for solution in self.solution.gas.values():
            solution.gas.fill(self._fill_log10_number_density)
            if perturb_gas_log10:
                solution.gas.perturb(perturb_gas_log10)
            solution.gas.clip(self._min_log10_number_density, self._max_log10_number_density)

        for constraint in constraints.gas_constraints:
            self.solution.gas[constraint.species].gas.value = constraint.get_log10_value(
                temperature=temperature, pressure=pressure
            )

        for solution in self.solution.condensed.values():
            solution.activity.fill(self._fill_log10_activity)
            solution.mass.fill(self._fill_log10_number_density)
            solution.stability.fill(self._fill_log10_stability)
            if perturb_gas_log10:
                solution.mass.perturb(perturb_gas_log10)
                # solution.stability.perturb(perturb_gas_log10)

            # TODO: This imposes the activity value if stable, but might be in conflict with
            # stability criteria for unstable condensates?
            # solution.activity.clip(maximum_value=0)

            # Satisfy auxilliary equation by construction
            # solution.stability.value = log10_TAU - solution.mass.value

        for constraint in constraints.activity_constraints:
            self.solution.condensed[constraint.species].activity.value = (
                constraint.get_log10_value(temperature=temperature, pressure=pressure)
            )

        # TODO: Testing. If the solver fails it could be because one or several of the condensed
        # species are unstable. Just randomly guess here.
        # if perturb_gas_log10:
        # for species in self.solution.activity.data:
        #     stability: str = random.choice(["stable", "unstable"])
        #     if stability == "stable":
        #         self.solution.activity.data[species] = 0
        #         self.solution.mass.data[species] = 19
        #     else:
        #         self.solution.activity.data[species] = -24
        #         self.solution.mass.data[species] = -16
        # self.solution.condensed.perturb_values(10)

    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb_gas_log10: float = 0,
        attempt: int = 0,
    ) -> npt.NDArray[np.float_]:
        """Gets the log10 value of the initial solution.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb_gas_log10: Maximum log10 value to perturb the gas number densities. Defaults to
                0.
            attempt: Solution attempt number

        Returns:
            The initial solution
        """
        # Only perturb the value after the first attempt
        if attempt == 0:
            perturb_value = 0
        else:
            perturb_value = perturb_gas_log10

        self.process_data(
            constraints,
            temperature=temperature,
            pressure=pressure,
            perturb_gas_log10=perturb_value,
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
        **kwargs: Optional keyword arguments to pass through to the base class.

    Attributes:
        value: A dictionary used to compute the initial solution
        solution: The initial solution
    """

    @override
    def __init__(self, value: dict | None = None, *, species: Species, **kwargs):
        if value is None:
            value_dict: dict = {}
        else:
            value_dict = value
        super().__init__(value_dict, species=species, **kwargs)

    def _get_log10_values(
        self,
        species: ChemicalSpecies,
        prefix: str,
    ) -> float | None:
        """Gets log10 values.

        Args:
            species_list: List of species
            prefix: Key prefix

        Returns:
            Log10 values or None
        """

        key: ChemicalSpecies | str = f"{prefix}{species.name}" if prefix else species

        try:
            output: float | None = np.log10(self.value[key])
        except KeyError:
            # Ignore missing keys. These are later filled with fill values.
            output = None

        return output

    @override
    def set_data(self, *args, **kwargs) -> None:
        del args
        del kwargs

        for species, solution in self.solution.gas.items():
            value: float | None = self._get_log10_values(species, "")
            if value is not None:
                solution.gas.value = value

        for species, solution in self.solution.condensed.items():
            value: float | None = self._get_log10_values(species, ACTIVITY_PREFIX)
            if value is not None:
                solution.activity.value = value
            value: float | None = self._get_log10_values(species, CONDENSED_MASS_PREFIX)
            if value is not None:
                solution.mass.value = value
            value: float | None = self._get_log10_values(species, STABILITY_PREFIX)
            if value is not None:
                solution.stability.value = value

        # self.solution.gas.data = self._get_log10_values(self._species.gas_species, "")

        # self.solution.condensed.activity.data = self._get_log10_values(
        #     self._species.condensed_species, ""
        # )
        # self.solution.condensed.data = self._get_log10_values(
        #     self._species.condensed_species, MASS_PREFIX
        # )
        # self.solution.stability.data = self._get_log10_values(
        #     self._species.condensed_species, STABILITY_PREFIX
        # )


class InitialSolutionRegressor(InitialSolution[Output]):
    """A regressor to compute the initial solution

    Importantly, the type and order of constraints must be the same in the new model as the
    previous model, but the values of the constraints can be different.

    Args:
        value: Output for constructing the regressor
        species: Species
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model. Defaults to True.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True.
            Defaults to 100.
        partial_fit: Partial fit the regressor during the model run. Defaults to True.
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor. Defaults to 500.
        **kwargs: Optional keyword arguments to pass through to the base class.

    Attributes:
        value: Output for constructing the regressor
        solution: The initial solution
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
        species: Species,
        fit: bool = True,
        fit_batch_size: int = 100,
        partial_fit: bool = True,
        partial_fit_batch_size: int = 500,
        solution_override: InitialSolutionDict | None = None,
        **kwargs,
    ):
        super().__init__(value, species=species, **kwargs)
        self.fit: bool = fit
        # Ensure consistency of arguments and correct handling of fit versus partial refit.
        self.fit_batch_size: int = fit_batch_size if self.fit else 0
        self.partial_fit: bool = partial_fit
        self.partial_fit_batch_size: int = partial_fit_batch_size
        if solution_override is None:
            self._solution_override: InitialSolutionDict | None = None
        else:
            self._solution_override = solution_override
        self._fit(self.value)

    @classmethod
    def from_pickle(cls, pickle_file: Path | str, **kwargs) -> InitialSolutionRegressor:
        """Creates a regressor from output read from a pickle file.

        Args:
            pickle_file: Pickle file of the output from a previous (or similar) model run. The
                constraints must be the same as the new model and in the same order.
            **kwargs: Arbitrary keyword arguments to pass through to the constructor

        Returns:
            An initial solution regressor
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
    def set_data(
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

        # FIXME: Regressor doesn't return a low enough log10 value
        # if abs(self.solution.data[13]) < 1e-2:
        #     logger.warning("Implementing hack")
        #     self.solution.activity.data[self.species.get_species_from_name("H2O_l")] = -24
        # if abs(self.solution.data[14]) < 1e-2:
        #     logger.warning("Implementing hack")
        #     self.solution.activity.data[self.species.get_species_from_name("C_cr")] = -24

    @override
    def process_data(
        self, constraints: SystemConstraints, *, temperature: float, pressure: float, **kwargs
    ) -> None:
        """Includes a user-specified override to the initial solution"""
        super().process_data(constraints, temperature=temperature, pressure=pressure, **kwargs)

        # FIXME:
        # if self._solution_override is not None:
        #     self._solution_override.set_data(
        #         constraints,
        #         temperature=temperature,
        #         pressure=pressure,
        #     )
        #     self.solution.merge(self._solution_override.solution)

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


class InitialSolutionLast(InitialSolutionProtocol):
    """An initial solution that uses the previous solution as the current solution guess.

    This is useful for incrementing through a grid of parameters, where the previous solution is
    probably a reasonable initial estimate for the current solution.

    Args:
        value: An initial solution until `switch_iteration` is reached. Defaults to None, meaning
            that :class:`InitialSolutionDict` is used with default arguments.
        species: Species
        switch_iteration: Iteration number to switch the initial solution to the previous solution.
            Defaults to 1.
        **kwargs: Optional keyword arguments to instantiate :class:`InitialSolutionDict`

    Attributes:
        value: An initial solution for the first solution only
        solution: The initial solution
    """

    def __init__(
        self,
        value: InitialSolutionProtocol | None = None,
        *,
        species: Species,
        switch_iteration: int = 1,
        **kwargs,
    ):
        if value is None:
            value_start: InitialSolutionProtocol = InitialSolutionDict(species=species, **kwargs)
        else:
            value_start = value
        self.value: InitialSolutionProtocol = value_start
        self._species: Species = species
        self._switch_iteration: int = switch_iteration
        self._kwargs = kwargs

    @property
    def species(self) -> Species:
        return self.value.species

    @override
    def get_log10_value(self, *args, **kwargs) -> npt.NDArray[np.float_]:
        return self.value.get_log10_value(*args, **kwargs)

    @override
    def update(self, output: Output) -> None:
        if output.size == self._switch_iteration:
            value_dict: dict[ChemicalSpecies | str, float] = output["raw_solution"][-1]
            # InitialSolutionDict takes the log10, so we must raise 10 to the values.
            value_dict = {key: 10**value for key, value in value_dict.items()}
            # Convert species from strings to objects
            for species in self.species.data:
                value_dict[species] = value_dict.pop(species.name)

            self.value = InitialSolutionDict(value_dict, species=self.species, **self._kwargs)
        else:
            self.value.update(output)


class InitialSolutionSwitchRegressor(InitialSolutionProtocol):
    """An initial solution that uses an initial solution before switching to a regressor.

    Args:
        value: An initial solution until `switch_iteration` is reached. Defaults to None.
        species: Species
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model. Defaults to True.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True.
            Defaults to 100.
        partial_fit: Partial fit the regressor during the model run. Defaults to True.
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor. Defaults to 500.
        switch_iteration: Iteration number to switch the initial solution to the regressor.
            Defaults to 50.
        **kwargs: Optional keyword arguments to instantiate :class:`InitialSolutionDict`

    Attributes:
        value: An initial solution
        solution: The initial solution
    """

    def __init__(
        self,
        value: InitialSolutionProtocol | None = None,
        *,
        species: Species,
        fit: bool = True,
        fit_batch_size: int = 100,
        partial_fit: bool = True,
        partial_fit_batch_size: int = 500,
        switch_iteration: int = 50,
        **kwargs,
    ):
        if value is None:
            value_init: InitialSolutionProtocol = InitialSolutionDict(species=species, **kwargs)
        else:
            value_init = value
        self.value: InitialSolutionProtocol = value_init
        self._species: Species = species
        self._switch_iteration: int = switch_iteration
        self._fit: bool = fit
        self._fit_batch_size: int = fit_batch_size
        self._partial_fit: bool = partial_fit
        self._partial_fit_batch_size: int = partial_fit_batch_size
        self._kwargs: dict[str, Any] = kwargs

    @property
    def species(self) -> Species:
        return self.value.species

    @override
    def get_log10_value(self, *args, **kwargs) -> npt.NDArray[np.float_]:
        return self.value.get_log10_value(*args, **kwargs)

    @override
    def update(self, output: Output) -> None:
        if output.size == self._switch_iteration:
            self.value = InitialSolutionRegressor(
                output,
                species=self._species,
                fit=self._fit,
                fit_batch_size=self._fit_batch_size,
                partial_fit=self._partial_fit,
                partial_fit_batch_size=self._partial_fit_batch_size,
                **self._kwargs,
            )
        else:
            self.value.update(output)


class InitialSolutionSwitchOnFail(InitialSolutionProtocol):
    """An initial solution that switches to a different initial solution after too many fails.

    Args:
        value: An initial solution until `switch_fails` is reached. Defaults to None.
        species: Species
        value_on_fail: An initial solution after `switch_fails` is reached. Defaults to None.
        switch_fails: Number of fails before switching the initial solution. Defaults to 1.
        **kwargs: Optional keyword arguments to instantiate :class:`InitialSolutionDict`

    Attributes:
        value: An initial solution before too many fails are reached
        value_on_fail: An initial solution after too many fails are reached
        solution: The initial solution
    """

    def __init__(
        self,
        value: InitialSolutionProtocol | None = None,
        *,
        species: Species,
        value_on_fail: InitialSolutionProtocol | None = None,
        switch_fails: int = 1,
        **kwargs,
    ):
        if value is None:
            value_init: InitialSolutionProtocol = InitialSolutionDict(species=species, **kwargs)
        else:
            value_init = value
        self.value: InitialSolutionProtocol = value_init
        if value_on_fail is None:
            value_on_fail_init: InitialSolutionProtocol = InitialSolutionDict(
                species=species, **kwargs
            )
        else:
            value_on_fail_init = value_on_fail
        self.value_on_fail: InitialSolutionProtocol = value_on_fail_init
        self._switch_fails: int = switch_fails
        self._kwargs: dict[str, Any] = kwargs

    @property
    def species(self) -> Species:
        return self.value.species

    @override
    def get_log10_value(self, *args, attempt: int, **kwargs) -> npt.NDArray[np.float_]:
        if attempt < self._switch_fails:
            logger.warning("Before switch on fail, attempt = %d", attempt)
            return self.value.get_log10_value(*args, attempt=attempt, **kwargs)
        else:
            attempt = attempt - self._switch_fails
            logger.warning("After switch on fail, attempt = %d", attempt)
            return self.value_on_fail.get_log10_value(*args, attempt=attempt, **kwargs)

    @override
    def update(self, output: Output) -> None:
        self.value.update(output)
        self.value_on_fail.update(output)
