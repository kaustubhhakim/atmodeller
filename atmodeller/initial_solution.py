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
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import spmatrix
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from atmodeller import INITIAL_SOLUTION_MAX_LOG10, INITIAL_SOLUTION_MIN_LOG10
from atmodeller.constraints import SystemConstraints
from atmodeller.core import Species
from atmodeller.output import Output

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class InitialSolution(ABC):
    """Initial solution

    Note:
        Activity constraints are imposed directly by a private method
        :meth:`InitialSolution._conform_to_constraints` since their exact solution is known a
        priori.

    Args:
        value: Some value (object) used to compute the initial solution
        species: Species in the interior-atmosphere system
        min_log10: Minimuim log10 value of the initial solution. Defaults to
            ``INITIAL_SOLUTION_MIN_LOG10``
        max_log10: Maximum log10 value of the initial solution. Defaults to
            ``INITIAL_SOLUTION_MAX_LOG10``
        **kwargs: Catches unused keyword arguments from a child constructors

    Attributes:
        value: Some value (object) used to compute the initial solution
        species: Species in the interior-atmosphere system
        min_log10: Minimum log10 value of the initial solution
        max_log10: Maximum log10 value of the initial solution
        **kwargs: Catches unused keyword arguments from a child constructors
    """

    def __init__(
        self,
        value: Any,
        *,
        species: Species,
        min_log10: float = INITIAL_SOLUTION_MIN_LOG10,
        max_log10: float = INITIAL_SOLUTION_MAX_LOG10,
        **kwargs,
    ):
        del kwargs
        logger.info("Creating %s", self.__class__.__name__)
        self.value: Any = value
        self.species: Species = species
        self.min_log10: float = min_log10
        self.max_log10: float = max_log10

    @abstractmethod
    def get_value(
        self, constraints: SystemConstraints, temperature: float, pressure: float
    ) -> np.ndarray:
        """Computes the raw value of the initial solution.

        Args:
            constraints: Constraints on the interior-atmosphere system
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The initial solution
        """

    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        perturb: bool = False,
        perturb_log10: float = 2,
    ) -> np.ndarray:
        """Computes the log10 value of the initial solution with additional processing.

        Args:
            constraints: Constraints on the interior-atmosphere system
            temperature: Temperature in K
            pressure: Pressure in bar
            perturb: Randomaly perturb the log10 value by `perturb_log10`. Defaults to False
            perturb_log10: Maximum absolute log10 value to perturb the initial solution. Defaults
                to 2.

        Returns
            The log10 initial solution adhering to bounds and the system constraints
        """
        value: np.ndarray = self.get_value(constraints, temperature, pressure)
        log10_value: np.ndarray = np.log10(value)

        if perturb:
            logger.info(
                "Randomly perturbing the initial solution by a maximum of %f log10 units",
                perturb_log10,
            )
            log10_value += perturb_log10 * (2 * np.random.rand(log10_value.size) - 1)

        if np.any((log10_value < self.min_log10) | (log10_value > self.max_log10)):
            logger.warning("Initial solution has values outside the min and max thresholds")
            logger.warning(
                "Clipping the initial solution between %f and %f", self.min_log10, self.max_log10
            )
            log10_value = np.clip(log10_value, self.min_log10, self.max_log10)

        self._conform_to_constraints(log10_value, constraints, temperature, pressure)

        return log10_value

    def update(self, output: Output) -> None:
        """Updates the initial solution.

        The base class does nothing.

        Args;
            output: output
        """
        del output

    def _conform_to_constraints(
        self,
        initial_solution: np.ndarray,
        constraints: SystemConstraints,
        temperature: float,
        pressure: float,
    ) -> None:
        """Conforms the initial solution to activity, pressure and fugacity constraints.

        For simplicity, impose both pressure and fugacity constraints as pressure constraints.

        Args:
            initial_solution: Initial solution to conform to the constraints (in-place).
            constraints: Constraints on the interior-atmosphere system
            temperature: Temperature in K
            pressure: Pressure in bar
        """
        for constraint in constraints.reaction_network_constraints:
            index: int = self.species.indices[constraint.species]
            logger.debug("Setting %s %d", constraint.species, index)
            initial_solution[index] = constraint.get_log10_value(
                temperature=temperature, pressure=pressure
            )

        logger.debug("Conform initial solution to constraints = %s", initial_solution)


class InitialSolutionConstant(InitialSolution):
    """A constant value for the initial solution

    The default value, which is applied to each gas species individually, is chosen to be within
    an order of magnitude of the solar system rocky planets, i.e. 10 bar.

    Args:
        value: A constant pressure for the initial condition in bar. Defaults to 10.
        **kwargs: Keyword arguments to pass through to base class

    Attributes:
        value: A constant pressure for the initial condition in bar. Defaults to 10.
    """

    def __init__(self, value: float = 10, **kwargs):
        super().__init__(value, **kwargs)

    @override
    def get_value(self, *args, **kwargs) -> np.ndarray:
        """See base class."""
        del args
        del kwargs
        logger.debug("%s: value = %s", self.__class__.__name__, self.value)

        return self.value * np.ones(self.species.number)


class InitialSolutionDict(InitialSolution):
    """A dictionary of species and their values for the initial solution

    Args:
        value: A dictionary of species and values for one or several gas species
        fill_value: Initial value for species that are not specified in `value`. Defaults to 1 bar.
        **kwargs: Keyword arguments to pass through to base class

    Attributes:
        value: A dictionary of species and values for all the gas species
    """

    def __init__(self, value: dict[str, float | int], fill_value: float = 1, **kwargs):
        super().__init__(value, **kwargs)
        self._fill_missing_species(fill_value)
        self._value: np.ndarray = np.array(list(self.value.values()))

    def _fill_missing_species(self, fill_value: float) -> None:
        """Fills missing species values.

        Args:
            fill_value: Fill value to use for missing species
        """
        for species in self.species.formulas:
            if species not in self.value:
                self.value[species] = fill_value

        # Must maintain order with species.
        self.value = {key: self.value[key] for key in self.species.formulas}

    @override
    def get_value(self, *args, **kwargs) -> np.ndarray:
        """See base class."""
        del args
        del kwargs
        logger.debug("%s: value = %s", self.__class__.__name__, self._value)

        return self._value


class InitialSolutionRegressor(InitialSolution):
    """A regressor to compute the initial solution

    Args:
        value: Output for constructing the regressor
        species_fill: Dictionary of missing species and their initial values. Defaults to None.
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model. Defaults to True.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True.
            Defaults to 100.
        partial_fit: Partial fit the regressor during the model run. Defaults to True.
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor. Defaults to 500.
        **kwargs: Keyword arguments to pass through to base class

    Attributes:
        value: Output for constructing the regressor
        species_fill: Dictionary of missing species and their initial values
        fit: Fit the regressor during the model run
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True
        partial_fit: Partial fit the regressor during the model run
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor
    """

    def __init__(
        self,
        value: Output,
        *,
        species_fill: dict[str, float] | None = None,
        fit: bool = True,
        fit_batch_size: int = 100,
        partial_fit: bool = True,
        partial_fit_batch_size: int = 500,
        **kwargs,
    ):
        super().__init__(value, **kwargs)
        self.species_fill: dict[str, float] = species_fill if species_fill is not None else {}
        self.fit: bool = fit
        # Ensure consistency of arguments and correct handling of fit versus partial refit.
        self.fit_batch_size: int = fit_batch_size if self.fit else 0
        self.partial_fit: bool = partial_fit
        self.partial_fit_batch_size: int = partial_fit_batch_size
        self._reg: MultiOutputRegressor  # For typing
        self._solution_scalar: StandardScaler  # For typing
        self._constraints_scalar: StandardScaler  # For typing
        self._conform_output_to_species()
        self._fit(self.value)

    @classmethod
    def from_pickle(cls, pickle_file: Path | str, **kwargs) -> InitialSolutionRegressor:
        """Creates a regressor from output read from a pickle file.

        Args:
            pickle_file: Pickle file of the output from a previous (or similar) model run.
                Importantly, the reaction network must be the same (same number of species in the
                same order) and the constraints must be the same (also in the same order).
            **kwargs: Arbitrary keyword arguments to pass through to constructor

        Returns:
            A regressor
        """
        output: Output = Output.read_pickle(pickle_file)

        return cls(output, **kwargs)

    def _conform_output_to_species(self) -> None:
        """Conforms the output (initial regressor data) to the species for the new model.

        Ensures that the column order is correct when initialising the regressor for the new model.
        Fills in initial values for missing species with user-prescribed data and excludes species
        that are not in the new model.
        """
        output_dataframes: dict[str, pd.DataFrame] = self.value.to_dataframes()
        solution_df: pd.DataFrame = output_dataframes["solution"]
        conformed_solution_df: pd.DataFrame = solution_df.copy()

        if self.species is None:
            species_formulas: list[str] = self.value.species
        else:
            species_formulas = self.species.formulas

        if self.species_fill:
            fill_df: pd.DataFrame = pd.DataFrame(
                {col: [self.species_fill[col]] * len(solution_df) for col in self.species_fill}
            )
            logger.debug("fill_df = %s", fill_df)

            # Add columns that don't exist and replace those that do.
            for column in fill_df.columns:
                conformed_solution_df[column] = fill_df[column]

            # Reorder the columns based on the order of the species for the new model
            conformed_solution_df = conformed_solution_df[species_formulas]
            logger.debug("conformed_solution_df = %s", conformed_solution_df)

            assert species_formulas == list(conformed_solution_df.columns)

        # Set the conformed solution to the output so it is used to construct the initial regressor
        self.value["solution"] = conformed_solution_df.to_dict(orient="records")

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
        output_dataframes: dict[str, pd.DataFrame] = output.to_dataframes()

        constraints: pd.DataFrame = output_dataframes["constraints"]
        if start_index is not None and end_index is not None:
            constraints = constraints.iloc[start_index:end_index]
        constraints_log10_values: np.ndarray = np.log10(constraints.values)
        self._constraints_scalar = StandardScaler().fit(constraints_log10_values)
        constraints_scaled: np.ndarray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )

        solution: pd.DataFrame = output_dataframes["solution"]
        logger.debug("solution = %s", solution)

        if start_index is not None and end_index is not None:
            solution = solution.iloc[start_index:end_index]
        solution_log10_values: np.ndarray = np.log10(solution.values)
        self._solution_scalar = StandardScaler().fit(solution_log10_values)
        solution_scaled: np.ndarray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

        constraint_names = list(constraints.columns)
        logger.info("%s: Found constraints = %s", self.__class__.__name__, constraint_names)
        species_names = list(solution.columns)
        logger.info("%s: Found species = %s", self.__class__.__name__, species_names)

        base_regressor: SGDRegressor = SGDRegressor()
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
        output_dataframes: dict[str, pd.DataFrame] = output.to_dataframes()

        constraints: pd.DataFrame = output_dataframes["constraints"].iloc[start_index:end_index]
        logger.debug(constraints)
        constraints_log10_values: np.ndarray = np.log10(constraints.values)
        constraints_scaled: np.ndarray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )

        solution: pd.DataFrame = output_dataframes["solution"].iloc[start_index:end_index]
        logger.debug(solution)
        solution_log10_values: np.ndarray = np.log10(solution.values)
        solution_scaled: np.ndarray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

        self._reg.partial_fit(constraints_scaled, solution_scaled)

    @override
    def get_value(
        self, constraints: SystemConstraints, temperature: float, pressure: float
    ) -> np.ndarray:
        """See base class."""
        evaluated_constraints_log10: dict[str, float] = constraints.evaluate_log10(
            temperature=temperature, pressure=pressure
        )
        values_constraints_log10: np.ndarray = np.array(list(evaluated_constraints_log10.values()))
        values_constraints_log10 = values_constraints_log10.reshape(1, -1)
        constraints_scaled: np.ndarray | spmatrix = self._constraints_scalar.transform(
            values_constraints_log10
        )
        solution_scaled: np.ndarray | spmatrix = self._reg.predict(constraints_scaled)
        solution_original: np.ndarray | spmatrix = self._solution_scalar.inverse_transform(
            solution_scaled
        )

        assert isinstance(solution_original, np.ndarray)
        value: np.ndarray = 10 ** solution_original.flatten()
        logger.debug("%s: value = %s", self.__class__.__name__, value)

        return value

    def action_fit(self, output: Output) -> tuple[int, int] | None:
        """Determines if a fit of the regressor is necessary.

        Args:
            output: Output

        Returns:
            A tuple: (start_index, end_index) or None
        """
        trigger_fit: bool = self.fit_batch_size == output.size

        if self.fit and trigger_fit:
            return (0, output.size)

    def action_partial_fit(self, output: Output) -> tuple[int, int] | None:
        """Determines if a partial refit of the regressor is necessary.

        Args:
            output: Output

        Returns:
            A tuple: (start_index, end_index) or None
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
        """See base class."""
        action_fit: tuple[int, int] | None = self.action_fit(output)
        action_partial_fit: tuple[int, int] | None = self.action_partial_fit(output)

        if action_fit is not None:
            self._fit(output, start_index=action_fit[0], end_index=action_fit[1])

        if action_partial_fit is not None:
            self._partial_fit(
                output, start_index=action_partial_fit[0], end_index=action_partial_fit[1]
            )


class InitialSolutionSwitchRegressor(InitialSolution):
    """An initial solution that uses constant initial value(s) before switching to a regressor.

    Args:
        value: Initial constant regressor
        fit_batch_size: Number of simulations to generate before fitting the regressor. Defaults
            to 100.
        **kwargs: Arbitrary keyword arguments to pass to :class:`InitialSolutionRegressor`

    Attributes:
        value: Initial constant regressor
        fit_batch_size: Number of simulations to generate before fitting the regressor.
    """

    def __init__(self, value: InitialSolution, fit_batch_size: int = 100, **kwargs):
        self.value: InitialSolution  # For typing
        super().__init__(value, **kwargs)
        self.fit_batch_size: int = fit_batch_size
        # Store to instantiate regressor once the switch occurs.
        self._kwargs: dict[str, Any] = kwargs

    @override
    def get_value(self, *args, **kwargs) -> ndarray:
        """See base class."""
        return self.value.get_value(*args, **kwargs)

    @override
    def update(self, output: Output, *args, **kwargs) -> None:
        """See base class."""
        if output.size == self.fit_batch_size:
            # The fit keyword argument of InitialSolutionRegressor is effectively ignored
            # because the fit is done once when InitialSolutionRegressor is instantiated and
            # action_fit cannot be triggered regardless of the value of fit.
            self.value = InitialSolutionRegressor(output, **self._kwargs)
        else:
            self.value.update(output, *args, **kwargs)
