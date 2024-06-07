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
from collections.abc import Mapping
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
from atmodeller.core import Species
from atmodeller.interfaces import TypeChemicalSpecies_co
from atmodeller.output import Output

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T")

MIN_LOG10: float = -12
"""Minimum log10 of the initial solution

Motivated by typical values of oxygen fugacity at the iron-wustite buffer
"""
MAX_LOG10: float = 5
"""Maximum log10 of the initial solution"""


class InitialSolution(ABC, Generic[T]):
    """Initial solution

    Args:
        value: An object or value used to compute the initial solution
        species: Species
        min_log10: Minimum log10 value. Defaults to :data:`MIN_LOG10`.
        max_log10: Maximum log10 value. Defaults to :data:`MAX_LOG10`.

    Attributes:
        value: An object or value used to compute the initial solution
    """

    def __init__(
        self,
        value: T,
        *,
        species: Species,
        min_log10: float = MIN_LOG10,
        max_log10: float = MAX_LOG10,
    ):
        logger.info("Creating %s", self.__class__.__name__)
        self.value: T = value
        self._species: Species = species
        self._min_log10: float = min_log10
        self._max_log10: float = max_log10

    @property
    def species(self) -> Species:
        """Species"""
        return self._species

    @property
    def min_log10(self) -> float:
        """Minimum log10 value"""
        return self._min_log10

    @property
    def max_log10(self) -> float:
        """Maximum log10 value"""
        return self._max_log10

    @abstractmethod
    def get_value(
        self, constraints: SystemConstraints, temperature: float, pressure: float
    ) -> npt.NDArray[np.float_]:
        """Computes the raw value of the initial solution for the reaction network

        This does not deal with the degree of condensation, which is instead exclusively treated
        by :meth:`~get_log10_value`.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The raw initial solution excluding the degree of condensation (if relevant)
        """

    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        degree_of_condensation_number: int,
        number_of_condensed_species: int,
        perturb: bool = False,
        perturb_log10: float = 2,
    ) -> npt.NDArray[np.float_]:
        """Computes the log10 value of the initial solution with additional processing.

        Args:
            constraints: Constraints
            temperature: Temperature in K
            pressure: Pressure in bar
            degree_of_condensation_number: Number of elements to solve for the degree of
                condensation
            number_of_condensed_species: Number of condensed species to solve for the condensate
                stability factors (lambda factors)
            perturb: Randomly perturb the log10 value by `perturb_log10`. Defaults to False.
            perturb_log10: Maximum absolute log10 value to perturb the initial solution. Defaults
                to 2.

        Returns
            The log10 initial solution adhering to bounds and the constraints
        """
        value: npt.NDArray[np.float_] = self.get_value(constraints, temperature, pressure)
        log10_value: npt.NDArray[np.float_] = np.log10(value)

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

        # Apply constraints from the reaction network (activities and fugacities)
        for constraint in constraints.reaction_network_constraints:
            index: int = self.species.species_index(constraint.species)
            logger.debug("Setting %s %d", constraint.species, index)
            log10_value[index] = constraint.get_log10_value(
                temperature=temperature, pressure=pressure
            )
        logger.debug("Conform initial solution to constraints = %s", log10_value)

        # When condensates and mass constraints are present, we assume an initial degree of
        # condensation of 0.5 for each element. Recall that the (log10) solution quantity is:
        # beta = log10(mu) = log10(d/(1-d)), where beta is the log10 mass of the condensed element,
        # and d is the degree of condensation. Hence d = 0.5 gives mu = 1 gives beta = 0
        log_degree_of_condensation: npt.NDArray = np.zeros(degree_of_condensation_number)
        log10_value = np.append(log10_value, log_degree_of_condensation)

        # Small lambda factors assume the condensates are stable, which is probably a reasonable
        # assumption given that the user has chosen to include them in the species list.
        log_lambda: npt.NDArray = -12 * np.ones(number_of_condensed_species)
        log10_value = np.append(log10_value, log_lambda)

        return log10_value

    def update(self, output: Output) -> None:
        """Updates the initial solution.

        This base class does nothing.

        Args;
            output: output
        """
        del output


class InitialSolutionConstant(InitialSolution[npt.NDArray[np.float_]]):
    """A constant value for the initial solution

    The default value, which is applied to each species individually, is chosen to be within an
    order of magnitude of the solar system rocky planets, i.e. 10 bar.

    Args:
        value: A constant pressure for the initial condition in bar. Defaults to 10.
        species: Species
        min_log10: Minimum log10 value. Defaults to :data:`MIN_LOG10`.
        max_log10: Maximum log10 value. Defaults to :data:`MAX_LOG10`.

    Attributes:
        value: A constant pressure for the initial condition in bar
    """

    @override
    def __init__(
        self,
        value: float = 10,
        *,
        species: Species,
        min_log10: float = MIN_LOG10,
        max_log10: float = MAX_LOG10,
    ):
        value_array: npt.NDArray = value * np.ones(species.number_species())
        super().__init__(value_array, species=species, min_log10=min_log10, max_log10=max_log10)
        logger.debug("initial_solution = %s", self.asdict())

    def asdict(self) -> dict[str, float]:
        """Dictionary of the initial solution"""
        return dict(zip(self.species.names, self.value))

    @override
    def get_value(self, *args, **kwargs) -> npt.NDArray[np.float_]:
        del args
        del kwargs
        logger.debug("%s: value = %s", self.__class__.__name__, self.value)

        # TODO: Treat activities and fugacities differently in terms of numerical value?
        return self.value


class InitialSolutionDict(InitialSolution[npt.NDArray[np.float_]]):
    """A dictionary of species and their values for the initial solution

    Args:
        value: A dictionary of species and values
        species: Species
        min_log10: Minimum log10 value. Defaults to :data:`MIN_LOG10`.
        max_log10: Maximum log10 value. Defaults to :data:`MAX_LOG10`.
        fill_value: Initial value for species that are not specified in `value`. Defaults to 1.

    Attributes:
        value: A dictionary of species and values for all the gas species
    """

    @override
    def __init__(
        self,
        value: Mapping[TypeChemicalSpecies_co, float],
        *,
        species: Species,
        min_log10: float = MIN_LOG10,
        max_log10: float = MAX_LOG10,
        fill_value: float = 1,
    ):
        species_dict: dict[TypeChemicalSpecies_co, float] = {
            unique_species: fill_value for unique_species in species
        }
        species_dict |= value
        species_ic: npt.NDArray[np.float_] = np.array(list(species_dict.values()))
        super().__init__(species_ic, species=species, min_log10=min_log10, max_log10=max_log10)
        logger.debug("initial_solution = %s", self.asdict())

    def asdict(self) -> dict[str, float]:
        """Dictionary of the initial solution"""
        return dict(zip(self.species.names, self.value))

    @override
    def get_value(self, *args, **kwargs) -> npt.NDArray:
        del args
        del kwargs
        logger.debug("%s: value = %s", self.__class__.__name__, self.value)

        return self.value


class InitialSolutionRegressor(InitialSolution[Output]):
    """A regressor to compute the initial solution

    Args:
        value: Output for constructing the regressor
        species: Species
        min_log10: Minimum log10 value. Defaults to :data:`MIN_LOG10`.
        max_log10: Maximum log10 value. Defaults to :data:`MAX_LOG10`.
        species_fill: Dictionary of missing species and their initial values. Defaults to None.
        fill_value: Initial value for species that are not specified in `species_fill`. Defaults to
            1.
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model. Defaults to True.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit is True.
            Defaults to 100.
        partial_fit: Partial fit the regressor during the model run. Defaults to True.
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor. Defaults to 500.

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
        species: Species,
        min_log10: float = MIN_LOG10,
        max_log10: float = MAX_LOG10,
        species_fill: dict[TypeChemicalSpecies_co, float] | None = None,
        fill_value: float = 1,
        fit: bool = True,
        fit_batch_size: int = 100,
        partial_fit: bool = True,
        partial_fit_batch_size: int = 500,
    ):
        self.fit: bool = fit
        # Ensure consistency of arguments and correct handling of fit versus partial refit.
        self.fit_batch_size: int = fit_batch_size if self.fit else 0
        self.partial_fit: bool = partial_fit
        self.partial_fit_batch_size: int = partial_fit_batch_size

        self._conform_solution(
            value, species=species, species_fill=species_fill, fill_value=fill_value
        )
        super().__init__(value, species=species, min_log10=min_log10, max_log10=max_log10)
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
        output: Output = Output.read_pickle(pickle_file)

        return cls(output, **kwargs)

    def _conform_solution(
        self,
        output: Output,
        *,
        species: Species,
        species_fill: dict[TypeChemicalSpecies_co, float] | None,
        fill_value: float,
    ) -> None:
        """Conforms the solution in output to the species and their fill values

        Args:
            output: Output
            species: Species
            species_fill: Dictionary of missing species and their initial values. Defaults to None.
            fill_value: Initial value for species that are not specified in `species_fill`.
        """
        solution: pd.DataFrame = output.to_dataframes()["solution"].copy()
        logger.debug("solution = %s", solution)

        species_fill_: dict[TypeChemicalSpecies_co, float] = (
            species_fill if species_fill is not None else {}
        )
        initial_solution_dict: InitialSolutionDict = InitialSolutionDict(
            species_fill_,
            species=species,
            fill_value=fill_value,
        )
        fill_df: pd.DataFrame = pd.DataFrame(initial_solution_dict.asdict(), index=[0])
        fill_df = fill_df.loc[fill_df.index.repeat(len(solution))].reset_index(drop=True)
        logger.debug("fill_df = %s", fill_df)

        # Preference the values in the solution and fill missing species
        conformed_solution: pd.DataFrame = solution.combine_first(fill_df)[species.names]
        logger.debug("conformed_solution = %s", conformed_solution)
        output["solution"] = conformed_solution.to_dict(orient="records")

    def _get_log10_values(
        self,
        output: Output,
        name: str,
        start_index: int | None,
        end_index: int | None,
    ) -> npt.NDArray[np.float_]:
        """Gets log10 values of either the constraints or the solution from `output`

        Args:
            output: Output
            name: solution or constraints
            start_index: Start index for fit. Defaults to None, meaning use all available data.
            end_index: End index for fit. Defaults to None, meaning use all available data.

        Returns:
            Log10 values of either the solution or constraints depending on `name`
        """
        output_dataframes: dict[str, pd.DataFrame] = output.to_dataframes()
        data: pd.DataFrame = output_dataframes[name]
        if start_index is not None and end_index is not None:
            data = data.iloc[start_index:end_index]
        data_log10_values: npt.NDArray = np.log10(data.values)

        return data_log10_values

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

        constraints_log10_values: npt.NDArray = self._get_log10_values(
            output, "constraints", start_index, end_index
        )
        self._constraints_scalar = StandardScaler().fit(constraints_log10_values)
        constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )

        solution_log10_values: npt.NDArray = self._get_log10_values(
            output, "solution", start_index, end_index
        )
        self._solution_scalar = StandardScaler().fit(solution_log10_values)
        solution_scaled: npt.NDArray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

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

        constraints_log10_values: npt.NDArray = self._get_log10_values(
            output, "constraints", start_index, end_index
        )
        constraints_scaled: npt.NDArray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )
        solution_log10_values: npt.NDArray = self._get_log10_values(
            output, "solution", start_index, end_index
        )
        solution_scaled: npt.NDArray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

        self._reg.partial_fit(constraints_scaled, solution_scaled)

    @override
    def get_value(
        self, constraints: SystemConstraints, temperature: float, pressure: float
    ) -> npt.NDArray:
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

        value: npt.NDArray = 10 ** solution_original.flatten()
        logger.debug("%s: value = %s", self.__class__.__name__, value)

        return value

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


class InitialSolutionSwitchRegressor(InitialSolution[InitialSolution]):
    """An initial solution that uses an initial solution before switching to a regressor.

    Args:
        value: An initial solution
        species: Species
        min_log10: Minimum log10 value. Defaults to :data:`MIN_LOG10`.
        max_log10: Maximum log10 value. Defaults to :data:`MAX_LOG10`.
        fit_batch_size: Number of simulations to generate before fitting the regressor. Defaults
            to 100.
        **kwargs: Keyword arguments that are specific to :class:`InitialSolutionRegressor`

    Attributes:
        value: An initial solution
        fit_batch_size: Number of simulations to generate before fitting the regressor
    """

    @override
    def __init__(
        self,
        value: InitialSolution,
        *,
        species: Species,
        min_log10: float = MIN_LOG10,
        max_log10: float = MAX_LOG10,
        fit_batch_size: int = 100,
        **kwargs,
    ):
        super().__init__(value, species=species, min_log10=min_log10, max_log10=max_log10)
        self._fit_batch_size: int = fit_batch_size
        # Store to instantiate regressor once the switch occurs.
        self._kwargs: dict[str, Any] = kwargs

    @override
    def get_value(self, *args, **kwargs) -> npt.NDArray:
        return self.value.get_value(*args, **kwargs)

    @override
    def update(self, output: Output, *args, **kwargs) -> None:
        if output.size == self._fit_batch_size:
            # The fit keyword argument of InitialSolutionRegressor is effectively ignored because
            # the fit is done once when InitialSolutionRegressor is instantiated and action_fit
            # cannot be triggered regardless of the value of fit.
            self.value = InitialSolutionRegressor(
                output,
                species=self.species,
                min_log10=self.min_log10,
                max_log10=self.max_log10,
                **self._kwargs,
            )
        else:
            self.value.update(output, *args, **kwargs)
