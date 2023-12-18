"""Initial condition

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from atmodeller.interfaces import GetValueABC
from atmodeller.output import Output

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class InitialConditionABC(GetValueABC):
    """Initial condition base class

    The return type also permits np.ndarray whereas the base class only permits float.
    """

    def __post_init__(self):
        logger.info("Creating %s", self.__class__.__name__)

    def get_value(self, *args, **kwargs) -> np.ndarray | float:
        """Computes the value for given input arguments.

        See base class.
        """
        ...

    def get_log10_value(
        self, evaluated_log10_constraints: np.ndarray, *args, **kwargs
    ) -> np.ndarray | float:
        """Computes the log10 value for given input arguments.

        Args:
            evaluated_log10_constraints: An array of the log10 constraints evaluated at current
                conditions
            *args: Arbitrary positional arguments
            **kwargs: Arbitrary keyword arguments

        Returns
            The initial condition
        """
        return super().get_log10_value(evaluated_log10_constraints, *args, **kwargs)

    def update(self, output: Output, *args, **kwargs) -> InitialConditionABC:
        """Updates the current initial condition (i.e. self) or returns a new initial condition.

        Args;
            output: output
            *args: Arbitrary positional arguments
            **kwargs: Arbitrary keyword arguments

        Returns:
            An InitialConditionABC (can be self)
        """
        ...


@dataclass
class InitialConditionRegressor(InitialConditionABC):
    """A regressor to compute the initial condition.

    Args:
        pickle_file: Pickle file of the output from a previous (or similar) model run. Importantly,
            the reaction network must be the same (same number of species in the same order) and
            the constraints must be the same (also in the same order).
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model. Defaults to True.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit = True.
            Defaults to 50.
        partial_fit: Partial fit the regressor during the model run. Defaults to True.
        partial_fit_batch_size: Number of solutions to calculate before partial refit of the
            regressor. Defaults to 50.

    Attributes:
        pickle_file: Pickle file
        fit: Fit the regressor during the model run. This will replace the original regressor by a
            regressor trained only on the data from the current model.
        fit_batch_size: Number of solutions to calculate before fitting model data if fit = True.
        partial_fit: Partial fit the regressor during the model run.
        partial_fit_batch_size: Number of solutions before partial refit of the regressor
        constraint_names: Names of the constraints (and their order) in the output
        species_names: Names of the species (and their order) in the output
    """

    pickle_file: Path | str
    _: KW_ONLY
    fit: bool = True
    fit_batch_size: int = 50
    partial_fit: bool = True
    partial_fit_batch_size: int = 50
    constraint_names: list[str] = field(init=False)
    species_names: list[str] = field(init=False)
    _reg: MultiOutputRegressor = field(init=False)
    _solution_scalar: StandardScaler = field(init=False)
    _constraints_scalar: StandardScaler = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        # Ensure consistency of arguments and correct handling of fit versus partial refit.
        if not self.fit:
            self.fit_batch_size = 0
        with open(self.pickle_file, "rb") as handle:
            output_data: dict[str, pd.DataFrame] = pickle.load(handle)

        logger.info("%a: Reading data from %s", self.__class__.__name__, self.pickle_file)
        self._fit(output_data)

    def _fit(
        self,
        output_data: dict[str, pd.DataFrame],
        start_index: int | None = None,
        end_index: int | None = None,
    ) -> None:
        """Fits the regressor

        Args:
            output_data: Output data
            start_index: Start index for fit. Defaults to None, meaning use all available data.
            end_index: End index for fit. Defaults to None, meaning use all available data.
        """
        logger.info("%s: Fit (%s, %s)", self.__class__.__name__, start_index, end_index)

        constraints: pd.DataFrame = output_data["constraints"]
        if start_index is not None and end_index is not None:
            constraints = constraints.iloc[start_index:end_index]
        constraints_log10_values: np.ndarray = np.log10(constraints.values)
        self._constraints_scalar = StandardScaler().fit(constraints_log10_values)
        constraints_scaled: np.ndarray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )

        solution: pd.DataFrame = output_data["solution"]
        if start_index is not None and end_index is not None:
            solution = solution.iloc[start_index:end_index]
        solution_log10_values: np.ndarray = np.log10(solution.values)
        self._solution_scalar = StandardScaler().fit(solution_log10_values)
        solution_scaled: np.ndarray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

        self.constraint_names = list(constraints.columns)
        logger.info("%s: Found constraints = %s", self.__class__.__name__, self.constraint_names)
        self.species_names = list(solution.columns)
        logger.info("%s: Found species = %s", self.__class__.__name__, self.species_names)

        base_regressor: SGDRegressor = SGDRegressor()
        multi_output_regressor: MultiOutputRegressor = MultiOutputRegressor(base_regressor)
        multi_output_regressor.partial_fit(constraints_scaled, solution_scaled)

        self._reg = multi_output_regressor

    def _partial_fit(
        self,
        output_data: dict[str, pd.DataFrame],
        start_index: int,
        end_index: int,
    ) -> None:
        """Partial fits the regressor

        Args:
            output_data: Output data
            start_index: Start index for partial fit
            end_index: End index for partial fit
        """
        logger.info("%s: Partial fit (%d, %d)", self.__class__.__name__, start_index, end_index)

        constraints: pd.DataFrame = output_data["constraints"].iloc[start_index:end_index]
        logger.debug(constraints)
        constraints_log10_values: np.ndarray = np.log10(constraints.values)
        constraints_scaled: np.ndarray | spmatrix = self._constraints_scalar.transform(
            constraints_log10_values
        )

        solution: pd.DataFrame = output_data["solution"].iloc[start_index:end_index]
        logger.debug(solution)
        solution_log10_values: np.ndarray = np.log10(solution.values)
        solution_scaled: np.ndarray | spmatrix = self._solution_scalar.transform(
            solution_log10_values
        )

        self._reg.partial_fit(constraints_scaled, solution_scaled)

    def get_value(self, evaluated_log10_constraints: np.ndarray) -> np.ndarray:
        """Computes the value.

        Args:
            evaluated_log10_constraints: Log10 of the constraints evaluated at current conditions

        Returns:
            A guess for the initial solution
        """
        constraints: np.ndarray = evaluated_log10_constraints.reshape(1, -1)
        constraints_scaled: np.ndarray | spmatrix = self._constraints_scalar.transform(constraints)
        solution_scaled: np.ndarray | spmatrix = self._reg.predict(constraints_scaled)
        solution_original: np.ndarray | spmatrix = self._solution_scalar.inverse_transform(
            solution_scaled
        )

        assert isinstance(solution_original, np.ndarray)
        value: np.ndarray = 10 ** solution_original.flatten()
        logger.debug("%s: value = %s", self.__class__.__name__, value)

        return value

    def action_fit(self, output: Output) -> tuple[int, int] | None:
        """Is a fit required.

        Args:
            output: Output

        Returns:
            A tuple: (start_index, end_index) or None if nothing to do
        """
        trigger_fit: bool = self.fit_batch_size == output.size
        if self.fit and trigger_fit:
            return (0, output.size)

    def action_partial_fit(self, output: Output) -> tuple[int, int] | None:
        """Is a partial fit required.

        Args:
            output: Output

        Returns:
            A tuple: (start_index, end_index) or None if nothing to do
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

    def update(self, output: Output) -> InitialConditionABC:
        """See base class."""
        action_fit: tuple[int, int] | None = self.action_fit(output)
        action_partial_fit: tuple[int, int] | None = self.action_partial_fit(output)

        if action_fit is not None:
            output_data: dict[str, pd.DataFrame] = output.as_dict()
            self._fit(output_data, start_index=action_fit[0], end_index=action_fit[1])

        if action_partial_fit is not None:
            output_data: dict[str, pd.DataFrame] = output.as_dict()
            self._partial_fit(
                output_data, start_index=action_partial_fit[0], end_index=action_partial_fit[1]
            )

        return self


@dataclass
class InitialConditionConstant(InitialConditionABC):
    """A constant value for the initial condition

    This only needs to return a reasonable initial guess for the pressure of gas species. The
    activity of condensed phases is included by InteriorAtmosphereSystem and therefore does not
    need to be considered here.

    The default value, which is applied to each gas species individually, is chosen to be within
    an order of magnitude of the solar system rocky planets, i.e. 10 bar.

    Args:
        value: A constant pressure for the initial condition in bar. Defaults to 10.
    """

    value: float = 10

    def get_value(self, *args, **kwargs) -> float:
        del args
        del kwargs
        logger.debug("%s: value = %s", self.__class__.__name__, self.value)

        return self.value


# TODO: New class to switch from constant to regressor

# def updates does nothing for a constant value (see base class)

# if output.size == 10:
#     logger.warning("here")
#     file_prefix: Path | str = Path("test_restart")
#     output.to_pickle(file_prefix)
#     test = InitialConditionRegressor(file_prefix.with_suffix(".pkl"), fit=False)
#     return test
# else:
#     return self
