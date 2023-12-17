"""Initial condition

See the LICENSE file for licensing information.
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from atmodeller.interfaces import GetValueABC
from atmodeller.output import Output

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class InitialConditionABC(GetValueABC):
    """Initial condition base class

    The return types accommodate np.ndarray whereas the base class only accommodates float.
    """

    def __post_init__(self):
        logger.info("Creating %s", self.__class__.__name__)

    def get_value(self, *args, **kwargs) -> np.ndarray | float:
        """Computes the value for given input arguments.

        See base class.
        """
        ...

    def get_log10_value(self, *args, **kwargs) -> np.ndarray | float:
        """Computes the log10 value for given input arguments.

        See base class.
        """
        return super().get_log10_value(*args, **kwargs)

    def update(self, output: Output, *args, **kwargs) -> None:
        """Updates the initial condition"""


@dataclass
class InitialConditionRegressor(InitialConditionABC):
    """Applies a regressor to compute the initial condition"""

    pickle_file: Path | str
    partial_fit_batch_size: int = 10
    species_names: list[str] = field(init=False)
    _reg: MultiOutputRegressor = field(init=False)
    _solution_scalar: StandardScaler = field(init=False)
    _constraints_scalar: StandardScaler = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        with open(self.pickle_file, "rb") as handle:
            output_data: dict[str, pd.DataFrame] = pickle.load(handle)

        logger.info("Reading data from %s", self.pickle_file)

        constraints: pd.DataFrame = output_data["constraints"]
        constraints_log10_values: np.ndarray = np.log10(constraints.values)
        self._constraints_scalar = StandardScaler().fit(constraints_log10_values)
        constraints_scaled = self._constraints_scalar.transform(constraints_log10_values)

        solution: pd.DataFrame = output_data["solution"]
        solution_log10_values: np.ndarray = np.log10(solution.values)
        self._solution_scalar = StandardScaler().fit(solution_log10_values)
        solution_scaled = self._solution_scalar.transform(solution_log10_values)

        self.species_names = list(solution.columns)
        logger.info("Found species = %s", self.species_names)

        base_regressor = SGDRegressor(max_iter=1000, tol=1e-3)

        multi_output_regressor = MultiOutputRegressor(base_regressor)

        multi_output_regressor.partial_fit(constraints_scaled, solution_scaled)

        self._reg = multi_output_regressor

    def get_value(self, constraints_evaluate_log10: dict[str, float]) -> np.ndarray:
        """Computes the value.

        Args:
            constraints_evaluate_log10: Log10 of the constraints evaluated at current conditions

        Returns:
            A guess for the initial solution
        """
        predict_in: np.ndarray = np.array(list(constraints_evaluate_log10.values())).reshape(1, -1)
        predict_in_scaled: np.ndarray = cast(
            np.ndarray, self._constraints_scalar.transform(predict_in)
        )
        solution_scaled: np.ndarray = cast(np.ndarray, self._reg.predict(predict_in_scaled))
        solution_original: np.ndarray = cast(
            np.ndarray, self._solution_scalar.inverse_transform(solution_scaled)
        )

        value = 10 ** solution_original.flatten()

        logger.warning("%s: value = %s", self.__class__.__name__, value)

        return value

    def update(self, output: Output) -> None:
        if not (output.size % self.partial_fit_batch_size):
            count: int = output.size // self.partial_fit_batch_size - 1
            logger.warning("%s: partial refit (%d)", self.__class__.__name__, count)
            start_index: int = count * self.partial_fit_batch_size
            end_index: int = start_index + self.partial_fit_batch_size
            logger.warning("start_index = %d, end_index = %d", start_index, end_index)
            constraints: pd.DataFrame = output.constraints.iloc[start_index:end_index]
            logger.warning(constraints)
            constraints_scaled = self._constraints_scalar.transform(np.log10(constraints.values))
            solution: pd.DataFrame = output.solution.iloc[start_index:end_index]
            solution_scaled = self._solution_scalar.transform(np.log10(solution.values))
            logger.warning("Refitting initial condition regressor (%d)", count)
            self._reg.fit(constraints_scaled, solution_scaled)


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

    # def updates does nothing for a constant value (see base class)
