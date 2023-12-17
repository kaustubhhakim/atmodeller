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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from atmodeller.interfaces import GetValueABC

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

    def update(self, *args, **kwargs) -> None:
        """Updates the initial condition"""
        ...


@dataclass
class InitialConditionRegressor(InitialConditionABC):
    """Applies a regressor to compute the initial condition"""

    pickle_file: Path | str
    species_names: list[str] = field(init=False)
    _reg: MultiOutputRegressor = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        with open(self.pickle_file, "rb") as handle:
            output_data: dict[str, pd.DataFrame] = pickle.load(handle)

        solution: pd.DataFrame = output_data["solution"]
        constraints: pd.DataFrame = output_data["constraints"]

        logger.info("Reading data from %s", self.pickle_file)
        self.species_names = list(solution.columns)
        logger.info("Found species = %s", self.species_names)

        base_regressor = SGDRegressor(max_iter=1000, tol=1e-3)
        multi_output_regressor = MultiOutputRegressor(
            make_pipeline(StandardScaler(), base_regressor)
        )
        multi_output_regressor.fit(np.log10(constraints.values), np.log10(solution.values))
        self._reg = multi_output_regressor

    def get_value(self, constraints_evaluate_log10: dict[str, float]) -> np.ndarray:
        """Computes the value.

        Args:
            constraints_evaluate_log10: Log10 of the constraints evaluated at current conditions

        Returns:
            A guess for the initial solution
        """
        predict_in: np.ndarray = np.array(list(constraints_evaluate_log10.values())).reshape(1, -1)
        prediction: np.ndarray = cast(np.ndarray, self._reg.predict(predict_in)).flatten()
        value = 10**prediction

        logger.debug("%s: value = %s", self.__class__.__name__, value)

        return value


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

    def update(self, *args, **kwargs) -> None:
        """Update does nothing for a constant value"""
        del args
        del kwargs
