"""Initial condition

See the LICENSE file for licensing information.
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from atmodeller.interfaces import GetValueABC

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class InitialConditionABC(GetValueABC):
    """Initial condition base class"""

    def __post_init__(self):
        logger.info("Creating %s", self.__class__.__name__)

    def get_value(self, species: str, **kwargs) -> float:
        ...

    def update(self, *args, **kwargs) -> None:
        ...


@dataclass
class InitialConditionRegressor(InitialConditionABC):
    """Applies a regressor to compute the initial condition"""

    pickle_file: Path | str
    data: dict[str, Pipeline] = field(init=False, default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        with open(self.pickle_file, "rb") as handle:
            output_data: dict[str, pd.DataFrame] = pickle.load(handle)

        solution: pd.DataFrame = output_data["solution"]
        constraints: pd.DataFrame = output_data["constraints"]

        logger.info("Reading data from %s", self.pickle_file)
        solution_species: list[str] = list(solution.columns)
        logger.info("Found species = %s", solution_species)

        for species in solution_species:
            reg: Pipeline = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
            pressure: pd.Series = solution[species]
            reg.fit(np.log10(constraints.values), np.log10(pressure))
            self.data[species] = reg

        logger.info(self.data)

    def get_value(self, species: str, constraints_evaluate_log10: dict[str, float]) -> float:
        """Computes the value.

        Args:
            species: Species (chemical formula)
            constraints_evaluate_log10: Log10 of the constraints evaluated at current conditions

        Returns:
            An initial guess for the species
        """
        predict_in: np.ndarray = np.array(list(constraints_evaluate_log10.values())).reshape(1, -1)
        reg: Pipeline = self.data[species]
        # Since predict_in is always a np.ndarray I think the return type is also always a
        # np.ndarray. But in general it could be a tuple.
        prediction: np.ndarray | tuple = reg.predict(predict_in)
        prediction_array: np.ndarray = np.array(prediction)
        assert prediction_array.size == 1
        value = 10 ** float(prediction_array)

        logger.debug("%s: species = %s, value = %s", self.__class__.__name__, species, value)

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

    def get_value(self, species: str, *args, **kwargs) -> float:
        del args
        del kwargs

        logger.debug("%s: species = %s, value = %s", self.__class__.__name__, species, self.value)

        return self.value

    def update(self, *args, **kwargs) -> None:
        """Update does nothing for a constant value"""
        del args
        del kwargs
