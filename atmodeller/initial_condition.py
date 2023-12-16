"""Initial condition

See the LICENSE file for licensing information.
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

logger: logging.Logger = logging.getLogger(__name__)


class InitialCondition(Protocol):
    def __call__(self, *args, **kwargs) -> float | np.ndarray:
        ...


@dataclass
class InitialConditionRegressor:
    """Applies a regressor to compute the initial condition"""

    pickle_file: Path | str
    data: dict[str, Pipeline] = field(init=False, default_factory=dict)

    def __post_init__(self):
        logger.info("Creating an initial condition")
        with open(self.pickle_file, "rb") as handle:
            data: dict[str, pd.DataFrame] = pickle.load(handle)

        solution: pd.DataFrame = data["solution"]
        constraints: pd.DataFrame = data["constraints"]

        logger.info("Reading data from %s", self.pickle_file)
        solution_species: list[str] = list(solution.columns)
        logger.info("Found species = %s", solution_species)

        for species in solution_species:
            reg: Pipeline = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
            pressure: pd.Series = solution[species]
            reg.fit(np.log10(constraints.values), np.log10(pressure.values))
            self.data[species] = reg

        logger.info(self.data)

    def __call__(
        self, chemical_formula: str, constraints_evaluate_log10: dict[str, float]
    ) -> float | np.ndarray:
        predict_in: np.ndarray = np.array(list(constraints_evaluate_log10.values())).reshape(1, -1)
        reg: Pipeline = self.data[chemical_formula]
        value: np.ndarray = reg.predict(predict_in)
        logger.debug("Value = %s, %s", value, type(value))

        return value


class InitialConditionArray:
    """An initial condition defined by an array or dictionary"""


class InitialConditionConstant:
    """A constant initial condition"""

    def __call__(self, *args, **kwargs) -> float | np.ndarray:
        del args
        del kwargs
        value: float = 1
        return value
