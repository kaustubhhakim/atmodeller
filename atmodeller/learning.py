"""Machine learning

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger: logging.Logger = logging.getLogger(__name__)
import numpy as np


@dataclass
class Learning:
    """Machine learning

    Args:
        constraints: Constraints dataframe
        solution: Solution dataframe
    """

    constraints: pd.DataFrame
    solution: pd.DataFrame

    def __post_init__(self):
        # We actually solve for log10
        self.constraints = np.log10(self.constraints)
        self.solution = np.log10(self.solution)

    def get_regressor(self, species: str):
        """Gets the regressor.

        Args:
            species: Species to regress

        Returns:
            regression pipeline
        """
        # Always scale the input. The most convenient way is to use a pipeline.

        # FIXME: improve training? some predictions are negative!

        reg = make_pipeline(StandardScaler(), SGDRegressor(max_iter=1000, tol=1e-3))
        outcome: pd.Series = self.solution[species]
        # print(self.constraints.values)
        # print(outcome.values)
        reg.fit(self.constraints.values, outcome.values)

        # Below is for a quick sanity check to make sure the predictions and training compare
        # reasonably. For chemical network, probably don't need a super-accurate initial guess
        # anyway
        logger.warning("predictions = %s'", reg.predict(self.constraints.values))
        logger.warning("original = %s", outcome)

        return reg

    # Now let's train on a bit more data incrementally
    # predictors_extra, outcomes_extra = generate_synthetic_data_extra()
