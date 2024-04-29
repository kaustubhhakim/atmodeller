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
"""Combines thermodynamic datasets"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from atmodeller.thermodata.holland import ThermodynamicDatasetHollandAndPowell
from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDataset,
)
from atmodeller.thermodata.janaf import ThermodynamicDatasetJANAF

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if TYPE_CHECKING:
    from atmodeller.core import ChemicalSpecies


logger: logging.Logger = logging.getLogger(__name__)


class ThermodynamicDatasetCombined(ThermodynamicDataset):
    """Combines thermodynamic data from multiple datasets.

    Args:
        datasets: A list of thermodynamic data to use. Defaults to Holland and Powell and JANAF.
    """

    _DATA_SOURCE: str = "Combined"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15
    """Enthalpy reference temperature may be different for all combined datasets. Defauls to JANAF.
    """
    _STANDARD_STATE_PRESSURE: float = 1
    """Standard state pressure may be different for all combined datasets. Defaults to JANAF"""

    def __init__(
        self,
        datasets: list[ThermodynamicDataset] | None = None,
    ):
        if datasets is None:
            self.datasets: list[ThermodynamicDataset] = []
            self.add_dataset(ThermodynamicDatasetHollandAndPowell())
            self.add_dataset(ThermodynamicDatasetJANAF())
        else:
            self.datasets = datasets

    def add_dataset(self, dataset: ThermodynamicDataset) -> None:
        """Adds a thermodynamic dataset

        Args:
            dataset: A thermodynamic dataset
        """
        if len(self.datasets) >= 1:
            logger.warning("Combining different thermodynamic data may result in inconsistencies")
        logger.info("Adding thermodynamic data: %s", dataset.data_source)
        self.datasets.append(dataset)

    @override
    def get_species_data(
        self, species: ChemicalSpecies, **kwargs
    ) -> ThermodynamicDataForSpeciesProtocol | None:
        for dataset in self.datasets:
            if dataset is not None:
                return dataset.get_species_data(species, **kwargs)

        raise KeyError(f"Thermodynamic data for {species.formula} is not available in any dataset")
