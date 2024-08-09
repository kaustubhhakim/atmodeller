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
"""Thermodynamic data from Holland and Powell :cite:p:`HP91,HP98`"""

# Convenient to use symbols so pylint: disable=C0103

from __future__ import annotations

import importlib.resources
import logging
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from jaxtyping import ArrayLike

from atmodeller.thermodata import DATA_DIRECTORY
from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesABC,
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDataset,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if TYPE_CHECKING:
    from atmodeller.interfaces import ChemicalSpecies

logger: logging.Logger = logging.getLogger(__name__)

HOLLAND_FILENAME: str = "holland_Mindata161127.csv"
"""Filename of the thermodynamic data from :cite:t:`HP91,HP98`"""


class ThermodynamicDatasetHollandAndPowell(ThermodynamicDataset):
    """The thermodynamic dataset from :cite:t:`HP91,HP98`.

    See also the equations in :cite:t:`P78{Appendix A}`.

    Attributes:
        data: Thermodynamic data used for calculations
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298
    _STANDARD_STATE_PRESSURE: float = 1

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(HOLLAND_FILENAME)
        )
        with data as data_path:
            logger.info("Reading thermodynamic data for %s from %s", self.data_source, data_path)
            self.data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        self.data["name of phase component"] = self.data["name of phase component"].str.strip()
        self.data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        self.data.drop(columns="Abbreviation", inplace=True)
        self.data.set_index("name of phase component", inplace=True)
        self.data = self.data.loc[:, :"Vmax"]
        self.data = self.data.astype(float)

    @override
    def get_species_data(
        self,
        species: ChemicalSpecies,
        *,
        name: str | None = None,
        **kwargs,
    ) -> ThermodynamicDataForSpeciesProtocol | None:
        """Gets the thermodynamic data for a species.

        Args:
            species: A chemical species
            name: Select the record that matches this name. Defaults to None. This is used of
                preference if available, otherwise the formula name of the species is used to
                search for the thermodynamic data.
            **kwargs: Catches unused keyword arguments.

        Returns:
            Thermodynamic data for the species or None if not available
        """
        del kwargs
        search_name: str = name if name is not None else species.hill_formula

        try:
            logger.debug(
                "Searching for %s (name = %s) in %s",
                species.hill_formula,
                search_name,
                self.data_source,
            )
            phase_data: pd.Series = cast(pd.Series, self.data.loc[search_name])
            logger.debug("Thermodynamic data found = %s", phase_data)

            return self.ThermodynamicDataForSpecies(
                species, self.data_source, phase_data, self.enthalpy_reference_temperature
            )

        except KeyError:
            logger.warning("Thermodynamic data not found")
            return None

    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesABC):
        """Thermodynamic data for a species

        Args:
            species: A chemical species
            data_source: Source of the thermodynamic data
            data: Data used for thermodynamic calculations
            enthalpy_reference_temperature: Enthalpy reference temperature

        Attributes:
            species: A chemical species
            data_source: Source of the thermodynamic data
            data: Data used for thermodynamic calculations
            enthalpy_reference_temperature: Enthalpy reference temperature
            dKdP: Derivative of bulk modulus (K) with respect to pressure. Set to 4.
            dKdT_factor: Factor for computing the temperature-dependence of K. Set to -1.5e-4.
        """

        @override
        def __init__(
            self,
            species: ChemicalSpecies,
            data_source: str,
            data: pd.Series,
            enthalpy_reference_temperature: float,
        ):
            super().__init__(species, data_source, data)
            self.enthalpy_reference_temperature: float = enthalpy_reference_temperature
            self.dKdP: float = 4.0
            self.dKdT_factor: float = -1.5e-4

        @override
        def get_formation_gibbs(self, *, temperature: float, pressure: ArrayLike) -> ArrayLike:
            gibbs: ArrayLike = self._get_enthalpy(temperature) - temperature * self._get_entropy(
                temperature
            )

            if self.species.phase == "cr" or self.species.phase == "l":
                gibbs = gibbs + self._get_volume_pressure_integral(temperature, pressure)

            # logger.debug(
            #     "Species = %s, standard Gibbs energy of formation = %f",
            #     self.species.hill_formula,
            #     gibbs,
            # )

            return gibbs

        def _get_enthalpy(self, temperature: float) -> float:
            """Calculates the enthalpy at temperature.

            Args:
                temperature: Temperature in K

            Returns:
                Enthalpy in J
            """
            enthalpy0 = self.data["Hf"]  # J
            # Coefficients for calculating the heat capacity
            a = self.data["a"]  # J/K
            b = self.data["b"]  # J/K^2
            c = self.data["c"]  # J K
            d = self.data["d"]  # J K^(-1/2)

            enthalpy_integral: float = (
                enthalpy0
                + a * (temperature - self.enthalpy_reference_temperature)
                + b / 2 * (temperature**2 - self.enthalpy_reference_temperature**2)
                - c * (1 / temperature - 1 / self.enthalpy_reference_temperature)
                + 2 * d * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return enthalpy_integral

        def _get_entropy(self, temperature: float) -> float:
            r"""Calculates the entropy at temperature.

            Args:
                temperature: Temperature in K

            Returns:
                Entropy in :math:`\mathrm{J}\mathrm{K}^{-1}`
            """
            entropy0 = self.data["S"]  # J/K
            # Coefficients for calculating the heat capacity
            a = self.data["a"]  # J/K
            b = self.data["b"]  # J/K^2
            c = self.data["c"]  # J K
            d = self.data["d"]  # J K^(-1/2)

            entropy_integral: float = (
                entropy0
                + a * np.log(temperature / self.enthalpy_reference_temperature)
                + b * (temperature - self.enthalpy_reference_temperature)
                - c / 2 * (1 / temperature**2 - 1 / self.enthalpy_reference_temperature**2)
                - 2 * d * (1 / temperature**0.5 - 1 / self.enthalpy_reference_temperature**0.5)
            )
            return entropy_integral

        def _get_volume_at_temperature(self, temperature: float) -> float:
            r"""Calculates the volume at temperature.

            The exponential arises from the strict derivation, but often an expansion is performed:

            .. math::

                \exp(x) = 1+x

            as in :cite:t:`HP98`. Below, the exponential is retained, but note that the equation in
            :cite:t:`HP98{p311}` is expanded.

            Args:
                temperature: Temperature in K

            Returns:
                Volume in :math:`\mathrm{J}\mathrm{bar}^{-1}
            """
            volume0 = self.data["V"]  # J/bar
            # Thermal expansivity in 1/K
            alpha0 = self.data["a0"]

            volume: float = volume0 * np.exp(
                alpha0 * (temperature - self.enthalpy_reference_temperature)
                - 2 * 10.0 * alpha0 * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return volume

        def _get_bulk_modulus_at_temperature(self, temperature: float) -> float:
            """Calculates the bulk modulus at temperature :cite:p:`HP98{p312 in the text}`.

            Args:
                temperature: Temperature in K

            Returns:
                Bulk modulus in bar
            """
            # Bulk modulus in bar
            bulk_modulus0 = self.data["K"]
            bulk_modulus: float = bulk_modulus0 * (
                1 + self.dKdT_factor * (temperature - self.enthalpy_reference_temperature)
            )
            return bulk_modulus

        def _get_volume_pressure_integral(
            self, temperature: float, pressure: ArrayLike
        ) -> ArrayLike:
            """Computes the volume-pressure integral :cite:p:`HP98{p312}`

            Args:
                temperature: Temperature in K
                pressure: Pressure in bar

            Returns:
                The volume-pressure integral in J
            """
            volume: float = self._get_volume_at_temperature(temperature)
            bulk_modulus: float = self._get_bulk_modulus_at_temperature(temperature)
            # Uses P-1 instead of P.
            integral_vp: ArrayLike = (
                volume
                * bulk_modulus
                / (self.dKdP - 1)
                * (
                    (1 + self.dKdP * (pressure - 1.0) / bulk_modulus) ** (1.0 - 1.0 / self.dKdP)
                    - 1
                )
            )

            return integral_vp
