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
"""Core"""

from __future__ import annotations

import importlib.resources
import logging
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from atmodeller import DATA_DIRECTORY
from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesABC,
    ThermodynamicDatasetABC,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.core import ChemicalComponent, CondensedSpecies
    from atmodeller.interior_atmosphere import InteriorAtmosphereSystem, Planet


class ThermodynamicDatasetHollandAndPowell(ThermodynamicDatasetABC):
    """Thermodynamic dataset from :cite:t:`HP91,HP98`.

    The book 'Equilibrium thermodynamics in petrology: an introduction' by R. Powell also has
    a useful appendix A with equations.
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    def __init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath("Mindata161127.csv")
        )
        with data as data_path:
            self.data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        self.data["name of phase component"] = self.data["name of phase component"].str.strip()
        self.data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        self.data.drop(columns="Abbreviation", inplace=True)
        self.data.set_index("name of phase component", inplace=True)
        self.data = self.data.loc[:, :"Vmax"]
        self.data = self.data.astype(float)

    @override
    def get_species_data(
        self, species: ChemicalComponent, name: str | None = None, **kwargs
    ) -> ThermodynamicDataForSpeciesABC | None:
        """See base class."""
        del kwargs

        try:
            phase_data: pd.Series | None = self.data.loc[name]
            logger.debug(
                "Thermodynamic data for %s (%s) found in %s",
                species.formula,
                name,
                self.data_source,
            )

            return self.ThermodynamicDataForSpecies(
                species, self.data_source, phase_data, self.enthalpy_reference_temperature
            )

        except KeyError:
            phase_data = None
            logger.warning(
                "Thermodynamic data for %s (%s) not found in %s",
                species.formula,
                name,
                self.data_source,
            )

            return None

    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesABC):
        """Thermodynamic data for a species

        Args:
            species: Species
            data_source: Source of the thermodynamic data
            data: Data used to compute the Gibbs energy of formation
            enthalpy_reference_temperature: Enthalpy reference temperature

        Attributes:
            species: Species
            data_source: Source of the thermodynamic data
            data: Data used to compute the Gibbs energy of formation
            enthalpy_reference_temperature: Enthalpy reference temperature
            dkdp: Derivative of bulk modulus (K) with respect to pressure. Set to 4.
            dkdt_factor: Factor for computing the temperature-dependence of K. Set to 1.5e-4.
        """

        @override
        def __init__(
            self,
            species: ChemicalComponent,
            data_source: str,
            data: pd.Series,
            enthalpy_reference_temperature: float,
        ):
            super().__init__(species, data_source, data)
            self.enthalpy_reference_temperature: float = enthalpy_reference_temperature
            self.dkdp: float = 4.0
            self.dkdt_factor: float = -1.5e-4

        @override
        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            """See base class"""
            gibbs: float = self._get_enthalpy(temperature) - temperature * self._get_entropy(
                temperature
            )

            if isinstance(self.species, CondensedSpecies):
                gibbs += self._get_volume_pressure_integral(temperature, pressure)

            logger.debug(
                "Species = %s, standard Gibbs energy of formation = %f",
                self.species.formula,
                gibbs,
            )

            return gibbs

        def _get_enthalpy(self, temperature: float) -> float:
            """Calculates the enthalpy at temperature.

            Args:
                temperature: Temperature in kelvin

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
            """Calculates the entropy at temperature.

            Args:
                temperature: Temperature in kelvin

            Returns:
                Entropy in J/K
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
            """Calculates the volume at temperature.

            The exponential arises from the strict derivation, but often an expansion is performed
            where exp(x) = 1+x as in Holland and Powell (1998). Below the exp term is retained, but
            the equation in Holland and Powell (1998) p311 is expanded.

            Args:
                temperature: Temperature in kelvin

            Returns:
                Volume in J/bar
            """
            volume0 = self.data["V"]  # J/bar
            alpha0 = self.data["a0"]  # K^(-1), thermal expansivity

            volume: float = volume0 * np.exp(
                alpha0 * (temperature - self.enthalpy_reference_temperature)
                - 2 * 10.0 * alpha0 * (temperature**0.5 - self.enthalpy_reference_temperature**0.5)
            )
            return volume

        def _get_bulk_modulus_at_temperature(self, temperature: float) -> float:
            """Calculates the bulk modulus at temperature.

            Holland and Powell (1998), p312 in the text

            Args:
                temperature: Temperature in kelvin

            Returns:
                Bulk modulus in bar
            """
            bulk_modulus0 = self.data["K"]  # Bulk modulus in bar
            bulk_modulus: float = bulk_modulus0 * (
                1 + self.dkdt_factor * (temperature - self.enthalpy_reference_temperature)
            )
            return bulk_modulus

        def _get_volume_pressure_integral(self, temperature: float, pressure: float) -> float:
            """Computes the volume-pressure integral.

            Holland and Powell (1998), p312.

            Args:
                temperature: Temperature in kelvin
                pressure: Pressure in bar

            Returns:
                The volume-pressure integral
            """
            volume: float = self._get_volume_at_temperature(temperature)
            bulk_modulus: float = self._get_bulk_modulus_at_temperature(temperature)
            integral_vp: float = (
                volume
                * bulk_modulus
                / (self.dkdp - 1)
                * (
                    (1 + self.dkdp * (pressure - 1.0) / bulk_modulus) ** (1.0 - 1.0 / self.dkdp)
                    - 1
                )
            )  # J, use P-1.0 instead of P.
            return integral_vp
