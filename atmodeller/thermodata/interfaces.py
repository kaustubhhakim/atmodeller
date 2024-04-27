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
"""Interfaces for thermodynamic data"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.core import ChemicalComponent


class ThermodynamicDataForSpeciesProtocol(Protocol):
    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float: ...


class ThermodynamicDataForSpeciesABC(ABC):
    """Thermodynamic data for a species to compute the Gibbs energy of formation.

    Args:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation
        *args: Arbitrary positional arguments used by child classes
        **kwargs: Arbitrary keyword arguments used by child classes

    Attributes:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation
    """

    def __init__(self, species: ChemicalComponent, data_source: str, data: Any, *args, **kwargs):
        del args
        del kwargs
        self.species: ChemicalComponent = species
        self.data_source: str = data_source
        self.data: Any = data

    @abstractmethod
    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol.

        Args:
            temperature: Temperature in kelvin
            pressure: Total pressure in bar

        Returns:
            The standard Gibbs free energy of formation in J/mol
        """


class ThermodynamicDatasetABC(ABC):
    """Thermodynamic dataset"""

    _DATA_SOURCE: str
    # JANAF standards below. May be overwritten by child classes.
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @abstractmethod
    def get_species_data(
        self, species: ChemicalComponent, **kwargs
    ) -> ThermodynamicDataForSpeciesABC | None:
        """Gets the thermodynamic data for a species or None if not available

        Args:
            species: Species
            **kwargs: Arbitrary keyword arguments

        Returns:
            Thermodynamic data for the species or None if not available
        """

    @property
    def data_source(self) -> str:
        """The source of the data."""
        return self._DATA_SOURCE

    @property
    def enthalpy_reference_temperature(self) -> float:
        """Enthalpy reference temperature in kelvin"""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def standard_state_pressure(self) -> float:
        """Standard state pressure in bar"""
        return self._STANDARD_STATE_PRESSURE
