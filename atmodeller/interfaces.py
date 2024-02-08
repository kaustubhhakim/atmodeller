"""Interfaces

Copyright 2024 Dan J. Bower

This file is part of Atmodeller.

Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU 
General Public License as published by the Free Software Foundation, either version 3 of the 
License, or (at your option) any later version.

Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License along with Atmodeller. If not, 
see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from atmodeller.core import ChemicalComponent

logger: logging.Logger = logging.getLogger(__name__)


class GetValueABC(ABC):
    """An object with a get_value method."""

    @abstractmethod
    def get_value(self, *args, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            An evaluation based on the provided arguments
        """
        ...

    def get_log10_value(self, *args, **kwargs) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            *args: Positional arguments only
            **kwargs: Keyword arguments only

        Returns:
            An evaluation of the log10 value based on the provided arguments
        """
        return np.log10(self.get_value(*args, **kwargs))


@dataclass(kw_only=True, frozen=True)
class ConstraintABC(GetValueABC):
    """A constraint to apply to an interior-atmosphere system.

    Args:
        name: The name of the constraint, which should be one of: 'activity', 'fugacity',
            'pressure', or 'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.

    Attributes:
        name: The name of the constraint
        species: The species to constrain
    """

    name: str
    species: str

    @property
    def full_name(self) -> str:
        """Combines the species name and constraint name to give a unique descriptive name."""
        if self.species:
            full_name: str = f"{self.species}_"
        else:
            full_name = ""
        full_name += self.name

        return full_name


@dataclass(frozen=True)
class ThermodynamicDataForSpeciesProtocol(Protocol):
    """Protocol for a class with a method that returns the Gibbs energy of formation for a species.

    Args:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation

    Attributes:
        species: Species
        data_source: Source of the thermodynamic data
        data: Data used to compute the Gibbs energy of formation
    """

    species: ChemicalComponent
    data_source: str
    data: Any

    def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol.

        Args:
            temperature: Temperature in kelvin
            pressure: Total pressure in bar

        Returns:
            The standard Gibbs free energy of formation in J/mol
        """
        ...


class ThermodynamicDatasetABC(ABC):
    """Thermodynamic dataset base class"""

    _DATA_SOURCE: str
    # JANAF standards below. May be overwritten by child classes.
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @abstractmethod
    def get_data(self, species: ChemicalComponent) -> ThermodynamicDataForSpeciesProtocol | None:
        """Gets the thermodynamic data for a species, otherwise None if not available

        Args:
            species: Species

        Returns:
            Thermodynamic data for the species, otherwise None is not available
        """
        ...

    @property
    def DATA_SOURCE(self) -> str:
        """Identifies the source of the data."""
        return self._DATA_SOURCE

    @property
    def ENTHALPY_REFERENCE_TEMPERATURE(self) -> float:
        """Enthalpy reference temperature in kelvin"""
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def STANDARD_STATE_PRESSURE(self) -> float:
        """Standard state pressure in bar"""
        return self._STANDARD_STATE_PRESSURE
