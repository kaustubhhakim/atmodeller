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
"""Constraints for the interior-atmosphere system"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from collections import UserList
from typing import Generic, TypeVar

import numpy as np

from atmodeller.core import GasSpecies, _ChemicalSpecies, _CondensedSpecies
from atmodeller.interfaces import (
    ActivityConstraintProtocol,
    ConstraintProtocol,
    ElementConstraintProtocol,
    FugacityConstraintProtocol,
    SpeciesConstraintProtocol,
    TotalPressureConstraintProtocol,
)
from atmodeller.thermodata.redox_buffers import _RedoxBuffer
from atmodeller.utilities import filter_by_type

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ConstraintProtocol)
U = TypeVar("U", bound=_ChemicalSpecies)


class ElementMassConstraint(ElementConstraintProtocol):
    """An element mass constraint

    Args:
        element: The element whose mass to constrain
        value: The value of the mass
    """

    def __init__(self, element: str, value: float):
        self._element: str = element
        self._value: float = value
        self._constraint: str = "mass"

    @property
    def constraint(self) -> str:
        """Name of the constraint"""
        return self._constraint

    @property
    def element(self) -> str:
        """Element whose mass to constrain"""
        return self._element

    @property
    def name(self) -> str:
        """Unique name of the constraint"""
        return f"{self.element}_{self.constraint}"

    def get_value(self, *args, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The evaluated value
        """
        del args
        del kwargs

        return self._value

    def get_log10_value(self, *args, **kwargs) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The log10 evaluated value
        """
        return np.log10(self.get_value(*args, **kwargs))


class _SpeciesConstraint(ABC, Generic[U]):
    """A species constraint

    Args:
        species: The species to constrain
        constraint: The type of constraint, which should be activity, fugacity, pressure or mass,
            depending on the phase of the species.
    """

    def __init__(self, species: U, constraint: str):
        self._species: U = species
        self._constraint: str = constraint

    @property
    def constraint(self) -> str:
        return self._constraint

    @property
    def name(self) -> str:
        """Name of the constraint"""
        return f"{self.species.name}_{self.constraint}"

    @property
    def species(self) -> U:
        """Species to constrain"""
        return self._species

    @abstractmethod
    def get_value(self, *args, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            The evaluated value
        """

    def get_log10_value(self, *args, **kwargs) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            *args: Positional arguments only
            **kwargs: Keyword arguments only

        Returns:
            The log10 evaluated value
        """
        return np.log10(self.get_value(*args, **kwargs))


class _SpeciesConstantConstraint(_SpeciesConstraint[U]):
    """A species constraint of a constant value

    Args:
        species: The species to constrain
        constraint: The type of constraint, which should be activity, fugacity, pressure or mass,
            depending on the phase of the species.
        value: The constant value
    """

    @override
    def __init__(self, species: U, value: float, constraint: str):
        super().__init__(species, constraint)
        self._value: float = value

    @override
    def get_value(self, *args, **kwargs) -> float:
        del args
        del kwargs

        return self._value


class ActivityConstraint(_SpeciesConstantConstraint[_CondensedSpecies]):
    """A constant activity

    Args:
        species: The species to constrain
        value: The activity. Defaults to unity for ideal behaviour.
    """

    @override
    def __init__(self, species: _CondensedSpecies, value: float = 1, constraint: str = "activity"):
        super().__init__(species, value, constraint)

    def activity(self, *args, **kwargs) -> float:
        return self.get_value(*args, **kwargs)

    # TODO: For the purposes of calculation can I call the activity the fugacity?


class FugacityConstraint(_SpeciesConstantConstraint[GasSpecies]):
    """A constant fugacity constraint

    Args:
        species: The species to constrain
        value: The fugacity in bar
    """

    @override
    def __init__(self, species: GasSpecies, value: float, constraint: str = "fugacity"):
        super().__init__(species, value, constraint)

    def fugacity(self, *args, **kwargs) -> float:
        return self.get_value(*args, **kwargs)

    def pressure(self, *args, **kwargs) -> float:
        fugacity: float = self.fugacity(*args, **kwargs)
        fugacity /= self.species.eos.fugacity_coefficient(*args, **kwargs)

        return fugacity


class BufferedFugacityConstraint(_SpeciesConstraint[GasSpecies]):
    """A buffered fugacity constraint

    Args:
        species: The species to constrain
        value: The redox buffer
    """

    @override
    def __init__(self, species: GasSpecies, value: _RedoxBuffer, constraint: str = "fugacity"):
        super().__init__(species, constraint)
        self._value: _RedoxBuffer = value

    @override
    def get_value(self, *args, **kwargs) -> float:
        return self._value.get_value(*args, **kwargs)

    def fugacity(self, *args, **kwargs) -> float:
        return self.get_value(*args, **kwargs)


class MassConstraint(_SpeciesConstantConstraint):
    """A constant mass constraint

    Args:
        species: The species to constrain
        value: The mass in kg
    """

    @override
    def __init__(self, species: _ChemicalSpecies, value: float, constraint: str = "mass"):
        super().__init__(species, value, constraint)

    def mass(self, *args, **kwargs) -> float:
        return self.get_value(*args, **kwargs)


class PressureConstraint(_SpeciesConstantConstraint[GasSpecies]):
    """A constant pressure constraint

    Args:
        species: The species to constrain
        value: The pressure in bar
    """

    @override
    def __init__(self, species: GasSpecies, value: float, constraint: str = "pressure"):
        super().__init__(species, value, constraint)

    def fugacity(self, *args, **kwargs) -> float:
        fugacity: float = self.pressure(*args, **kwargs)
        fugacity *= self.species.eos.fugacity_coefficient(*args, **kwargs)

        return fugacity

    def pressure(self, *args, **kwargs) -> float:
        del args
        del kwargs

        return self._value

    @override
    def get_value(self, *args, **kwargs) -> float:
        return self.fugacity(*args, **kwargs)


class TotalPressureConstraint(ConstraintProtocol):
    """A total pressure constraint

    Args:
        value: The total pressure in bar
    """

    @override
    def __init__(self, value: float):
        self._value: float = value
        self._constraint: str = "total_pressure"
        self._name: str = "total_pressure"

    @property
    def constraint(self) -> str:
        return self._constraint

    @property
    def name(self) -> str:
        return self._name

    def get_value(self, *args, **kwargs) -> float:
        del args
        del kwargs

        return self._value

    def get_log10_value(self, *args, **kwargs) -> float:
        return np.log10(self.get_value(*args, **kwargs))

    def total_pressure(self, *args, **kwargs) -> float:
        return self.get_value(*args, **kwargs)


class SystemConstraints(UserList):
    """A collection of constraints

    It provides methods to filter constraints based on their types, such as activity, fugacity,
    mass, pressure, and reaction network constraints.

    Args:
        initlist: Initial list of constraints. Defaults to None.

    Attributes:
        data: A list of constraints
    """

    def __init__(self, initlist=None):
        self.data: list[ConstraintProtocol]
        super().__init__(initlist)

    @property
    def activity_constraints(self) -> list[ActivityConstraintProtocol]:
        """Constraints related to activity"""
        return list(filter_by_type(self, ActivityConstraintProtocol).values())

    @property
    def fugacity_constraints(self) -> list[FugacityConstraintProtocol]:
        """Constraints related to fugacity"""
        out = list(filter_by_type(self, FugacityConstraintProtocol).values())

        return out

    # TODO: Currently only for element mass constraints, but should be generalised to include
    # species mass constraints
    @property
    def mass_constraints(self) -> list[ElementMassConstraint]:
        """Constraints related to mass conservation"""
        return list(filter_by_type(self, ElementMassConstraint).values())

    @property
    def total_pressure_constraint(self) -> list[TotalPressureConstraintProtocol]:
        """Total pressure constraint"""
        total_pressure: list[TotalPressureConstraintProtocol] = list(
            filter_by_type(self, TotalPressureConstraintProtocol).values()
        )
        if len(total_pressure) > 1:
            msg: str = "You can only specify a maximum of one total pressure constraint"
            logger.error(msg)
            raise ValueError(msg)

        return total_pressure

    @property
    def reaction_network_constraints(self) -> list[SpeciesConstraintProtocol]:
        """Constraints related to the reaction network"""
        filtered_entries: list[SpeciesConstraintProtocol] = []
        filtered_entries.extend(self.fugacity_constraints)
        filtered_entries.extend(self.activity_constraints)

        return filtered_entries

    @property
    def number_reaction_network_constraints(self) -> int:
        """Number of constraints related to the reaction network"""
        return len(self.reaction_network_constraints)

    def evaluate(self, temperature: float, pressure: float) -> dict[str, float]:
        """Evaluates all constraints.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            A dictionary of the evaluated constraints (same order as the constraints list)
        """
        evaluated_constraints: dict[str, float] = {}
        for constraint in self.data:
            evaluated_constraints[constraint.name] = constraint.get_value(
                temperature=temperature,
                pressure=pressure,
            )

        return evaluated_constraints

    def evaluate_log10(self, temperature: float, pressure: float) -> dict[str, float]:
        """Evaluates all constraints and returns the log10 values.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            A dictionary of the log10 evaluated constraints (same order as the constraints list)
        """
        return {
            key: np.log10(value) for key, value in self.evaluate(temperature, pressure).items()
        }
