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
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass, field

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class Constraint(ABC):
    """A constraint to apply to an interior-atmosphere system.

    Args:
        name: The name of the constraint, which should be one of: 'activity', 'fugacity',
            'pressure', or 'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.
    """

    name: str = ""
    """Name of the constraint"""
    species: str = ""
    """Species to constrain"""

    @property
    def full_name(self) -> str:
        """Combines the species name and constraint name to give a unique descriptive name."""
        if self.species:
            full_name: str = f"{self.species}_"
        else:
            full_name = ""
        full_name += self.name

        return full_name

    @abstractmethod
    def get_value(self, *args, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            An evaluation based on the provided arguments
        """

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
class ConstantConstraint(Constraint):
    """A constraint of a constant value

    Args:
        name: The name of the constraint, which should be one of: 'activity', 'fugacity',
            'pressure', or 'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.
        value: The constant value, which is usually in kg for masses and bar for pressures or
            fugacities.
    """

    value: float
    """Constant value of the constraint"""

    def get_value(self, **kwargs) -> float:
        """Returns the constant value. See base class."""
        del kwargs
        return self.value


@dataclass(kw_only=True, frozen=True)
class ActivityConstraint(ConstantConstraint):
    """A constant activity

    Args:
        species: The species to constrain
        value: The constant value. Defaults to unity for ideal behaviour.
    """

    name: str = field(init=False, default="activity")
    value: float = 1.0


@dataclass(kw_only=True, frozen=True)
class FugacityConstraint(ConstantConstraint):
    """A constant fugacity constraint. See base class."""

    name: str = field(init=False, default="fugacity")


@dataclass(kw_only=True, frozen=True)
class PressureConstraint(ConstantConstraint):
    """A constant pressure constraint. See base class."""

    name: str = field(init=False, default="pressure")


@dataclass(kw_only=True, frozen=True)
class TotalPressureConstraint(ConstantConstraint):
    """Total pressure constraint. See base class.

    'species' is not required so is set to an empty string
    """

    species: str = field(init=False, default="")
    name: str = field(init=False, default="total_pressure")


@dataclass(kw_only=True, frozen=True)
class MassConstraint(ConstantConstraint):
    """A constant mass constraint. See base class."""

    name: str = field(init=False, default="mass")


class SystemConstraints(UserList):
    """A collection of constraints for an interior-atmosphere system.

    A collection of constraints that can be applied to an interior-atmosphere system. It provides
    methods to filter constraints based on their types, such as activity, fugacity, mass
    conservation, pressure, and reaction network constraints.

    Args:
        initlist: Initial list of constraints. Defaults to None.

    Attributes:
        data: A list of constraints for the interior-atmosphere system
        names: A list of unique names of the constraints
    """

    def __init__(self, initlist=None):
        self.data: list[Constraint]
        super().__init__(initlist)

    @property
    def full_names(self) -> list[str]:
        return [constraint.full_name for constraint in self.data]

    @property
    def activity_constraints(self) -> list[Constraint]:
        """Constraints related to activity"""
        return self._filter_by_name("activity")

    @property
    def fugacity_constraints(self) -> list[Constraint]:
        """Constraints related to fugacity"""
        return self._filter_by_name("fugacity")

    @property
    def mass_constraints(self) -> list[Constraint]:
        """Constraints related to mass conservation"""
        return self._filter_by_name("mass")

    @property
    def pressure_constraints(self) -> list[Constraint]:
        """Constraints related to pressure"""
        return self._filter_by_name("pressure")

    @property
    def total_pressure_constraint(self) -> list[Constraint]:
        """Total pressure constraint"""
        total_pressure: list[Constraint] = self._filter_by_name("total_pressure")
        if len(total_pressure) > 1:
            msg: str = "You can only specify zero or one total pressure constraints"
            logger.error(msg)
            raise ValueError(msg)

        return total_pressure

    @property
    def reaction_network_constraints(self) -> list[Constraint]:
        """Constraints related to the reaction network"""
        filtered_entries: list[Constraint] = self.fugacity_constraints
        filtered_entries.extend(self.pressure_constraints)
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
            evaluated_constraints[constraint.full_name] = constraint.get_value(
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

    def _filter_by_name(self, name: str) -> list[Constraint]:
        """Filters the constraints by a given name.

        Args:
            name: The filter string (e.g., activity, fugacity, pressure, mass)

        Returns:
            A list of filtered constraints
        """
        filtered: list = []
        for entry in self.data:
            if entry.name == name:
                filtered.append(entry)

        return filtered
