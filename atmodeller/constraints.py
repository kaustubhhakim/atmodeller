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

from atmodeller.core import GasSpecies
from atmodeller.interfaces import (
    ChemicalSpecies,
    CondensedSpecies,
    ConstraintProtocol,
    ElementConstraintProtocol,
    ReactionNetworkConstraintProtocol,
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
U = TypeVar("U", bound=ChemicalSpecies)


class ElementMassConstraint(ElementConstraintProtocol):
    """An element mass constraint

    Args:
        element: The element whose mass to constrain
        value: The mass in kg
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
    def mass(self) -> float:
        """Value of the mass constraint in kg"""
        return self._value

    @property
    def name(self) -> str:
        """Unique name of the constraint"""
        return f"{self.element}_{self.constraint}"

    def get_value(self, *args, **kwargs) -> float:
        """Computes the value for given input arguments.

        Args:
            *args: Catches unused positional arguments
            **kwargs: Catches unused keyword arguments

        Returns:
            The evaluated value
        """
        del args
        del kwargs

        return self.mass

    def get_log10_value(self, *args, **kwargs) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            *args: Catches unused positional arguments
            **kwargs: Catches unused keyword arguments

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
        """Unique name by combing the unique species name and the constraint name"""
        return f"{self.species.name}_{self.constraint}"

    @property
    def species(self) -> U:
        """Species to constrain"""
        return self._species

    @abstractmethod
    def get_value(self, temperature: float, pressure: float) -> float:
        """Computes the value for given input arguments.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The evaluated value
        """

    @abstractmethod
    def get_log10_value(self, temperature: float, pressure: float) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The log10 evaluated value
        """


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
    def get_value(self, temperature: float, pressure: float) -> float:
        del temperature
        del pressure

        return self._value

    @override
    def get_log10_value(self, temperature: float, pressure: float) -> float:
        return np.log10(self.get_value(temperature, pressure))


class ActivityConstraint(_SpeciesConstantConstraint[CondensedSpecies]):
    """A constant activity

    Args:
        species: The species to constrain
        value: The activity. Defaults to unity for ideal behaviour.
    """

    @override
    def __init__(
        self,
        species: CondensedSpecies,
        value: float = 1,
    ):
        super().__init__(species, value, "activity")

    def activity(self, temperature: float, pressure: float) -> float:
        """Value of the activity constraint

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Activity
        """
        return self.get_value(temperature, pressure)

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Alias for the value of the activity constraint, which is anyhow related to fugacity

        This allows this constraint to satisfy
        :class:`atmodeller.interfaces.FugacityConstraintProtocol`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity (activity)
        """
        return self.activity(temperature, pressure)


class FugacityConstraint(_SpeciesConstantConstraint[GasSpecies]):
    """A constant fugacity constraint

    Args:
        species: The species to constrain
        value: The fugacity in bar
    """

    @override
    def __init__(self, species: GasSpecies, value: float):
        super().__init__(species, value, "fugacity")

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Value of the fugacity constraint in bar

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        return self.get_value(temperature, pressure)

    def pressure(self, temperature: float, pressure: float) -> float:
        """Value of the pressure constraint in bar

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Pressure in bar
        """
        fugacity: float = self.fugacity(temperature, pressure)
        fugacity /= self.species.eos.fugacity_coefficient(temperature, pressure)

        return fugacity


class BufferedFugacityConstraint(_SpeciesConstraint[GasSpecies]):
    """A buffered fugacity constraint

    Args:
        species: The species to constrain
        value: The redox buffer
    """

    @override
    def __init__(self, species: GasSpecies, value: _RedoxBuffer):
        super().__init__(species, "fugacity")
        self._value: _RedoxBuffer = value

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Value of the fugacity constraint in bar

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        return self.get_value(temperature, pressure)

    @override
    def get_value(self, temperature: float, pressure: float) -> float:
        return self._value.get_value(temperature, pressure)

    @override
    def get_log10_value(self, temperature: float, pressure: float) -> float:
        return np.log10(self.get_value(temperature, pressure))


class MassConstraint(_SpeciesConstantConstraint):
    """A constant mass constraint

    Args:
        species: The species to constrain
        value: The mass in kg
    """

    @override
    def __init__(self, species: ChemicalSpecies, value: float):
        super().__init__(species, value, "mass")

    def mass(self, *args, **kwargs) -> float:
        """Value of the mass constraint in kg"""
        return self.get_value(*args, **kwargs)


class PressureConstraint(_SpeciesConstantConstraint[GasSpecies]):
    """A constant pressure constraint

    Args:
        species: The species to constrain
        value: The pressure in bar
    """

    @override
    def __init__(self, species: GasSpecies, value: float):
        super().__init__(species, value, "pressure")

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Value of the fugacity constraint in bar

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """
        fugacity: float = self.pressure(temperature, pressure)
        fugacity *= self.species.eos.fugacity_coefficient(temperature, pressure)

        return fugacity

    def pressure(self, *args, **kwargs) -> float:
        """Value of the pressure constraint in bar

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Pressure in bar
        """
        del args
        del kwargs

        return self._value

    @override
    def get_value(self, temperature: float, pressure: float) -> float:
        return self.fugacity(temperature, pressure)


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

    @override
    def get_value(self, *args, **kwargs) -> float:
        del args
        del kwargs

        return self._value

    @override
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
    """

    # UserList itself is not a generic class, so this is for typing:
    data: list[ConstraintProtocol]
    """List of constraints"""

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
            raise ValueError("More than one total pressure constraint prescribed")

        return total_pressure

    @property
    def reaction_network_constraints(self) -> list[ReactionNetworkConstraintProtocol]:
        """Constraints related to the reaction network"""
        return list(filter_by_type(self, ReactionNetworkConstraintProtocol).values())

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
