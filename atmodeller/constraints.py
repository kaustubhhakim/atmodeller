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
from molmass import Formula

from atmodeller import AVOGADRO
from atmodeller.core import GasSpecies, Species
from atmodeller.interfaces import (
    ActivityConstraintProtocol,
    ChemicalSpecies,
    CondensedSpecies,
    ConstraintProtocol,
    GasConstraintProtocol,
    MassConstraintProtocol,
    ReactionNetworkConstraintProtocol,
)
from atmodeller.thermodata.interfaces import RedoxBufferProtocol
from atmodeller.utilities import UnitConversion, filter_by_type, get_number_density

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

T = TypeVar("T", bound=ConstraintProtocol)
U = TypeVar("U", bound=ChemicalSpecies)
V = TypeVar("V")


class ElementMassConstraint(MassConstraintProtocol):
    """An element mass constraint

    Args:
        element: The element whose mass to constrain
        value: The mass in kg
    """

    def __init__(self, element: str, value: float):
        self._formula: Formula = Formula(element)
        self._value: float = value
        self._constraint: str = "mass"

    @property
    def constraint(self) -> str:
        """Name of the constraint"""
        return self._constraint

    @property
    def element(self) -> str:
        """Element whose mass to constrain"""
        return str(self._formula)

    @property
    def mass(self) -> float:
        """Value of the mass constraint in kg"""
        return self._value

    @property
    def molar_mass(self) -> float:
        r"""Molar mass in :math:\mathrm{kg}\mathrm{mol}^{-1}"""
        return UnitConversion.g_to_kg(self._formula.mass)

    @property
    def name(self) -> str:
        """Unique name of the constraint"""
        return f"{self.element}_{self.constraint}"

    @property
    def log10_number_of_molecules(self) -> float:
        """Log10 number of molecules"""
        return np.log10(self.mass) + np.log10(AVOGADRO) - np.log10(self.molar_mass)

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


class _SpeciesConstraint(ABC, Generic[U, V]):
    """A species constraint

    Args:
        species: The species to constrain
        value: Object to compute the value of the constraint
    """

    constraint_type: str = "default"

    def __init__(self, species: U, value: V):
        self._species: U = species
        self._value: V = value
        self._constraint: str = self.constraint_type

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

    def get_log10_value(self, temperature: float, pressure: float) -> float:
        """Computes the log10 value for given input arguments.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The log10 evaluated value
        """
        return np.log10(self.get_value(temperature, pressure))


class ActivityConstraint(_SpeciesConstraint[CondensedSpecies, float]):
    """An activity

    Args:
        species: The species to constrain
        value: The activity. Defaults to unity for ideal behaviour.
    """

    constraint_type: str = "activity"

    def activity(self, temperature: float, pressure: float) -> float:
        """Value of the activity constraint

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Activity
        """
        del temperature
        del pressure

        return self._value

    @override
    def get_value(self, temperature: float, pressure: float) -> float:
        return self.activity(temperature, pressure)


class _FugacityConstraint(_SpeciesConstraint[GasSpecies, V]):
    """A fugacity constraint

    Args:
        species: The species to constrain
        value: Object to compute the fugacity
    """

    constraint_type: str = "fugacity"

    @abstractmethod
    def fugacity(self, temperature: float, pressure: float) -> float:
        """Value of the fugacity constraint in bar

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Fugacity in bar
        """

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

    @override
    def get_value(self, temperature: float, pressure: float) -> float:
        return get_number_density(temperature, self.fugacity(temperature, pressure))


class FugacityConstraint(_FugacityConstraint[float]):
    """A constant fugacity constraint

    Args:
        species: The species to constrain
        value: The fugacity in bar
    """

    @override
    def fugacity(self, temperature: float, pressure: float) -> float:
        del temperature
        del pressure

        return self._value


class BufferedFugacityConstraint(_FugacityConstraint[RedoxBufferProtocol]):
    """A buffered fugacity constraint

    Args:
        species: The species to constrain
        value: The redox buffer
    """

    @override
    def fugacity(self, temperature: float, pressure: float) -> float:
        return self._value.get_value(temperature, pressure)


class MassConstraint(_SpeciesConstraint[ChemicalSpecies, float]):
    """A mass constraint

    Args:
        species: The species to constrain
        value: The mass in kg
    """

    constraint_type: str = "mass"

    def mass(self, *args, **kwargs) -> float:
        """Value of the mass constraint in kg"""
        return self.get_value(*args, **kwargs)


class PressureConstraint(_SpeciesConstraint[GasSpecies, float]):
    """A pressure constraint

    Args:
        species: The species to constrain
        value: The pressure in bar
    """

    constraint_type: str = "pressure"

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

    def get_value(self, temperature: float, pressure: float) -> float:
        return get_number_density(temperature, self.fugacity(temperature, pressure))


# FIXME: Update to number densities
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

    Args:
        initlist: Initial list of constraints. Defaults to None.
    """

    # UserList itself is not a generic class, so this is for typing:
    data: list[ConstraintProtocol]
    """List of constraints"""

    def add_activity_constraints(self, species: Species) -> None:
        """Adds activity constraints

        These constraints allow condensed phases to either be stable (activity of unity) or
        unstable (activity less than unity).

        Args:
            species: Species
        """
        for condensed_species in species.condensed_species:
            if condensed_species not in self.constrained_species:
                # FIXME: Maybe don't set by default?
                logger.debug("Automatically adding activity constraint for %s", condensed_species)
                self.append(ActivityConstraint(condensed_species, 1))
            else:
                logger.debug("Activity constraint for %s already included", condensed_species)

    @property
    def constrained_species(self) -> list[ChemicalSpecies]:
        """Constraints applied to species

        Species constraints are only applied in the context of the reaction network
        """
        return [constraint.species for constraint in self.reaction_network_constraints]

    @property
    def mass_constraints(self) -> list[ElementMassConstraint]:
        """Constraints related to element mass conservation"""
        return list(filter_by_type(self, ElementMassConstraint).values())

    @property
    def total_pressure_constraint(self) -> list[TotalPressureConstraint]:
        """Total pressure constraint"""
        total_pressure: list[TotalPressureConstraint] = list(
            filter_by_type(self, TotalPressureConstraint).values()
        )
        if len(total_pressure) > 1:
            raise ValueError("More than one total pressure constraint prescribed")

        return total_pressure

    @property
    def activity_constraints(self) -> list[ActivityConstraintProtocol]:
        """Constraints related to condensed species activities"""
        return list(filter_by_type(self, ActivityConstraintProtocol).values())

    @property
    def gas_constraints(self) -> list[GasConstraintProtocol]:
        """Constraints related to gas species fugacities and pressures"""
        return list(filter_by_type(self, GasConstraintProtocol).values())

    @property
    def reaction_network_constraints(self) -> list[ReactionNetworkConstraintProtocol]:
        """Constraints related to the reaction network"""
        constraints: list[ReactionNetworkConstraintProtocol] = []
        constraints.extend(self.activity_constraints)
        constraints.extend(self.gas_constraints)

        return constraints

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
            A dictionary of the evaluated constraints in the same order as the constraints
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
            A dictionary of the log10 evaluated constraints in the same order as the constraints
        """
        return {
            key: np.log10(value) for key, value in self.evaluate(temperature, pressure).items()
        }
