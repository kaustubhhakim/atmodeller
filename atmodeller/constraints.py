"""Constraints for the system of equations

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


This module defines constraints that can be applied to an interior-atmosphere system.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class Constraint(ABC):
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
class ConstantConstraint(Constraint):
    """A constraint of a constant value

    Args:
        name: The name of the constraint, which should be one of: 'activity', 'fugacity',
            'pressure', or 'mass'.
        species: The species to constrain, typically representing a species for 'pressure' or
            'fugacity' constraints or an element for 'mass' constraints.
        value: The constant value, which is usually in kg for masses and bar for pressures or
            fugacities.

    Attributes:
        name: The name of the constraint
        species: The species to constrain
        value: The constant value
    """

    value: float

    def get_value(self, **kwargs) -> float:
        """Returns the constant value. See base class."""
        del kwargs
        return self.value


@dataclass(kw_only=True, frozen=True)
class ActivityConstant(ConstantConstraint):
    """A constant activity

    Args:
        species: The species to constrain
        value: The constant value. Defaults to unity for ideal behaviour.

    Attributes:
        species: The species to constrain
        value: The constant value
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

    Note that condensed species activity constraints are inherent to the species and are appended
    to this list of user-prescribed constraints before the interior-atmosphere system is solved.

    Args:
        initlist: Initial list of constraints. Defaults to None.

    Attributes:
        data: A list of constraints for the interior-atmosphere system
        names: A list of unique names of the constraints
        activity_constraints: Activity constraints
        fugacity_constraints: Fugacity constraints
        mass_constraints: Mass constraints
        pressure_constraints: Pressure constraints
        total_pressure_constraint: Total pressure constraint
        reaction_network_constraints: Constraints for a reaction network
        number_reaction_network_constraints: Number of reaction network constraints
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


@dataclass(kw_only=True, frozen=True)
class RedoxBuffer(Constraint):
    """A mineral redox buffer that constrains a fugacity as a function of temperature

    Args:
        log10_shift: Log10 shift relative to the buffer
        pressure: Optional constant pressure in bar to always use to evaluate the redox buffer.
            Defaults to None, meaning that the input pressure is used instead.
    """

    log10_shift: float = 0
    pressure: float | None = None
    name: str = field(init=False, default="fugacity")

    @abstractmethod
    def get_buffer_log10_value(self, *, temperature: float, pressure: float, **kwargs) -> float:
        """Log10 value at the buffer

        Args:
            temperature: Temperature
            pressure: Pressure
            **kwargs: Arbitrary keyword arguments

        Returns:
            log10 of the fugacity at the buffer
        """

    def get_log10_value(self, *, temperature: float, pressure: float, **kwargs) -> float:
        """Log10 value including any shift

        Args:
            temperature: Temperature
            pressure: Pressure
            **kwargs: Arbitrary keyword arguments

        Returns:
            Log10 of the fugacity including any shift
        """
        # Below the input pressure value is overridden if self.pressure is not None.
        if self.pressure is not None:
            pressure = self.pressure
            logger.debug(
                "Evaluate %s at constant pressure = %f", self.__class__.__name__, pressure
            )
        log10_value: float = self.get_buffer_log10_value(
            temperature=temperature, pressure=pressure, **kwargs
        )
        log10_value += self.log10_shift
        return log10_value

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """See base class"""
        log10_value: float = self.get_log10_value(temperature=temperature, pressure=pressure)
        value: float = 10**log10_value
        return value


@dataclass(kw_only=True, frozen=True)
class OxygenFugacityBuffer(RedoxBuffer):
    """A mineral redox buffer that constrains oxygen fugacity as a function of temperature."""

    species: str = field(init=False, default="O2")


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintHirschmann(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from O'Neill and Pownceby (1993) and Hirschmann et al. (2008).

    https://ui.adsabs.harvard.edu/abs/1993CoMP..114..296O/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del kwargs
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintOneill(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from O'Neill and Eggins (2002).

    Gibbs energy of reaction is at 1 bar. See Table 6.
    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del pressure
        del kwargs
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintBallhaus(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from Ballhaus et al. (1991).

    https://ui.adsabs.harvard.edu/abs/1991CoMP..107...27B/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del kwargs
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintFischer(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from Fischer et al. (2011).

    See Table S2 in supplementary materials.
    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract

    See base class.
    """

    def get_buffer_log10_value(
        self, *, temperature: float, pressure: float = 1, **kwargs
    ) -> float:
        """See base class."""

        del kwargs
        pressure_GPa: float = UnitConversion.bar_to_GPa(pressure)
        a_P: float = 6.44059 + 0.00463099 * pressure_GPa
        b_P: float = (
            -28.1808
            + 0.556272 * pressure_GPa
            - 0.00143757 * pressure_GPa**2
            + 4.0256e-6 * pressure_GPa**3
            - 5.4861e-9 * pressure_GPa**4  # Note typo in Table S2. Must be pressure**4.
        )
        b_P *= 1000 / temperature
        fugacity: float = a_P + b_P

        return fugacity
