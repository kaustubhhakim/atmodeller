"""Constraints for the system of equations.

This module defines constraints that can be applied to an interior-atmosphere system.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import logging
from collections import UserList
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import ConstantConstraint, ConstraintABC
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class FugacityConstraint(ConstantConstraint):
    """A constant fugacity constraint. See base class."""

    name: str = field(init=False, default="fugacity")


@dataclass(kw_only=True, frozen=True)
class PressureConstraint(ConstantConstraint):
    """A constant pressure constraint. See base class."""

    name: str = field(init=False, default="pressure")


@dataclass(kw_only=True, frozen=True)
class MassConstraint(ConstantConstraint):
    """A constant mass constraint. See base class."""

    name: str = field(init=False, default="mass")


class SystemConstraints(UserList):
    """A collection of constraints for an interior-atmosphere system.

    A collection of constraints that can be applied to an interior-atmosphere system. It provides
    methods to filter constraints based on their types, such as fugacity, mass conservation,
    pressure, and reaction network constraints.

    Note that solid activity constraints are inherent to the species and are combined with these
    user-prescribed constraints when the interior-atmosphere system is solved.

    Args:
        initlist: Initial list of constraints. Defaults to None.

    Attributes:
        data: A list of constraints for the interior-atmosphere system.
    """

    def __init__(self, initlist=None):
        self.data: list[ConstraintABC]
        super().__init__(initlist)

    def _filter_by_name(self, name: str) -> list[ConstraintABC]:
        """Filters the constraints by a given name.

        Args:
            name: The filter string (e.g., fugacity, pressure, mass).

        Returns:
            A list of filtered constraints.
        """
        filtered: list = []
        for entry in self.data:
            if entry.name == name:
                filtered.append(entry)

        return filtered

    @property
    def fugacity_constraints(self) -> list[ConstraintABC]:
        """Constraints related to fugacity."""
        return self._filter_by_name("fugacity")

    @property
    def mass_constraints(self) -> list[ConstraintABC]:
        """Constraints related to mass conservation."""
        return self._filter_by_name("mass")

    @property
    def pressure_constraints(self) -> list[ConstraintABC]:
        """Constraints related to pressure."""
        return self._filter_by_name("pressure")

    @property
    def reaction_network_constraints(self) -> list[ConstraintABC]:
        """Constraints related to the reaction network."""
        filtered_entries: list[ConstraintABC] = self.fugacity_constraints
        filtered_entries.extend(self.pressure_constraints)

        return filtered_entries

    @property
    def number_reaction_network_constraints(self) -> int:
        """Number of constraints related to the reaction network."""
        return len(self.reaction_network_constraints)


@dataclass(kw_only=True, frozen=True)
class RedoxBuffer(ConstraintABC):
    """A mineral redox buffer that constrains a fugacity as a function of temperature.

    Args:
        log10_shift: Log10 shift relative to the buffer.
    """

    log10_shift: float = 0
    name: str = field(init=False, default="fugacity")


@dataclass(kw_only=True, frozen=True)
class OxygenFugacityBuffer(RedoxBuffer):
    """A mineral redox buffer that constraints oxygen fugacity as a function of temperature."""

    species: str = field(init=False, default="O2")


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintHirschmann(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from O'Neill and Pownceby (1993) and Hirschmann et al. (2008).

    https://ui.adsabs.harvard.edu/abs/1993CoMP..114..296O/abstract

    See base class.
    """

    def get_value(self, *, temperature: float, pressure: float = 1, **kwargs) -> float:
        """See base class."""

        del kwargs
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )
        fugacity += self.log10_shift
        fugacity = 10**fugacity

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintOneill(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from O'Neill and Eggins (2002).

    Gibbs energy of reaction is at 1 bar. See Table 6.
    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract

    See base class.
    """

    def get_value(self, *, temperature: float, pressure: float = 1, **kwargs) -> float:
        """See base class."""

        del pressure
        del kwargs
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )
        fugacity += self.log10_shift
        fugacity = 10**fugacity

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintBallhaus(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from Ballhaus et al. (1991).

    https://ui.adsabs.harvard.edu/abs/1991CoMP..107...27B/abstract

    See base class.
    """

    def get_value(self, *, temperature: float, pressure: float = 1, **kwargs) -> float:
        """See base class."""

        del kwargs
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )
        fugacity += self.log10_shift
        fugacity = 10**fugacity

        return fugacity


@dataclass(kw_only=True, frozen=True)
class IronWustiteBufferConstraintFischer(OxygenFugacityBuffer):
    """Iron-wustite buffer (fO2) from Fischer et al. (2011).

    See Table S2 in supplementary materials.
    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract

    See base class.
    """

    def get_value(self, *, temperature: float, pressure: float = 1, **kwargs) -> float:
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
        fugacity += self.log10_shift
        fugacity = 10**fugacity
        return fugacity
