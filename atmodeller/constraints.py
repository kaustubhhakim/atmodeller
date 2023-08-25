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
from typing import Type, TypeVar

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import BufferedFugacity, SystemConstraint
from atmodeller.utilities import UnitConversion, filter_by_type

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class ValueConstraint:
    """A value constraint to apply to the system.

    A constraint that associates a specific value with a species in the system. It is used to
    enforce a particular value for a certain property, such as pressure or fugacity.

    Args:
        species: The species for which the constraint applies.
        value: The imposed value associated with the species.

    Attributes:
        species: The species for which the constraint applies.
        value: The imposed value associated with the species.
    """

    species: str
    value: float

    def get_value(self, **kwargs) -> float:
        """Retrieve the imposed value associated with the species.

        Args:
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The imposed value associated with the species.
        """
        del kwargs
        return self.value


@dataclass(kw_only=True)
class ReactionNetworkConstraint(ValueConstraint):
    """A value constraint applied to a reaction network.

    A constraint that applies specifically to a reaction network within the system. It extends the
    functionality of the ValueConstraint by targeting constraints related to reaction networks.

    Attributes:
        species: The species for which the constraint applies.
        value: The imposed value associated with the species.
    """


@dataclass(kw_only=True)
class FugacityConstraint(ReactionNetworkConstraint):
    """A constraint for fugacity.

    A constraint that enforces a specific fugacity value for a species within the context of a
    reaction network. It is a specialized form of ReactionNetworkConstraint.

    Attributes:
        species: The species for which the fugacity constraint applies.
        value: The imposed fugacity value associated with the species.
    """


@dataclass(kw_only=True)
class PressureConstraint(ReactionNetworkConstraint):
    """A constraint for pressure.

    A constraint that enforces a specific pressure value within the context of a reaction network.
    It is a specialized form of ReactionNetworkConstraint.

    Attributes:
        species: The species for which the pressure constraint applies.
        value: The imposed pressure value associated with the species.
    """


@dataclass(kw_only=True)
class MassConstraint(ValueConstraint):
    """A constraint for mass conservation.

    A constraint that enforces a specific mass value for a species as a part of mass conservation.
    It is a specialized form of ValueConstraint.

    Attributes:
        species: The species for which the mass conservation constraint applies.
        value: The imposed mass value associated with the species.
    """


class SystemConstraints(UserList):
    """Collection of constraints for an interior-atmosphere system.

    A collection of constraints that can be applied to an interior-atmosphere system. It provides
    methods to filter constraints based on their types, such as fugacity, mass conservation,
    pressure, and reaction network constraints.

    Args:
        initlist: Initial list of constraints. Defaults to None.

    Attributes:
        data: List of constraints contained in the system.
    """

    def __init__(self, initlist=None):
        self.data: list[SystemConstraint]
        super().__init__(initlist)

    @property
    def fugacity_constraints(self) -> list[FugacityConstraint]:
        """Constraints related to fugacity."""
        return filter_by_type(self.data, FugacityConstraint)

    @property
    def mass_constraints(self) -> list[MassConstraint]:
        """Constraints related to mass conservation."""
        return filter_by_type(self.data, MassConstraint)

    @property
    def pressure_constraints(self) -> list[PressureConstraint]:
        """Constraints related to pressure."""
        return filter_by_type(self.data, PressureConstraint)

    @property
    def reaction_network_constraints(self) -> list[ReactionNetworkConstraint]:
        """Constraints related to the reaction network."""
        return filter_by_type(self.data, ReactionNetworkConstraint)

    @property
    def number_reaction_network_constraints(self) -> int:
        """Number of constraints related to the reaction network."""
        return len(self.reaction_network_constraints)


class IronWustiteBufferHirschmann(BufferedFugacity):
    """Iron-wustite buffer (fO2) from O'Neill and Pownceby (1993) and Hirschmann et al. (2008).

    https://ui.adsabs.harvard.edu/abs/1993CoMP..114..296O/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )
        return fugacity


class IronWustiteBufferOneill(BufferedFugacity):
    """Iron-wustite buffer (fO2) from O'Neill and Eggins (2002).

    Gibbs energy of reaction is at 1 bar. See Table 6.
    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        del pressure
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )
        return fugacity


class IronWustiteBufferBallhaus(BufferedFugacity):
    """Iron-wustite buffer (fO2) from Ballhaus et al. (1991).

    https://ui.adsabs.harvard.edu/abs/1991CoMP..107...27B/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )
        return fugacity


class IronWustiteBufferFischer(BufferedFugacity):
    """Iron-wustite buffer (fO2) from Fischer et al. (2011).

    See Table S2 in supplementary materials.
    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
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
        buffer: float = a_P + b_P
        return buffer


@dataclass(kw_only=True)
class BufferedFugacityConstraint(FugacityConstraint):
    """A buffered fugacity constraint to apply to an interior-atmosphere system.

    A constraint that applies a buffered fugacity requirement to a species within an
    interior-atmosphere system. The buffered fugacity is controlled by a BufferedFugacity model,
    with an optional log10 shift applied.

    Args:
        species: The species for which the buffered fugacity constraint applies. Defaults to 'O2'.
        fugacity: A BufferedFugacity model representing the buffer. Defaults to
            IronWustiteBufferHirschmann.
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.

    Attributes:
        species: The species that is buffered by the given buffer.
        fugacity: The BufferedFugacity model that defines the buffer.
        log10_shift: The log10 shift applied to the buffered fugacity.
    """

    species: str = "O2"
    value: BufferedFugacity = field(default_factory=IronWustiteBufferHirschmann)
    log10_shift: float = 0

    def get_value(self, *, temperature: float, pressure: float = 1, **kwargs) -> float:
        """Calculate the buffered fugacity value based on the provided temperature and pressure.

        Args:
            temperature: The temperature at which to calculate the fugacity.
            pressure: The pressure at which to calculate the fugacity. Defaults to 1.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            The calculated buffered fugacity value.
        """
        del kwargs
        value: float = 10 ** self.value(
            temperature=temperature, pressure=pressure, log10_shift=self.log10_shift
        )
        return value
