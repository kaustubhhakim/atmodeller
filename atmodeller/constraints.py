"""Constraints for the system of equations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import UserList
from dataclasses import dataclass, field
from typing import Protocol, Type, TypeVar

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


class SystemConstraint(Protocol):
    """A value constraint to apply to an interior-atmosphere system.

    Args:
        species: The species to constrain. Usually a species for a pressure or fugacity constraint
            or an element for a mass constraint.
        value: Imposed value in kg for masses and bar for pressures or fugacities.

    Attributes:
        species: The species to constrain. Usually a species for a pressure or fugacity constraint
            or an element for a mass constraint.
        value: Imposed value in kg for masses and bar for pressures or fugacities.
    """

    @property
    def species(self) -> str:
        ...

    def get_value(self, **kwargs) -> float:
        ...


T = TypeVar("T", bound=SystemConstraint)


class SystemConstraints(UserList):
    def __init__(self, initlist=None):
        self.data: list[SystemConstraint]
        super().__init__(initlist)

    def _filter_by_type(self, class_type: Type[T]) -> list[T]:
        return [constraint for constraint in self if isinstance(constraint, class_type)]

    @property
    def fugacity_constraints(self) -> list[FugacityConstraint]:
        """Constraints for fugacity."""
        return self._filter_by_type(FugacityConstraint)

    @property
    def mass_constraints(self) -> list[MassConstraint]:
        """Constraints for mass conservation."""
        return self._filter_by_type(MassConstraint)

    @property
    def pressure_constraints(self) -> list[PressureConstraint]:
        """Constraints for pressure."""
        return self._filter_by_type(PressureConstraint)

    @property
    def reaction_network_constraints(self) -> list[ReactionNetworkConstraint]:
        """Constraints for the reaction network."""
        return self._filter_by_type(ReactionNetworkConstraint)

    @property
    def number_reaction_network_constraints(self) -> int:
        return len(self.reaction_network_constraints)


@dataclass(kw_only=True)
class ValueConstraint:
    species: str
    value: float

    def get_value(self, **kwargs) -> float:
        del kwargs
        return self.value


@dataclass(kw_only=True)
class ReactionNetworkConstraint(ValueConstraint):
    """A value constraint applied to a reaction network."""


@dataclass(kw_only=True)
class FugacityConstraint(ReactionNetworkConstraint):
    """TODO."""


@dataclass(kw_only=True)
class PressureConstraint(ReactionNetworkConstraint):
    """TODO."""


@dataclass(kw_only=True)
class MassConstraint(ValueConstraint):
    """TODO."""


class BufferedFugacity(ABC):
    """Abstract base class for calculating buffered fugacity based on temperature and pressure.

    This class defines a method to calculate the log10(fugacity) of a buffer substance in terms of
    temperature. Subclasses must implement the '_fugacity' method to provide the specific
    calculation.

    Attributes:
        None
    """

    @abstractmethod
    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """Calculates the log10(fugacity) of the buffer in terms of temperature.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in bar. Defaults to 1 bar.

        Returns:
            Log10 of the fugacity.
        """
        raise NotImplementedError

    def __call__(
        self, *, temperature: float, pressure: float = 1, log10_shift: float = 0
    ) -> float:
        """Calculates the log10(fugacity) of the buffer plus an optional shift.

        Args:
            temperature: Temperature in Kelvin.
            pressure: Pressure in bar. Defaults to 1 bar.
            log10_shift: Log10 shift. Defaults to 0.

        Returns:
            Log10 of the fugacity including the shift.
        """
        return self._fugacity(temperature=temperature, pressure=pressure) + log10_shift


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

    Args:
        species: The species that is buffered by 'buffer'. Defaults to 'O2'.
        fugacity: A BufferedFugacity. Defaults to IronWustiteBufferHirschmann
        log10_shift: Log10 shift relative to the buffer. Defaults to 0.

    Attributes:
        species: The species that is buffered by 'buffer'.
        fugacity: A BufferedFugacity.
        log10_shift: Log10 shift relative to the buffer.
    """

    species: str = "O2"
    value: BufferedFugacity = field(default_factory=IronWustiteBufferHirschmann)
    log10_shift: float = 0

    def get_value(self, *, temperature: float, pressure: float = 1, **kwargs) -> float:
        del kwargs
        value: float = 10 ** self.value(
            temperature=temperature, pressure=pressure, log10_shift=self.log10_shift
        )
        return value
