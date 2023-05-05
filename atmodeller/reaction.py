"""Oxygen fugacity buffers and gas phase reactions."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class _OxygenFugacity(ABC):
    """Oxygen fugacity base class."""

    @abstractmethod
    def _buffer(self, *, temperature: float) -> float:
        """Log10(fO2) of the buffer in terms of temperature.

        Args:
            temperature: Temperature.

        Returns:
            oxygen_fugacity of the buffer.
        """
        raise NotImplementedError

    def __call__(self, *, temperature: float, fo2_shift: float = 0) -> float:
        """log10 of fo2."""
        return self._buffer(temperature=temperature) + fo2_shift


class IronWustiteBufferOneill(_OxygenFugacity):
    """Iron-wustite buffer from O'Neill and Eggin (2002)."""

    def _buffer(self, *, temperature: float) -> float:
        """See base class."""
        buffer: float = (
            2
            * (
                -244118
                + 115.559 * temperature
                - 8.474 * temperature * np.log(temperature)
            )
            / (np.log(10) * 8.31441 * temperature)
        )
        return buffer


class IronWustiteBufferFischer(_OxygenFugacity):
    """Iron-wustite buffer from Fischer et al. (2011)."""

    def _buffer(self, *, temperature: float) -> float:
        """See base class."""
        buffer: float = 6.94059 - 28.1808 * 1e3 / temperature
        return buffer


@dataclass
class _EquilibriumConstant:
    """Parameters that define the equilibrium constant for a reaction."""

    temperature_factor: float
    constant: float
    fo2_stoichiometry: float
    oxygen_fugacity: _OxygenFugacity = field(default_factory=IronWustiteBufferOneill)

    def fo2_log10(self, *, temperature: float, fo2_shift: float = 0) -> float:
        """Oxygen fugacity.

        Args:
            temperature: Temperature.
            fo2_shift: log10 shift relative to the buffer.

        Returns:
            Log10 of the oxygen fugacity.
        """
        return self.oxygen_fugacity(temperature=temperature, fo2_shift=fo2_shift)

    def equilibrium_constant_log10(self, *, temperature: float) -> float:
        """Log10 of the equilibrium constant.

        Args:
            temperature: Temperature.

        Returns:
            Log10 of the equilibrium constant of the reaction.
        """
        return self.temperature_factor / temperature + self.constant

    def equilibrium_constant(self, *, temperature: float) -> float:
        """Equilibrium constant.

        Args:
            temperature: Temperature.

        Returns:
            The equilibrium constant of the reaction.
        """
        return 10 ** self.equilibrium_constant_log10(temperature=temperature)

    def modified_equilibrium_constant_log10(
        self, *, temperature: float, fo2_shift: float
    ) -> float:
        """Log10 of the 'modified' equilibrium constant, which includes oxygen fugacity.

        Args:
            temperature: Temperature.
            fo2_shift: log10 shift relative to the buffer.

        Returns:
            Log10 of the 'modified' equilibrium constant.
        """
        return self.equilibrium_constant_log10(
            temperature=temperature
        ) - self.fo2_stoichiometry * self.fo2_log10(
            temperature=temperature, fo2_shift=fo2_shift
        )

    def modified_equilibrium_constant(
        self, *, temperature: float, fo2_shift: float
    ) -> float:
        """The 'modified' equilibrium constant, which includes oxygen fugacity.

        Args:
            temperature: Temperature.
            fo2_shift: log10 shift relative to the buffer.

        Returns:
            The 'modified' equilibrium constant.
        """
        return 10.0 ** self.modified_equilibrium_constant_log10(
            temperature=temperature, fo2_shift=fo2_shift
        )


@dataclass
class JanafC(_EquilibriumConstant):
    """JANAF log10Keq, 1500 < K < 3000 for CO2 = CO + 0.5 fo2."""

    temperature_factor: float = -14467.511400133637
    constant: float = 4.348135473316284
    fo2_stoichiometry: float = 0.5


@dataclass
class SchaeferC(_EquilibriumConstant):
    """Schaefer log10Keq for CO2 = CO + 0.5 fo2."""

    temperature_factor: float = -14787
    constant: float = 4.5472
    fo2_stoichiometry: float = 0.5


@dataclass
class SchaeferCH4(_EquilibriumConstant):
    """Schaefer log10Keq for CO2 + 2H2 = CH4 + fo2."""

    temperature_factor: float = -16276
    constant: float = -5.4738
    fo2_stoichiometry: float = 1


@dataclass
class JanafH(_EquilibriumConstant):
    """JANAF log10Keq, 1500 < K < 3000 for H2O = H2 + 0.5 fo2."""

    temperature_factor: float = -13152.477779978302
    constant: float = 3.038586383273608
    fo2_stoichiometry: float = 0.5


@dataclass
class SchaeferH(_EquilibriumConstant):
    """Schaefer log10Keq for H2O = H2 + 0.5 fo2."""

    temperature_factor: float = -12794
    constant: float = 2.7768
    fo2_stoichiometry: float = 0.5
