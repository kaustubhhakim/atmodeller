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
        """Log10(fo2) of the buffer in terms of temperature.

        Args:
            temperature: Temperature.

        Returns:
            Log10 of the oxygen fugacity.
        """
        raise NotImplementedError

    def __call__(self, *, temperature: float, fo2_shift: float = 0) -> float:
        """log10(fo2) plus an optional shift.

        Args:
            temperature: Temperature.
            fo2_shift: Log10 shift.

        Returns:
            Log10 of the oxygen fugacity including a shift.
        """
        return self._buffer(temperature=temperature) + fo2_shift


class IronWustiteBufferOneill(_OxygenFugacity):
    """Iron-wustite buffer from O'Neill and Eggins (2002). See Table 6.

    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract
    """

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
    """Iron-wustite buffer from Fischer et al. (2011).

    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract
    """

    def _buffer(self, *, temperature: float) -> float:
        """See base class."""
        buffer: float = 6.94059 - 28.1808 * 1e3 / temperature
        return buffer


@dataclass
class _Reaction:
    """A gas phase reaction.

    Args:
        temperature_factor: Factor to multiply 1/temperature for the equilibrium constant.
        constant: Constant factor to add for the equilibrium constant.
        fo2_stoichiometry: The stoichiometry of oxygen in the reaction.
        oxygen_fugacity: The oxygen fugacity model to use for the reaction.
    """

    temperature_factor: float
    constant: float
    fo2_stoichiometry: float
    oxygen_fugacity: _OxygenFugacity = field(default_factory=IronWustiteBufferOneill)

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
        ) - self.fo2_stoichiometry * self.oxygen_fugacity(
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
class JanafC(_Reaction):
    """CO2 = CO + 0.5 fo2.

    JANAF log10Keq, 1500 < T < 3000. Fit by P. Sossi.
    """

    temperature_factor: float = -14467.511400133637
    constant: float = 4.348135473316284
    fo2_stoichiometry: float = 0.5


@dataclass
class JanafH(_Reaction):
    """H2O = H2 + 0.5 fo2.

    JANAF log10Keq, 1500 < T < 3000. Fit by P. Sossi.
    """

    temperature_factor: float = -13152.477779978302
    constant: float = 3.038586383273608
    fo2_stoichiometry: float = 0.5


@dataclass
class IvtanthermoC(_Reaction):
    """CO2 = CO + 0.5 fo2.

    IVANTHERMO log10Keq, 298.15 < T < 2000. Fit by Schaefer and Fegley (2017).

    https://ui.adsabs.harvard.edu/abs/2017ApJ...843..120S/abstract
    """

    temperature_factor: float = -14787
    constant: float = 4.5472
    fo2_stoichiometry: float = 0.5


@dataclass
class IvtanthermoCH4(_Reaction):
    """CO2 + 2H2 = CH4 + fo2.

    IVANTHERMO log10Keq, 298.15 < T < 2000. Fit by Schaefer and Fegley (2017).

    https://ui.adsabs.harvard.edu/abs/2017ApJ...843..120S/abstract
    """

    temperature_factor: float = -16276
    constant: float = -5.4738
    fo2_stoichiometry: float = 1


@dataclass
class IvtanthermoH(_Reaction):
    """H2O = H2 + 0.5 fo2.

    IVANTHERMO log10Keq, 298.15 < T < 2000. Fit by Schaefer and Fegley (2017).

    https://ui.adsabs.harvard.edu/abs/2017ApJ...843..120S/abstract
    """

    temperature_factor: float = -12794
    constant: float = 2.7768
    fo2_stoichiometry: float = 0.5


@dataclass
class MolarMasses:
    """Molar masses of atoms and molecules in kg/mol.

    There is a library that could do this, but it would add a dependency and there is always a
    risk it wouldn't be supported in the future:

    https://pypi.org/project/molmass/
    """

    # Define atoms here.
    # pylint: disable=invalid-name
    C: float = 12.0107e-3
    H: float = 1.0079e-3
    N: float = 14.0067e-3
    O: float = 15.9994e-3
    S: float = 32.065e-3

    def __post_init__(self):
        # Define molecules here.
        self.CH4: float = self.C + self.H * 4
        self.CO: float = self.C + self.O
        self.CO2: float = self.C + 2 * self.O
        self.H2: float = self.H * 2
        self.H2O: float = self.H * 2 + self.O
        self.N2: float = self.N * 2
        self.SO2: float = self.S + 2 * self.O
