"""Oxygen fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        self.CH4: float = self.C + 4 * self.H
        self.CO: float = self.C + self.O
        self.CO2: float = self.C + 2 * self.O
        self.H2: float = self.H * 2
        self.H2O: float = self.H * 2 + self.O
        self.N2: float = self.N * 2
        self.NH3: float = self.N + 3 * self.H
        self.O2: float = self.O * 2
        self.SO2: float = self.S + 2 * self.O


@dataclass(frozen=True)
class FormationEquilibriumConstants:
    """Formation equilibrium constants.

    These parameters result from a linear fit in temperature space to the log Kf column in the
    JANAF data tables for a given molecule. See the jupyter notebook in 'janaf/'.

    log10(Kf) = a + b/T

    In the future we could use the Shomate equation to calculate the equilibrium of the gas phase
    reactions.
    """

    # Want to use molecule names therefore pylint: disable=invalid-name
    C: tuple[float, float] = (0, 0)
    CH4: tuple[float, float] = (-5.830066176470588, 4829.067647058815)
    CO: tuple[float, float] = (4.319860294117643, 6286.120588235306)
    CO2: tuple[float, float] = (-0.028289705882357442, 20753.870588235302)
    H2: tuple[float, float] = (0, 0)
    H2O: tuple[float, float] = (-3.0385132352941198, 13152.698529411768)
    N2: tuple[float, float] = (0, 0)
    O2: tuple[float, float] = (0, 0)
    # TODO: Commented out by Laura so check values.
    # NH3: tuple[float, float] = (-45.9, 192.77)

class Solubility(ABC):
    """Solubility base class."""

    def power_law(self, pressure: float, constant: float, exponent: float) -> float:
        """Power law. Pressure in bar and returns ppmw."""
        return constant * pressure**exponent

    @abstractmethod
    def _solubility(self, pressure: float, temperature: float) -> float:
        raise NotImplementedError

    def __call__(self, pressure: float, *args) -> float:
        """Dissolved volatile concentration in ppmw in the melt."""
        return self._solubility(pressure, *args)


class NoSolubility(Solubility):
    """No solubility."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del pressure
        del temperature
        return 0.0


class AnorthiteDiopsideH2O(Solubility):
    """Newcombe et al. (2017)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 727, 0.5)


class PeridotiteH2O(Solubility):
    """Sossi et al. (2022)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 534, 0.5)


class BasaltDixonH2O(Solubility):
    """Dixon et al. (1995) refit by Paolo Sossi."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 965, 0.5)


class BasaltWilsonH2O(Solubility):
    """Hamilton (1964) and Wilson and Head (1981)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 215, 0.7)


class LunarGlassH2O(Solubility):
    """Newcombe et al. (2017)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 683, 0.5)


class BasaltDixonCO2(Solubility):
    """Dixon et al. (1995)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        ppmw: float = (
            (3.8e-7) * pressure * np.exp(-23 * (pressure - 1) / (83.15 * temperature))
        )
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)
        return ppmw


class LibourelN2(Solubility):
    """Libourel et al. (2003)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        ppmw: float = self.power_law(pressure, 0.0611, 1.0)
        return ppmw
