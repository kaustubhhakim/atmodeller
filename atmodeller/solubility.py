"""Solubility laws."""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

class _Solubility(ABC):
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


class AnorthiteDiopsideH2O(_Solubility):
    """Newcombe et al. (2017)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 727, 0.5)


class PeridotiteH2O(_Solubility):
    """Sossi et al. (2022)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 534, 0.5)


class BasaltDixonH2O(_Solubility):
    """Dixon et al. (1995) refit by Paolo Sossi."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 965, 0.5)


class BasaltWilsonH2O(_Solubility):
    """Hamilton (1964) and Wilson and Head (1981)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 215, 0.7)


class LunarGlassH2O(_Solubility):
    """Newcombe et al. (2017)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        return self.power_law(pressure, 683, 0.5)


class BasaltDixonCO2(_Solubility):
    """Dixon et al. (1995)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        ppmw: float = (
            (3.8e-7) * pressure * np.exp(-23 * (pressure - 1) / (83.15 * temperature))
        )
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)
        return ppmw


class LibourelN2(_Solubility):
    """Libourel et al. (2003)."""

    def _solubility(self, pressure: float, temperature: float) -> float:
        del temperature
        ppmw: float = self.power_law(pressure, 0.0611, 1.0)
        return ppmw
