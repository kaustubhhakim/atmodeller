"""Solubility laws and reactions."""

import logging
from abc import ABC, abstractmethod

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class _OxygenFugacity(ABC):
    """Oxygen fugacity base class."""

    @abstractmethod
    def _buffer(self, temperature: float):
        """Log10(fO2) of the buffer in terms of temperature."""
        raise NotImplementedError

    def __call__(self, temperature: float, fo2_shift: float = 0) -> float:
        """log10 of fo2."""
        return self._buffer(temperature) + fo2_shift


class IronWustiteBufferOneill(_OxygenFugacity):
    """Iron-wustite buffer from O'Neill and Eggin (2002)."""

    def _buffer(self, temperature: float) -> float:
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

    def _buffer(self, temperature: float) -> float:
        buffer: float = 6.94059 - 28.1808 * 1e3 / temperature
        return buffer


class ModifiedKeq:
    """Modified equilibrium constant, i.e. includes fo2.

    Args:
        keq_model: Equilibrium model to use. Options are give below __call__.
        fo2_model: fo2 model to use. See class _OxygenFugacity for options. Defaults to 'oneill'.
    """

    def __init__(self, keq_model: str):
        self.fo2: _OxygenFugacity = IronWustiteBufferOneill()
        self._callmodel = getattr(self, keq_model)

    def __call__(self, temperature: float, fo2_shift: float = 0) -> float:
        fo2: float = self.fo2(temperature, fo2_shift)
        keq, fo2_stoich = self._callmodel(temperature)
        geq: float = 10 ** (keq - fo2_stoich * fo2)
        return geq

    # For the methods below, the second entry in the tuple is the stoichiometry of O2.

    def schaefer_ch4(self, temperature: float) -> tuple[float, float]:
        """Schaefer log10Keq for CO2 + 2H2 = CH4 + fo2."""
        return (-16276 / temperature - 5.4738, 1)

    def schaefer_c(self, temperature: float) -> tuple[float, float]:
        """Schaefer log10Keq for CO2 = CO + 0.5 fo2."""
        return (-14787 / temperature + 4.5472, 0.5)

    def schaefer_h(self, temperature: float) -> tuple[float, float]:
        """Schaefer log10Keq for H2O = H2 + 0.5 fo2."""
        return (-12794 / temperature + 2.7768, 0.5)

    def janaf_c(self, temperature: float) -> tuple[float, float]:
        """JANAF log10Keq, 1500 < K < 3000 for CO2 = CO + 0.5 fo2."""
        return (-14467.511400133637 / temperature + 4.348135473316284, 0.5)

    def janaf_h(self, temperature: float) -> tuple[float, float]:
        """JANAF log10Keq, 1500 < K < 3000 for H2O = H2 + 0.5 fo2."""
        return (-13152.477779978302 / temperature + 3.038586383273608, 0.5)


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
