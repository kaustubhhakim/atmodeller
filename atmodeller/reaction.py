"""Oxygen fugacity buffers and gas phase reactions."""

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