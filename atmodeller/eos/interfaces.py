"""Interfaces for real gas equations of state.

See the LICENSE file for licensing information.
"""

from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.interfaces import GetValueABC


@dataclass(kw_only=True)
class FugacityModelABC(GetValueABC):
    """A fugacity model.

    This base class requires a specification for the volume and volume integral, since then the
    fugacity and related quantities can be computed using the standard relation:

    RTlnf = integral(VdP).

    Args:
        scaling: Scaling depending on the units of the coefficients that are used in this model.
            For pressure in bar this is unity and for pressure in kbar this is kilo. Defaults to
            unity.

    Attributes:
        scaling: Scaling depending on the units of pressure.
        GAS_CONSTANT: Gas constant with the appropriate units.
    """

    scaling: float = 1
    GAS_CONSTANT: float = field(init=False, default=GAS_CONSTANT)

    def __post_init__(self):
        """Scales the GAS_CONSTANT to ensure it has the correct units."""
        self.GAS_CONSTANT /= self.scaling

    def get_value(self, *, temperature: float, pressure: float) -> float:
        """Evaluates the fugacity coefficient at temperature and pressure.

        Note that the input 'pressure' must ALWAYS be in bar, so it is scaled here using
        'self.scaling' since self.fugacity_coefficient requires the internal units of pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in bar.

        Returns:
            Fugacity coefficient evaluated at temperature and pressure, which is dimensionaless.
        """
        pressure /= self.scaling
        fugacity_coefficient: float = self.fugacity_coefficient(temperature, pressure)

        return fugacity_coefficient

    def ln_fugacity(self, temperature: float, pressure: float) -> float:
        """Natural log of the fugacity.

        The fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and f0 is the
        pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity.
        """
        ln_fugacity: float = self.volume_integral(temperature, pressure) / (
            self.GAS_CONSTANT * temperature
        )

        return ln_fugacity

    def fugacity(self, temperature: float, pressure: float) -> float:
        """Fugacity in the same units as the input pressure.

        Note that the fugacity term in the exponential is non-dimensional (f'), where f'=f/f0 and
        f0 is the pure gas fugacity at reference pressure of 1 bar under which f0 = P0 = 1 bar.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity.
        """
        fugacity: float = np.exp(self.ln_fugacity(temperature, pressure))  # bar
        fugacity /= self.scaling  # to units of input pressure for consistency.

        return fugacity

    def fugacity_coefficient(self, temperature: float, pressure: float) -> float:
        """Fugacity coefficient.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            fugacity coefficient, which is non-dimensional.
        """
        fugacity_coefficient: float = self.fugacity(temperature, pressure) / pressure

        return fugacity_coefficient

    @abstractmethod
    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        ...

    @abstractmethod
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        ...
