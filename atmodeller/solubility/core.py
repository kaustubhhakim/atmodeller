#
# Copyright 2024 Dan J. Bower, Fabian L. Seidler
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Core classes and functions for solubility laws.

Units for temperature and pressure are K and bar, respectively.
"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer
from atmodeller.utilities import power_law

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier


class Solubility(eqx.Module):
    """Solubility interface

    :meth:`~Solubility.jax_concentration` is defined in order to allow arguments to be passed by
    position to lax.switch.
    """

    @abstractmethod
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
        fO2: ArrayLike | None = None,
    ) -> Array:
        """Concentration in ppmw

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K. Defaults to None for not used.
            pressure: Pressure in bar. Defaults to None for not used.
            fO2: fO2 in bar. Defaults to None for not used.

        Returns:
            Concentration in ppmw
        """

    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> Array:
        """Wrapper to pass concentration arguments by position to use with JAX lax.switch

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K
            pressure: Pressure in bar
            fO2: fO2 in bar

        Returns:
            Concentration in ppmw
        """
        return self.concentration(fugacity, temperature=temperature, pressure=pressure, fO2=fO2)


class NoSolubility(Solubility):
    """No solubility"""

    @override
    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
        fO2: ArrayLike | None = None,
    ) -> ArrayLike:
        del fugacity
        del temperature
        del pressure
        del fO2

        return jnp.array(0.0)  # For JAX compatibility


class SolubilityPowerLaw(Solubility):
    """A solubility power law

    Args:
        constant: Constant
        exponent: Exponent
    """

    constant: float
    """Constant"""
    exponent: float
    """Exponent"""

    @override
    def concentration(self, fugacity: ArrayLike, *args, **kwargs) -> ArrayLike:
        del args
        del kwargs

        return power_law(fugacity, self.constant, self.exponent)


class SolubilityPowerLawLog10(Solubility):
    """A solubility power law with log10 coefficients

    Args:
        log10_constant: Log10 constant
        log10_exponent: Log10 exponent

    """

    log10_constant: float
    """Log10 constant"""
    log10_exponent: float
    """Log10 exponent"""

    @override
    def concentration(self, fugacity: ArrayLike, **kwargs) -> ArrayLike:
        del kwargs

        return jnp.power(10, (self.log10_constant + self.log10_exponent * jnp.log10(fugacity)))


def fO2_temperature_correction(
    fO2: ArrayLike,
    *,
    temperature: ArrayLike,
    pressure: ArrayLike,
    reference_temperature: ArrayLike,
) -> Array:
    """Applies a temperature correction to fO2.

    Some experimentally derived solubility laws operate on absolute fO2, which depends on
    temperature and pressure. A temperature correction has to be applied to maintain the same
    fO2 shift at arbitrary temperature.

    Args:
        fO2: Absolute oxygen fugacity at `temperature` in bar
        temperature: Temperature in K
        pressure: Absolute pressure in bar
        reference_temperature: Reference temperature, which is usually the temperature at which the
            experiment was performed.

    Returns:
        Adjusted fO2
    """
    iron_wustite_buffer: RedoxBufferProtocol = IronWustiteBuffer()
    logiw_fugacity_at_current_temp: ArrayLike = iron_wustite_buffer.log10_fugacity(
        temperature, pressure
    )
    fo2_shift: Array = jnp.log10(fO2) - logiw_fugacity_at_current_temp

    logiw_fugacity_at_reference_temp: ArrayLike = iron_wustite_buffer.log10_fugacity(
        reference_temperature, pressure
    )
    adjusted_fo2: Array = jnp.power(10, logiw_fugacity_at_reference_temp + fo2_shift)

    return adjusted_fo2
