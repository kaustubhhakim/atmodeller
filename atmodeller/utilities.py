"""Utilities.

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

from molmass import Formula

from atmodeller import OCEAN_MOLES

logger: logging.Logger = logging.getLogger(__name__)


class UnitConversion:
    """Unit conversions."""

    @classmethod
    def bar_to_Pa(cls, value_bar: float = 1) -> float:
        """bar to Pa."""
        return value_bar * 1e5

    @classmethod
    def bar_to_GPa(cls, value_bar: float = 1) -> float:
        """Bar to GPa."""
        return cls.bar_to_Pa(value_bar) * 1.0e-9

    @classmethod
    def fraction_to_ppm(cls, value_fraction: float = 1) -> float:
        """Mole or mass fraction to parts-per-million by mole or mass, respectively."""
        return value_fraction * 1.0e6

    @classmethod
    def g_to_kg(cls, value_grams: float = 1) -> float:
        """Grams to kilograms."""
        return value_grams * 1.0e-3

    @classmethod
    def ppm_to_fraction(cls, value_ppm: float = 1) -> float:
        """Parts-per-million by mole or mass to mole or mass fraction, respectively."""
        return 1 / cls.fraction_to_ppm(value_ppm)

    @classmethod
    def weight_percent_to_ppmw(cls, value_weight_percent: float = 1) -> float:
        """Weight percent to parts-per-million by weight"""
        return value_weight_percent * 1.0e4


def earth_oceans_to_kg(number_of_earth_oceans: float = 1) -> float:
    h_grams: float = number_of_earth_oceans * OCEAN_MOLES * Formula("H2").mass
    h_kg: float = UnitConversion().g_to_kg(h_grams)
    return h_kg
