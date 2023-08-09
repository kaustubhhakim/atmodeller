"""Utilities."""

import logging

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
    def ppm_to_fraction(cls, value_ppm: float = 1) -> float:
        """Parts-per-million by mole or mass to mole or mass fraction, respectively."""
        return 1 / cls.fraction_to_ppm(value_ppm)

    @classmethod
    def weight_precent_to_ppmw(cls, value_weight_percent: float = 1) -> float:
        """Weight percent to parts-per-million by weight"""
        return value_weight_percent * 1.0e4
