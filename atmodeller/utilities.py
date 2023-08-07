"""Utilities."""

import logging

logger: logging.Logger = logging.getLogger(__name__)


class UnitConversion:
    """Unit conversions."""

    @staticmethod
    def bar_to_GPa(value_bar: float) -> float:
        """Bar to GPa."""
        return value_bar * 1.0e-4

    @staticmethod
    def mole_fraction_to_ppm(value_mole_fraction: float) -> float:
        """Mole fraction to parts-per-million."""
        return value_mole_fraction * 1.0e6

    @staticmethod
    def weight_precent_to_ppmw(value_weight_percent: float) -> float:
        """Weight percent to parts-per-million by weight"""
        return value_weight_percent * 1.0e4
