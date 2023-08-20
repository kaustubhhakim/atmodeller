"""Utilities."""

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


# TODO: Replace with molmass.
# @dataclass
# class MolarMasses:
#     """Molar masses of atoms and molecules in kg/mol.

#     Note some AI-generated and should be checked for correctness.

#     There is a library that could do this, but it would add a dependency and there is always a
#     risk it wouldn't be supported in the future:

#     https://pypi.org/project/molmass/
#     """

#     # Define atoms here.
#     H: float = 1.0079e-3
#     He: float = 4.0026e-3
#     Li: float = 6.941e-3
#     Be: float = 9.0122e-3
#     B: float = 10.81e-3
#     C: float = 12.0107e-3
#     N: float = 14.0067e-3
#     O: float = 15.9994e-3
#     F: float = 18.9984e-3
#     Ne: float = 20.1797e-3
#     Na: float = 22.9897e-3
#     Mg: float = 24.305e-3
#     Al: float = 26.9815e-3
#     Si: float = 28.0855e-3
#     P: float = 30.9738e-3
#     S: float = 32.065e-3
#     Cl: float = 35.453e-3
#     K: float = 39.0983e-3
#     Ar: float = 39.948e-3
#     Ca: float = 40.078e-3
#     Sc: float = 44.9559e-3
#     Ti: float = 47.867e-3
#     V: float = 50.9415e-3
#     Cr: float = 51.9961e-3
#     Mn: float = 54.938e-3
#     Fe: float = 55.845e-3
#     Ni: float = 58.6934e-3
#     Co: float = 58.9332e-3
#     Cu: float = 63.546e-3
#     Zn: float = 65.38e-3
#     Ga: float = 69.723e-3
#     Ge: float = 72.63e-3
#     As: float = 74.9216e-3
#     Se: float = 78.96e-3
#     Br: float = 79.904e-3
#     Kr: float = 83.798e-3
#     Rb: float = 85.4678e-3
#     Sr: float = 87.62e-3
#     Y: float = 88.9059e-3
#     Zr: float = 91.224e-3
#     Nb: float = 92.9064e-3
#     Mo: float = 95.94e-3
#     Tc: float = 98.0e-3
#     Ru: float = 101.07e-3
#     Rh: float = 102.9055e-3
#     Pd: float = 106.42e-3
#     Ag: float = 107.8682e-3
#     Cd: float = 112.411e-3
#     In: float = 114.818e-3
#     Sn: float = 118.71e-3
#     Sb: float = 121.76e-3
#     I: float = 126.9045e-3
#     Te: float = 127.6e-3
#     Xe: float = 131.293e-3

#     def __post_init__(self):
#         # This is for convenience, since the number of moles in Earth's ocean are given in terms of
#         # H2. In which case, the mass of H2 is useful to have direct access to in order to compute
#         # the mass of H in an Earth ocean.
#         self.H2: float = 2 * self.H
