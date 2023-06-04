"""Oxygen fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)


class OxygenFugacity(ABC):
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


class IronWustiteBufferOneill(OxygenFugacity):
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


class IronWustiteBufferFischer(OxygenFugacity):
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

    Note some AI-generated and should be checked for correctness.

    There is a library that could do this, but it would add a dependency and there is always a
    risk it wouldn't be supported in the future:

    https://pypi.org/project/molmass/
    """

    # Define atoms here.
    # pylint: disable=invalid-name
    H: float = 1.0079e-3
    He: float = 4.0026e-3
    Li: float = 6.941e-3
    Be: float = 9.0122e-3
    B: float = 10.81e-3
    C: float = 12.0107e-3
    N: float = 14.0067e-3
    O: float = 15.9994e-3
    F: float = 18.9984e-3
    Ne: float = 20.1797e-3
    Na: float = 22.9897e-3
    Mg: float = 24.305e-3
    Al: float = 26.9815e-3
    Si: float = 28.0855e-3
    P: float = 30.9738e-3
    S: float = 32.065e-3
    Cl: float = 35.453e-3
    K: float = 39.0983e-3
    Ar: float = 39.948e-3
    Ca: float = 40.078e-3
    Sc: float = 44.9559e-3
    Ti: float = 47.867e-3
    V: float = 50.9415e-3
    Cr: float = 51.9961e-3
    Mn: float = 54.938e-3
    Fe: float = 55.845e-3
    Ni: float = 58.6934e-3
    Co: float = 58.9332e-3
    Cu: float = 63.546e-3
    Zn: float = 65.38e-3
    Ga: float = 69.723e-3
    Ge: float = 72.63e-3
    As: float = 74.9216e-3
    Se: float = 78.96e-3
    Br: float = 79.904e-3
    Kr: float = 83.798e-3
    Rb: float = 85.4678e-3
    Sr: float = 87.62e-3
    Y: float = 88.9059e-3
    Zr: float = 91.224e-3
    Nb: float = 92.9064e-3
    Mo: float = 95.94e-3
    Tc: float = 98.0e-3
    Ru: float = 101.07e-3
    Rh: float = 102.9055e-3
    Pd: float = 106.42e-3
    Ag: float = 107.8682e-3
    Cd: float = 112.411e-3
    In: float = 114.818e-3
    Sn: float = 118.71e-3
    Sb: float = 121.76e-3
    I: float = 126.9045e-3
    Te: float = 127.6e-3
    Xe: float = 131.293e-3

    def __post_init__(self):
        # Define molecules here. Of course, for a given molecule name this could be automatically
        # determined using basic string operations, thereby avoiding this requirement to manually
        # compute the masses of molecules.
        self.CH4: float = self.C + 4 * self.H
        self.CO: float = self.C + self.O
        self.CO2: float = self.C + 2 * self.O
        self.H2: float = self.H * 2
        self.H2O: float = self.H * 2 + self.O
        self.N2: float = self.N * 2
        self.NH3: float = self.N + 3 * self.H
        self.O2: float = self.O * 2
        self.SO2: float = self.S + 2 * self.O
        self.F2: float = self.F * 2
        self.NaCl: float = self.Na + self.Cl
        self.CaCO3: float = self.Ca + self.C + 3 * self.O
        self.KBr: float = self.K + self.Br
        self.MgO: float = self.Mg + self.O
        self.PCl5: float = self.P + 5 * self.Cl
        self.SiO2: float = self.Si + 2 * self.O
        self.SF6: float = self.S + 6 * self.F
        self.Al2O3: float = 2 * self.Al + 3 * self.O
        self.COCl2: float = self.C + self.O + 2 * self.Cl
        self.HF: float = self.H + self.F
        self.HBr: float = self.H + self.Br
        self.SiF4: float = self.Si + 4 * self.F
        self.NaHCO3: float = self.Na + self.H + self.C + 3 * self.O
        self.MgCl2: float = self.Mg + 2 * self.Cl
        self.P2O5: float = 2 * self.P + 5 * self.O
        self.SO3: float = self.S + 3 * self.O
        self.SO4: float = self.S + 4 * self.O
        self.Na2CO3: float = 2 * self.Na + self.C + 3 * self.O
        self.K2SO4: float = 2 * self.K + self.S + 4 * self.O
        self.H2SO4: float = self.H * 2 + self.S + 4 * self.O
        self.KOH: float = self.K + self.O + self.H
        self.NaOH: float = self.Na + self.O + self.H
        self.CaO: float = self.Ca + self.O
        self.P4O10: float = 4 * self.P + 10 * self.O
        self.SiO: float = self.Si + self.O
        self.P4S3: float = 4 * self.P + 3 * self.S
        self.SiS2: float = self.Si + 2 * self.S
        self.Sb2S3: float = 2 * self.Sb + 3 * self.S
        self.As2S3: float = 2 * self.As + 3 * self.S
        self.SnO2: float = self.Sn + 2 * self.O
        self.Sb2O3: float = 2 * self.Sb + 3 * self.O
        self.As2O5: float = 2 * self.As + 5 * self.O
        self.SnCl4: float = self.Sn + 4 * self.Cl
        self.SbCl3: float = self.Sb + 3 * self.Cl
        self.I2: float = self.I * 2
        self.TeO2: float = self.Te + 2 * self.O
        self.XeF4: float = self.Xe + 4 * self.F


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
