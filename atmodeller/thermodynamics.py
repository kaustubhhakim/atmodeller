"""Oxygen fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

#For unit conversion commonly used in Solubility class:
bar_to_GPa: float = 0.0001 #bar/GPa 
molefrac_to_ppm: float = 1e6 #ppm/molefrac 
wtperc_to_ppm : float = 1e4 #weight percent (wt. %) to ppm

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
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
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
        # This is for convenience, since the number of moles in Earth's ocean are given in terms of
        # H2. In which case, the mass of H2 is useful to have direct access to in order to compute
        # the mass of H in an Earth ocean.
        self.H2: float = 2 * self.H


@dataclass(frozen=True)
class GibbsConstants:
    """Standard Gibbs free energy of formation fitting constants.

    These parameters result from a linear fit in temperature space to the delta-f G column in the
    JANAF data tables for a given molecule. See the jupyter notebook in 'docs/'.

    G = aT + b

    where a = -S (standard entropy of formation) and b = H (standard enthalpy of formation).

    In the future we could use the Shomate equation to calculate the equilibrium of the gas phase
    reactions.
    """

    # Want to use molecule names therefore pylint: disable=invalid-name
    C: tuple[float, float] = (0, 0)
    CH4: tuple[float, float] = (0.11162272058823527, -92.46793382352928)
    CO: tuple[float, float] = (-0.08269582352941174, -120.3571470588237)
    CO2: tuple[float, float] = (0.0005482647058825124, -397.33434558823564)
    H2: tuple[float, float] = (0, 0)
    H2O: tuple[float, float] = (0.05817258823529415, -251.80119852941178)
    N2: tuple[float, float] = (0, 0)
    NH3: tuple[float, float] = (0.11667370588235293, -54.026338235293984)
    O2: tuple[float, float] = (0, 0)
    SO: tuple[float, float] = (-0.004834249999999983, -59.61862500000004)
    SO2: tuple[float, float] = (0.0725481764705883, -361.0377720588236)
    S2: tuple[float, float] = (0, 0)
    H2S: tuple[float, float] = (0.04955845588235296, -90.57971323529412)
    S: tuple[float, float] = (-0.06120597058823535, 217.87218382352947)
    Cl: tuple[float, float] = (-0.06116926470588236, 127.34372058823527)
    Cl2: tuple[float, float] = (0, 0)
    HCl: tuple[float, float] = (-0.005433338235294086, -95.73480147058832)
    F: tuple[float, float] = (-0.06466483823529413, 84.63857352941173)
    F2: tuple[float, float] = (0, 0)
    HF: tuple[float, float] = (-0.0015432205882352377, -277.91656617647067)

class Solubility(ABC):
    """Solubility base class."""

    def power_law(self, pressure: float, constant: float, exponent: float) -> float:
        """Power law. Pressure in bar and returns ppmw."""
        return constant * pressure**exponent

    @abstractmethod
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        #Note: fo2 is in log10fo2
        raise NotImplementedError

    def __call__(self, pressure: float, temperature: float, fo2: float, *args) -> float:
        """Dissolved volatile concentration in ppmw in the melt."""
        return self._solubility(pressure, temperature, fo2, *args)


class NoSolubility(Solubility):
    """No solubility."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del pressure
        del temperature
        del fo2
        return 0.0


class AnorthiteDiopsideH2O(Solubility):
    """Newcombe et al. (2017)."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        return self.power_law(pressure, 727, 0.5)


class PeridotiteH2O(Solubility):
    """Sossi et al. (2022)."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        return self.power_law(pressure, 534, 0.5)


class BasaltDixonH2O(Solubility):
    """Dixon et al. (1995) refit by Paolo Sossi."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        return self.power_law(pressure, 965, 0.5)


class BasaltWilsonH2O(Solubility):
    """Hamilton (1964) and Wilson and Head (1981)."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        return self.power_law(pressure, 215, 0.7)


class LunarGlassH2O(Solubility):
    """Newcombe et al. (2017)."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        return self.power_law(pressure, 683, 0.5)


class BasaltDixonCO2(Solubility):
    """Dixon et al. (1995)."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del fo2
        ppmw: float = (3.8e-7) * pressure * np.exp(-23 * (pressure - 1) / (83.15 * temperature))
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)
        return ppmw


class BasaltLibourelN2(Solubility):
    """Libourel et al. (2003), basalt (tholeiitic) magmas"""
    #Eq. 23, includes dependence on pressure and fO2:
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature 
        ppmw: float = (0.0611*pressure)+(((10**fo2)**-0.75)*5.97e-10*(pressure**0.5)) 
        return ppmw
    #Eq. 19 for relatively oxidizing conditions (air to IW), only has pressure dependence 
    #def _solubility(self, pressure: float, temperature: float) -> float:
    #    del temperature
    #    ppmw: float = self.power_law(pressure, 0.0611, 1.0)
    #    return ppmw

class BasaltH2(Solubility):
    """Hirschmann et al. 2012 for Basalt"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        """Power law fit to Figure 5, basalt Pure H2 curve"""
        del temperature
        del fo2
        pressure_GPa: float = pressure*bar_to_GPa
        #Fitting coefficients, determined in solubility_fits.ipynb
        ppm: float = self.power_law(pressure, 6479.75, 1.20) 
        return ppm

    def _solubility_v2(self, pressure: float, temperature: float, fo2: float) -> float:
        """Taking fit from Fig. 4 for Basalt (with fH2(P) fitted from Tables 1 and 2)"""
        del temperature
        del fo2
        pressure_GPa: float = pressure*bar_to_GPa
        fH2 = self.power_law(pressure_GPa, 7458.81, 2.01) #bars; power-law fit 
        molefrac: float = np.exp(-11.403-(0.76*pressure_GPa))*fH2
        ppm: float = molefrac* molefrac_to_ppm #CHECK, is there an extra step to make this ppmw?
        return ppm


class AndesiteH2(Solubility):
    """Hirschmann et al. 2012, Using the fit from Fig. 4 for Andesite (with fH2(P) fitted from Tables 1 and 2)"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature 
        del fo2
        pressure_GPa: float = pressure*bar_to_GPa
        fH2 = self.power_law(pressure_GPa, 7856.31, 2.17) #bars; power-law fit 
        molefrac: float = np.exp(-10.591-(0.81*pressure_GPa))*fH2
        ppm: float = molefrac* molefrac_to_ppm #CHECK, is there an extra step to make this ppmw?
        return ppm
    
class PeridotiteH2(Solubility):
    """Hirschmann et al. 2012 for Peridotite: Fitting power law to Figure 5, Peridotite Pure H2 curve"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        pressure_GPa: float = pressure*bar_to_GPa
        ppm: float = self.power_law(pressure, 1722.31, 1.03)
        return ppm
    
class ObsidianH2(Solubility):
    """Gaillard et al. 2003, valid for pressures from 0.02-70 bar; power law fit to Table 4 data"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        ppmw: float = self.power_law(pressure, 0.163, 1.252) 
        return ppmw 
    
class AndesiteSO2(Solubility):
    """Boulliung & Wood 2022, Fitting S (ppm) vs. Temperature"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del pressure
        del fo2
        a, b = [-0.29028571428571454, 528.3908571428574] #from Table 3, least squares linear fit
        ppm: float = (a*temperature) + b
        return ppm
    
class BasaltSO2(Solubility):
    """Boulliung & Wood 2022, Fitting S (ppm) vs. Temperature"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del pressure
        del fo2
        ppm = 0.25 * np.exp(1.2249*(-1.1 - 5.5976 - (24505/temperature) + (0.8099*np.log10(temperature)))) #Fit from Figure 3, using FMQ(temperature) from O'Neill 1987a
        return ppm
    
class MercuryMagmaS(Solubility):
    """Namur et al. 2016, S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like magmas; Check, I think this would mainly apply to H2S but maybe also S2 and S"""
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        a, b, c, d  = [7.25, -2.54e4, 0.04, -0.551] #coefficients from eq. 10 of Namur+2016
        wt_perc = np.exp(a + (b/temperature)+ ((c*pressure)/temperature) + (d*fo2))
        ppmw = wt_perc * wtperc_to_ppm 
        return ppmw 
   
basalt_container: dict = {'H2O':BasaltDixonH2O(), 'CO2':BasaltDixonCO2(), 'H2': BasaltH2(), 'N2': BasaltLibourelN2(), 'SO2': BasaltSO2()}
andesite_container: dict = {'H2':AndesiteH2(), 'SO2':AndesiteSO2()}
peridotite_container: dict = {'H2O':PeridotiteH2O(), 'H2':PeridotiteH2()}
anorthdiop_containter: dict = {'H2O': AnorthiteDiopsideH2O()}
reducedmagma_container: dict = {'H2S': MercuryMagmaS()}

master_container: dict = {'basalt':basalt_container, 'andesite':andesite_container, 'peridotite':peridotite_container, 'anorthiteDiopsideEuctectic': anorthdiop_containter, 'reducedmagma': reducedmagma_container}



