"""Oxygen fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from atmodeller import DATA_ROOT_PATH

# For unit conversions.
bar_to_GPa: float = 0.0001  # bar/GPa
molefrac_to_ppm: float = 1e6  # ppm/molefrac
wtperc_to_ppm: float = 1e4  # weight percent (wt. %) to ppm

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


class StandardGibbsFreeEnergyOfFormation(ABC):
    """Standard Gibbs free energy of formation base class."""

    def __init__(self):
        self.data: pd.DataFrame = self._read_thermodynamic_data()

    @abstractmethod
    def _read_thermodynamic_data(self) -> pd.DataFrame:
        """Reads and returns the thermodynamic data."""

    @abstractmethod
    def get(self, molecule: str, *, temperature: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol"""


class StandardGibbsFreeEnergyOfFormationLinear(StandardGibbsFreeEnergyOfFormation):
    """Standard Gibbs free energy of formation from a linear fit of JANAF data wrt. temperature.

    See the comments in the data file that is parsed by __init__
    """

    # Temperature range used to fit the JANAF data.
    TEMPERATURE_HIGH: float = 3000  # K
    TEMPERATURE_LOW: float = 1500  # K

    def _read_thermodynamic_data(self) -> pd.DataFrame:
        data_path: Path = DATA_ROOT_PATH / Path("gibbs_linear.csv")  # type: ignore
        data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        data.set_index("species", inplace=True)
        data = data.astype(float)
        return data

    def get(self, molecule: str, *, temperature: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol.

        G = aT + b
        where a = -S (standard entropy of formation) and b = H (standard enthalpy of formation).

        Args:
            molecule: Molecule.
            temperature: Temperature.

        Returns:
            The standard Gibbs free energy of formation.
        """
        try:
            formation_constants: tuple[float, float] = tuple(self.data.loc[molecule].tolist())
        except KeyError:
            logger.error("Thermodynamic data not available for %s", molecule)
            raise

        if (temperature < self.TEMPERATURE_LOW) or (temperature > self.TEMPERATURE_HIGH):
            msg: str = f"Temperature must be in the range {self.TEMPERATURE_LOW} K to "
            msg += f"{self.TEMPERATURE_HIGH} K"
            raise ValueError(msg)

        gibbs: float = formation_constants[0] * temperature + formation_constants[1]
        gibbs *= 1000  # To convert from kJ to J.
        logger.debug("Molecule = %s, standard Gibbs energy of formation = %f", molecule, gibbs)

        return gibbs


class StandardGibbsFreeEnergyOfFormationHolland(StandardGibbsFreeEnergyOfFormation):
    """Standard Gibbs free energy of formation from Holland and Powell (1998).

    See the comments in the data file that is parsed by __init__
    """

    def _read_thermodynamic_data(self) -> pd.DataFrame:
        data_path: Path = DATA_ROOT_PATH / Path("Mindata161127.csv")  # type: ignore
        data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        data["name of phase component"] = data["name of phase component"].str.strip()
        data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        data.drop(columns="Abbreviation", inplace=True)
        data.set_index("name of phase component", inplace=True)
        data = data.loc[:, :"Vmax"]
        data = data.astype(float)
        return data

    def get(self, molecule: str, *, temperature: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol

        Args:
            molecule: Molecule.
            temperature: Temperature.

        Returns:
            The standard Gibbs free energy of formation.
        """
        try:
            data: pd.Series = self.data.loc[molecule]
        except KeyError:
            logger.error("Thermodynamic data not available for %s", molecule)
            raise

        temp_ref: float = 298  # K

        H = data.get("Hf")  # J
        S = data.get("S")  # J/K
        a = data.get("a")  # J/K           Coeff for calc heat capacity.
        b = data.get("b")  # J/K^2         Coeff for calc heat capacity.
        c = data.get("c")  # J K           Coeff for calc heat capacity.
        d = data.get("d")  # J K^(-1/2)    Coeff for calc heat capacity.

        integral_H: float = (
            H
            + a * (temperature - temp_ref)  # type: ignore a is a float.
            + b / 2 * (temperature**2 - temp_ref**2)  # type: ignore b is a float.
            - c * (1 / temperature - 1 / temp_ref)  # type: ignore c is a float.
            + 2 * d * (temperature**0.5 - temp_ref**0.5)  # type: ignore d is a float.
        )
        integral_S: float = (
            S
            + a * np.log(temperature / temp_ref)  # type: ignore a is a float.
            + b * (temperature - temp_ref)  # type: ignore b is a float.
            - c / 2 * (1 / temperature**2 - 1 / temp_ref**2)  # type: ignore c is a float.
            - 2 * d * (1 / temperature**0.5 - 1 / temp_ref**0.5)  # type: ignore d is a float.
        )

        gibbs: float = integral_H - temperature * integral_S
        logger.debug("Molecule = %s, standard Gibbs energy of formation = %f", molecule, gibbs)

        return gibbs


class Solubility(ABC):
    """Solubility base class."""

    def power_law(self, pressure: float, constant: float, exponent: float) -> float:
        """Power law. Pressure in bar and returns ppmw."""
        return constant * pressure**exponent

    @abstractmethod
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        # Note: fo2 is in log10fo2.
        raise NotImplementedError

    def __call__(self, pressure: float, temperature: float, fo2: float, *args) -> float:
        """Dissolved volatile concentration in ppmw in the melt."""
        return self._solubility(pressure, temperature, fo2, *args)


class NoSolubility(Solubility):
    """No solubility."""

    def _solubility(self, *args) -> float:
        del args
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
    """Libourel et al. (2003), basalt (tholeiitic) magmas."""

    # Eq. 23, includes dependence on pressure and fO2:
    # def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
    #     del temperature
    #     ppmw: float = (0.0611 * pressure) + (((10**fo2) ** -0.75) * 5.97e-10 * (pressure**0.5))
    #     return ppmw

    # TODO: Dan reinstated the law with fO2 as a temporary measure until we integrate fO2 into the
    # workflow.
    # Eq. 19 for relatively oxidizing conditions (air to IW), only has pressure dependence
    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        ppmw: float = self.power_law(pressure, 0.0611, 1.0)
        return ppmw


class BasaltH2(Solubility):
    """Hirschmann et al. 2012 for Basalt."""

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        """Power law fit to Figure 5, basalt Pure H2 curve."""
        del temperature
        del fo2
        # TODO: Maggie to check, variable is not currently used.
        pressure_gpa: float = pressure * bar_to_GPa  # pylint: disable=unused-variable
        # Fitting coefficients, determined in solubility_fits.ipynb
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = self.power_law(pressure, 6479.75, 1.20)
        return ppm

    def _solubility_v2(self, pressure: float, temperature: float, fo2: float) -> float:
        """Taking fit from Fig. 4 for Basalt (with fH2(P) fitted from Tables 1 and 2)."""
        del temperature
        del fo2
        pressure_gpa: float = pressure * bar_to_GPa
        fh2 = self.power_law(pressure_gpa, 7458.81, 2.01)  # bars; power-law fit
        molefrac: float = np.exp(-11.403 - (0.76 * pressure_gpa)) * fh2
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = molefrac * molefrac_to_ppm  # CHECK, is there an extra step to make this ppmw?
        return ppm


class AndesiteH2(Solubility):
    """Hirschmann et al. 2012.

    Using the fit from Fig. 4 for Andesite (with fH2(P) fitted from Tables 1 and 2).
    """

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        pressure_gpa: float = pressure * bar_to_GPa
        fh2 = self.power_law(pressure_gpa, 7856.31, 2.17)  # bars; power-law fit
        molefrac: float = np.exp(-10.591 - (0.81 * pressure_gpa)) * fh2
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = molefrac * molefrac_to_ppm  # CHECK, is there an extra step to make this ppmw?
        return ppm


class PeridotiteH2(Solubility):
    """Hirschmann et al. 2012 for Peridotite.

    Fitting power law to Figure 5, Peridotite Pure H2 curve.
    """

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        # TODO: Maggie to check, variable is not currently used.
        pressure_gpa: float = pressure * bar_to_GPa  # pylint: disable=unused-variable
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = self.power_law(pressure, 1722.31, 1.03)
        return ppm


class ObsidianH2(Solubility):
    """Gaillard et al. 2003.

    Valid for pressures from 0.02-70 bar; power law fit to Table 4 data.
    """

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del temperature
        del fo2
        ppmw: float = self.power_law(pressure, 0.163, 1.252)
        return ppmw


class AndesiteSO2(Solubility):
    """Boulliung & Wood 2022.

    Fitting S (ppm) vs. Temperature.
    """

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del pressure
        del fo2
        # from Table 3, least squares linear fit.
        temperature_factor, constant = [-0.29028571428571454, 528.3908571428574]
        ppm: float = (temperature_factor * temperature) + constant
        return ppm


class BasaltSO2(Solubility):
    """Boulliung & Wood 2022.

    Fitting S (ppm) vs. Temperature.
    """

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        del pressure
        del fo2
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = 0.25 * np.exp(
            1.2249 * (-1.1 - 5.5976 - (24505 / temperature) + (0.8099 * np.log10(temperature)))
        )  # Fit from Figure 3, using FMQ(temperature) from O'Neill 1987a
        return ppm


class MercuryMagmaS(Solubility):
    """Namur et al. 2016.

    S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like magmas.
    """

    # TODO: Maggie to check, I think this would mainly apply to H2S but maybe also S2 and S.

    def _solubility(self, pressure: float, temperature: float, fo2: float) -> float:
        # pylint: disable=invalid-name
        a, b, c, d = [7.25, -2.54e4, 0.04, -0.551]  # coefficients from eq. 10 of Namur+2016
        wt_perc: float = np.exp(a + (b / temperature) + ((c * pressure) / temperature) + (d * fo2))
        ppmw: float = wt_perc * wtperc_to_ppm
        return ppmw


# Dictionaries of self-consistent solubility laws for a given composition.
andesite_solubilities: dict[str, Solubility] = {"H2": AndesiteH2(), "SO2": AndesiteSO2()}
anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltDixonH2O(),
    "CO2": BasaltDixonCO2(),
    "H2": BasaltH2(),
    "N2": BasaltLibourelN2(),
    "SO2": BasaltSO2(),
}
peridotite_solubilities: dict[str, Solubility] = {"H2O": PeridotiteH2O(), "H2": PeridotiteH2()}
reducedmagma_solubilities: dict[str, Solubility] = {"H2S": MercuryMagmaS()}

# Dictionary of all the composition solubilities. Lowercase key name by convention. All of the
# dictionaries with self-consistent solubility laws for a given composition (above) should be
# included in this dictionary.
# TODO: Dan, auto-assemble this dictionary rather than required the user to add?
composition_solubilities: dict[str, dict[str, Solubility]] = {
    "basalt": basalt_solubilities,
    "andesite": andesite_solubilities,
    "peridotite": peridotite_solubilities,
    "anorthiteDiopsideEuctectic": anorthdiop_solubilities,
    "reducedmagma": reducedmagma_solubilities,
}
