"""Fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from thermochem import janaf

from atmodeller import DATA_ROOT_PATH, GAS_CONSTANT, GRAVITATIONAL_CONSTANT
from atmodeller.utilities import MolarMasses, UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet.

    Default values are for a reduced (at the Iron-Wustite buffer) and fully molten Earth.

    Args:
        mantle_mass: Mass of the planetary mantle. Defaults to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass. Defaults to
            Earth.
        surface_radius: Radius of the planetary surface. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
        melt_composition: Melt composition of the planet. Default is None.

    Attributes:
        mantle_mass: Mass of the planetary mantle.
        mantle_melt_fraction: mass fraction of the mantle that is molten.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass.
        surface_radius: Radius of the planetary surface.
        surface_temperature: Temperature of the planetary surface.
        melt_composition: Melt composition of the planet.
        planet_mass: Mass of the planet.
        surface_gravity: Gravitational acceleration at the planetary surface.
        surface_area: Surface area of the planetary surface.
    """

    mantle_mass: float = 4.208261222595111e24  # kg, Earth's mantle mass
    mantle_melt_fraction: float = 1.0  # Completely molten
    core_mass_fraction: float = 0.295334691460966  # Earth's core mass fraction
    surface_radius: float = 6371000.0  # m, Earth's radius
    surface_temperature: float = 2000.0  # K
    melt_composition: Union[str, None] = None
    planet_mass: float = field(init=False)
    surface_gravity: float = field(init=False)

    def __post_init__(self):
        logger.info("Creating a new planet")
        self.planet_mass = self.mantle_mass / (1 - self.core_mass_fraction)
        self.surface_gravity = GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2
        logger.info("Mantle mass (kg) = %f", self.mantle_mass)
        logger.info("Mantle melt fraction = %f", self.mantle_melt_fraction)
        logger.info("Core mass fraction = %f", self.core_mass_fraction)
        logger.info("Planetary radius (m) = %f", self.surface_radius)
        logger.info("Planetary mass (kg) = %f", self.planet_mass)
        logger.info("Surface temperature (K) = %f", self.surface_temperature)
        logger.info("Surface gravity (m/s^2) = %f", self.surface_gravity)
        logger.info("Melt Composition = %s", self.melt_composition)

    @property
    def surface_area(self):
        """Surface area of the planet."""
        return 4.0 * np.pi * self.surface_radius**2


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the molecule or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: "Molecule", element: Optional[str] = None, **kwargs) -> float:
        """Wrapper to return the mass of either the molecule or one of its elements.

        Args:
            element: Returns the mass of this element. Defaults to None to return the molecule
                mass.
            **kwargs: Catches keyword arguments to forward to func.

        Returns:
            Mass of either the molecule or element.
        """
        mass: float = func(self, **kwargs)
        if element is not None:
            mass *= self.element_masses.get(element, 0) / self.molar_mass

        return mass

    return mass_wrapper


class Solubility(ABC):
    """Solubility base class."""

    def power_law(self, fugacity: float, constant: float, exponent: float) -> float:
        """Power law. Fugacity in bar and returns ppmw."""
        return constant * fugacity**exponent

    @abstractmethod
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        raise NotImplementedError

    def __call__(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Dissolved volatile concentration in ppmw in the melt."""
        solubility: float = self._solubility(fugacity, temperature, fugacities_dict)
        logger.debug(
            "%s, f = %f, T = %f, ppmw = %f",
            self.__class__.__name__,
            fugacity,
            temperature,
            solubility,
        )
        return solubility


class NoSolubility(Solubility):
    """No solubility."""

    def _solubility(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return 0.0


@dataclass(kw_only=True, frozen=True)
class MoleculeOutput:
    """Output for a molecule."""

    mass_in_atmosphere: float  # kg
    mass_in_solid: float  # kg
    mass_in_melt: float  # kg
    ppmw_in_solid: float  # ppm by weight
    ppmw_in_melt: float  # ppm by weight
    pressure_in_atmosphere: float  # bar

    @property
    def mass_in_total(self):
        """Total mass in all reservoirs in kg."""
        return self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


@dataclass(kw_only=True)
class Molecule:
    """A molecule and its properties.

    Args:
        name: Chemical formula of the molecule.
        solubility: Solubility model. Defaults to no solubility.
        solid_melt_distribution_coefficient: Distribution coefficient. Defaults to 0.

    Attributes:
        name: Chemical formula of the molecule.
        solubility: Solubility model.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
        elements: The elements and their (stoichiometric) counts in the molecule.
        element_masses: The elements and their total masses in the molecule.
        molar_mass: Molar mass of the molecule.
        output: To store calculated values for output.
    """

    name: str
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0
    elements: dict[str, int] = field(init=False)
    element_masses: dict[str, float] = field(init=False)
    molar_mass: float = field(init=False)
    output: MoleculeOutput | None = field(init=False, default=None)

    def __post_init__(self):
        logger.info("Creating a molecule: %s", self.name)
        masses: MolarMasses = MolarMasses()
        self.elements = self._count_elements()
        self.element_masses = {
            key: value * getattr(masses, key) for key, value in self.elements.items()
        }
        self.molar_mass = sum(self.element_masses.values())

    @property
    def is_diatomic(self) -> bool:
        """Is the molecule diatomic.

        Useful for obtaining the appropriate JANAF data for the Gibbs free energy.
        """
        if len(self.elements) == 1 and list(self.elements.values())[0] == 2:
            return True
        else:
            return False

    def _count_elements(self) -> dict[str, int]:
        """Counts the number of atoms.

        Returns:
            A dictionary of the elements and their stoichiometric counts.
        """
        element_count: dict[str, int] = {}
        current_element: str = ""
        current_count: str = ""

        for char in self.name:
            if char.isupper():
                if current_element != "":
                    count = int(current_count) if current_count else 1
                    element_count[current_element] = element_count.get(current_element, 0) + count
                    current_count = ""
                current_element = char
            elif char.islower():
                current_element += char
            elif char.isdigit():
                current_count += char

        if current_element != "":
            count: int = int(current_count) if current_count else 1
            element_count[current_element] = element_count.get(current_element, 0) + count
        logger.debug("element count = \n%s", element_count)
        return element_count

    @_mass_decorator
    def mass(
        self,
        *,
        planet: Planet,
        partial_pressure_bar: float,
        atmosphere_mean_molar_mass: float,
        fugacities_dict: dict[str, float],
        element: Optional[str] = None,
    ) -> float:
        """Total mass.

        Args:
            planet: Planet properties.
            partial_pressure_bar: Partial pressure in bar.
            atmosphere_mean_molar_mass: Mean molar mass of the atmosphere.
            fugacities_dict: Dictionary of all the species and their partial pressures.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the molecule (element=None) or element (element=element).
        """

        del element

        # Atmosphere.
        mass_in_atmosphere: float = (
            UnitConversion.bar_to_Pa(partial_pressure_bar) / planet.surface_gravity
        )
        mass_in_atmosphere *= planet.surface_area * self.molar_mass / atmosphere_mean_molar_mass

        # Melt.
        prefactor: float = planet.mantle_mass * planet.mantle_melt_fraction
        ppmw_in_melt: float = self.solubility(
            partial_pressure_bar,
            planet.surface_temperature,
            fugacities_dict,
        )
        mass_in_melt = prefactor * ppmw_in_melt * UnitConversion.ppm_to_fraction()

        # Solid.
        prefactor: float = planet.mantle_mass * (1 - planet.mantle_melt_fraction)
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass_in_solid = prefactor * ppmw_in_solid * UnitConversion.ppm_to_fraction()

        self.output = MoleculeOutput(
            mass_in_atmosphere=mass_in_atmosphere,
            mass_in_solid=mass_in_solid,
            mass_in_melt=mass_in_melt,
            ppmw_in_solid=ppmw_in_solid,
            ppmw_in_melt=ppmw_in_melt,
            pressure_in_atmosphere=partial_pressure_bar,
        )

        return self.output.mass_in_total


class BufferedFugacity(ABC):
    """Buffered fugacity base class."""

    @abstractmethod
    def _fugacity(self, *, temperature: float) -> float:
        """Log10(fugacity) of the buffer in terms of temperature.

        Args:
            temperature: Temperature.

        Returns:
            Log10 of the fugacity.
        """
        raise NotImplementedError

    def __call__(self, *, temperature: float, fugacity_log10_shift: float = 0) -> float:
        """log10(fugacity) plus an optional shift.

        Args:
            temperature: Temperature.
            fugacity_log10_shift: Log10 shift.

        Returns:
            Log10 of the fugacity including a shift.
        """
        return self._fugacity(temperature=temperature) + fugacity_log10_shift


class IronWustiteBufferHirschmann(BufferedFugacity):
    """Iron-wustite buffer (fO2) from O'Neill and Pownceby (1993) and Hirschmann et al. (2008)."""

    def _fugacity(self, *, temperature: float) -> float:
        """See base class."""
        # TODO: Assuming one bar (total?) pressure
        total_pressure: float = 1  # bar
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (total_pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )
        return fugacity


class IronWustiteBufferOneill(BufferedFugacity):
    """Iron-wustite buffer (fO2) from O'Neill and Eggins (2002). See Table 6.

    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract
    """

    def _fugacity(self, *, temperature: float) -> float:
        """See base class."""
        fugacity: float = (
            2
            * (-244118 + 115.559 * temperature - 8.474 * temperature * np.log(temperature))
            / (np.log(10) * GAS_CONSTANT * temperature)
        )
        return fugacity


class IronWustiteBufferFischer(BufferedFugacity):
    """Iron-wustite buffer (fO2) from Fischer et al. (2011). See Table S2 in supplementary materials.

    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract
    """

    def _fugacity(self, *, temperature: float) -> float:
        """See base class."""
        # Collapsed polynomial since it is evaluated at P=0 GPa (i.e. no pressure dependence).
        buffer: float = 6.44059 - 28.1808 * 1e3 / temperature
        return buffer


class StandardGibbsFreeEnergyOfFormation(ABC):
    """Standard Gibbs free energy of formation base class."""

    def __init__(self):
        self.data: pd.DataFrame = self._read_thermodynamic_data()

    @abstractmethod
    def _read_thermodynamic_data(self) -> pd.DataFrame:
        """Reads and returns the thermodynamic data."""

    @abstractmethod
    def get(self, molecule: Molecule, *, temperature: float) -> float:
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

    def get(self, molecule: Molecule, *, temperature: float) -> float:
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
            formation_constants: tuple[float, float] = tuple(self.data.loc[molecule.name].tolist())
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


class StandardGibbsFreeEnergyOfFormationJANAF(StandardGibbsFreeEnergyOfFormation):
    """Standard Gibbs free energy of formation from the JANAF tables."""

    def _read_thermodynamic_data(self) -> pd.DataFrame:
        """Data is downloaded when required."""
        ...

    def get(self, molecule: Molecule, *, temperature: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol.

        In the JANAF tables, we have generally chosen the ideal diatomic gas for the reference
        state of permanent gases such as O2, N2, Cl2 etc. (quoting from the JANAF documentation).

        Returns:
            The standard Gibbs free energy of formation.
        """

        db = janaf.Janafdb()
        if molecule.is_diatomic:
            phase = db.getphasedata(formula=molecule.name, phase="ref")
        else:
            phase = db.getphasedata(formula=molecule.name, phase="g")

        logger.debug("Phase = %s", phase)
        gibbs: float = phase.DeltaG(temperature)

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

    def get(self, molecule: Molecule, *, temperature: float) -> float:
        """Gets the standard Gibbs free energy of formation in J/mol

        Args:
            molecule: Molecule.
            temperature: Temperature.

        Returns:
            The standard Gibbs free energy of formation.
        """
        try:
            data: pd.Series = self.data.loc[molecule.name]
        except KeyError:
            logger.error("Thermodynamic data not available for %s", molecule.name)
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
        logger.debug(
            "Molecule = %s, standard Gibbs energy of formation = %f", molecule.name, gibbs
        )

        return gibbs


class AnorthiteDiopsideH2O(Solubility):
    """Newcombe et al. (2017).

    https://ui.adsabs.harvard.edu/abs/2017GeCoA.200..330N/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 727, 0.5)


class PeridotiteH2O(Solubility):
    """Sossi et al. (2023).

    https://ui.adsabs.harvard.edu/abs/2023E%26PSL.60117894S/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 524, 0.5)


class BasaltDixonH2O(Solubility):
    """Dixon et al. (1995) refit by Paolo Sossi."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 965, 0.5)


class BasaltWilsonH2O(Solubility):
    """Hamilton (1964) and Wilson and Head (1981)."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 215, 0.7)


class LunarGlassH2O(Solubility):
    """Newcombe et al. (2017).

    https://ui.adsabs.harvard.edu/abs/2017GeCoA.200..330N/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 683, 0.5)


class BasaltDixonCO2(Solubility):
    """Dixon et al. (1995)."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del fugacities_dict
        ppmw: float = (3.8e-7) * fugacity * np.exp(-23 * (fugacity - 1) / (83.15 * temperature))
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)
        return ppmw


class BasaltLibourelN2(Solubility):
    """Libourel et al. (2003), basalt (tholeiitic) magmas.

    Eq. 23, includes dependence on pressure and fO2.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        ppmw: float = self.power_law(fugacity, 0.0611, 1.0)
        # TODO: Could add fO2 lower and upper bounds.
        if "O2" in fugacities_dict:
            # TODO: Confirm fO2 and not log10fO2 or lnfO2?
            constant: float = (fugacities_dict["O2"] ** -0.75) * 5.97e-10
            ppmw += self.power_law(fugacity, constant, 0.5)
        return ppmw


class BasaltH2(Solubility):
    """Hirschmann et al. 2012 for Basalt."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Power law fit to Figure 5, basalt Pure H2 curve."""
        del temperature
        del fugacities_dict
        # TODO: Maggie to check, variable is not currently used.
        pressure_gpa: float = UnitConversion.bar_to_GPa(
            fugacity
        )  # pylint: disable=unused-variable
        # Fitting coefficients, determined in solubility_fits.ipynb
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = self.power_law(fugacity, 6479.75, 1.20)
        return ppm

    def _solubility_v2(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Taking fit from Fig. 4 for Basalt (with fH2(P) fitted from Tables 1 and 2)."""
        del temperature
        del fugacities_dict
        pressure_gpa: float = UnitConversion.bar_to_GPa(fugacity)
        fh2 = self.power_law(pressure_gpa, 7458.81, 2.01)  # bars; power-law fit
        molefrac: float = np.exp(-11.403 - (0.76 * pressure_gpa)) * fh2
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = UnitConversion.fraction_to_ppm(
            molefrac
        )  # CHECK, is there an extra step to make this ppmw?
        return ppm


class AndesiteH2(Solubility):
    """Hirschmann et al. 2012.

    Using the fit from Fig. 4 for Andesite (with fH2(P) fitted from Tables 1 and 2).
    """

    def _solubility(self, fugacity: float, temperature: float, fugacities_dict: float) -> float:
        del temperature
        del fugacities_dict
        pressure_gpa: float = UnitConversion.bar_to_GPa(fugacity)
        fh2 = self.power_law(pressure_gpa, 7856.31, 2.17)  # bars; power-law fit
        molefrac: float = np.exp(-10.591 - (0.81 * pressure_gpa)) * fh2
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = UnitConversion.fraction_to_ppm(
            molefrac
        )  # CHECK, is there an extra step to make this ppmw?
        return ppm


class PeridotiteH2(Solubility):
    """Hirschmann et al. 2012 for Peridotite.

    Fitting power law to Figure 5, Peridotite Pure H2 curve.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        # TODO: Maggie to check, variable is not currently used.
        pressure_gpa: float = UnitConversion.bar_to_GPa(
            fugacity
        )  # pylint: disable=unused-variable
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = self.power_law(fugacity, 1722.31, 1.03)
        return ppm


class ObsidianH2(Solubility):
    """Gaillard et al. 2003.

    Valid for pressures from 0.02-70 bar; power law fit to Table 4 data.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        ppmw: float = self.power_law(fugacity, 0.163, 1.252)
        return ppmw


class AndesiteSO2(Solubility):
    """Boulliung & Wood 2022.

    Fitting S (ppm) vs. Temperature.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del fugacity
        del fugacities_dict
        # from Table 3, least squares linear fit.
        temperature_factor, constant = (-0.29028571428571454, 528.3908571428574)
        # TODO: Maggie to check, ppm or ppmw? Probably use ppmw to be explicit if by weight.
        ppm: float = (temperature_factor * temperature) + constant
        return ppm


class BasaltSO2(Solubility):
    """Boulliung & Wood 2022.

    Fitting S (ppm) vs. Temperature.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del fugacity
        del fugacities_dict
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

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        a, b, c, d = [7.25, -2.54e4, 0.04, -0.551]  # Coeffs from eq. 10 (Namur et al., 2016).
        # TODO: Confirm fO2 and not log10fO2 or lnfO2?
        # FIXME: How to deal if fO2 not available?  Drop last term?
        wt_perc: float = np.exp(
            a + (b / temperature) + ((c * fugacity) / temperature) + (d * fugacities_dict["O2"])
        )
        ppmw: float = UnitConversion.weight_precent_to_ppmw(wt_perc)
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
