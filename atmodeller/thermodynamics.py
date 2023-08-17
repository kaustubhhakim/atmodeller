"""Fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Protocol, Union

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


class Solubility(ABC):
    """Solubility base class."""

    def power_law(self, fugacity: float, constant: float, exponent: float) -> float:
        """Power law. Fugacity in bar and returns ppmw."""
        return constant * fugacity**exponent

    @abstractmethod
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Dissolved volatile concentration in ppmw in the melt.

        Args:
            fugacity: Fugacity of the species.
            temperature: Temperature.
            fugacities_dict: Fugacities of all other species in the system.

        Returns:
            ppmw of the species in the melt.
        """
        raise NotImplementedError

    def __call__(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Dissolved volatile concentration in ppmw in the melt.

        See self._solubility.
        """
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


@dataclass(kw_only=True)
class MoleculeOutput:
    """Output for a solution."""

    mass_in_atmosphere: float  # kg
    mass_in_solid: float  # kg
    mass_in_melt: float  # kg
    moles_in_atmosphere: float
    moles_in_melt: float
    moles_in_solid: float
    ppmw_in_solid: float  # ppm by weight
    ppmw_in_melt: float  # ppm by weight
    pressure_in_atmosphere: float  # bar
    volume_mixing_ratio: float
    mass_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the molecule or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: "GasPhase", element: Optional[str] = None, **kwargs) -> float:
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


@dataclass(kw_only=True)
class PhaseProtocol(Protocol):
    """A thermodynamic phase."""

    name: str
    phase: str


@dataclass(kw_only=True)
class SolidPhase(PhaseProtocol):
    """A solid phase."""

    name: str
    phase: str = field(init=False, default="solid")

    def __post_init__(self):
        logger.info("Creating a solid phase: %s", self.name)


@dataclass(kw_only=True)
class GasPhase(PhaseProtocol):
    """A gas phase and its properties.

    Args:
        name: Chemical formula.
        solubility: Solubility model. Defaults to no solubility.
        solid_melt_distribution_coefficient: Distribution coefficient. Defaults to 0.

    Attributes:
        name: Chemical formula.
        solubility: Solubility model.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
        elements: The elements and their (stoichiometric) counts.
        hill_formula: The Hill formula.
        element_masses: The elements and their total masses.
        molar_mass: Molar mass.
        output: To store calculated values for output.
    """

    name: str
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0
    elements: dict[str, int] = field(init=False)
    hill_formula: str = field(init=False)
    element_masses: dict[str, float] = field(init=False)
    molar_mass: float = field(init=False)
    phase: str = field(init=False, default="gas")

    def __post_init__(self):
        logger.info("Creating a gas phase: %s", self.name)
        masses: MolarMasses = MolarMasses()
        self.elements = self._count_elements()
        self.hill_formula = self._get_hill_formula()
        self.element_masses = {
            key: value * getattr(masses, key) for key, value in self.elements.items()
        }
        self.molar_mass = sum(self.element_masses.values())

    @property
    def is_diatomic(self) -> bool:
        """Is the molecule diatomic.

        Useful for obtaining the appropriate JANAF data for the Gibbs free energy of formation.
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
        elements: dict[str, int] = {}
        current_element: str = ""
        current_count: str = ""

        for char in self.name:
            if char.isupper():
                if current_element != "":
                    count = int(current_count) if current_count else 1
                    elements[current_element] = elements.get(current_element, 0) + count
                    current_count = ""
                current_element = char
            elif char.islower():
                current_element += char
            elif char.isdigit():
                current_count += char

        if current_element != "":
            count: int = int(current_count) if current_count else 1
            elements[current_element] = elements.get(current_element, 0) + count
        logger.debug("element count = \n%s", elements)
        return elements

    def _get_hill_formula(self) -> str:
        """Get the Hill empirical formula for this molecule.

        JANAF uses the Hill empirical formula to index its data tables.

        Returns:
            The Hill empirical formula.
        """
        if "C" in self.elements:
            ordered_elements = ["C"]
        else:
            ordered_elements = []

        if "H" in self.elements:
            ordered_elements.append("H")

        ordered_elements.extend(sorted(self.elements.keys() - {"C", "H"}))

        formula_string: str = "".join(
            [
                element + (str(self.elements[element]) if self.elements[element] > 1 else "")
                for element in ordered_elements
            ]
        )
        logger.debug("JANAF formula string = %s", formula_string)

        return formula_string

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
        volume_mixing_ratio = partial_pressure_bar / sum(fugacities_dict.values())

        moles_in_atmosphere: float = mass_in_atmosphere / self.molar_mass

        # Melt.
        prefactor: float = planet.mantle_mass * planet.mantle_melt_fraction
        ppmw_in_melt: float = self.solubility(
            partial_pressure_bar,
            planet.surface_temperature,
            fugacities_dict,
        )
        mass_in_melt = prefactor * ppmw_in_melt * UnitConversion.ppm_to_fraction()

        moles_in_melt: float = mass_in_melt / self.molar_mass

        # Solid.
        prefactor: float = planet.mantle_mass * (1 - planet.mantle_melt_fraction)
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass_in_solid = prefactor * ppmw_in_solid * UnitConversion.ppm_to_fraction()
        moles_in_solid: float = mass_in_solid / self.molar_mass

        self.output = MoleculeOutput(
            mass_in_atmosphere=mass_in_atmosphere,
            mass_in_solid=mass_in_solid,
            mass_in_melt=mass_in_melt,
            moles_in_atmosphere=moles_in_atmosphere,
            moles_in_melt=moles_in_melt,
            moles_in_solid=moles_in_solid,
            ppmw_in_solid=ppmw_in_solid,
            ppmw_in_melt=ppmw_in_melt,
            pressure_in_atmosphere=partial_pressure_bar,
            volume_mixing_ratio=volume_mixing_ratio,
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
        # total_pressure is set to 1 bar.
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


class IronWustiteBufferBallhaus(BufferedFugacity):
    """Iron-wustite buffer (fO2) from Ballhaus et al. (1991).

    https://ui.adsabs.harvard.edu/abs/1991CoMP..107...27B/abstract
    """

    def _fugacity(self, *, temperature: float) -> float:
        """See base class."""
        # total_pressure is set to 1 bar.
        total_pressure: float = 1  # bar
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * total_pressure / temperature
            + 3e-6 * total_pressure
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


class StandardGibbsFreeEnergyOfFormationProtocol(Protocol):
    """Standard Gibbs free energy of formation protocol."""

    @property
    def name(self) -> str:
        """Returns a string identifying the source of the data."""
        ...

    def get(self, molecule: PhaseProtocol, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            molecule: A Molecule.
            temperature: Temperature.
            pressure: Pressure (total) is relevant for condensed phases, but not for ideal gases.

        Returns:
            The standard Gibbs free energy of formation (J/mol).

        Raises:
            KeyError: Thermodynamic data is not available for the molecule.
        """
        ...


class StandardGibbsFreeEnergyOfFormationJANAF(StandardGibbsFreeEnergyOfFormationProtocol):
    """Standard Gibbs free energy of formation from the JANAF tables."""

    name: str = "JANAF"
    ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    STANDARD_STATE_PRESSURE: float = 1  # bar

    def get(self, molecule: PhaseProtocol, *, temperature: float, pressure: float) -> float:
        """See base class.

        In the JANAF tables, we have generally chosen the ideal diatomic gas for the reference
        state of permanent gases such as O2, N2, Cl2 etc. (quoting from the JANAF documentation).
        """

        del pressure

        db = janaf.Janafdb()

        def get_phase_data(phase):
            try:
                phase_data = db.getphasedata(formula=molecule.hill_formula, phase=phase)
            except ValueError:
                return None
            return phase_data

        if molecule.is_diatomic:
            phase_data_ref = get_phase_data("ref")
            phase_data_g = get_phase_data("g")
            if phase_data_ref is None and phase_data_g is None:
                msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                    molecule.name,
                    molecule.hill_formula,
                    self.name,
                )
                logger.warning(msg)
                raise KeyError(msg)
            phase = phase_data_ref or phase_data_g
        else:
            phase = get_phase_data("g")
            if phase is None:
                msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                    molecule.name,
                    molecule.hill_formula,
                    self.name,
                )
                logger.warning(msg)
                raise KeyError(msg)

        assert phase is not None
        logger.debug("Phase = %s", phase)
        gibbs: float = phase.DeltaG(temperature)

        logger.debug(
            "Molecule = %s, standard Gibbs energy of formation = %f", molecule.name, gibbs
        )

        return gibbs


class StandardGibbsFreeEnergyOfFormationHollandAndPowell(
    StandardGibbsFreeEnergyOfFormationProtocol
):
    """Standard Gibbs free energy of formation from Holland and Powell (1998).

    See the comments in the data file that is parsed by __init__
    """

    name: str = "Holland and Powell"
    ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K

    def __init__(self):
        data_path: Path = DATA_ROOT_PATH / Path("Mindata161127.csv")  # type: ignore
        data: pd.DataFrame = pd.read_csv(data_path, comment="#")
        data["name of phase component"] = data["name of phase component"].str.strip()
        data.rename(columns={"Unnamed: 1": "Abbreviation"}, inplace=True)
        data.drop(columns="Abbreviation", inplace=True)
        data.set_index("name of phase component", inplace=True)
        data = data.loc[:, :"Vmax"]
        data = data.astype(float)
        self.data = data

    def get(
        self, molecule: PhaseProtocol, *, temperature: float, pressure: Union[float, None] = None
    ) -> float:
        """See base class."""
        try:
            data: pd.Series = self.data.loc[molecule.name]
        except KeyError as exc:
            msg: str = "Thermodynamic data for %s (%s) is not available in %s" % (
                molecule.name,
                molecule.hill_formula,
                self.name,
            )
            logger.error(msg)
            raise KeyError from exc

        H = data.get("Hf")  # J
        S = data.get("S")  # J/K
        V = data.get("V")  # J/bar
        a = data.get("a")  # J/K           Coeff for calc heat capacity.
        b = data.get("b")  # J/K^2         Coeff for calc heat capacity.
        c = data.get("c")  # J K           Coeff for calc heat capacity.
        d = data.get("d")  # J K^(-1/2)    Coeff for calc heat capacity.
        alpha0 = data.get("a0")  # K^(-1), thermal expansivity
        K = data.get("K")  # bar, bulk modulus

        integral_H: float = (
            H
            + a * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)  # type: ignore a is a float.
            + b / 2 * (temperature**2 - self.ENTHALPY_REFERENCE_TEMPERATURE**2)  # type: ignore b is a float.
            - c * (1 / temperature - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE)  # type: ignore c is a float.
            + 2 * d * (temperature**0.5 - self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)  # type: ignore d is a float.
        )
        integral_S: float = (
            S
            + a * np.log(temperature / self.ENTHALPY_REFERENCE_TEMPERATURE)  # type: ignore a is a float.
            + b * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)  # type: ignore b is a float.
            - c / 2 * (1 / temperature**2 - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE**2)  # type: ignore c is a float.
            - 2 * d * (1 / temperature**0.5 - 1 / self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)  # type: ignore d is a float.
        )

        gibbs: float = integral_H - temperature * integral_S
        logger.debug(
            "Molecule = %s, standard Gibbs energy of formation = %f", molecule.name, gibbs
        )

        # TODO: Is there a better way to determine if the phase is condensed or not?
        if V:
            logger.info("Condensed phase so including volume-pressure integral")
            assert pressure is not None
            # Volume at T.
            # TODO: Why the exponential?  Seems different to the paper (check with Meng).
            V_T = V * np.exp(
                alpha0 * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)  # type: ignore
                - 2
                * 10.0
                * alpha0  # type: ignore
                * (temperature**0.5 - self.ENTHALPY_REFERENCE_TEMPERATURE**0.5)
            )
            dKdp: float = 4.0  # dimensionless, derivative of bulk modulus w.r.t. pressure
            # Derivative of bulk modulus w.r.t. temperature, from Holland and Powell (1998).
            dKdt: float = -K * 1.5e-4  # type: ignore
            K_T: float = K + dKdt * (temperature - self.ENTHALPY_REFERENCE_TEMPERATURE)  # type: ignore
            integral_VP: float = (
                V_T
                * K_T
                / (dKdp - 1)
                * ((1 + dKdp * (pressure - 1.0) / K_T) ** (1.0 - 1.0 / dKdp) - 1)
            )  # J, use P-1.0 instead of P
        else:
            logger.info("Ideal gas")
            integral_VP = 0

        gibbs += integral_VP

        return gibbs


@dataclass
class StandardGibbsFreeEnergyOfFormation(StandardGibbsFreeEnergyOfFormationProtocol):
    """Combined thermodynamic data that uses multiple datasets."""

    name: str = "Combined"
    datasets: list[StandardGibbsFreeEnergyOfFormationProtocol] = field(default_factory=list)
    STANDARD_STATE_PRESSURE: float = field(init=False, default=1)  # bar

    def __post_init__(self):
        if not self.datasets:
            self.add_dataset(StandardGibbsFreeEnergyOfFormationHollandAndPowell())
            self.add_dataset(StandardGibbsFreeEnergyOfFormationJANAF())

    def add_dataset(self, dataset: StandardGibbsFreeEnergyOfFormationProtocol):
        self.datasets.append(dataset)

    def get(self, molecule: PhaseProtocol, *, temperature: float, pressure: float) -> float:
        for dataset in self.datasets:
            try:
                gibbs: float = dataset.get(molecule, temperature=temperature, pressure=pressure)
                return gibbs
            except KeyError:
                continue

        msg: str = "Thermodynamic data for %s (%s) is not available in any dataset" % (
            molecule.name,
            molecule.hill_formula,
        )
        logger.error(msg)
        raise KeyError(msg)


# region Andesite solubility


class AndesiteH2(Solubility):
    """Hirschmann et al. 2012.

    Fit to fH2 vs. H2 concentration from Table 2.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        # TODO: Maggie to add comment what the next line is (or remove if no longer required).
        # ppmw: float = self.power_law(fugacity, 34.43369241, 0.49459427)
        ppmw: float = 10 ** (1.20257736 * np.log10(np.sqrt(fugacity)) + 1.01058631)
        return ppmw


class AndesiteS_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (31984.243 / temperature)
        logS_wtp: float = (
            logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        )
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteS_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-).

    Using expression in abstract for S wt% and the expression for sulfide capacity. Composition
    for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """fugacity is fS2."""
        logCs: float = 0.225 - (8876.5 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteS(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteS_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteS_Sulfate()

    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)
        return solubility


# endregion


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


# region Basalt solubility


class BasaltDixonCO2(Solubility):
    """Dixon et al. (1995)."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del fugacities_dict
        ppmw: float = (3.8e-7) * fugacity * np.exp(-23 * (fugacity - 1) / (83.15 * temperature))
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)
        return ppmw


class BasaltDixonH2O(Solubility):
    """Dixon et al. (1995) refit by Paolo Sossi."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 965, 0.5)


class BasaltH2(Solubility):
    """Hirschmann et al. 2012 for Basalt.

    Fit to fH2 vs. H2 concentration from Table 2.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        # TODO: Maggie to add comment what the next line is (or remove if no longer required).
        # ppmw: float = self.power_law(fugacity, 53.65376426, 0.38365457)
        ppmw: float = 10 ** (1.04827856 * np.log10(np.sqrt(fugacity)) + 1.10083602)
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
            constant: float = (fugacities_dict["O2"] ** -0.75) * 5.97e-10
            ppmw += self.power_law(fugacity, constant, 0.5)
        return ppmw


class BasaltS_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (31532.862 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class BasaltS_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = 0.225 - (7817.134 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class BasaltS(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltS_Sulfide()
        self.sulfate_solubility: Solubility = BasaltS_Sulfate()

    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)
        return solubility


class BasaltWilsonH2O(Solubility):
    """Hamilton (1964) and Wilson and Head (1981)."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 215, 0.7)


class TBasaltS_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Trachy-Basalt from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (32446.366 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class TBasaltS_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for Trachy-basalt from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = 0.225 - (7842.5 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


# endregion


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


class MercuryMagmaS(Solubility):
    """Namur et al. 2016.

    S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like magmas.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        a, b, c, d = [7.25, -2.54e4, 0.04, -0.551]  # Coeffs from eq. 10 (Namur et al., 2016).
        # FIXME: How to deal if fO2 not available?  Drop last term?
        wt_perc: float = np.exp(
            a
            + (b / temperature)
            + ((c * fugacity) / temperature)
            + (d * np.log10(fugacities_dict["O2"]))
        )
        ppmw: float = UnitConversion.weight_percent_to_ppmw(wt_perc)
        return ppmw


class PeridotiteH2O(Solubility):
    """Sossi et al. (2023).

    Power law parameters are in the abstract:
    https://ui.adsabs.harvard.edu/abs/2023E%26PSL.60117894S/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 524, 0.5)


class SilicicMeltsH2(Solubility):
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


# Dictionaries of self-consistent solubility laws for a given composition.
andesite_solubilities: dict[str, Solubility] = {
    "H2": AndesiteH2(),
    "O2S": AndesiteS(),
    "OS": AndesiteS(),
    "S2": AndesiteS(),
}
anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltDixonH2O(),
    "CO2": BasaltDixonCO2(),
    "H2": BasaltH2(),
    "N2": BasaltLibourelN2(),
    "O2S": BasaltS(),
    "OS": BasaltS(),
    "S2": BasaltS(),
}
peridotite_solubilities: dict[str, Solubility] = {"H2O": PeridotiteH2O()}
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
