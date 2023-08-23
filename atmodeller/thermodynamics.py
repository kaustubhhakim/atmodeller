"""Fugacity buffers, gas phase reactions, and solubility laws."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import Callable, Optional, Protocol, Union

import numpy as np
import pandas as pd
from molmass import Formula
from thermochem import janaf

from atmodeller import DATA_ROOT_PATH, GAS_CONSTANT, GRAVITATIONAL_CONSTANT
from atmodeller.solubilities import NoSolubility, Solubility
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet.

    Default values are for a fully molten Earth.

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
    """

    mantle_mass: float = 4.208261222595111e24  # kg, Earth's mantle mass
    mantle_melt_fraction: float = 1.0  # Completely molten
    core_mass_fraction: float = 0.295334691460966  # Earth's core mass fraction
    surface_radius: float = 6371000.0  # m, Earth's radius
    surface_temperature: float = 2000.0  # K
    melt_composition: Union[str, None] = None

    def __post_init__(self):
        logger.info("Creating a new planet")
        logger.info("Mantle mass (kg) = %f", self.mantle_mass)
        logger.info("Mantle melt fraction = %f", self.mantle_melt_fraction)
        logger.info("Core mass fraction = %f", self.core_mass_fraction)
        logger.info("Planetary radius (m) = %f", self.surface_radius)
        logger.info("Planetary mass (kg) = %f", self.planet_mass)
        logger.info("Surface temperature (K) = %f", self.surface_temperature)
        logger.info("Surface gravity (m/s^2) = %f", self.surface_gravity)
        logger.info("Melt Composition = %s", self.melt_composition)

    @property
    def planet_mass(self) -> float:
        """Mass of the planet."""
        return self.mantle_mass / (1 - self.core_mass_fraction)

    @property
    def surface_area(self) -> float:
        """Surface area of the planet."""
        return 4.0 * np.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> float:
        """Surface gravity of the planet."""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2


class BufferedFugacity(ABC):
    """Buffered fugacity base class."""

    @abstractmethod
    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """Log10(fugacity) of the buffer in terms of temperature.

        Args:
            temperature: Temperature (K).
            pressure: Pressure (bar). Defaults to 1.

        Returns:
            Log10 of the fugacity.
        """
        raise NotImplementedError

    def __call__(
        self, *, temperature: float, pressure: float = 1, fugacity_log10_shift: float = 0
    ) -> float:
        """log10(fugacity) plus an optional shift.

        Args:
            temperature: Temperature (K).
            pressure: Pressure (bar). Defaults to 1.
            fugacity_log10_shift: Log10 shift. Defaults to 0.

        Returns:
            Log10 of the fugacity including the shift.
        """
        return self._fugacity(temperature=temperature, pressure=pressure) + fugacity_log10_shift


class IronWustiteBufferHirschmann(BufferedFugacity):
    """Iron-wustite buffer (fO2) from O'Neill and Pownceby (1993) and Hirschmann et al. (2008).

    https://ui.adsabs.harvard.edu/abs/1993CoMP..114..296O/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )
        return fugacity


class IronWustiteBufferOneill(BufferedFugacity):
    """Iron-wustite buffer (fO2) from O'Neill and Eggins (2002).

    Gibbs energy of reaction is at 1 bar. See Table 6.

    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        del pressure
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

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        fugacity: float = (
            14.07
            - 28784 / temperature
            - 2.04 * np.log10(temperature)
            + 0.053 * pressure / temperature
            + 3e-6 * pressure
        )
        return fugacity


class IronWustiteBufferFischer(BufferedFugacity):
    """Iron-wustite buffer (fO2) from Fischer et al. (2011).

    See Table S2 in supplementary materials.

    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract
    """

    def _fugacity(self, *, temperature: float, pressure: float = 1) -> float:
        """See base class."""
        pressure_GPa: float = UnitConversion.bar_to_GPa(pressure)
        a_P: float = 6.44059 + 0.00463099 * pressure_GPa
        b_P: float = (
            -28.1808
            + 0.556272 * pressure_GPa
            - 0.00143757 * pressure_GPa**2
            + 4.0256e-6 * pressure_GPa**3
            - 5.4861e-9 * pressure_GPa**4  # Note typo in Table S2. Must be pressure**4.
        )
        b_P *= 1000 / temperature
        buffer: float = a_P + b_P
        return buffer


@dataclass(kw_only=True)
class GasSpeciesOutput:
    """Output for a gas species."""

    mass_in_atmosphere: float  # kg
    mass_in_solid: float  # kg
    mass_in_melt: float  # kg
    moles_in_atmosphere: float  # moles
    moles_in_melt: float  # moles
    moles_in_solid: float  # moles
    ppmw_in_solid: float  # ppm by weight
    ppmw_in_melt: float  # ppm by weight
    pressure_in_atmosphere: float  # bar
    volume_mixing_ratio: float  # dimensionless
    mass_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the gas species or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: "GasSpecies", element: Optional[str] = None, **kwargs) -> float:
        """Wrapper to return the mass of either the gas species or one of its elements.

        Args:
            element: Returns the mass of this element. Defaults to None to return the gas species
                mass.
            **kwargs: Catches keyword arguments to forward to func.

        Returns:
            Mass of either the gas species or element.
        """
        mass: float = func(self, **kwargs)
        if element is not None:
            try:
                mass *= (
                    UnitConversion.g_to_kg(self.formula.composition()[element].mass)
                    / self.molar_mass
                )
            except KeyError:  # Element not in formula so must be zero.
                mass = 0

        return mass

    return mass_wrapper


class Ideality(ABC):
    """Ideality base class."""

    @abstractmethod
    def _ideality(self, *, temperature: float, pressure: float) -> float:
        """Ideality.

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            ideality.
        """
        raise NotImplementedError

    def __call__(self, *, temperature: float, pressure: float) -> float:
        """Ideality."""
        return self._ideality(temperature=temperature, pressure=pressure)


class Ideal(Ideality):
    """Ideal solution (activity of unity) or ideal gas (fugacity coefficient of unity)."""

    def _ideality(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return 1.0


@dataclass(kw_only=True)
class ChemicalComponent(ABC):
    """A chemical component and its properties.

    Args:
        chemical_formula: Chemical formula (e.g. CO2, C, CH4, etc.).
        common_name: Common name for locating Gibbs data in the thermodynamic database.

    Attributes:
        chemical_formula: Chemical formula.
        common_name: Common name in the thermodynamic database.
        formula: Formula.
    """

    chemical_formula: str
    common_name: str
    formula: Formula = field(init=False)
    ideality: Ideality = field(default_factory=Ideal)
    # TODO: select source of thermodynamic data for this species.  Do all the reading in/caching
    # to set up the interpolation functions
    # TODO: Option to specify if "ideal" or not to determine if a linear solve can be performed.

    def __post_init__(self):
        logger.info(
            "Creating a %s: %s (%s)",
            self.__class__.__name__,
            self.common_name,
            self.chemical_formula,
        )
        self.formula = Formula(self.chemical_formula)

    @property
    @abstractmethod
    def phase(self) -> str:
        """Returns the phase (solid, gas, etc.)."""
        ...

    @property
    def molar_mass(self) -> float:
        """Returns the molar mass in kg/mol."""
        return UnitConversion.g_to_kg(self.formula.mass)

    @property
    def hill_formula(self) -> str:
        """Returns the Hill formula."""
        return self.formula.formula

    @property
    def is_homonuclear_diatomic(self) -> bool:
        """Returns True if the species is homonuclear diatomic or False if not."""

        composition = self.formula.composition()

        if len(list(composition.keys())) == 1 and list(composition.values())[0].count == 2:
            return True
        else:
            return False

    @cached_property
    def modified_hill_formula(self) -> str:
        """Returns the modified Hill formula.

        JANAF uses the modified Hill formula to index its data tables. In short, H, if present,
        should appear after C (if C is present), otherwise it must be the first element.
        """
        elements: dict[str, int] = {
            element: properties.count for element, properties in self.formula.composition().items()
        }

        if "C" in elements:
            ordered_elements: list[str] = ["C"]
        else:
            ordered_elements = []

        if "H" in elements:
            ordered_elements.append("H")

        ordered_elements.extend(sorted(elements.keys() - {"C", "H"}))

        formula_string: str = "".join(
            [
                element + (str(elements[element]) if elements[element] > 1 else "")
                for element in ordered_elements
            ]
        )
        logger.debug("Modified Hill formula = %s", formula_string)

        return formula_string


@dataclass(kw_only=True)
class SolidSpecies(ChemicalComponent):
    """A solid species."""

    # TODO: Check what this means if two solids are in equilibrium (solid solution). Presumably
    # then the mole fractions become relevant.
    activity: Ideality = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.activity = self.ideality  # Ideality relates to activity.

    @property
    def phase(self) -> str:
        return "solid"

    # TODO: Automatically flag a constraint for unity activity when solving the system?
    # or some message to the logger?


@dataclass(kw_only=True)
class GasSpecies(ChemicalComponent):
    """A gas species.

    Args:
        chemical_formula: Chemical formula (e.g. CO2, C, CH4, etc.).
        solubility: Solubility model. Defaults to no solubility.
        fugacity_coefficient: Fugacity coefficient model.
        solid_melt_distribution_coefficient: Distribution coefficient. Defaults to 0.

    Attributes:
        chemical_formula: Chemical formula.
        common_name: Common name in the thermodynamic database.
        solubility: Solubility model.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
        output: To store calculated values for output.
    """

    common_name: str = field(init=False)
    solubility: Solubility = field(default_factory=NoSolubility)
    # species fugacity = fugacity coefficient * species partial pressure
    fugacity_coefficient: Ideality = field(init=False)
    solid_melt_distribution_coefficient: float = 0

    def __post_init__(self):
        self.common_name = self.chemical_formula
        super().__post_init__()
        self.fugacity_coefficient = self.ideality  # Ideality relates to fugacity.

    @property
    def phase(self) -> str:
        return "gas"

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
            element: Returns the mass for an element. Defaults to None to return the species mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the species (element=None) or element (element=element).
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

        self.output = GasSpeciesOutput(
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


class StandardGibbsFreeEnergyOfFormationProtocol(Protocol):
    """Standard Gibbs free energy of formation protocol."""

    _DATA_SOURCE: str
    _ENTHALPY_REFERENCE_TEMPERATURE: float  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @property
    def DATA_SOURCE(self) -> str:
        """Returns a string identifying the source of the data."""
        return self._DATA_SOURCE

    @property
    def ENTHALPY_REFERENCE_TEMPERATURE(self) -> float:
        return self._ENTHALPY_REFERENCE_TEMPERATURE

    @property
    def STANDARD_STATE_PRESSURE(self) -> float:
        return self._STANDARD_STATE_PRESSURE

    def get(self, species: ChemicalComponent, *, temperature: float, pressure: float) -> float:
        """Returns the standard Gibbs free energy of formation in units of J/mol.

        Args:
            species: A species.
            temperature: Temperature.
            pressure: Pressure (total) is relevant for condensed phases, but not for ideal gases.

        Returns:
            The standard Gibbs free energy of formation (J/mol).

        Raises:
            KeyError: Thermodynamic data is not available for the species.
        """
        ...


class StandardGibbsFreeEnergyOfFormationJANAF(StandardGibbsFreeEnergyOfFormationProtocol):
    """Standard Gibbs free energy of formation from the JANAF tables."""

    _DATA_SOURCE: str = "JANAF"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    def get(self, species: ChemicalComponent, *, temperature: float, pressure: float) -> float:
        """See base class.

        In the JANAF tables, we have generally chosen the ideal diatomic gas for the reference
        state of permanent gases such as O2, N2, Cl2 etc. (quoting from the JANAF documentation).
        """

        del pressure

        db = janaf.Janafdb()

        def get_phase_data(phase):
            try:
                phase_data = db.getphasedata(formula=species.modified_hill_formula, phase=phase)
            except ValueError:
                return None
            return phase_data

        if species.phase == "gas":
            if species.is_homonuclear_diatomic:
                phase_data_ref = get_phase_data("ref")
                phase_data_g = get_phase_data("g")
                if phase_data_ref is None and phase_data_g is None:
                    msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                        species.common_name,
                        species.modified_hill_formula,
                        self.DATA_SOURCE,
                    )
                    logger.warning(msg)
                    raise KeyError(msg)
                phase = phase_data_ref or phase_data_g
            else:
                phase = get_phase_data("g")
                if phase is None:
                    msg = "Thermodynamic data for %s (%s) is not available in %s" % (
                        species.common_name,
                        species.modified_hill_formula,
                        self.DATA_SOURCE,
                    )
                    logger.warning(msg)
                    raise KeyError(msg)
        else:  # if species.phase == "solid":
            phase = get_phase_data("ref")

        assert phase is not None
        logger.debug(
            "Thermodynamic data for %s (%s) = %s",
            species.common_name,
            species.modified_hill_formula,
            phase,
        )
        gibbs: float = phase.DeltaG(temperature)

        logger.debug(
            "standard Gibbs energy of formation for %s (%s) = %f",
            species.common_name,
            species.modified_hill_formula,
            gibbs,
        )

        return gibbs


class StandardGibbsFreeEnergyOfFormationHollandAndPowell(
    StandardGibbsFreeEnergyOfFormationProtocol
):
    """Standard Gibbs free energy of formation from Holland and Powell (1998).

    See the comments in the data file that is parsed by __init__
    """

    _DATA_SOURCE: str = "Holland and Powell"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

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
        self,
        species: ChemicalComponent,
        *,
        temperature: float,
        pressure: Union[float, None] = None,
    ) -> float:
        """See base class."""
        try:
            data: pd.Series = self.data.loc[species.common_name]
        except KeyError as exc:
            msg: str = "Thermodynamic data for %s (%s) is not available in %s" % (
                species.common_name,
                species.hill_formula,
                self.DATA_SOURCE,
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
            "Species = %s, standard Gibbs energy of formation = %f", species.common_name, gibbs
        )

        if species.phase == "solid":
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

        # TODO: maybe test magnitude of integral_VP to still allow linear solve.
        gibbs += integral_VP

        return gibbs


class StandardGibbsFreeEnergyOfFormation(StandardGibbsFreeEnergyOfFormationProtocol):
    """Combined thermodynamic data that uses multiple datasets.

    Args:
        datasets: A list of thermodynamic data to use.
    """

    _DATA_SOURCE: str = "Combined"
    _STANDARD_STATE_PRESSURE: float = 1  # bar
    # TODO: Check: Taking JANAF temperature.  Same for Holland and Powell (1998)?
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K

    def __init__(
        self, datasets: Union[list[StandardGibbsFreeEnergyOfFormationProtocol], None] = None
    ):
        if datasets is None:
            self.datasets: list[StandardGibbsFreeEnergyOfFormationProtocol] = []
            self.add_dataset(StandardGibbsFreeEnergyOfFormationHollandAndPowell())
            self.add_dataset(StandardGibbsFreeEnergyOfFormationJANAF())
        else:
            self.datasets = datasets

    def add_dataset(self, dataset: StandardGibbsFreeEnergyOfFormationProtocol) -> None:
        if len(self.datasets) >= 1:
            logger.warning("Combining different thermodynamic data may result in consistencies")
        logger.info("Adding thermodynamic data: %s", dataset.DATA_SOURCE)
        self.datasets.append(dataset)

    def get(self, species: ChemicalComponent, *, temperature: float, pressure: float) -> float:
        for dataset in self.datasets:
            try:
                gibbs: float = dataset.get(species, temperature=temperature, pressure=pressure)
                return gibbs
            except KeyError:
                continue

        msg: str = "Thermodynamic data for %s (%s) is not available in any dataset" % (
            species.common_name,
            species.hill_formula,
        )
        logger.error(msg)
        raise KeyError(msg)
