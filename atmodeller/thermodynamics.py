"""Thermodynamic classes, including fugacity buffers and Gibbs energies.

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
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

import numpy as np
import pandas as pd
from thermochem import janaf

from atmodeller import DATA_ROOT_PATH
from atmodeller.interfaces import (
    ChemicalComponent,
    NoSolubility,
    Solubility,
    StandardGibbsFreeEnergyOfFormationProtocol,
)
from atmodeller.utilities import UnitConversion

if TYPE_CHECKING:
    from atmodeller.core import InteriorAtmosphereSystem, Planet
    from atmodeller.interfaces import SystemConstraint


logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SolidSpecies(ChemicalComponent):
    """A solid species.

    For a solid species, 'self.ideality' refers to its activity, where the activity is equal to the
    activity coefficient multiplied by the species' volume mixing ratio.

    Args:
        chemical_formula: Chemical formula (e.g., CO2, C, CH4, etc.).
        common_name: Common name for locating Gibbs data in the thermodynamic database.
        ideality: Ideality object representing activity for thermodynamic calculations. Defaults to
            Ideal.

    Attributes:
        chemical_formula: Chemical formula.
        common_name: Common name in the thermodynamic database.
        formula: Formula object derived from the chemical formula.
        ideality: Ideality object representing activity for thermodynamic calculations.
    """

    @property
    def activity(self) -> SystemConstraint:
        """Activity."""
        return self.ideality


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
    fugacity: float  # bar
    fugacity_coefficient: float  # dimensionless
    pressure_in_atmosphere: float  # bar
    volume_mixing_ratio: float  # dimensionless
    mass_in_total: float = field(init=False)

    def __post_init__(self):
        self.mass_in_total = self.mass_in_atmosphere + self.mass_in_melt + self.mass_in_solid


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the gas species or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: GasSpecies, element: Optional[str] = None, **kwargs) -> float:
        """Wrapper to return the mass of either the gas species or one of its elements.

        Args:
            element: Returns the mass of this element. Defaults to None to return the species mass.
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
            except KeyError:  # Element not in formula so mass is zero.
                mass = 0

        return mass

    return mass_wrapper


@dataclass(kw_only=True)
class GasSpecies(ChemicalComponent):
    """A gas species.

    For a gas species, 'self.ideality' refers to its fugacity coefficient, where the fugacity is
    equal to the fugacity coefficient multiplied by the species' partial pressure.

    Args:
        chemical_formula: Chemical formula (e.g. CO2, C, CH4, etc.).
        solubility: Solubility model. Defaults to no solubility.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
            Defaults to 0.
        ideality: Ideality object representing the fugacity coefficient for thermodynamic
            calculations. Defaults to Ideal (i.e., unity).

    Attributes:
        chemical_formula: Chemical formula.
        common_name: Common name in the thermodynamic database.
        formula: Formula object derived from the chemical formula.
        ideality: Ideality object representing the fugacity coefficient for thermodynamic
            calculations.
        solubility: Solubility model.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
        output: To store calculated values for output.
    """

    common_name: str = field(init=False)
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0

    def __post_init__(self):
        self.common_name = self.chemical_formula
        super().__post_init__()

    @property
    def fugacity_coefficient(self) -> SystemConstraint:
        """Fugacity coefficient."""
        return self.ideality

    @_mass_decorator
    def mass(
        self,
        *,
        planet: Planet,
        system: InteriorAtmosphereSystem,
        element: Optional[str] = None,
    ) -> float:
        """Calculates the total mass of the species or element.

        Args:
            planet: Planet properties.
            system: Interior atmosphere system.
            element: Returns the mass for an element. Defaults to None to return the species mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the species (element=None) or element (element=element).
        """

        del element

        pressure: float = system.pressures_dict[self.chemical_formula]
        fugacity: float = system.fugacities_dict[self.chemical_formula]
        fugacity_coefficient: float = system.fugacity_coefficients_dict[self.chemical_formula]

        # Atmosphere.
        mass_in_atmosphere: float = UnitConversion.bar_to_Pa(pressure) / planet.surface_gravity
        mass_in_atmosphere *= (
            planet.surface_area * self.molar_mass / system.atmospheric_mean_molar_mass
        )
        volume_mixing_ratio: float = pressure / system.total_pressure
        moles_in_atmosphere: float = mass_in_atmosphere / self.molar_mass

        # Melt.
        prefactor: float = planet.mantle_mass * planet.mantle_melt_fraction
        ppmw_in_melt: float = self.solubility(
            fugacity,
            planet.surface_temperature,
            system.fugacities_dict,
        )
        mass_in_melt: float = prefactor * ppmw_in_melt * UnitConversion.ppm_to_fraction()
        moles_in_melt: float = mass_in_melt / self.molar_mass

        # Solid.
        prefactor: float = planet.mantle_mass * (1 - planet.mantle_melt_fraction)
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass_in_solid: float = prefactor * ppmw_in_solid * UnitConversion.ppm_to_fraction()
        moles_in_solid: float = mass_in_solid / self.molar_mass

        self.output: GasSpeciesOutput = GasSpeciesOutput(
            mass_in_atmosphere=mass_in_atmosphere,
            mass_in_solid=mass_in_solid,
            mass_in_melt=mass_in_melt,
            moles_in_atmosphere=moles_in_atmosphere,
            moles_in_melt=moles_in_melt,
            moles_in_solid=moles_in_solid,
            ppmw_in_solid=ppmw_in_solid,
            ppmw_in_melt=ppmw_in_melt,
            fugacity=fugacity,
            fugacity_coefficient=fugacity_coefficient,
            pressure_in_atmosphere=pressure,
            volume_mixing_ratio=volume_mixing_ratio,
        )

        return self.output.mass_in_total


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

        if isinstance(species, GasSpecies):
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
        else:
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

        if isinstance(species, SolidSpecies):
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


class StandardGibbsFreeEnergyOfFormation(StandardGibbsFreeEnergyOfFormationProtocol):
    """Combines thermodynamic data that uses multiple datasets.

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
        """Adds a thermodynamic dataset.

        Args:
            dataset: A thermodynamic dataset.
        """
        if len(self.datasets) >= 1:
            logger.warning("Combining different thermodynamic data may result in consistencies")
        logger.info("Adding thermodynamic data: %s", dataset.DATA_SOURCE)
        self.datasets.append(dataset)

    def get(self, species: ChemicalComponent, *, temperature: float, pressure: float) -> float:
        """See base class."""
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
