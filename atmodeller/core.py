"""Core classes and functions for modeling interior-atmosphere systems.

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
import pprint
from collections import UserList
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.optimize import fsolve

from atmodeller import GRAVITATIONAL_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.interfaces import ChemicalComponent, NoSolubility
from atmodeller.reaction_network import ReactionNetwork
from atmodeller.solubilities import composition_solubilities
from atmodeller.thermodynamics import (
    GasSpecies,
    SolidSpecies,
    StandardGibbsFreeEnergyOfFormationJANAF,
    StandardGibbsFreeEnergyOfFormationProtocol,
)
from atmodeller.utilities import filter_by_type

if TYPE_CHECKING:
    from atmodeller.core import Planet
    from atmodeller.interfaces import Solubility

logger: logging.Logger = logging.getLogger(__name__)


class Species(UserList):
    """Collections of species for an interior-atmosphere system.

    A collection of species. It provides methods to filter species based on their phases (solid,
    gas).

    Args:
        initlist: Initial list of species. Defaults to None.

    Attributes:
        data: List of species contained in the system.
    """

    def __init__(self, initlist: Union[list[ChemicalComponent], None] = None):
        self.data: list[ChemicalComponent]  # For typing.
        super().__init__(initlist)

    @property
    def number(self) -> int:
        """Number of species."""
        return len(self)

    @property
    def gas_species(self) -> dict[int, GasSpecies]:
        """Gas species."""
        return filter_by_type(self, GasSpecies)

    @property
    def number_gas_species(self) -> int:
        """Number of gas species."""
        return len(self.gas_species)

    @property
    def solid_species(self) -> dict[int, SolidSpecies]:
        """Solid species."""
        return filter_by_type(self, SolidSpecies)

    @property
    def number_solid_species(self) -> int:
        """Number of solid species."""
        return len(self.solid_species)

    @property
    def indices(self) -> dict[str, int]:
        """Indices of the species."""
        return {
            chemical_formula: index
            for index, chemical_formula in enumerate(self.chemical_formulas)
        }

    @property
    def chemical_formulas(self) -> list[str]:
        """Chemical formulas of the species."""
        return [species.chemical_formula for species in self.data]

    def conform_solubilities_to_planet_composition(self, planet: Planet) -> None:
        """Ensure that the solubilities of the species are consistent with the planet composition.

        Args:
            planet: A planet.
        """
        if planet.melt_composition is not None:
            msg: str = (
                # pylint: disable=consider-using-f-string
                "Setting solubilities to be consistent with the melt composition (%s)"
                % planet.melt_composition
            )
            logger.info(msg)
            try:
                solubilities: dict[str, Solubility] = composition_solubilities[
                    planet.melt_composition.casefold()
                ]
            except KeyError:
                logger.error("Cannot find solubilities for %s", planet.melt_composition)
                raise

            for species in self.gas_species.values():
                try:
                    species.solubility = solubilities[species.chemical_formula]
                    logger.info(
                        "Found solubility law for %s: %s",
                        species.chemical_formula,
                        species.solubility.__class__.__name__,
                    )
                except KeyError:
                    logger.info("No solubility law for %s", species.chemical_formula)
                    species.solubility = NoSolubility()

    def _species_sorter(self, species: ChemicalComponent) -> tuple[int, str]:
        """Sorter for the species.

        Sorts first by species complexity and second by species name.

        Args:
            species: Species.

        Returns:
            A tuple to sort first by number of elements and second by species name.
        """
        return (species.formula.atoms, species.chemical_formula)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet.

    Defines the properties of a planet that are relevant for interior modeling. It provides default
    values suitable for modelling a fully molten Earth-like planet.

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
        """Mass of the planet in SI units."""
        return self.mantle_mass / (1 - self.core_mass_fraction)

    @property
    def surface_area(self) -> float:
        """Surface area of the planet in SI units."""
        return 4.0 * np.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> float:
        """Surface gravity of the planet in SI units."""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2


@dataclass(kw_only=True)
class InteriorAtmosphereSystem:
    """An interior-atmosphere system.

    Args:
        species: A list of species.
        gibbs_data: Standard Gibbs free energy of formation. Defaults to JANAF.
        planet: A planet. Defaults to a molten Earth.

    Attributes:
        species: A list of species, possibly reordered compared to the input arg.
        gibbs_data: Standard Gibbs free energy of formation.
        planet: A planet.
    """

    species: Species
    gibbs_data: StandardGibbsFreeEnergyOfFormationProtocol = field(
        default_factory=StandardGibbsFreeEnergyOfFormationJANAF
    )
    planet: Planet = field(default_factory=Planet)
    _reaction_network: ReactionNetwork = field(init=False)
    _solution: np.ndarray = field(init=False)

    def __post_init__(self):
        logger.info("Creating an interior-atmosphere system")
        self.species.conform_solubilities_to_planet_composition(self.planet)
        self._reaction_network = ReactionNetwork(species=self.species, gibbs_data=self.gibbs_data)
        # Initialise solution to zero.
        self._solution = np.zeros_like(self.species, dtype=np.float_)

    @property
    def log10_pressures(self) -> np.ndarray:
        """Log10 pressures."""
        return self._solution

    @property
    def pressures(self) -> np.ndarray:
        """Pressures."""
        return 10**self.log10_pressures

    @property
    def pressures_dict(self) -> dict[str, float]:
        """Pressures of all species in a dictionary."""
        # TODO: Activity for solid (or remove from output).
        output: dict[str, float] = {}
        for chemical_formula, pressure in zip(self.species.chemical_formulas, self.pressures):
            output[chemical_formula] = pressure

        return output

    @property
    def fugacity_coefficients_dict(self) -> dict[str, float]:
        """Fugacity coefficients in a dictionary."""
        output: dict[str, float] = {
            species.chemical_formula: species.ideality.get_value(
                temperature=self.planet.surface_temperature, pressure=self.total_pressure
            )
            for species in self.species.data
        }
        return output

    @property
    def fugacities_dict(self) -> dict[str, float]:
        """Fugacities of all species in a dictionary."""
        output: dict[str, float] = {}
        for key, value in self.fugacity_coefficients_dict.items():
            output[key] = value * self.pressures_dict[key]
        return output

    @property
    def total_pressure(self) -> float:
        """Total pressure."""
        return sum(self.pressures)

    @property
    def atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere."""
        mu_atmosphere: float = 0
        for index, species in enumerate(self.species):
            mu_atmosphere += species.molar_mass * self.pressures[index]
        mu_atmosphere /= self.total_pressure

        return mu_atmosphere

    @property
    def output(self) -> dict:
        """Convenient output for analysis."""
        output_dict: dict = {}
        output_dict["total_pressure_in_atmosphere"] = self.total_pressure
        output_dict["mean_molar_mass_in_atmosphere"] = self.atmospheric_mean_molar_mass
        for species in self.species.gas_species.values():
            output_dict[species.chemical_formula] = species.output
        # TODO: Dan to add elemental outputs as well.
        return output_dict

    def isclose(
        self, target_dict: dict[str, float], rtol: float = 1.0e-5, atol: float = 1.0e-8
    ) -> np.bool_:
        """Determines if the solution pressures are close to target values within a tolerance."""

        if len(self.pressures_dict) != len(target_dict):
            return np.bool_(False)

        target_pressures: np.ndarray = np.array(
            [target_dict[species.formula.formula] for species in self.species]
        )
        isclose: np.bool_ = np.isclose(
            target_pressures, self.pressures, rtol=rtol, atol=atol
        ).all()

        return isclose

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        initial_log10_pressures: Union[np.ndarray, None] = None,
    ) -> None:
        """Solves the system to determine the partial pressures with provided constraints.

        Args:
            constraints: Constraints for the system of equations.
            initial_log10_pressures: Initial guess for the log10 pressures. Defaults to None.

        Returns:
            The pressures in bar.
        """

        constraints = self._assemble_constraints(constraints)
        self._solution = self._solve_fsolve(
            constraints=constraints, initial_log10_pressures=initial_log10_pressures
        )

        # Recompute quantities that depend on the solution, since species.mass is not called for
        # the reaction network.
        for species in self.species.gas_species.values():
            species.mass(
                planet=self.planet,
                system=self,
            )

        logger.info(pprint.pformat(self.pressures_dict))

    def _assemble_constraints(self, constraints: SystemConstraints) -> SystemConstraints:
        """Combines the user-prescribed constraints with intrinsic constraints (solid activities).

        Args:
            constraints: Constraints as prescribed by the user.

        Returns:
            Constraints list including solid activities.
        """
        logger.info("Assembling constraints")
        for solid in self.species.solid_species.values():
            constraints.append(solid.activity)
        logger.info("Constraints: %s", pprint.pformat(constraints))

        return constraints

    def _solve_fsolve(
        self,
        *,
        constraints: SystemConstraints,
        initial_log10_pressures: Union[np.ndarray, None],
    ) -> np.ndarray:
        """Solves the non-linear system of equations.

        Args:
            constraints: Constraints for the system of equations.
            initial_log10_pressures: Initial guess for the log10 pressures.
        """

        if initial_log10_pressures is None:
            initial_log10_pressures = np.ones_like(self.species, dtype=np.float_)
        logger.debug("initial_log10_pressures = %s", initial_log10_pressures)
        ier: int = 0
        # Count the number of attempts to solve the system by randomising the initial condition.
        ic_count: int = 1
        # Maximum number of attempts to solve the system by randomising the initial condition.
        ic_count_max: int = 10
        sol: np.ndarray = np.zeros_like(initial_log10_pressures)
        infodict: dict = {}

        coefficient_matrix: np.ndarray = self._reaction_network.get_coefficient_matrix(
            constraints=constraints
        )

        while ier != 1 and ic_count <= ic_count_max:
            sol, infodict, ier, mesg = fsolve(
                self._objective_func,
                initial_log10_pressures,
                args=(constraints, coefficient_matrix),
                full_output=True,
            )
            logger.info(mesg)
            if ier != 1:
                logger.info(
                    "Retrying with a new randomised initial condition (attempt %d)",
                    ic_count,
                )
                # Increase or decrease the magnitude of all pressures.
                initial_log10_pressures *= 2 * np.random.random_sample()
                logger.debug("initial_log10_pressures = %s", initial_log10_pressures)
                ic_count += 1

        if ic_count == ic_count_max:
            logger.error("Maximum number of randomised initial conditions has been exceeded")
            raise RuntimeError("Solution cannot be found (ic_count == ic_count_max)")

        logger.info("Number of function calls = %d", infodict["nfev"])
        logger.info("Final objective function evaluation = %s", infodict["fvec"])

        return sol

    def _objective_func(
        self,
        log10_pressures: np.ndarray,
        constraints: SystemConstraints,
        coefficient_matrix: np.ndarray,
    ) -> np.ndarray:
        """Objective function for the non-linear system.

        Args:
            log10_pressures: Log10 of the pressures of each species.
            constraints: Constraints for the system of equations.
            coefficient_matrix: Coefficient matrix.

        Returns:
            The solution, which is the log10 of the pressures for each species.
        """
        self._solution = log10_pressures

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = self._reaction_network.get_residual(
            system=self, constraints=constraints, coefficient_matrix=coefficient_matrix
        )

        # Compute residual for the mass balance.
        residual_mass: np.ndarray = np.zeros(len(constraints.mass_constraints), dtype=np.float_)
        for constraint_index, constraint in enumerate(constraints.mass_constraints.values()):
            for species in self.species.gas_species.values():
                residual_mass[constraint_index] += species.mass(
                    planet=self.planet,
                    system=self,
                    element=constraint.species,
                )
            # Mass values are constant so no need to pass any arguments to get_value().
            residual_mass[constraint_index] -= constraint.get_value()
            # Normalise by target mass to compute a relative residual.
            residual_mass[constraint_index] /= constraint.get_value()
        logger.debug("Residual_mass = %s", residual_mass)

        # Combined residual.
        residual: np.ndarray = np.concatenate((residual_reaction, residual_mass))
        logger.debug("Residual = %s", residual)

        return residual
