#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Interior atmosphere system"""

from __future__ import annotations

import logging
import pprint
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError
from scipy.optimize import OptimizeResult, root
from sklearn.metrics import mean_squared_error

from atmodeller import AVOGADRO, GAS_CONSTANT
from atmodeller.constraints import SystemConstraints, TotalPressureConstraint
from atmodeller.core import GasSpecies, Planet, Species
from atmodeller.initial_solution import InitialSolutionDict, InitialSolutionProtocol
from atmodeller.output import Output
from atmodeller.reaction_network import ReactionNetworkWithCondensateStability
from atmodeller.solution import Solution
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


# TODO: Renamed, was element_moles
# @property
# def moles_of_elements(self) -> float:
#     """Total number of moles of elements in the atmosphere"""
#     number_density: float = 0
#     for element in self.species.elements():
#         number_density += self.element_gas_number_density(element)["atmosphere_number_density"]

#     moles = self.number_density_to_moles(number_density)

#     return moles


@dataclass(kw_only=True)
class InteriorAtmosphereSystem:
    """An interior-atmosphere system

    Args:
        species: A list of species
        planet: A planet. Defaults to a molten Earth
        initial_solution: Initial solution. Defaults to a constant for all species.
    """

    species: Species
    """A list of species"""
    planet: Planet = field(default_factory=Planet)
    """A planet"""
    initial_solution: InitialSolutionProtocol | None = None
    """Initial solution"""
    output: Output = field(init=False, default_factory=Output)
    """Output data"""
    _reaction_network: ReactionNetworkWithCondensateStability = field(init=False)
    # Convenient to set and update on this instance.
    _constraints: SystemConstraints = field(init=False, default_factory=SystemConstraints)
    _solution: Solution = field(init=False)
    _residual: npt.NDArray = field(init=False)
    _attempted_solves: int = field(init=False, default=0)
    _failed_solves: int = field(init=False, default=0)

    def __post_init__(self):
        logger.info("Creating an interior-atmosphere system")
        self.species.conform_solubilities_to_composition(self.planet.melt_composition)
        if self.initial_solution is None:
            self.initial_solution = InitialSolutionDict(species=self.species)
        self._reaction_network = ReactionNetworkWithCondensateStability(self.species)
        self._solution = Solution(self.species)

    @property
    def constraints(self) -> SystemConstraints:
        """Constraints"""
        return self._constraints

    @property
    def failed_solves(self) -> int:
        """Number of failed solves"""
        percentage_failed: float = self._failed_solves * 100 / self.number_of_attempted_solves
        logger.info(
            "%d failed solves from a total attempted of %d (%.1f %%)",
            self._failed_solves,
            self.number_of_attempted_solves,
            percentage_failed,
        )

        return self._failed_solves

    @property
    def number_of_attempted_solves(self) -> int:
        """The total number of systems with attempted solves"""
        return self._attempted_solves

    @property
    def number_of_solves(self) -> int:
        """The total number of systems solved"""
        return self.output.size

    @property
    def solution(self) -> Solution:
        """The solution"""
        return self._solution

    @property
    def gas_volume(self) -> float:
        """Total volume of the atmosphere

        Derived using the mechanical pressure balance due to the weight of the atmosphere and the
        ideal gas equation of state.
        """
        volume: float = self.planet.surface_area / self.planet.surface_gravity
        volume *= (
            GAS_CONSTANT * self.planet.surface_temperature / self.solution.gas.mean_molar_mass
        )

        return volume

    def residual_dict(self) -> dict[str, float]:
        """Residual of the objective function

        The order of the constraints must align with the order in which they are assembled.
        """
        output: dict[str, float] = {}
        for index, reaction in enumerate(self._reaction_network.reactions().values()):
            output[reaction] = self._residual[index]
        for index, constraint in enumerate(self.constraints.reaction_network_constraints):
            row_index: int = self._reaction_network.number_reactions + index
            output[constraint.name] = self._residual[row_index]
        for index, constraint in enumerate(self.constraints.mass_constraints):
            row_index = (
                self._reaction_network.number_reactions
                + self.constraints.number_reaction_network_constraints
                + index
            )
            output[constraint.name] = self._residual[row_index]
        for index, constraint in enumerate(self.constraints.total_pressure_constraint):
            output[constraint.name] = self._residual[-1]  # Always last index if applied

        output["rms"] = np.sqrt(np.mean(np.array(list(output.values())) ** 2))

        return output

    def condensed_element_masses(self) -> dict[str, float]:
        """Calculates the number densities of elements in all condensed species.

        Returns:
            Dictionary of elements and their number densities in all condensed species
        """
        condensed_element_masses: dict[str, float] = {
            element: 0 for element in self.species.elements()
        }
        for species in self.species.condensed_species:
            number_density: float = self.solution.condensed[species].mass.physical
            for element, value in species.composition().items():
                condensed_element_masses[element] += number_density * value.count

        logger.debug("condensed_element_masses = %s", condensed_element_masses)

        return condensed_element_masses

    def gas_species_reservoir_masses(
        self,
        species: GasSpecies,
    ) -> dict[str, float]:
        """Calculates the masses of the gas species in the atmosphere-mantle system

        Additional quantities are saved during the calculation for subsequent (and self-consistent)
        output.

        Args:
            species: A gas species

        Returns:
            A dictionary that includes the reservoir masses of the species
        """
        temperature: float = self.planet.surface_temperature
        pressure: float = self.solution.gas.gas_pressure(temperature)

        output: dict[str, float] = {}
        output["atmosphere_number_density"] = self.solution.gas[species].physical
        output["pressure"] = self.solution.gas[species].pressure(temperature)
        output["fugacity"] = self.solution.gas[species].fugacity(
            temperature,
            pressure,
        )

        output["melt_ppmw"] = species.solubility.concentration(
            fugacity=output["fugacity"],
            temperature=temperature,
            pressure=self.solution.gas.gas_pressure(temperature),
            **self.solution.gas.fugacities_by_hill_formula(temperature, pressure),
        )
        # Numerator to molecules
        output["melt_number_density"] = (
            output["melt_ppmw"] * UnitConversion.ppm_to_fraction() * AVOGADRO / species.molar_mass
        )
        output["melt_number_density"] *= self.planet.mantle_melt_mass / self.gas_volume

        output["solid_ppmw"] = output["melt_ppmw"] * species.solid_melt_distribution_coefficient
        output["solid_number_density"] = (
            output["solid_ppmw"] * UnitConversion.ppm_to_fraction() * AVOGADRO / species.molar_mass
        )
        output["solid_number_density"] *= self.planet.mantle_solid_mass / self.gas_volume

        return output

    def element_number_density_in_gas_species_reservoirs(self, species: GasSpecies, element: str):
        """Calculates the number density of an element in the reservoirs of a gas species.

        Args:
            species: A gas species
            element: Compute the number density for the element in the species.

        Returns:
            Element number density in the gas species reservoirs
        """
        output: dict[str, float] = self.gas_species_reservoir_masses(species)

        number_density: dict[str, float] = {
            "atmosphere_number_density": output["atmosphere_number_density"],
            "melt_number_density": output["melt_number_density"],
            "solid_number_density": output["solid_number_density"],
        }

        try:
            element_count: int = species.composition()[element].count
        except KeyError:
            # Element not in formula so number density is zero.
            element_count = 0

        for key in number_density:
            number_density[key] *= element_count

        return number_density

    def element_gas_number_density(self, element: str) -> dict[str, float]:
        """Calculates the number density of an element in all gas species in each reservoir.

        Args:
            element: Element to compute the number density for.

        Returns:
            Gas reservoir number densities of the element
        """
        number_density: dict[str, float] = {
            "atmosphere_number_density": 0,
            "melt_number_density": 0,
            "solid_number_density": 0,
        }

        for species in self.species.gas_species:
            species_number_density: dict[str, float] = (
                self.element_number_density_in_gas_species_reservoirs(species, element)
            )
            for key, value in species_number_density.items():
                number_density[key] += value

        logger.debug("element_gas_number_density for %s = %s", element, number_density)

        return number_density

    def element_number_density(self, element: str) -> dict[str, float]:
        """Calculates the number density of an element.

        Args:
            element: Element to compute the number density for.

        Returns:
            Total mass of the element
        """
        element_number_density: dict[str, float] = self.element_gas_number_density(element)
        element_number_density["condensed"] = self.condensed_element_masses()[element]

        logger.debug("element_number_density for %s = %s", element, element_number_density)

        return element_number_density

    def get_number_density_residual(self) -> npt.NDArray[np.float_]:
        """Returns the residual vector of the number density balance."""
        residual_number_density: npt.NDArray[np.float_] = np.zeros(
            len(self.constraints.mass_constraints), dtype=np.float_
        )
        for index, constraint in enumerate(self.constraints.mass_constraints):
            residual_number_density[index] = np.log10(
                sum(self.element_number_density(constraint.element).values())
            )
            residual_number_density[index] -= constraint.log10_number_of_molecules - np.log10(
                self.gas_volume
            )

        logger.debug("residual_number_density = %s", residual_number_density)

        return residual_number_density

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        initial_solution: InitialSolutionProtocol | None = None,
        extra_output: dict[str, float] | None = None,
        max_attempts: int = 20,
        perturb_gas_log10: float = 2.0,
        errors: str = "ignore",
        method: str = "hybr",
        tol: float | None = None,
        **options,
    ) -> None:
        """Solves the system to determine the activities and partial pressures with constraints.

        Args:
            constraints: Constraints for the system of equations
            initial_solution: Initial condition for this solve only. Defaults to 'None', meaning
                that the default (self.initial_solution) is used.
            extra_output: Extra data to write to the output
            max_attempts: Maximum number of attempts to randomise the initial condition to find a
                solution if the initial guess fails. Defaults to 10.
            perturb_gas_log10: Maximum log10 perturbation to apply to the pressures on failure.
                Defaults to 2.0.
            errors: Either 'raise' solver errors or 'ignore'. Defaults to 'ignore'.
            method: Type of solver. Defaults to 'hybr'.
            tol: Tolerance for termination. Defaults to None.
            **options: Keyword arguments for solver options. Available keywords depend on method.
        """
        logger.info("Solving system number %d", self.number_of_solves)
        self._attempted_solves += 1

        self._constraints = constraints
        self._constraints.add_activity_constraints(self.species)

        if initial_solution is None:
            initial_solution = self.initial_solution
        assert initial_solution is not None

        coefficient_matrix: npt.NDArray = self._reaction_network.get_coefficient_matrix(
            self.constraints
        )
        activity_modifier: npt.NDArray = self._reaction_network.get_activity_modifier(
            self.constraints
        )
        equilibrium_modifier: npt.NDArray = self._reaction_network.get_equilibrium_modifier(
            self.constraints
        )

        for attempt in range(max_attempts):
            logger.info("Attempt %d/%d", attempt + 1, max_attempts)

            # The only constraints that require pressure are the fugacity constraints, so for the
            # purpose of determining the initial solution we evaluate them (if present) at 1 bar to
            # ensure the initial solution is bounded.
            log_solution: npt.NDArray = initial_solution.get_log10_value(
                self.constraints,
                temperature=self.planet.surface_temperature,
                pressure=1,
                perturb_gas_log10=perturb_gas_log10,
                attempt=attempt,
            )
            logger.info("Initial solution = %s", log_solution)
            try:
                sol = root(
                    self._objective_func,
                    log_solution,
                    args=(
                        coefficient_matrix,
                        activity_modifier,
                        equilibrium_modifier,
                    ),
                    method=method,
                    tol=tol,
                    options=options,
                )
                logger.info(sol["message"])
                logger.debug("sol = %s", sol)

            except TypeError as exc:
                msg: str = (
                    f"{exc}\nAdditional context: Number of unknowns and constraints must be equal"
                )
                raise ValueError(msg) from exc

            except LinAlgError:
                if errors == "raise":
                    raise
                else:
                    logger.warning("Linear algebra error")
                    sol = OptimizeResult()
                    sol.success = False

            if sol.success:
                logger.info("Actual solution = %s", sol.x)
                error: npt.NDArray = np.sqrt(mean_squared_error(sol.x, log_solution))
                logger.info(
                    "%s: RMSE (actual vs initial) = %s",
                    self.initial_solution.__class__.__name__,
                    error,
                )
                self._log_solution = sol.x
                self._residual = sol.fun
                # FIXME: Need to add back in
                # self.output.add(self, extra_output)
                initial_solution.update(self.output)
                logger.info(pprint.pformat(self.solution_dict()))
                break
            else:
                logger.warning("The solver failed.")

        if not sol.success:
            msg: str = f"Solver failed after {max_attempts} attempt(s) (errors = {errors})"
            self._failed_solves += 1
            if errors == "raise":
                logger.error(msg)
                logger.error("constraints = %s", self.constraints)
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
                logger.warning("constraints = %s", self.constraints)
                logger.warning("Continuing with next solve")

    def _objective_func(
        self,
        log_solution: npt.NDArray,
        coefficient_matrix: npt.NDArray,
        activity_modifier: npt.NDArray,
        equilibrium_modifier: npt.NDArray,
    ) -> npt.NDArray[np.float_]:
        """Objective function for the non-linear system.

        Args:
            log_solution: Log10 of the activities and pressures of each species
            coefficient_matrix: Coefficient matrix
            activity_modifier: Activity modifier matrix for condensate stability
            equilibrium_modifier: Equilibrium modifier matrix for condensate stability

        Returns:
            The solution, which is the log10 of the activities and pressures for each species
        """
        # This must be set here.
        self._solution.data = log_solution

        logger.debug("log_solution passed into _objective_func = %s", log_solution)

        temperature: float = self.planet.surface_temperature
        pressure: float = self.solution.gas.gas_pressure(temperature)

        # Compute residual for the reaction network.
        residual_reaction: npt.NDArray = self._reaction_network.get_residual(
            temperature=temperature,
            pressure=pressure,
            constraints=self.constraints,
            coefficient_matrix=coefficient_matrix,
            activity_modifier=activity_modifier,
            equilibrium_modifier=equilibrium_modifier,
            solution=self.solution,
        )

        # Compute residual for the mass balance.
        residual_number_density: npt.NDArray[np.float_] = self.get_number_density_residual()

        # Compute residual for the total pressure (if relevant).
        residual_total_pressure: npt.NDArray = np.zeros(
            len(self.constraints.total_pressure_constraint), dtype=np.float_
        )
        if len(self.constraints.total_pressure_constraint) > 0:
            constraint: TotalPressureConstraint = self.constraints.total_pressure_constraint[0]
            residual_total_pressure[0] += np.log10(
                self.solution.gas.gas_number_density
            ) - constraint.get_log10_value(temperature=temperature, pressure=pressure)

        # Combined residual
        residual: npt.NDArray = np.concatenate(
            (residual_reaction, residual_number_density, residual_total_pressure)
        )
        logger.debug("residual = %s", residual)

        return residual

    def solution_dict(self) -> dict[str, float]:
        """Solution in a dictionary"""
        return self.solution.solution_dict(self.planet.surface_temperature, self.gas_volume)

    def isclose(
        self,
        target_dict: dict[str, float],
        rtol: float = 1.0e-5,
        atol: float = 1.0e-8,
    ) -> np.bool_:
        """Determines if the solution pressures are close to target values within a tolerance.

        Args:
            target_dict: Dictionary of the target values
            rtol: Relative tolerance. Defaults to 1.0E-5.
            atol: Absolute tolerance. Defaults to 1.0E-8.

        Returns:
            True if the solution is close to the target, otherwise False
        """
        if len((self.solution_dict())) != len(target_dict):
            return np.bool_(False)

        target_values: list = list(dict(sorted(target_dict.items())).values())
        solution_values: list = list(dict(sorted(self.solution_dict().items())).values())
        isclose: np.bool_ = np.isclose(target_values, solution_values, rtol=rtol, atol=atol).all()

        return isclose

    def isclose_tolerance(self, target_dict: dict[str, float], message: str = "") -> float | None:
        """Writes a log message with the tightest tolerance that is satisfied.

        Args:
            target_dict: Dictionary of the target values
            message: Message prefix to write to the logger when a tolerance is satisfied. Defaults
                to an empty string.

        Returns:
            The tightest tolerance satisfied
        """
        for log_tolerance in (-6, -5, -4, -3, -2, -1):
            tol: float = 10**log_tolerance
            if self.isclose(target_dict, rtol=tol, atol=tol):
                logger.info("%s (tol = %f)".lstrip(), message, tol)
                return tol

        logger.info("%s (no tolerance < 0.1 satisfied)".lstrip(), message)
