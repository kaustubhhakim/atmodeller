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
import pandas as pd
from molmass import Formula
from scipy.linalg import LinAlgError
from scipy.optimize import OptimizeResult, root
from sklearn.metrics import mean_squared_error

from atmodeller.constraints import SystemConstraints
from atmodeller.core import GasSpecies, Planet, Solution, Species
from atmodeller.initial_solution import InitialSolutionConstant
from atmodeller.interfaces import (
    CondensedSpecies,
    InitialSolutionProtocol,
    TotalPressureConstraintProtocol,
)
from atmodeller.output import Output
from atmodeller.reaction_network import ReactionNetworkWithCondensateStability
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


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
    _failed_solves: int = 0

    def __post_init__(self):
        logger.info("Creating an interior-atmosphere system")
        self.species.conform_solubilities_to_composition(self.planet.melt_composition)
        if self.initial_solution is None:
            self.initial_solution = InitialSolutionConstant(species=self.species)
        self._reaction_network = ReactionNetworkWithCondensateStability(species=self.species)

    @property
    def atmosphere_element_moles(self) -> float:
        """Total number of moles of elements in the atmosphere"""
        element_moles: float = 0
        for element in self.species.elements():
            molar_mass: float = UnitConversion.g_to_kg(Formula(element).mass)
            element_moles += self.element_gas_mass(element)["atmosphere_mass"] / molar_mass

        return element_moles

    @property
    def atmosphere_mass(self) -> float:
        """Total mass of the atmosphere"""
        mass: float = (
            UnitConversion.bar_to_Pa(self.atmosphere_pressure) / self.planet.surface_gravity
        )
        mass *= self.planet.surface_area

        return mass

    @property
    def atmosphere_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere"""
        return self.solution.gas_mean_molar_mass

    @property
    def atmosphere_pressure(self) -> float:
        """Total pressure of the atmosphere"""
        return self.solution.total_pressure

    @property
    def atmosphere_species_moles(self) -> float:
        """Total number of moles of species in the atmosphere"""
        species_moles: float = 0
        for species in self.species.gas_species:
            species_moles += (
                self.gas_species_reservoir_masses(species)["atmosphere_mass"] / species.molar_mass
            )

        return species_moles

    @property
    def constraints(self) -> SystemConstraints:
        """Constraints"""
        return self._constraints

    @property
    def failed_solves(self) -> int:
        """Number of failed solves"""
        fraction: float = self._failed_solves / self.number_of_solves
        logger.info(
            "%d failed solves from a total of %d (%f %%)",
            self._failed_solves,
            self.number_of_solves,
            fraction,
        )

        return self._failed_solves

    @property
    def number_of_solves(self) -> int:
        """The total number of systems solved"""
        return self.output.size

    @property
    def solution(self) -> Solution:
        """The solution"""
        return self._solution

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

        return output

    def solution_dict(self) -> dict[str, float]:
        """Solution in a dictionary"""
        return self._solution.solution_dict()

    def condensed_species_masses(self) -> dict[CondensedSpecies, dict[str, float]]:
        """Computes the masses of condensed species and their elements

        This follows the approach outlined in :cite:t:`KSP24{Appendix B}`, albeit simplified under
        the assumption of a handful of linearly independent condensates that can be solved for in
        a single solve, i.e. no iteration is required.

        Returns:
            Dictionary of condensed species and their element masses
        """
        condensed_species_masses: dict[CondensedSpecies, dict[str, float]] = {}
        condensed_elements: list[str] = self.solution.condensed_elements_to_solve
        condensed_species: list[CondensedSpecies] = self.solution.condensed_species_to_solve
        mapping: npt.NDArray = np.zeros((len(condensed_elements), len(condensed_species)))

        if mapping.size < 1:
            return condensed_species_masses

        # Assemble matrices
        # TODO: If both H and O prescribed as constraints, then drop one based on the one that is
        # the minimum stoichiometry bottleneck for H2O. So keep the minimum and drop the maximum.
        # Then determine the amount of the other quantity and compare to the prescribed constraint
        # Should correct original constraint to be self-consistent.
        element_condensed_mass: list[float] = []
        element_condensed_moles: list[float] = []
        for ii, condensed_element in enumerate(condensed_elements):
            condensed_mass: float = self.element_condensed_mass(condensed_element)
            element_condensed_mass.append(condensed_mass)
            condensed_moles: float = condensed_mass / UnitConversion.g_to_kg(
                Formula(condensed_element).mass
            )
            element_condensed_moles.append(condensed_moles)
            for jj, species in enumerate(condensed_species):
                if condensed_element in species.composition():
                    mapping[ii, jj] = species.composition()[condensed_element].fraction

        element_condensed_mass_dict: dict[str, float] = dict(
            zip(condensed_elements, element_condensed_mass)
        )
        element_condensed_moles_dict: dict[str, float] = dict(
            zip(condensed_elements, element_condensed_moles)
        )

        logger.debug("element_condensed_mass = %s", element_condensed_mass_dict)
        logger.debug("element_condensed_moles = %s", element_condensed_moles_dict)
        logger.debug("mapping = %s", mapping)

        # For debugging element partitioning in condensates.
        if "H" in element_condensed_mass_dict and "O" in element_condensed_mass_dict:
            logger.debug("Compute H/O ratio in condensed phase")
            HO_ratio_mass: float = (  # pylint: disable=invalid-name
                element_condensed_mass_dict["H"] / element_condensed_mass_dict["O"]
            )
            HO_ratio_moles: float = (  # pylint: disable=invalid-name
                element_condensed_moles_dict["H"] / element_condensed_moles_dict["O"]
            )
            logger.debug("H/O (mass) = %f, H/O (moles) = %f", HO_ratio_mass, HO_ratio_moles)
            logger.debug(
                "O/H (mass) = %f, O/H (moles) = %f", 1 / HO_ratio_mass, 1 / HO_ratio_moles
            )

        # Count how many condensates can be associated with each element
        associations: npt.NDArray = np.count_nonzero(mapping, axis=1)
        logger.debug("associations = %s", associations)
        # Enforce conditions for a single solve, which avoids the complication of having to iterate
        try:
            assert np.all(associations == 1)
        except AssertionError as exc:
            raise AssertionError(
                "Every element can only be associated with a single condensate"
            ) from exc

        element_condensed_mass_ar: npt.NDArray = np.array(
            element_condensed_mass, dtype=np.float_
        ).reshape(-1, 1)
        condensed_masses: npt.NDArray = np.linalg.solve(
            mapping, element_condensed_mass_ar
        ).flatten()

        # Necessary to back-compute some species that might not be constrained by mass balance,
        # for example oxygen is often constrained by fO2 and not by abundance, but we want to know
        # how much oxygen is in the system.
        for nn, species in enumerate(condensed_species):
            composition: pd.DataFrame = species.composition().dataframe()
            composition["Mass"] = condensed_masses[nn] * composition["Fraction"]
            logger.debug("composition = %s", composition)
            condensed_species_masses[species] = composition.to_dict()["Mass"]

        return condensed_species_masses

    def condensed_element_masses(self) -> dict[str, float]:
        """Calculates the computed and implied masses of elements in all condensed species.

        TODO: Different to element_condensed_mass this computes the implied masses as well.

        Returns:
            Dictionary of elements and their masses in all condensed species
        """
        condensed_species_masses = self.condensed_species_masses()
        condensed_element_masses: dict[str, float] = {}
        for element_masses in condensed_species_masses.values():
            for element, value in element_masses.items():
                if element in condensed_element_masses:
                    condensed_element_masses[element] += value
                else:
                    condensed_element_masses[element] = value

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
        output: dict[str, float] = {}
        output["pressure"] = self.solution.gas_pressures[species]
        output["fugacity"] = self.solution.gas_fugacities[species]

        # Atmosphere
        output["atmosphere_mass"] = (
            UnitConversion.bar_to_Pa(output["pressure"]) / self.planet.surface_gravity
        )
        output["atmosphere_mass"] *= (
            self.planet.surface_area * species.molar_mass / self.atmosphere_molar_mass
        )

        # Melt
        output["melt_ppmw"] = species.solubility.concentration(
            fugacity=output["fugacity"],
            temperature=self.planet.surface_temperature,
            pressure=self.atmosphere_pressure,
            **self.solution.gas_fugacities_by_hill_formula,
        )
        output["melt_mass"] = (
            self.planet.mantle_melt_mass * output["melt_ppmw"] * UnitConversion.ppm_to_fraction()
        )

        # Trapped in the solid mantle
        output["solid_ppmw"] = output["melt_ppmw"] * species.solid_melt_distribution_coefficient
        output["solid_mass"] = (
            self.planet.mantle_solid_mass * output["solid_ppmw"] * UnitConversion.ppm_to_fraction()
        )

        return output

    def element_mass_in_gas_species_reservoirs(self, species: GasSpecies, element: str):
        """Calculates the mass of an element in the reservoirs of a gas species.

        Args:
            species: A gas species
            element: Compute the mass for the element in the species.

        Returns:
            Element mass in the gas species reservoirs
        """
        output: dict[str, float] = self.gas_species_reservoir_masses(species)

        mass: dict[str, float] = {
            "atmosphere_mass": output["atmosphere_mass"],
            "melt_mass": output["melt_mass"],
            "solid_mass": output["solid_mass"],
        }

        try:
            mass_scale_factor: float = (
                UnitConversion.g_to_kg(species.composition()[element].mass) / species.molar_mass
            )
        except KeyError:  # Element not in formula so mass is zero.
            mass_scale_factor = 0
        for key in mass:
            mass[key] *= mass_scale_factor

        return mass

    def element_gas_mass(self, element: str) -> dict[str, float]:
        """Calculates the mass of an element in all gas species in each reservoir.

        Args:
            element: Element to compute the mass for.

        Returns:
            Gas reservoir masses of the element
        """
        mass: dict[str, float] = {"atmosphere_mass": 0, "melt_mass": 0, "solid_mass": 0}

        for species in self.species.gas_species:
            species_mass: dict[str, float] = self.element_mass_in_gas_species_reservoirs(
                species, element
            )
            for key, value in species_mass.items():
                mass[key] += value

        logger.debug("element_gas_mass for %s = %s", element, mass)

        return mass

    def element_condensed_mass(self, element: str) -> float:
        """Calculates the mass of an element in all condensed species.

        Args:
            element: Element to compute the mass for.

        Returns:
            Condensed mass of the element
        """
        if element in self.solution.condensed_elements_to_solve:
            mass = sum(self.element_gas_mass(element).values())
            mass *= 10 ** self.solution._beta_solution[element]
        else:
            mass = 0

        logger.debug("element_condensed_mass for %s = %s", element, mass)

        return mass

    def element_mass(self, element: str) -> dict[str, float]:
        """Calculates the mass of an element.

        Args:
            element: Element to compute the mass for.

        Returns:
            Total mass of the element
        """
        element_mass: dict[str, float] = self.element_gas_mass(element)
        element_mass["condensed"] = self.element_condensed_mass(element)

        logger.debug("element_mass for %s = %s", element, element_mass)

        return element_mass

    def get_mass_residual(self):
        """Returns the residual vector of the mass balance."""

        residual_mass: npt.NDArray = np.zeros(
            len(self.constraints.mass_constraints), dtype=np.float_
        )

        # Mass constraints are currently only ever specified in terms of elements. Hence
        # constraint.species is an element.
        for constraint_index, mass_constraint in enumerate(self.constraints.mass_constraints):
            residual_mass[constraint_index] = np.log10(
                sum(self.element_mass(mass_constraint.element).values())
            )
            residual_mass[constraint_index] -= mass_constraint.get_log10_value()

        logger.debug("residual_mass = %s", residual_mass)

        return residual_mass

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        initial_solution: InitialSolutionProtocol | None = None,
        extra_output: dict[str, float] | None = None,
        max_attempts: int = 10,
        perturb_log10: float = 2.0,
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
            perturb_log10: Maximum log10 perturbation to apply to the initial condition on failure.
                Defaults to 2.0.
            errors: Either 'raise' solver errors or 'ignore'. Defaults to 'ignore'.
            method: Type of solver. Defaults to 'hybr'.
            tol: Tolerance for termination. Defaults to None.
            **options: Keyword arguments for solver options. Available keywords depend on method.
        """
        logger.info("Solving system number %d", self.number_of_solves)

        self._constraints = constraints
        self._constraints.add_activity_constraints(self.species)

        self._solution = Solution(self.species, self.constraints, self.planet.surface_temperature)

        if initial_solution is None:
            initial_solution = self.initial_solution
        assert initial_solution is not None

        # These matrices depend on the constraints, but can be computed once for any given solve
        coefficient_matrix: npt.NDArray = self._reaction_network.get_coefficient_matrix(
            constraints=self.constraints
        )
        activity_modifier: npt.NDArray = self._reaction_network.get_activity_modifier(
            constraints=constraints, solution=self.solution
        )
        equilibrium_modifier: npt.NDArray = self._reaction_network.get_equilibrium_modifier(
            constraints=constraints, solution=self.solution
        )

        # The only constraints that require pressure are the fugacity constraints, so for the
        # purpose of determining the initial solution we evaluate them (if present) at 1 bar to
        # ensure the initial solution is bounded.
        log_solution: npt.NDArray = initial_solution.get_log10_value(
            self.constraints,
            temperature=self.planet.surface_temperature,
            pressure=1,
            degree_of_condensation_number=self.solution.number_condensed_elements_to_solve,
            number_of_condensed_species=self.solution.number_condensed_species_to_solve,
        )

        for attempt in range(max_attempts):
            logger.info("Attempt %d/%d", attempt + 1, max_attempts)
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
                logger.debug("Actual solution = %s", sol.x)
                error: npt.NDArray = np.sqrt(mean_squared_error(sol.x, log_solution))
                logger.info(
                    "%s: RMSE (actual vs initial) = %s",
                    self.initial_solution.__class__.__name__,
                    error,
                )
                self._log_solution = sol.x
                self._residual = sol.fun
                self.output.add(self, extra_output)
                initial_solution.update(self.output)
                logger.info(pprint.pformat(self.solution.solution_dict()))
                break
            else:
                logger.warning("The solver failed.")
                if attempt < max_attempts - 1:
                    log_solution = initial_solution.get_log10_value(
                        self.constraints,
                        temperature=self.planet.surface_temperature,
                        pressure=1,
                        degree_of_condensation_number=self.solution.number_condensed_elements_to_solve,
                        number_of_condensed_species=self.solution.number_condensed_species_to_solve,
                        perturb=True,
                        perturb_log10=perturb_log10,
                    )

        if not sol.success:
            msg: str = f"Solver failed after {max_attempts} attempt(s) (errors = {errors})"
            self._failed_solves += 1
            if self.solution.number_condensed_elements_to_solve > 0:
                logger.info("Probably no solution for condensed species and imposed constraints")
                logger.info("Remove some condensed species and try again")

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
    ) -> npt.NDArray:
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

        # Compute residual for the reaction network.
        residual_reaction: npt.NDArray = self._reaction_network.get_residual(
            temperature=self.planet.surface_temperature,
            pressure=self.atmosphere_pressure,
            constraints=self.constraints,
            coefficient_matrix=coefficient_matrix,
            activity_modifier=activity_modifier,
            equilibrium_modifier=equilibrium_modifier,
            solution=self.solution,
        )

        # Compute residual for the mass balance.
        residual_mass: npt.NDArray = self.get_mass_residual()

        # Compute residual for the total pressure (if relevant).
        residual_total_pressure: npt.NDArray = np.zeros(
            len(self.constraints.total_pressure_constraint), dtype=np.float_
        )
        if len(self.constraints.total_pressure_constraint) > 0:
            constraint: TotalPressureConstraintProtocol = (
                self.constraints.total_pressure_constraint[0]
            )
            residual_total_pressure[0] += np.log10(
                self.atmosphere_pressure
            ) - constraint.get_log10_value(
                temperature=self.planet.surface_temperature, pressure=self.atmosphere_pressure
            )

        # Combined residual
        residual: npt.NDArray = np.concatenate(
            (residual_reaction, residual_mass, residual_total_pressure)
        )
        logger.debug("residual = %s", residual)

        return residual
