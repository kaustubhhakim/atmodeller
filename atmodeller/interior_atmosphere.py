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

import copy
import logging
import pprint
import sys
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError
from scipy.optimize import OptimizeResult, root
from sklearn.metrics import mean_squared_error

from atmodeller import GAS_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.core import GasSpecies, Planet, Solution, Species
from atmodeller.initial_solution import InitialSolutionConstant
from atmodeller.interfaces import (
    InitialSolutionProtocol,
    TotalPressureConstraintProtocol,
)
from atmodeller.output import Output
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

TAU: float = 1e-15
"""Tau factor for the calculation of the auxilliary equations for condensate stability"""
log10_TAU: float = np.log10(TAU)
"""Log10 of Tau"""


class ReactionNetwork:
    """Determines the reactions to solve a chemical network.

    Args:
        species: Species

    Attributes:
        species: Species
        reaction_matrix: The reaction stoichiometry matrix
    """

    def __init__(self, species: Species):
        self.species: Species = species
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self.species.names)
        self.reaction_matrix: npt.NDArray | None = self._partial_gaussian_elimination()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        if self.species.number_species() == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def _partial_gaussian_elimination(self) -> npt.NDArray | None:
        """Performs a partial gaussian elimination to determine the required reactions.

        The species composition matrix is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of species - number of
        elements. These reactions are given in the last r rows of the reduced matrix.

        Returns:
            A matrix of the reaction stoichiometry
        """
        if self.species.number_species() == 1:
            logger.debug("Only one species therefore no reactions")
            return None

        matrix1: npt.NDArray = self.species.composition_matrix()
        matrix2: npt.NDArray = np.eye(self.species.number_species())
        augmented_matrix: npt.NDArray = np.hstack((matrix1, matrix2))
        logger.debug("augmented_matrix = \n%s", augmented_matrix)

        # Forward elimination
        for i in range(self.species.number_elements()):  # Note only over the number of elements.
            # Check if the pivot element is zero.
            if augmented_matrix[i, i] == 0:
                # Swap rows to get a non-zero pivot element.
                nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
                augmented_matrix[[i, nonzero_row], :] = augmented_matrix[[nonzero_row, i], :]
            # Perform row operations to eliminate values below the pivot.
            for j in range(i + 1, self.species.number_species()):
                ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

        # Backward substitution
        for i in range(self.species.number_elements() - 1, -1, -1):
            # Normalize the pivot row.
            augmented_matrix[i] /= augmented_matrix[i, i]
            # Eliminate values above the pivot.
            for j in range(i - 1, -1, -1):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

        reduced_matrix1: npt.NDArray = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix: npt.NDArray = augmented_matrix[
            self.species.number_elements() :, matrix1.shape[1] :
        ]
        logger.debug("reduced_matrix1 = \n%s", reduced_matrix1)
        logger.debug("reaction_matrix = \n%s", reaction_matrix)

        return reaction_matrix

    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary"""
        reactions: dict[int, str] = {}
        if self.reaction_matrix is not None:
            for reaction_index in range(self.number_reactions):
                reactants: str = ""
                products: str = ""
                for species_index, species in enumerate(self.species.data):
                    coeff: float = self.reaction_matrix[reaction_index, species_index]
                    if coeff != 0:
                        if coeff < 0:
                            reactants += f"{abs(coeff)} {species.name} + "
                        else:
                            products += f"{coeff} {species.name} + "

                reactants = reactants.rstrip(" + ")
                products = products.rstrip(" + ")
                reaction: str = f"{reactants} = {products}"
                reactions[reaction_index] = reaction

        return reactions

    def _get_reaction_log10_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the log10 of the reaction equilibrium constant.

        From the Gibbs free energy, we can calculate logKf as:
        logKf = - G/(ln(10)*R*T)

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature
            pressure: Pressure

        Returns:
            log10 of the reaction equilibrium constant
        """
        gibbs_energy: float = self._get_reaction_gibbs_energy_of_formation(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        equilibrium_constant: float = -gibbs_energy / (np.log(10) * GAS_CONSTANT * temperature)

        return equilibrium_constant

    def _get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`
            temperature: Temperature
            pressure: Pressure

        Returns:
            The Gibb's free energy of the reaction
        """
        gibbs_energy: float = 0
        assert self.reaction_matrix is not None
        for species_index, species in enumerate(self.species.data):
            assert species.thermodata is not None
            gibbs_energy += self.reaction_matrix[
                reaction_index, species_index
            ] * species.thermodata.get_formation_gibbs(temperature=temperature, pressure=pressure)

        return gibbs_energy

    def get_coefficient_matrix(self, *, constraints: SystemConstraints) -> npt.NDArray:
        """Gets the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        coeff: npt.NDArray = np.zeros((nrows, self.species.number_species()))
        if self.reaction_matrix is not None:
            coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index = self.species.find_species(constraint.species)
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("species = %s", self.species.names)
        logger.debug("coefficient matrix = \n%s", coeff)

        return coeff

    def _assemble_right_hand_side_values(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
    ) -> npt.NDArray:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = self.number_reactions + constraints.number_reaction_network_constraints
        rhs: npt.NDArray = np.zeros(nrows, dtype=float)

        # Reactions
        for reaction_index in range(self.number_reactions):
            logger.debug(
                "Row %02d: Reaction %d: %s",
                reaction_index,
                reaction_index,
                self.reactions()[reaction_index],
            )
            rhs[reaction_index] = self._get_reaction_log10_equilibrium_constant(
                reaction_index=reaction_index,
                temperature=temperature,
                pressure=pressure,
            )

        # Constraints
        for index, constraint in enumerate(constraints.reaction_network_constraints):
            row_index: int = self.number_reactions + index
            logger.debug("Row %02d: Setting %s %s", row_index, constraint.species, constraint.name)
            rhs[row_index] = constraint.get_log10_value(temperature=temperature, pressure=pressure)

        logger.debug("RHS vector = %s", rhs)

        return rhs

    def _assemble_log_fugacity_coefficients(
        self,
        *,
        temperature: float,
        pressure: float,
    ) -> npt.NDArray:
        """Assembles the fugacity coefficient vector on the left-hand side of the equations.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The log10(fugacity coefficient) vector
        """

        # Initialise to ideal behaviour.
        fugacity_coefficients: npt.NDArray = np.ones_like(self.species, dtype=float)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for gas_species in self.species.gas_species:
            fugacity_coefficient: float = gas_species.eos.fugacity_coefficient(
                temperature=temperature, pressure=pressure
            )
            fugacity_coefficients[self.species.find_species(gas_species)] = fugacity_coefficient

        log_fugacity_coefficients: npt.NDArray = np.log10(fugacity_coefficients)
        logger.debug("Log10 fugacity coefficient vector = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray,
        solution: Solution,
        **kwargs,
    ) -> npt.NDArray:
        """Returns the residual vector of the reaction network.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations
            coefficient_matrix: Coefficient matrix
            solution: Solution to compute the residual for
            **kwargs: Catches additional keyword arguments used by child classes

        Returns:
            The residual vector of the reaction network
        """
        del kwargs
        log_fugacity_coefficients: npt.NDArray = self._assemble_log_fugacity_coefficients(
            temperature=temperature, pressure=pressure
        )
        rhs: npt.NDArray = self._assemble_right_hand_side_values(
            temperature=temperature, pressure=pressure, constraints=constraints
        )
        residual_reaction: npt.NDArray = (
            coefficient_matrix.dot(log_fugacity_coefficients)
            + coefficient_matrix.dot(solution.species_array)
            - rhs
        )

        logger.debug("residual_reaction = %s", residual_reaction)

        return residual_reaction


class ReactionNetworkWithCondensateStability(ReactionNetwork):
    """A reaction network with condensate stability

    This automatically determines condensate stability using the extended law of mass-action
    (xLMA) equations :cite:p:`LKS17`. Also see :cite:p:`KSP42`.

    Args:
        species: Species

    Attributes:
        species: Species
        reaction_matrix: The reaction stoichiometry matrix
    """

    def get_activity_modifier(
        self, *, constraints: SystemConstraints, solution: Solution
    ) -> npt.NDArray:
        """Gets the activity modifier matrix for condensate stability

        Args:
            constraints: Constraints
            solution: Solution

        Returns:
            Activity modifier matrix
        """
        coefficient_matrix: npt.NDArray = self.get_coefficient_matrix(constraints=constraints)
        activity_modifier: npt.NDArray = np.zeros_like(coefficient_matrix)
        for species in solution.condensed_species_to_solve:
            index: int = self.species.find_species(species)
            activity_modifier[:, index] = coefficient_matrix[:, index]

        logger.debug("activity_modifier = %s", activity_modifier)

        return activity_modifier

    def get_equilibrium_modifier(
        self, *, constraints: SystemConstraints, solution: Solution
    ) -> npt.NDArray:
        """Gets the equilibrium constant modifier matrix for condensate stability

        Args:
            constraints: Constraints
            solution: Solution

        Returns:
            Equilibrium constant modifier matrix
        """
        activity_modifier: npt.NDArray = self.get_activity_modifier(
            constraints=constraints, solution=solution
        )
        equilibrium_modifier: npt.NDArray = copy.deepcopy(activity_modifier)
        equilibrium_modifier[self.number_reactions :, :] = 0

        logger.debug("equilibrium_modifier = %s", equilibrium_modifier)

        return equilibrium_modifier

    def get_stability_residual(self, solution: Solution) -> npt.NDArray:
        """Returns the residual vector of condensate stability

        Args:
            solution: Solution to compute the residual for

        Returns:
            The residual vector of condensate stability
        """
        residual_stability: npt.NDArray = np.zeros(
            solution.number_condensed_species_to_solve, dtype=np.float_
        )
        for nn, species in enumerate(solution.condensed_species_to_solve):
            residual_stability[nn] = solution._lambda_solution[species] - log10_TAU
            # The xLMA usually uses the condensate number density or similar, but it's simpler to
            # satisfy the auxiliary equations using the condensed mass of elements in the
            # condensate, which we have direct access to.
            for element in species.elements:
                try:
                    residual_stability += solution._beta_solution[element]
                except KeyError:
                    pass

        logger.debug("residual_stability = %s", residual_stability)

        return residual_stability

    @override
    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray,
        solution: Solution,
        activity_modifier: npt.NDArray,
        equilibrium_modifier: npt.NDArray,
    ) -> npt.NDArray:

        # Residual of the reaction network without a stability consideration
        residual_reaction: npt.NDArray = super().get_residual(
            temperature=temperature,
            pressure=pressure,
            constraints=constraints,
            coefficient_matrix=coefficient_matrix,
            solution=solution,
        )

        # Reaction network correction factors for condensate stability
        residual_reaction += activity_modifier.dot(10**solution.lambda_array)
        residual_reaction -= equilibrium_modifier.dot(10**solution.lambda_array)

        # Residual for the auxiliary stability equations
        residual_stability: npt.NDArray = self.get_stability_residual(solution)

        residual: npt.NDArray = np.concatenate((residual_reaction, residual_stability))
        logger.debug("residual_reaction = %s", residual_reaction)

        return residual


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
    def solution(self) -> Solution:
        """The solution"""
        return self._solution

    @property
    def number_of_solves(self) -> int:
        """The total number of systems solved"""
        return self.output.size

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
            self.planet.surface_area * species.molar_mass / self.atmospheric_mean_molar_mass
        )

        # Melt
        output["melt_ppmw"] = species.solubility.concentration(
            fugacity=output["fugacity"],
            temperature=self.planet.surface_temperature,
            pressure=self.total_pressure,
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

    @property
    def total_mass(self) -> float:
        """Total mass"""
        mass: float = UnitConversion.bar_to_Pa(self.total_pressure) / self.planet.surface_gravity
        mass *= self.planet.surface_area

        return mass

    @property
    def total_pressure(self) -> float:
        """Total pressure"""
        return self.solution.total_pressure

    @property
    def atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere"""
        return self.solution.gas_mean_molar_mass

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        initial_solution: InitialSolutionProtocol | None = None,
        extra_output: dict[str, float] | None = None,
        max_attempts: int = 50,
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
            method: Type of solver. Defaults to 'hybr'.
            max_attempts: Maximum number of attempts to randomise the initial condition to find a
                solution if the initial guess fails.
            perturb_log10: Maximum log10 perturbation to apply to the initial condition on failure.
                Defaults to 2.0.
            errors: Either 'raise' solver errors or 'ignore'. Defaults to 'ignore'.
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
            pressure=self.total_pressure,
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
                self.total_pressure
            ) - constraint.get_log10_value(
                temperature=self.planet.surface_temperature, pressure=self.total_pressure
            )

        # Combined residual
        residual: npt.NDArray = np.concatenate(
            (residual_reaction, residual_mass, residual_total_pressure)
        )
        logger.debug("residual = %s", residual)

        return residual
