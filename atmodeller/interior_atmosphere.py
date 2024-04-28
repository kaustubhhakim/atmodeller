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
from scipy.linalg import LinAlgError
from scipy.optimize import OptimizeResult, root
from sklearn.metrics import mean_squared_error

from atmodeller import GAS_CONSTANT, GRAVITATIONAL_CONSTANT
from atmodeller.constraints import Constraint, SystemConstraints
from atmodeller.core import Species
from atmodeller.initial_solution import InitialSolution, InitialSolutionConstant
from atmodeller.output import Output
from atmodeller.utilities import UnitConversion, dataclass_to_logger

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet

    Defines the properties of a planet that are relevant for interior modeling. It provides default
    values suitable for modelling a fully molten Earth-like planet.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to Earth.
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        surface_radius: Radius of the planetary surface in m. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
        melt_composition: Melt composition of the planet. Default is None.
    """

    planet_mass: float = 5.972e24
    """Mass of the planet in kg"""
    core_mass_fraction: float = 0.295334691460966
    """Mass fraction of the core relative to the planetary mass (kg/kg)"""
    mantle_melt_fraction: float = 1.0
    """Mass fraction of the mantle that is molten"""
    surface_radius: float = 6371000.0
    """Radius of the surface in m"""
    surface_temperature: float = 2000.0
    """Temperature of the surface in K"""
    melt_composition: str | None = None
    """Melt composition"""
    mantle_mass: float = field(init=False)
    """Mass of the mantle"""
    mantle_melt_mass: float = field(init=False)
    """Mass of the mantle that is molten"""
    mantle_solid_mass: float = field(init=False)
    """Mass of the mantle that is solid"""
    surface_area: float = field(init=False)
    """Surface area"""
    surface_gravity: float = field(init=False)
    """Surface gravity"""

    def __post_init__(self):
        self.mantle_mass = self.planet_mass * (1 - self.core_mass_fraction)
        self.mantle_melt_mass = self.mantle_mass * self.mantle_melt_fraction
        self.mantle_solid_mass = self.mantle_mass * (1 - self.mantle_melt_fraction)
        self.surface_area = 4.0 * np.pi * self.surface_radius**2
        self.surface_gravity = GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2
        logger.info("Creating a new planet")
        dataclass_to_logger(self, logger)


class _ReactionNetwork:
    """Determines the reactions to solve a chemical network.

    Args:
        species: Species

    Attributes:
        species: Species
        species_matrix: The stoichiometry matrix of the species in terms of elements
        reaction_matrix: The reaction stoichiometry matrix
    """

    def __init__(self, species: Species):
        self.species: Species = species
        logger.info("Creating a reaction network")
        logger.info("Species = %s", self.species.names)
        self.reaction_matrix: np.ndarray | None = self._partial_gaussian_elimination()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions()))

    @property
    def number_reactions(self) -> int:
        if self.species.number == 1:
            return 0
        else:
            assert self.reaction_matrix is not None
            return self.reaction_matrix.shape[0]

    def _partial_gaussian_elimination(self) -> np.ndarray | None:
        """Performs a partial gaussian elimination to determine the required reactions.

        A copy of `self.species_matrix` is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of species - number of
        elements. These reactions are given in the last r rows of the reduced matrix.

        Returns:
            A matrix of the reaction stoichiometry.
        """
        if self.species.number == 1:
            logger.debug("Only one species therefore no reactions")
            return None

        matrix1: np.ndarray = self.species.composition_matrix()
        matrix2: np.ndarray = np.eye(self.species.number)
        augmented_matrix: np.ndarray = np.hstack((matrix1, matrix2))
        logger.debug("augmented_matrix = \n%s", augmented_matrix)

        # Forward elimination
        for i in range(self.species.number_elements):  # Note only over the number of elements.
            # Check if the pivot element is zero.
            if augmented_matrix[i, i] == 0:
                # Swap rows to get a non-zero pivot element.
                nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
                augmented_matrix[[i, nonzero_row], :] = augmented_matrix[[nonzero_row, i], :]
            # Perform row operations to eliminate values below the pivot.
            for j in range(i + 1, self.species.number):
                ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("Augmented_matrix after forward elimination = \n%s", augmented_matrix)

        # Backward substitution
        for i in range(self.species.number_elements - 1, -1, -1):
            # Normalize the pivot row.
            augmented_matrix[i] /= augmented_matrix[i, i]
            # Eliminate values above the pivot.
            for j in range(i - 1, -1, -1):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("Augmented_matrix after backward substitution = \n%s", augmented_matrix)

        reduced_matrix1: np.ndarray = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix: np.ndarray = augmented_matrix[
            self.species.number_elements :, matrix1.shape[1] :
        ]
        logger.debug("Reduced_matrix1 = \n%s", reduced_matrix1)
        logger.debug("Reaction_matrix = \n%s", reaction_matrix)

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

    def get_coefficient_matrix(self, *, constraints: SystemConstraints) -> np.ndarray:
        """Builds the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        if nrows == self.species.number:
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.debug(msg)
        else:
            num: int = self.species.number - nrows
            logger.debug(
                "%d additional (mass) constraint(s) are necessary to solve the system", num
            )

        coeff: np.ndarray = np.zeros((nrows, self.species.number))
        if self.reaction_matrix is not None:
            coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index: int = self.species.indices[constraint.species]
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("Species = %s", self.species.formulas)
        logger.debug("Coefficient matrix = \n%s", coeff)

        return coeff

    def _assemble_right_hand_side_values(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
    ) -> np.ndarray:
        """Assembles the right-hand side vector of values for the system of equations.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar
            constraints: Constraints for the system of equations

        Returns:
            The right-hand side vector of values
        """
        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions
        rhs: np.ndarray = np.zeros(nrows, dtype=float)

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
            if constraint.name == "pressure":
                rhs[row_index] += np.log10(
                    self.species.gas_species_by_formula[
                        constraint.species
                    ].eos.fugacity_coefficient(temperature=temperature, pressure=pressure)
                )

        logger.debug("RHS vector = %s", rhs)

        return rhs

    def _assemble_log_fugacity_coefficients(
        self,
        *,
        temperature: float,
        pressure: float,
    ) -> np.ndarray:
        """Assembles the fugacity coefficient vector on the left-hand side of the equations.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            The log10(fugacity coefficient) vector
        """

        # Initialise to ideal behaviour.
        fugacity_coefficients: np.ndarray = np.ones_like(self.species, dtype=float)

        # Fugacity coefficients are only relevant for gas species. The initialisation of the array
        # above to unity ensures that the coefficients are all zero for condensed species, once the
        # log is taken.
        for index, gas_species in self.species.gas_species.items():
            fugacity_coefficient: float = gas_species.eos.fugacity_coefficient(
                temperature=temperature, pressure=pressure
            )
            fugacity_coefficients[index] = fugacity_coefficient

        log_fugacity_coefficients: np.ndarray = np.log10(fugacity_coefficients)
        logger.debug("Fugacity coefficient vector = %s", log_fugacity_coefficients)

        return log_fugacity_coefficients

    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: np.ndarray,
        log_solution: np.ndarray,
    ) -> np.ndarray:
        """Returns the residual vector of the reaction network.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar
            constraints: Constraints for the system of equations
            coefficient_matrix: Coefficient matrix
            log_solution: Estimated log10 solution to compute the residual for

        Returns:
            The residual vector of the reaction network
        """
        rhs: np.ndarray = self._assemble_right_hand_side_values(
            temperature=temperature, pressure=pressure, constraints=constraints
        )
        log_fugacity_coefficients: np.ndarray = self._assemble_log_fugacity_coefficients(
            temperature=temperature, pressure=pressure
        )
        residual_reaction: np.ndarray = (
            coefficient_matrix.dot(log_fugacity_coefficients)
            + coefficient_matrix.dot(log_solution)
            - rhs
        )
        logger.debug("Residual_reaction = %s", residual_reaction)

        return residual_reaction


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
    initial_solution: InitialSolution | None = None
    """Initial solution"""
    output: Output = field(init=False, default_factory=Output)
    """Output data"""
    _reaction_network: _ReactionNetwork = field(init=False)
    # Convenient to set and update on this instance.
    _constraints: SystemConstraints = field(init=False, default_factory=SystemConstraints)
    _log_solution: np.ndarray = field(init=False)
    _residual: np.ndarray = field(init=False)
    _failed_solves: int = 0

    def __post_init__(self):
        logger.info("Creating an interior-atmosphere system")
        self.species.conform_solubilities_to_composition(self.planet.melt_composition)
        if self.initial_solution is None:
            self.initial_solution = InitialSolutionConstant(species=self.species)
        self._reaction_network = _ReactionNetwork(species=self.species)
        self._log_solution = np.zeros_like(self.species, dtype=np.float_)

    @property
    def number_of_solves(self) -> int:
        """The total number of systems solved"""
        return self.output.size

    @property
    def constraints(self) -> SystemConstraints:
        """Constraints"""
        return self._constraints

    @property
    def log_solution(self) -> np.ndarray:
        """The solution.

        For gas species and condensed species the solution is the log10 activity and log10 partial
        pressure, respectively. Subsequent entries in the solution array relate to the degree of
        condensation for elements in condensed species, if applicable.
        """
        return self._log_solution

    @property
    def degree_of_condensation_elements(self) -> list[str]:
        """Elements to solve for the degree of condensation

        The elements for which to calculate the degree of condensation depends on both which
        elements are in condensed species and which mass constraints are applied.
        """
        condensation: list[str] = []
        for constraint in self.constraints.mass_constraints:
            if constraint.species in self.species.condensed_elements:
                condensation.append(constraint.species)

        return condensation

    @property
    def degree_of_condensation_number(self) -> int:
        """Number of elements to solve for the degree of condensation"""
        return len(self.degree_of_condensation_elements)

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
    def solution(self) -> np.ndarray:
        """Solution."""
        return 10**self.log_solution

    @property
    def residual_dict(self) -> dict[str, float]:
        """Residual of the objective function.

        The order of the constraints must align with the order in which they are assembled.
        """
        output: dict[str, float] = {}
        for index, reaction in enumerate(self._reaction_network.reactions().values()):
            output[reaction] = self._residual[index]
        for index, constraint in enumerate(self.constraints.reaction_network_constraints):
            row_index: int = self._reaction_network.number_reactions + index
            output[constraint.full_name] = self._residual[row_index]
        for index, constraint in enumerate(self.constraints.mass_constraints):
            row_index = (
                self._reaction_network.number_reactions
                + self.constraints.number_reaction_network_constraints
                + index
            )
            output[constraint.full_name] = self._residual[row_index]
        for index, constraint in enumerate(self.constraints.total_pressure_constraint):
            output[constraint.full_name] = self._residual[-1]  # Always last index if applied

        return output

    def solution_dict(self) -> dict[str, float]:
        """Solution for all species in a dictionary.

        This is convenient for a quick check of the solution, but in general you will use
        self.output to return a dictionary of all the data or export the data to Excel or a
        DataFrame.
        """
        output: dict[str, float] = {}
        # Gas species partial pressures
        for name, solution in zip(self.species.names, self.solution[: self.species.number]):
            output[name] = solution
        # Degree of condensation for elements in condensed species
        for degree_of_condensation, solution in zip(
            self.degree_of_condensation_elements, self.solution[self.species.number :]
        ):
            key: str = f"degree_of_condensation_{degree_of_condensation}"
            output[key] = solution / (1 + solution)

        return output

    @property
    def log10_fugacity_coefficients_dict(self) -> dict[str, float]:
        """Fugacity coefficients (relevant for gas species only) in a dictionary."""
        output: dict[str, float] = {
            species.formula: np.log10(
                species.eos.fugacity_coefficient(
                    temperature=self.planet.surface_temperature, pressure=self.total_pressure
                )
            )
            for species in self.species.gas_species.values()
        }
        return output

    @property
    def fugacities_dict(self) -> dict[str, float]:
        """Fugacities of all species in a dictionary."""
        output: dict[str, float] = {}
        for key, value in self.log10_fugacity_coefficients_dict.items():
            # TODO: Not clean to append _g suffix to denote gas phase.
            output[f"f{key}"] = 10 ** (np.log10(self.solution_dict()[f"{key}_g"]) + value)

        return output

    @property
    def total_mass(self) -> float:
        """Total mass."""
        mass: float = UnitConversion.bar_to_Pa(self.total_pressure) / self.planet.surface_gravity
        mass *= self.planet.surface_area
        return mass

    @property
    def total_pressure(self) -> float:
        """Total pressure."""
        indices: list[int] = list(self.species.gas_species.keys())

        return sum(float(self.solution[index]) for index in indices)

    @property
    def atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere."""
        mu_atmosphere: float = 0
        for index, species in self.species.gas_species.items():
            mu_atmosphere += species.molar_mass * self.solution[index]
        mu_atmosphere /= self.total_pressure

        return mu_atmosphere

    def isclose(
        self, target_dict: dict[str, float], rtol: float = 1.0e-5, atol: float = 1.0e-8
    ) -> np.bool_:
        """Determines if the solution pressures are close to target values within a tolerance.

        Args:
            target_dict: Dictionary of the target values
            rtol: Relative tolerance
            atol: Absolute tolerance

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
            message: Message prefix to write to the logger when a tolerance is satisfied

        Returns:
            The tightest tolerance satisfied
        """
        for log_tolerance in (-6, -5, -4, -3, -2, -1):
            tol: float = 10**log_tolerance
            if self.isclose(target_dict, rtol=tol, atol=tol):
                logger.info("%s (tol = %f)".lstrip(), message, tol)
                return tol

        logger.info("%s (no tolerance < 0.1 satisfied)".lstrip(), message)

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        initial_solution: InitialSolution | None = None,
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
        self.set_constraints(constraints)

        if self.degree_of_condensation_number > 0:
            logger.info("Solving for condensed species and mass constraints")
            logger.info("Reducing max_attempts to 1")
            max_attempts = 1

        if initial_solution is None:
            initial_solution = self.initial_solution
        assert initial_solution is not None

        coefficient_matrix: np.ndarray = self._reaction_network.get_coefficient_matrix(
            constraints=self.constraints
        )
        # The only constraints that require pressure are the fugacity constraints, so for the
        # purpose of determining the initial solution we evaluate them (if present) at 1 bar to
        # ensure the initial solution is bounded.
        log_solution: np.ndarray = initial_solution.get_log10_value(
            self.constraints,
            temperature=self.planet.surface_temperature,
            pressure=1,
            degree_of_condensation_number=self.degree_of_condensation_number,
        )

        for attempt in range(max_attempts):
            logger.info("Attempt %d/%d", attempt + 1, max_attempts)
            logger.info("Initial solution = %s", log_solution)
            try:
                sol = root(
                    self._objective_func,
                    log_solution,
                    args=(coefficient_matrix,),
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
                error: np.ndarray = np.sqrt(mean_squared_error(sol.x, log_solution))
                logger.info(
                    "%s: RMSE (actual vs initial) = %s",
                    self.initial_solution.__class__.__name__,
                    error,
                )
                self._log_solution = sol.x
                self._residual = sol.fun
                self.output.add(self, extra_output)
                initial_solution.update(self.output)
                logger.info(pprint.pformat(self.solution_dict()))
                break
            else:
                logger.warning("The solver failed.")
                if attempt < max_attempts - 1:
                    log_solution = initial_solution.get_log10_value(
                        self.constraints,
                        temperature=self.planet.surface_temperature,
                        pressure=1,
                        degree_of_condensation_number=self.degree_of_condensation_number,
                        perturb=True,
                        perturb_log10=perturb_log10,
                    )

        if not sol.success:
            msg: str = f"Solver failed after {max_attempts} attempt(s) (errors = {errors})"
            self._failed_solves += 1
            if self.degree_of_condensation_number > 0:
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

    def set_constraints(self, constraints: SystemConstraints) -> None:
        """Combines user-prescribed constraints with activity constraints.

        Args;
            constraints: Constraints for the system of equations
        """
        logger.debug("Set constraints")
        self._constraints = constraints

        for condensed_species in self.species.condensed_species.values():
            self._constraints.append(condensed_species.activity)
        logger.debug("Constraints: %s", pprint.pformat(self._constraints))

    def _objective_func(
        self,
        log_solution: np.ndarray,
        coefficient_matrix: np.ndarray,
    ) -> np.ndarray:
        """Objective function for the non-linear system.

        Args:
            log_solution: Log10 of the activities and pressures of each species
            coefficient_matrix: Coefficient matrix

        Returns:
            The solution, which is the log10 of the activities and pressures for each species
        """

        # This must be set here
        self._log_solution = log_solution

        # Exclude the degree of condensation for the reaction network
        log_solution_reaction: np.ndarray = log_solution[: self.species.number]

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = self._reaction_network.get_residual(
            temperature=self.planet.surface_temperature,
            pressure=self.total_pressure,
            constraints=self.constraints,
            coefficient_matrix=coefficient_matrix,
            log_solution=log_solution_reaction,
        )

        # Compute residual for the mass balance (if relevant).
        residual_mass: np.ndarray = np.zeros(
            len(self.constraints.mass_constraints), dtype=np.float_
        )

        for constraint_index, constraint in enumerate(self.constraints.mass_constraints):
            # Gas species
            for species in self.species.gas_species.values():
                residual_mass[constraint_index] += sum(
                    species.mass(
                        system=self,
                        element=constraint.species,
                    ).values()
                )

            residual_mass[constraint_index] = np.log10(residual_mass[constraint_index])

            # Condensed species
            for nn, condensed_element in enumerate(self.degree_of_condensation_elements):
                if condensed_element == constraint.species:
                    residual_mass[constraint_index] += np.log10(
                        self.solution[self.species.number + nn] + 1
                    )

            # Mass values are constant so no need to pass any arguments to get_value().
            residual_mass[constraint_index] -= constraint.get_log10_value()

        logger.debug("Residual_mass = %s", residual_mass)

        # Compute residual for the total pressure (if relevant).
        residual_total_pressure: np.ndarray = np.zeros(
            len(self.constraints.total_pressure_constraint), dtype=np.float_
        )
        if len(self.constraints.total_pressure_constraint) > 0:
            constraint: Constraint = self.constraints.total_pressure_constraint[0]
            residual_total_pressure[0] += (
                np.log10(self.total_pressure) - constraint.get_log10_value()
            )

        # Combined residual
        residual: np.ndarray = np.concatenate(
            (residual_reaction, residual_mass, residual_total_pressure)
        )
        logger.debug("Residual = %s", residual)

        return residual
