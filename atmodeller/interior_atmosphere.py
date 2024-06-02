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
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError
from scipy.optimize import OptimizeResult, root
from sklearn.metrics import mean_squared_error

from atmodeller import GAS_CONSTANT, GRAVITATIONAL_CONSTANT
from atmodeller.constraints import SystemConstraints
from atmodeller.core import GasSpecies, Solution, Species
from atmodeller.initial_solution import InitialSolutionConstant
from atmodeller.interfaces import (
    InitialSolutionProtocol,
    TotalPressureConstraintProtocol,
)
from atmodeller.output import Output
from atmodeller.utilities import UnitConversion, dataclass_to_logger

logger: logging.Logger = logging.getLogger(__name__)

TAU: float = 1e-15
"""Tau factor for the calculation of the auxilliary equations for condensate stability"""
log10_TAU: float = np.log10(TAU)
"""Log10 of Tau"""


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

        A copy of `self.species_matrix` is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of species - number of
        elements. These reactions are given in the last r rows of the reduced matrix.

        Returns:
            A matrix of the reaction stoichiometry.
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
        """Builds the coefficient matrix.

        Args:
            constraints: Constraints for the system of equations

        Returns:
            The coefficient matrix with the stoichiometry and constraints
        """

        nrows: int = constraints.number_reaction_network_constraints + self.number_reactions

        if nrows == self.species.number_species():
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.debug(msg)
        else:
            num: int = self.species.number_species() - nrows
            logger.debug(
                "%d additional (mass) constraint(s) are necessary to solve the system", num
            )

        coeff: npt.NDArray = np.zeros((nrows, self.species.number_species()))
        if self.reaction_matrix is not None:
            coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

        for index, constraint in enumerate(constraints.reaction_network_constraints):
            logger.debug("Apply %s constraint for %s", constraint.name, constraint.species)
            row_index: int = self.number_reactions + index
            species_index = self.species.find_species(constraint.species)
            logger.debug("Row %02d: Setting %s coefficient", row_index, constraint.species)
            coeff[row_index, species_index] = 1

        logger.debug("Species = %s", self.species.names)
        logger.debug("Coefficient matrix = \n%s", coeff)

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

    def get_stability_matrix(self, coefficient_matrix: npt.NDArray) -> npt.NDArray:
        """Gets the stability matrix used to implement condensate stability

        Args:
            coefficient_matrix: Coefficient matrix

        Returns:
            Stability matrix
        """
        stability_matrix: npt.NDArray = np.zeros_like(coefficient_matrix)
        for species in self.species.condensed_species:
            index: int = self.species.find_species(species)
            stability_matrix[:, index] = coefficient_matrix[:, index]
        stability_matrix[: self.number_reactions, :] = 0

        logger.debug("stability_matrix = %s", stability_matrix)

        return stability_matrix

    def get_residual(
        self,
        *,
        temperature: float,
        pressure: float,
        constraints: SystemConstraints,
        coefficient_matrix: npt.NDArray,
        stability_matrix: npt.NDArray,
        solution: Solution,
    ) -> npt.NDArray:
        """Returns the residual vector of the reaction network.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            constraints: Constraints for the system of equations
            coefficient_matrix: Coefficient matrix
            stability_matrix: Stability matrix for condensate stability
            solution: Solution to compute the residual for

        Returns:
            The residual vector of the reaction network
        """
        log_fugacity_coefficients: npt.NDArray = self._assemble_log_fugacity_coefficients(
            temperature=temperature, pressure=pressure
        )

        rhs: npt.NDArray = self._assemble_right_hand_side_values(
            temperature=temperature, pressure=pressure, constraints=constraints
        )

        residual_reaction: npt.NDArray = (
            coefficient_matrix.dot(log_fugacity_coefficients)
            + coefficient_matrix.dot(solution.species_array)
            + stability_matrix.dot(10**solution.lambda_array)
            - rhs
        )

        logger.debug("residual_reaction = %s", residual_reaction)

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
    initial_solution: InitialSolutionProtocol | None = None
    """Initial solution"""
    output: Output = field(init=False, default_factory=Output)
    """Output data"""
    _reaction_network: _ReactionNetwork = field(init=False)
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
        self._reaction_network = _ReactionNetwork(species=self.species)

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

    def mass(
        self,
        *,
        species: GasSpecies,
        element: Optional[str] = None,
    ) -> dict[str, float]:
        """Calculates the mass of the species or one of its elements in each reservoir

        Args:
            species: A gas species
            element: Compute the mass for an element in the species. Defaults to None to compute
                the species mass.

        Returns:
            Total reservoir masses of the species or element
        """
        pressure: float = self._solution.gas_pressures[species]
        fugacity: float = self._solution.gas_fugacities[species]

        # Atmosphere
        mass_in_atmosphere: float = (
            UnitConversion.bar_to_Pa(pressure) / self.planet.surface_gravity
        )
        mass_in_atmosphere *= (
            self.planet.surface_area * species.molar_mass / self.atmospheric_mean_molar_mass
        )

        # Melt
        ppmw_in_melt: float = species.solubility.concentration(
            fugacity=fugacity,
            temperature=self.planet.surface_temperature,
            pressure=self.total_pressure,
            **self._solution.gas_fugacities_by_hill_formula,
        )
        mass_in_melt: float = (
            self.planet.mantle_melt_mass * ppmw_in_melt * UnitConversion.ppm_to_fraction()
        )

        # Solid
        ppmw_in_solid: float = ppmw_in_melt * species.solid_melt_distribution_coefficient
        mass_in_solid: float = (
            self.planet.mantle_solid_mass * ppmw_in_solid * UnitConversion.ppm_to_fraction()
        )

        output: dict[str, float] = {
            "atmosphere": mass_in_atmosphere,
            "melt": mass_in_melt,
            "solid": mass_in_solid,
        }

        if element is not None:
            try:
                mass_scale_factor: float = (
                    UnitConversion.g_to_kg(species.composition()[element].mass)
                    / species.molar_mass
                )
            except KeyError:  # Element not in formula so mass is zero.
                mass_scale_factor = 0
            for key in output:
                output[key] *= mass_scale_factor

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
        return self._solution.total_pressure

    @property
    def atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere."""
        return self._solution.gas_molar_mass

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
        self._solution = Solution(self.species, self.constraints, self.planet.surface_temperature)

        if initial_solution is None:
            initial_solution = self.initial_solution
        assert initial_solution is not None

        coefficient_matrix: npt.NDArray = self._reaction_network.get_coefficient_matrix(
            constraints=self.constraints
        )
        stability_matrix: npt.NDArray = self._reaction_network.get_stability_matrix(
            coefficient_matrix
        )

        # The only constraints that require pressure are the fugacity constraints, so for the
        # purpose of determining the initial solution we evaluate them (if present) at 1 bar to
        # ensure the initial solution is bounded.
        log_solution: npt.NDArray = initial_solution.get_log10_value(
            self.constraints,
            temperature=self.planet.surface_temperature,
            pressure=1,
            degree_of_condensation_number=self._solution.number_condensed_elements,
            number_of_condensed_species=self.species.number_condensed_species,
        )

        for attempt in range(max_attempts):
            logger.info("Attempt %d/%d", attempt + 1, max_attempts)
            logger.info("Initial solution = %s", log_solution)
            try:
                sol = root(
                    self._objective_func,
                    log_solution,
                    args=(coefficient_matrix, stability_matrix),
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
                logger.info(pprint.pformat(self._solution.solution_dict()))
                break
            else:
                logger.warning("The solver failed.")
                if attempt < max_attempts - 1:
                    log_solution = initial_solution.get_log10_value(
                        self.constraints,
                        temperature=self.planet.surface_temperature,
                        pressure=1,
                        degree_of_condensation_number=self._solution.number_condensed_elements,
                        number_of_condensed_species=self.species.number_condensed_species,
                        perturb=True,
                        perturb_log10=perturb_log10,
                    )

        if not sol.success:
            msg: str = f"Solver failed after {max_attempts} attempt(s) (errors = {errors})"
            self._failed_solves += 1
            if self._solution.number_condensed_elements > 0:
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
        stability_matrix: npt.NDArray,
    ) -> npt.NDArray:
        """Objective function for the non-linear system.

        Args:
            log_solution: Log10 of the activities and pressures of each species
            coefficient_matrix: Coefficient matrix
            stability_matrix: Stability matrix for condensate stability

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
            stability_matrix=stability_matrix,
            solution=self._solution,
        )

        residual_stability: npt.NDArray = np.zeros(
            self.species.number_condensed_species, dtype=np.float_
        )
        for nn, species in enumerate(self.species.condensed_species):
            residual_stability[nn] = self._solution._lambda_solution[species] - log10_TAU
            for element in species.elements:
                try:
                    residual_stability += self._solution._beta_solution[element]
                except KeyError:
                    pass

        logger.debug("residual_stability = %s", residual_stability)

        # Compute residual for the mass balance (if relevant).
        residual_mass: npt.NDArray = np.zeros(
            len(self.constraints.mass_constraints), dtype=np.float_
        )

        # Recall that mass constraints are currently only ever specified in terms of elements.
        # Hence constraint.species is an element.
        for constraint_index, mass_constraint in enumerate(self.constraints.mass_constraints):
            # Gas species
            for species in self.species.gas_species:
                residual_mass[constraint_index] += sum(
                    self.mass(
                        species=species,
                        element=mass_constraint.element,
                    ).values()
                )

            residual_mass[constraint_index] = np.log10(residual_mass[constraint_index])

            # Condensed species
            for condensed_element in self._solution.condensed_elements:
                if condensed_element == mass_constraint.element:
                    residual_mass[constraint_index] += np.log10(
                        10 ** self._solution._beta_solution[condensed_element] + 1
                    )

            # Mass values are constant so no need to pass any arguments to get_value().
            residual_mass[constraint_index] -= mass_constraint.get_log10_value()

        logger.debug("residual_mass = %s", residual_mass)

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
            (residual_reaction, residual_stability, residual_mass, residual_total_pressure)
        )
        logger.debug("residual = %s", residual)

        return residual
