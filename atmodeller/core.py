"""Core classes and functions."""

from __future__ import annotations

import logging
import pprint
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.optimize import fsolve

from atmodeller import GAS_CONSTANT, GRAVITATIONAL_CONSTANT
from atmodeller.constraints import SystemConstraint
from atmodeller.reaction_network import ReactionNetwork
from atmodeller.solubilities import NoSolubility, composition_solubilities
from atmodeller.thermodynamics import (
    ChemicalComponent,
    GasSpecies,
    SolidSpecies,
    StandardGibbsFreeEnergyOfFormationJANAF,
    StandardGibbsFreeEnergyOfFormationProtocol,
)

if TYPE_CHECKING:
    from atmodeller.solubilities import Solubility

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


# @dataclass(kw_only=True)
# class ReactionNetwork:
#     """Determines the necessary reactions to solve a chemical network.

#     Args:
#         species: A list of species.
#         gibbs_data: Standard Gibbs free energy of formation.

#     Attributes:
#         species: A list of species.
#         gibbs_data: Standard Gibbs free energy of formation.
#         species_matrix: The stoichiometry matrix of the species in terms of elements.
#         reaction_matrix: The reaction stoichiometry matrix.
#     """

#     species: list[ChemicalComponent]
#     gibbs_data: StandardGibbsFreeEnergyOfFormationProtocol

#     def __post_init__(self):
#         # Avoid reordering since it is more intuitive for the order of the output to mirror the
#         # input.
#         # self.species.sort(key=self._species_sorter)
#         logger.info("Creating a reaction network")
#         logger.info("Species = %s", self.species_names)
#         self.species_matrix: np.ndarray = self.find_matrix()
#         self.reaction_matrix: np.ndarray = self.partial_gaussian_elimination()
#         logger.info("Reactions = \n%s", pprint.pformat(self.reactions))

#     @cached_property
#     def species_names(self) -> list[str]:
#         return [species.chemical_formula for species in self.species]

#     @cached_property
#     def number_reactions(self) -> int:
#         return self.number_species - self.number_unique_elements

#     @cached_property
#     def number_species(self) -> int:
#         return len(self.species)

#     @cached_property
#     def number_unique_elements(self) -> int:
#         return len(self.unique_elements)

#     @cached_property
#     def species_indices(self) -> dict[str, int]:
#         return {name: idx for idx, name in enumerate(self.species_names)}

#     @cached_property
#     def unique_elements(self) -> list[str]:
#         elements: list[str] = []
#         for species in self.species:
#             elements.extend(list(species.formula.composition().keys()))
#         unique_elements: list[str] = list(set(elements))
#         return unique_elements

#     def find_matrix(self) -> np.ndarray:
#         """Creates a matrix where species (rows) are split into their element counts (columns).

#         Returns:
#             For example, self.species = ['CO2', 'H2O'] would return:
#                 [[0, 1, 2],
#                  [2, 0, 1]]
#             if the columns represent the elements H, C, and O, respectively.
#         """
#         matrix: np.ndarray = np.zeros((self.number_species, self.number_unique_elements))
#         for species_index, species in enumerate(self.species):
#             for element_index, element in enumerate(self.unique_elements):
#                 try:
#                     count: int = species.formula.composition()[element].count
#                 except KeyError:
#                     count = 0
#                 matrix[species_index, element_index] = count
#         return matrix

#     def partial_gaussian_elimination(self) -> np.ndarray:
#         """Performs a partial gaussian elimination to determine the required reactions.

#         A copy of `self.species_matrix` is first (partially) reduced to row echelon form by
#         forward elimination, and then subsequently (partially) reduced to reduced row echelon form
#         by backward substitution. Applying the same operations to the identity matrix (as part of
#         the augmented matrix) reveals r reactions, where r = number of species - number of
#         elements. These reactions are given in the last r rows of the reduced matrix.

#         Returns:
#             A matrix of the reaction stoichiometry.
#         """
#         matrix1: np.ndarray = self.species_matrix
#         matrix2: np.ndarray = np.eye(self.number_species)
#         augmented_matrix: np.ndarray = np.hstack((matrix1, matrix2))
#         logger.debug("augmented_matrix = \n%s", augmented_matrix)

#         # Forward elimination.
#         for i in range(self.number_unique_elements):  # Note only over the number of elements.
#             # Check if the pivot element is zero.
#             if augmented_matrix[i, i] == 0:
#                 # Swap rows to get a non-zero pivot element.
#                 nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
#                 augmented_matrix[[i, nonzero_row]] = augmented_matrix[[nonzero_row, i]]
#             # Perform row operations to eliminate values below the pivot.
#             for j in range(i + 1, self.number_species):
#                 ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
#                 logger.debug("Ratio = \n%s", ratio)
#                 augmented_matrix[j] -= ratio * augmented_matrix[i]
#         logger.debug("Augmented_matrix after forward elimination = \n%s", augmented_matrix)

#         # Backward substitution.
#         for i in range(self.number_unique_elements - 1, -1, -1):
#             # Normalize the pivot row.
#             augmented_matrix[i] /= augmented_matrix[i, i]
#             # Eliminate values above the pivot.
#             for j in range(i - 1, -1, -1):
#                 if augmented_matrix[j, i] != 0:
#                     ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
#                     augmented_matrix[j] -= ratio * augmented_matrix[i]
#         logger.debug("Augmented_matrix after backward substitution = \n%s", augmented_matrix)

#         reduced_matrix1: np.ndarray = augmented_matrix[:, : matrix1.shape[1]]
#         reaction_matrix: np.ndarray = augmented_matrix[
#             self.number_unique_elements :, matrix1.shape[1] :
#         ]
#         logger.debug("Reduced_matrix1 = \n%s", reduced_matrix1)
#         logger.debug("Reaction_matrix = \n%s", reaction_matrix)

#         return reaction_matrix

#     @cached_property
#     def reactions(self) -> dict[int, str]:
#         """The reactions as a dictionary."""
#         reactions: dict[int, str] = {}
#         for reaction_index in range(self.number_reactions):
#             reactants: str = ""
#             products: str = ""
#             for species_index, species in enumerate(self.species):
#                 coeff: float = self.reaction_matrix[reaction_index, species_index]
#                 if coeff != 0:
#                     if coeff < 0:
#                         reactants += f"{abs(coeff)} {species.chemical_formula} + "
#                     else:
#                         products += f"{coeff} {species.chemical_formula} + "

#             reactants = reactants.rstrip(" + ")  # Removes the extra + at the end.
#             products = products.rstrip(" + ")  # Removes the extra + at the end.
#             reaction: str = f"{reactants} = {products}"
#             reactions[reaction_index] = reaction

#         return reactions

#     def get_reaction_log10_equilibrium_constant(
#         self, *, reaction_index: int, temperature: float, pressure: float
#     ) -> float:
#         """Gets the log10 of the reaction equilibrium constant.

#         From the Gibbs free energy, we can calculate logKf as:
#         logKf = - G/(ln(10)*R*T)

#         Args:
#             reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
#             temperature: Temperature.
#             pressure: Pressure.

#         Returns:
#             log10 of the reaction equilibrium constant.
#         """
#         gibbs_energy: float = self.get_reaction_gibbs_energy_of_formation(
#             reaction_index=reaction_index, temperature=temperature, pressure=pressure
#         )
#         equilibrium_constant: float = -gibbs_energy / (np.log(10) * GAS_CONSTANT * temperature)

#         return equilibrium_constant

#     def get_reaction_gibbs_energy_of_formation(
#         self, *, reaction_index: int, temperature: float, pressure: float
#     ) -> float:
#         """Gets the Gibb's free energy of formation for a reaction.

#         Args:
#             reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
#             temperature: Temperature.
#             pressure: Pressure.

#         Returns:
#             The Gibb's free energy of the reaction.
#         """
#         gibbs_energy: float = 0
#         for species_index, species in enumerate(self.species):
#             gibbs_energy += self.reaction_matrix[
#                 reaction_index, species_index
#             ] * self.gibbs_data.get(species, temperature=temperature, pressure=pressure)
#         return gibbs_energy

#     def get_reaction_equilibrium_constant(
#         self, *, reaction_index: int, temperature: float, pressure: float
#     ) -> float:
#         """Gets the equilibrium constant of a reaction Kf

#         Args:
#             reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
#             temperature: Temperature.
#             pressure: Pressure.

#         Returns:
#             The equilibrium constant of the reaction.
#         """
#         equilibrium_constant: float = 10 ** self.get_reaction_log10_equilibrium_constant(
#             reaction_index=reaction_index, temperature=temperature, pressure=pressure
#         )
#         return equilibrium_constant

#     def get_coefficient_matrix(self, *, constraints: list[SystemConstraint]) -> np.ndarray:
#         """Builds the coefficient matrix.

#         Args:
#             constraints: Constraints for the system of equations.

#         Returns:
#             The coefficient matrix with the stoichiometry and constraints.
#         """

#         solid_species = []
#         for species in self.species:
#             if isinstance(species, SolidSpecies):
#                 solid_species.append(species)
#         fugacity_constraints: list[SystemConstraint] = [
#             constraint for constraint in constraints if constraint.field == "fugacity"
#         ]
#         pressure_constraints: list[SystemConstraint] = [
#             constraint for constraint in constraints if constraint.field == "pressure"
#         ]

#         # fp_constraints = []
#         # for constraint in constraints:
#         #     if constraint.field == "fugacity" or constraint.field == "pressure":
#         #         fp_constraints.append(constraint)

#         # To maintain order fugacity then pressure, as for LHS and RHS vector.
#         fp_constraints: list = fugacity_constraints + pressure_constraints

#         nrows: int = len(solid_species) + len(fp_constraints) + self.number_reactions

#         if nrows == self.number_species:
#             msg: str = "The necessary number of constraints will be applied to "
#             msg += "the reaction network to solve the system"
#             logger.info(msg)
#         else:
#             num: int = self.number_species - nrows
#             # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
#             msg = "%d additional (mass) constraint(s) are necessary " % num
#             msg += "to solve the system"
#             logger.info(msg)

#         coeff: np.ndarray = np.zeros((nrows, self.number_species))
#         coeff[0 : self.number_reactions] = self.reaction_matrix.copy()

#         # Solid activities.
#         for index, species in enumerate(solid_species):
#             species_name: str = species.formula.formula
#             row_index: int = self.number_reactions + index
#             species_index: int = self.species_indices[species_name]
#             logger.info("Row %02d: Setting %s activity", row_index, species_name)
#             coeff[row_index, species_index] = 1

#         # Fugacity and pressure constraints.
#         for index, constraint in enumerate(fp_constraints):
#             row_index: int = self.number_reactions + len(solid_species) + index
#             species_index: int = self.species_indices[constraint.species]
#             logger.info("Row %02d: Setting %s %s", row_index, constraint.species, constraint.field)
#             coeff[row_index, species_index] = 1

#         logger.debug("Coefficient matrix = \n%s", coeff)

#         return coeff

#     def get_lhs_and_rhs_vectors(
#         self,
#         *,
#         system: InteriorAtmosphereSystem,
#         constraints: list[SystemConstraint],
#     ) -> tuple[np.ndarray, np.ndarray]:
#         """Builds the LHS non-ideal vector and the RHS vector.

#         Args:
#             system: Interior atmosphere system.
#             constraints: Constraints for the system of equations.

#         Returns:
#             The log10(fugacity coefficient) vector and the right-hand side vector.
#         """

#         solid_species: list[SolidSpecies] = [
#             species for species in self.species if isinstance(species, SolidSpecies)
#         ]
#         fugacity_constraints: list[SystemConstraint] = [
#             constraint for constraint in constraints if constraint.field == "fugacity"
#         ]
#         pressure_constraints: list[SystemConstraint] = [
#             constraint for constraint in constraints if constraint.field == "pressure"
#         ]
#         number_constraints: int = (
#             len(solid_species) + len(fugacity_constraints) + len(pressure_constraints)
#         )
#         nrows: int = number_constraints + self.number_reactions

#         if nrows == self.number_species:
#             msg: str = "The necessary number of constraints will be applied to "
#             msg += "the reaction network to solve the system"
#             logger.info(msg)
#         else:
#             num: int = self.number_species - nrows
#             # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
#             msg = "%d additional (mass) constraint(s) are necessary " % num
#             msg += "to solve the system"
#             logger.info(msg)

#         rhs: np.ndarray = np.zeros(nrows)
#         non_ideal: np.ndarray = np.ones_like(self.species, dtype=np.float_)

#         # Reactions.
#         for reaction_index in range(self.number_reactions):
#             logger.info(
#                 "Row %02d: Reaction %d: %s",
#                 reaction_index,
#                 reaction_index,
#                 self.reactions[reaction_index],
#             )
#             rhs[reaction_index] = self.get_reaction_log10_equilibrium_constant(
#                 reaction_index=reaction_index,
#                 temperature=system.planet.surface_temperature,
#                 pressure=system.total_pressure,
#             )

#         # Solid activities.
#         for index, species in enumerate(solid_species):
#             species_name: str = species.formula.formula
#             row_index: int = self.number_reactions + index
#             logger.info("Row %02d: Setting %s activity", row_index, species_name)
#             rhs[row_index] = np.log10(
#                 species.activity(
#                     temperature=system.planet.surface_temperature, pressure=system.total_pressure
#                 )
#             )

#         # Fugacity constraints.
#         for index, constraint in enumerate(fugacity_constraints):
#             row_index: int = self.number_reactions + len(solid_species) + index
#             logger.info("Row %02d: Setting %s fugacity", row_index, constraint.species)
#             rhs[row_index] = np.log10(
#                 constraint.get_value(
#                     temperature=system.planet.surface_temperature, pressure=system.total_pressure
#                 )
#             )

#         # Pressure constraints.
#         for index, constraint in enumerate(pressure_constraints):
#             row_index: int = (
#                 self.number_reactions + len(solid_species) + len(fugacity_constraints) + index
#             )
#             logger.info("Row %02d: Setting %s pressure", row_index, constraint.species)
#             rhs[row_index] = np.log10(
#                 constraint.get_value(
#                     temperature=system.planet.surface_temperature, pressure=system.total_pressure
#                 )
#             )
#             rhs[row_index] += np.log10(system.fugacity_coefficients_dict[constraint.species])

#         # The "non-ideal" LHS vector.
#         for index, species in enumerate(self.species):
#             # FIXME: Not for solid phases. Must be zero.
#             if species.phase == "solid":
#                 value: float = 1
#             else:
#                 value: float = system.fugacity_coefficients_dict[species.chemical_formula]
#             non_ideal[index] = value
#         non_ideal = np.log10(non_ideal)

#         logger.debug("Non-ideal vector = \n%s", non_ideal)
#         logger.debug("RHS vector = \n%s", rhs)

#         return rhs, non_ideal

#     def get_residual(
#         self,
#         *,
#         system: InteriorAtmosphereSystem,
#         constraints: list[SystemConstraint],
#         coefficient_matrix: np.ndarray,
#     ) -> np.ndarray:
#         """Returns the residual vector of the reaction network.

#         Args:
#             system: Interior atmosphere system.
#             constraints: Constraints for the system of equations.
#             coefficient_matrix: Coefficient matrix.

#         Returns:
#             The residual vector of the reaction network.
#         """

#         rhs, non_ideal = self.get_lhs_and_rhs_vectors(system=system, constraints=constraints)

#         residual_reaction: np.ndarray = (
#             coefficient_matrix.dot(non_ideal)
#             + coefficient_matrix.dot(system.log10_pressures)
#             - rhs
#         )
#         logger.debug("Residual_reaction = %s", residual_reaction)
#         return residual_reaction

#     def _species_sorter(self, species: ChemicalComponent) -> tuple[int, str]:
#         """Sorter for the species.

#         Sorts first by species complexity and second by species name.

#         Args:
#             species: Species.

#         Returns:
#             A tuple to sort first by number of elements and second by species name.
#         """
#         return (species.formula.atoms, species.chemical_formula)


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

    species: list[ChemicalComponent]
    gibbs_data: StandardGibbsFreeEnergyOfFormationProtocol = field(
        default_factory=StandardGibbsFreeEnergyOfFormationJANAF
    )
    planet: Planet = field(default_factory=Planet)
    _reaction_network: ReactionNetwork = field(init=False)
    _log10_pressures: np.ndarray = field(init=False)  # For the solution.

    def __post_init__(self):
        logger.info("Creating an interior-atmosphere system")
        self._conform_solubilities_to_composition()
        self._reaction_network = ReactionNetwork(species=self.species, gibbs_data=self.gibbs_data)
        self.species = self._reaction_network.species  # Note reordering by self._reaction_network.
        # Initialise solution to zero.
        self._log10_pressures = np.zeros_like(self.species, dtype=np.float_)

    @property
    def log10_pressures(self) -> np.ndarray:
        """Log10 pressures."""
        return self._log10_pressures

    @property
    def pressures(self) -> np.ndarray:
        """Pressures."""
        return 10**self.log10_pressures

    @property
    def pressures_dict(self) -> dict[str, float]:
        """Pressures of all species in a dictionary."""
        # TODO: Activity for solid (or remove from output).
        output: dict[str, float] = {}
        for species_name, pressure in zip(self._reaction_network.species_names, self.pressures):
            output[species_name] = pressure

        return output

    @property
    def fugacity_coefficients_dict(self) -> dict[str, float]:
        """Fugacity coefficients in a dictionary."""
        output: dict[str, float] = {
            species_name: species.ideality(
                temperature=self.planet.surface_temperature, pressure=self.total_pressure
            )
            for (species_name, species) in zip(self._reaction_network.species_names, self.species)
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
        for species in self.species:
            if species.phase == "gas":
                assert isinstance(species, GasSpecies)
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

    def _conform_solubilities_to_composition(self) -> None:
        """Ensure that the solubilities of the species are consistent with the melt composition."""
        if self.planet.melt_composition is not None:
            msg: str = (
                # pylint: disable=consider-using-f-string
                "Setting solubilities to be consistent with the melt composition (%s)"
                % self.planet.melt_composition
            )
            logger.info(msg)
            try:
                solubilities: dict[str, Solubility] = composition_solubilities[
                    self.planet.melt_composition.casefold()
                ]
            except KeyError:
                logger.error("Cannot find solubilities for %s", self.planet.melt_composition)
                raise

            for species in self.species:
                if species.phase == "gas":
                    assert isinstance(species, GasSpecies)
                    try:
                        species.solubility = solubilities[species.chemical_formula]
                        logger.info(
                            "Found Solubility law for %s: %s",
                            species.chemical_formula,
                            species.solubility.__class__.__name__,
                        )
                    except KeyError:
                        logger.info("No solubility law for %s", species.chemical_formula)
                        species.solubility = NoSolubility()

    def solve(
        self,
        constraints: list[SystemConstraint],
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

        logger.info("Constraints: %s", pprint.pformat(constraints))
        self._log10_pressures = self._solve_fsolve(
            constraints=constraints, initial_log10_pressures=initial_log10_pressures
        )

        # Recompute quantities that depend on the solution, since species.mass is not called for
        # the linear reaction network. TODO: Update this comment? Still relevant?
        for species in self.species:
            if species.phase == "gas":
                assert isinstance(species, GasSpecies)
                species.mass(
                    planet=self.planet,
                    system=self,
                )

        logger.info(pprint.pformat(self.pressures_dict))

    def _solve_fsolve(
        self,
        *,
        constraints: list[SystemConstraint],
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
        constraints: list[SystemConstraint],
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
        self._log10_pressures = log10_pressures

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = self._reaction_network.get_residual(
            system=self, constraints=constraints, coefficient_matrix=coefficient_matrix
        )

        mass_constraints: list[SystemConstraint] = [
            constraint for constraint in constraints if constraint.field == "mass"
        ]

        # Compute residual for the mass balance.
        residual_mass: np.ndarray = np.zeros_like(mass_constraints, dtype=np.float_)
        for constraint_index, constraint in enumerate(mass_constraints):
            for species in self.species:
                if species.phase == "gas":
                    assert isinstance(species, GasSpecies)
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
