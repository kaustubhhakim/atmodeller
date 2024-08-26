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

from atmodeller.constraints import SystemConstraints
from atmodeller.core import Planet, Species
from atmodeller.initial_solution import InitialSolutionProtocol
from atmodeller.output import Output
from atmodeller.reaction_network import ReactionNetworkWithMassBalance
from atmodeller.solution import Solution
from atmodeller.solver import Solver, SolverScipy

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphereSystem:
    """An interior-atmosphere system

    Args:
        species: Species
        planet: Planet

    Attributes:
        species: Species
        planet: Planet
        output: Output
    """

    def __init__(self, species: Species, planet: Planet):
        logger.info("Creating an interior-atmosphere system")
        self._species: Species = species
        self._planet: Planet = planet
        self._reaction_network: ReactionNetworkWithMassBalance = ReactionNetworkWithMassBalance(
            species, planet
        )
        self._output: Output = Output()

    @property
    def number_of_solves(self) -> int:
        """The total number of systems solved"""
        return self.output.size

    @property
    def output(self) -> Output:
        """Output"""
        return self._output

    def solve(
        self,
        solver: Solver = SolverScipy(),
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol | None = None,
        tol: float = 1.0e-8,
    ) -> Solution:
        """Solve

        Args:
            solver: Solver. Defaults to SolverScipy.
            constraints: Constraints for the system of equations
            initial_solution: Initial condition for this solve only. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.
        """
        logger.info("Solving system number %d", self.number_of_solves)

        solution, success = solver.solve(
            self._reaction_network,
            constraints=constraints,
            initial_solution=initial_solution,
            tol=tol,
        )

        # if not success:

        # TODO: Reinstate if the solve is successful then update output accordingly.
        # self.output.add(self, extra_output)
        # initial_solution.update(self.output)

        return solution, success


# @dataclass(kw_only=True)
# class InteriorAtmosphereSystem:
#     """An interior-atmosphere system

#     Args:
#         species: A list of species
#         planet: A planet. Defaults to a molten Earth
#         initial_solution: Initial solution. Defaults to a constant for all species.
#     """

#     species: Species
#     """A list of species"""
#     planet: Planet = field(default_factory=Planet)
#     """A planet"""
#     initial_solution: InitialSolutionProtocol | None = None
#     """Initial solution"""
#     output: Output = field(init=False, default_factory=Output)
#     """Output data"""
#     solution: Solution = field(init=False)
#     """Solution"""
#     _reaction_network: ReactionNetworkWithCondensateStability = field(init=False)
#     _constraints: SystemConstraints = field(init=False, default_factory=SystemConstraints)
#     _residual: jnp.ndarray = field(init=False)
#     _attempted_solves: int = field(init=False, default=0)
#     _failed_solves: int = field(init=False, default=0)

#     def __post_init__(self):
#         logger.info("Creating an interior-atmosphere system")
#         self.species.conform_solubilities_to_composition(self.planet.melt_composition)
#         if self.initial_solution is None:
#             self.initial_solution = InitialSolutionDict(species=self.species)
#         self._reaction_network = ReactionNetworkWithCondensateStability(self.species)
#         self.solution = Solution.create_from_species(species=self.species)
#         self.solution.planet = self.planet

#     @property
#     def number_of_solves(self) -> int:
#         """The total number of systems solved"""
#         return self.output.size

#     def residual_dict(self) -> dict[str, float]:
#         """Residual of the objective function

#         The order of the constraints must align with the order in which they are assembled.
#         """
#         output: dict[str, float] = {}
#         for index, reaction in enumerate(self._reaction_network.reactions().values()):
#             output[reaction] = self._residual[index].item()
#         for index, constraint in enumerate(self.constraints.reaction_network_constraints):
#             row_index: int = self._reaction_network.number_reactions + index
#             output[constraint.name] = self._residual[row_index].item()
#         for index, constraint in enumerate(self.constraints.mass_constraints):
#             row_index = (
#                 self._reaction_network.number_reactions
#                 + self.constraints.number_reaction_network_constraints
#                 + index
#             )
#             output[constraint.name] = self._residual[row_index].item()
#         for index, constraint in enumerate(self.constraints.total_pressure_constraint):
#             output[constraint.name] = self._residual[-1].item()  # Always last index if applied

#         output["rms"] = np.sqrt(np.mean(np.array(list(output.values())) ** 2))

#         return output

#     def solve(
#         self,
#         constraints: SystemConstraints,
#         *,
#         initial_solution: InitialSolutionProtocol | None = None,
#         extra_output: dict[str, float] | None = None,
#         max_attempts: int = 20,
#         perturb_log10_number_density: float = 2.0,
#         errors: str = "ignore",
#         method: str = "hybr",
#         tol: float | None = None,
#         **options,
#     ) -> None:
#         """Solves the system to determine the activities and partial pressures with constraints.

#         Args:
#             constraints: Constraints for the system of equations
#             initial_solution: Initial condition for this solve only. Defaults to 'None', meaning
#                 that the default (self.initial_solution) is used.
#             extra_output: Extra data to write to the output
#             max_attempts: Maximum number of attempts to randomise the initial condition to find a
#                 solution if the initial guess fails. Defaults to 10.
#             perturb_log10_number_density: Maximum log10 perturbation to apply to the pressures on
#                 failure. Defaults to 2.0.
#             errors: Either 'raise' solver errors or 'ignore'. Defaults to 'ignore'.
#             method: Type of solver. Defaults to 'hybr'.
#             tol: Tolerance for termination. Defaults to None.
#             **options: Keyword arguments for solver options. Available keywords depend on method.
#         """
#         logger.info("Solving system number %d", self.number_of_solves)
#         self._attempted_solves += 1

#         self._constraints = constraints
#         self._constraints.add_activity_constraints(self.species)

#         if initial_solution is None:
#             initial_solution = self.initial_solution
#         assert initial_solution is not None

#         # These can be determined once per solve because they depend on reaction stoichiometry and
#         # constraints, both of which are known and both of which are independent of the solution.
#         coefficient_matrix: jnp.ndarray = self._reaction_network.get_coefficient_matrix(
#             self.constraints
#         )
#         activity_modifier: jnp.ndarray = self._reaction_network.get_activity_modifier(
#             self.constraints
#         )
#         equilibrium_modifier: jnp.ndarray = self._reaction_network.get_equilibrium_modifier(
#             self.constraints
#         )

#         for attempt in range(max_attempts):
#             logger.info("Attempt %d/%d", attempt + 1, max_attempts)

#             # The only constraints that require pressure are the fugacity constraints, so for the
#             # purpose of determining the initial solution we evaluate them (if present) at 1 bar to
#             # ensure the initial solution is bounded.
#             log_solution: jnp.ndarray = initial_solution.get_log10_value(
#                 self.constraints,
#                 temperature=self.solution.atmosphere.temperature(),
#                 pressure=1,
#                 perturb_log10_number_density=perturb_log10_number_density,
#                 attempt=attempt,
#             )
#             try:
#                 sol = root(
#                     self._objective_func,
#                     log_solution,
#                     args=(
#                         coefficient_matrix,
#                         activity_modifier,
#                         equilibrium_modifier,
#                     ),
#                     method=method,
#                     tol=tol,
#                     options=options,
#                     # TODO: Add jac=
#                 )
#                 logger.info(sol["message"])
#                 logger.debug("sol = %s", sol)

#             except TypeError as exc:
#                 msg: str = (
#                     f"{exc}\nAdditional context: Number of unknowns and constraints must be equal"
#                 )
#                 raise ValueError(msg) from exc

#             except LinAlgError:
#                 if errors == "raise":
#                     raise
#                 else:
#                     logger.warning("Linear algebra error")
#                     sol = OptimizeResult()
#                     sol.success = False

#             if sol.success:
#                 # Below doesm't seem to be used anywhere
#                 # self._log_solution = sol.x
#                 self._residual = sol.fun
#                 residual_rmse: npt.NDArray[np.float_] = np.sqrt(
#                     np.sum(np.array(self._residual) ** 2)
#                 )
#                 logger.info("Residual RMSE = %.2e", residual_rmse)
#                 logger.info(
#                     "Actual solution = %s", pprint.pformat(self.solution.output_raw_solution())
#                 )
#                 initial_solution_rmse: npt.NDArray[np.float_] = np.sqrt(
#                     mean_squared_error(sol.x, np.array(log_solution))
#                 )
#                 logger.info(
#                     "Initial solution RMSE (%s) = %.2e",
#                     self.initial_solution.__class__.__name__,
#                     initial_solution_rmse,
#                 )
#                 self.output.add(self, extra_output)
#                 initial_solution.update(self.output)
#                 # logger.info(pprint.pformat(self.output_solution()))
#                 break
#             else:
#                 logger.warning("The solver failed.")

#         if not sol.success:
#             msg: str = f"Solver failed after {max_attempts} attempt(s) (errors = {errors})"
#             self._failed_solves += 1
#             if errors == "raise":
#                 logger.error(msg)
#                 logger.error("constraints = %s", self.constraints)
#                 raise RuntimeError(msg)
#             else:
#                 logger.warning(msg)
#                 logger.warning("constraints = %s", self.constraints)
#                 logger.warning("Continuing with next solve")

#     def _objective_func(
#         self,
#         log_solution: jnp.ndarray,
#         coefficient_matrix: jnp.ndarray,
#         activity_modifier: jnp.ndarray,
#         equilibrium_modifier: jnp.ndarray,
#     ) -> jnp.ndarray:
#         """Objective function for the non-linear system.

#         Args:
#             log_solution: Log10 of the activities and pressures of each species
#             coefficient_matrix: Coefficient matrix
#             activity_modifier: Activity modifier matrix for condensate stability
#             equilibrium_modifier: Equilibrium modifier matrix for condensate stability

#         Returns:
#             The solution, which is the log10 of the activities and pressures for each species
#         """
#         # This must be set here.
#         self.solution.value = log_solution

#         logger.debug("log_solution passed into _objective_func = %s", log_solution)

#         temperature: float = self.solution.atmosphere.temperature()
#         pressure: float = self.solution.atmosphere.pressure()

#         reaction_array: jnp.ndarray = self.get_reaction_array()
#         stability_array: jnp.ndarray = self.get_stability_array()

#         residual_reaction: jnp.ndarray = self._reaction_network.get_residual(
#             temperature=temperature,
#             pressure=pressure,
#             constraints=self.constraints,
#             coefficient_matrix=coefficient_matrix,
#             activity_modifier=activity_modifier,
#             equilibrium_modifier=equilibrium_modifier,
#             reaction_array=reaction_array,
#             stability_array=stability_array,
#         )

#         residual_stability_list: list[jnp.ndarray] = []
#         for collection in self.solution.condensed.values():
#             residual_stability_list.append(
#                 collection.stability.value - collection.tauc.value + collection.abundance.value
#             )
#         residual_stability = jnp.array(residual_stability_list)

#         residual_number_density_list: list[jnp.ndarray] = []
#         for constraint in self.constraints.mass_constraints:
#             res: jnp.ndarray = jnp.log10(self.solution.number_density(element=constraint.element))
#             res -= constraint.log10_number_of_molecules - jnp.log10(
#                 self.solution.atmosphere.volume()
#             )
#             residual_number_density_list.append(res)
#         residual_number_density = jnp.array(residual_number_density_list)

#         residual_total_pressure_list: list[jnp.ndarray] = []
#         if len(self.constraints.total_pressure_constraint) == 1:
#             constraint = self.constraints.total_pressure_constraint[0]
#             residual_total_pressure_list.append(
#                 jnp.log10(self.solution.atmosphere.number_density())
#                 - constraint.get_log10_value(temperature=temperature, pressure=pressure)
#             )
#             residual_total_pressure = jnp.array(residual_total_pressure_list)

#         # Concatenate all residuals
#         residual: jnp.ndarray = jnp.concatenate(
#             (
#                 residual_reaction,
#                 residual_stability,
#                 residual_number_density,
#                 residual_total_pressure,
#             )
#         )
#         logger.debug("residual = %s", residual)

#         return residual

#     def output_solution(self) -> dict[str, float]:
#         """Output the solution in a convenient form for comparison and benchmarking"""
#         return self.solution.output_solution()
