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
from atmodeller.initial_solution import InitialSolutionDict, InitialSolutionProtocol
from atmodeller.output import Output
from atmodeller.reaction_network import ReactionNetworkWithMassBalance
from atmodeller.solution import Solution
from atmodeller.solver import Solver, SolverScipy

logger: logging.Logger = logging.getLogger(__name__)


class InteriorAtmosphereSystem:
    """An interior-atmosphere system

    Args:
        species: Species
        planet: Planet. Defaults to a molten Earth.
    """

    def __init__(self, species: Species, planet: Planet = Planet()):
        logger.info("Creating an interior-atmosphere system")
        self._species: Species = species
        self._planet: Planet = planet
        self._reaction_network: ReactionNetworkWithMassBalance = ReactionNetworkWithMassBalance(
            species, planet
        )
        self._output: Output = Output()
        self._failed_solves: int = 0

    @property
    def number_of_failed_solves(self) -> int:
        """Number of failed solves"""
        return self._failed_solves

    @property
    def number_of_solves(self) -> int:
        """Number of systems solved"""
        return self.output.size

    @property
    def number_of_attempted_solves(self) -> int:
        """Number of attempted solves"""
        return self.number_of_solves + self.number_of_failed_solves

    @property
    def output(self) -> Output:
        """Output"""
        return self._output

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self._planet

    @property
    def species(self) -> Species:
        """Species"""
        return self._species

    @property
    def temperature(self) -> float:
        """Temperature"""
        return self._planet.surface_temperature

    def solve(
        self,
        constraints: SystemConstraints,
        *,
        solver: Solver = SolverScipy(),
        initial_solution: InitialSolutionProtocol | None = None,
        tol: float = 1.0e-8,
        errors: str = "ignore",
        extra_output: dict[str, float] | None = None,
    ) -> Solution:
        """Solve

        Args:
            constraints: Constraints for the system of equations
            solver: Solver. Defaults to SolverScipy.
            initial_solution: Initial solution. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.
            errors: Either `raise` solver errors or `ignore`. Defaults to `ignore`.
            extra_output: Extra output
        """
        logger.info("Solving system number %d", self.number_of_solves)

        if initial_solution is None:
            initial_solution_ = InitialSolutionDict(species=self._species, planet=self._planet)
        else:
            initial_solution_ = initial_solution

        solution, success = solver.solve(
            self._reaction_network,
            constraints=constraints,
            initial_solution=initial_solution_,
            tol=tol,
        )

        if success:
            residual_dict = self._reaction_network.get_residual_dict(solution, constraints)
            if residual_dict["rms"] > 1.0e-6:
                logger.warning("Solver reports success but RMSE = %0.2e", residual_dict["rms"])
            constraint_dict = constraints.evaluate(
                self.temperature, solution.atmosphere.pressure()
            )
            self.output.add(solution, residual_dict, constraint_dict, extra_output)
            initial_solution_.update(self.output)
        else:
            self._failed_solves = self._failed_solves + 1
            msg: str = f"{solver.__class__.__name__} failed"
            if errors == "raise":
                logger.error(msg)
                logger.error("constraints = %s", constraints)
                raise RuntimeError(msg)

            else:
                logger.warning(msg)
                logger.warning("constraints = %s", constraints)
                logger.warning("Continuing ...")

        return solution
