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
"""Solvers"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from optimistix._solution import Solution as Solution_optx
from scipy.optimize import OptimizeResult, root

from atmodeller.constraints import SystemConstraints
from atmodeller.initial_solution import InitialSolutionProtocol
from atmodeller.solution import Solution

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class Solver(ABC):
    """A solver"""

    @abstractmethod
    def get_residual(self, solution: Solution, constraints: SystemConstraints) -> Array:
        """Gets the residual

        Args:
            solution: Solution
            constraints: Constraints for the system of equations

        Returns:
            Residual array
        """

    def jacobian(self, args) -> Callable:
        """Creates a Jacobian callable function.

        Args:
            args: Tuple of other arguments

        Returns:
            A callable function that computes the Jacobian with respect to `solution_array`.
        """

        # Partially apply `args` to the `objective_function`
        def wrapped_objective(solution_array):
            return self.objective_function(solution_array, args)

        return jax.jacobian(wrapped_objective)

    def objective_function(self, solution_array: Array, args) -> Array:
        """Objective function

        Args:
            solution_array: Array of the solution
            args: Tuple of other arguments

        Returns:
            Residual array
        """
        logger.debug("log_solution passed into _objective_func = %s", solution_array)

        solution: Solution = Solution.create(self._species, self._planet)
        solution.value = solution_array

        # Required for optimistix
        constraints = args[0]
        residual: Array = self.get_residual(solution, constraints)

        # For Optimistix/JAX debugging
        jax.debug.print("solution into objective = {solution}", solution=solution.value)
        jax.debug.print("residual out of objective = {residual}", residual=residual)

        return residual

    @abstractmethod
    def solve(
        self,
        initial_solution: InitialSolutionProtocol | None = None,
        *,
        constraints: SystemConstraints,
        tol: float = 1.0e-8,
    ) -> tuple[Solution_optx, Callable, Solution]:
        """Solve

        Args:
            initial_solution: Initial solution
            constraints: Constraints for the system of equations
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Solver solution object, Jacobian function, atmodeller solution object
        """


class SolverOptimistix(Solver):

    def solve_optimistix(
        self,
        initial_solution: InitialSolutionProtocol | None = None,
        *,
        constraints: SystemConstraints,
        tol: float = 1.0e-8,
    ) -> tuple[Solution_optx, Callable, Solution]:
        """Solve using optimistix

        Args:
            constraints: Constraints for the system of equations
            initial_solution: Initial condition for this solve only. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Optimistix solution, Jacobian function, solution
        """

        if initial_solution is None:
            initial_solution = InitialSolutionDict(species=self._species)
        assert initial_solution is not None

        initial_solution_guess: Array = initial_solution.get_log10_value(
            constraints,
            temperature=self.temperature(),
            pressure=1,
        )

        # Other options if the surface is not well-behaved
        # solver = optx.BFGS(rtol=1e-3, atol=1e-3)
        # solver = optx.OptaxMinimiser(optax.adabelief(learning_rate=0.01), rtol=tol, atol=tol)
        # solver = optx.Dogleg(rtol=tol, atol=tol)
        # solver = optx.LevenbergMarquardt(rtol=tol, atol=tol)
        solver = optx.Newton(rtol=tol, atol=tol)
        # solver = optx.Chord(rtol=tol, atol=tol)
        sol = optx.root_find(
            self.objective_function,
            solver,
            initial_solution_guess,
            args=(constraints,),
            throw=True,
        )

        solution: Solution = Solution.create(self._species, self._planet)
        solution.value = jnp.array(sol.value)

        residual = self.get_residual(solution, constraints)
        rmse = np.sqrt(np.sum(np.array(residual) ** 2))
        # Success is indicated by no message
        if optx.RESULTS[sol.result] == "":
            logger.info("Success. RMSE = %0.2e, steps = %d", rmse, sol.stats["num_steps"])
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))

        # It is useful to also return the jacobian of this system for testing
        jacobian: Callable = self.jacobian((constraints,))
        logger.info("Jacobian = %s", jacobian(solution.value))

        return sol, jacobian, solution


class SolverScipy(Solver):

    def jacobian_scipy(self, solution_array: Array, args) -> Callable:
        # constraints = args[0]

        # Partially apply `args` to the `objective_function`
        def wrapped_objective(solution_array):
            return self.objective_function(solution_array, args)

        return jax.jacobian(wrapped_objective)(solution_array)

    @override
    def solve(
        self,
        initial_solution: InitialSolutionProtocol | None = None,
        *,
        constraints: SystemConstraints,
        tol: float = 1.0e-8,
    ) -> tuple[OptimizeResult, None, Solution]:
        """Solve using scipy

        Args:
            constraints: Constraints for the system of equations
            initial_solution: Initial condition for this solve only. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Scipy solution, Jacobian function (currently None), solution
        """

        # FIXME: When available, could pass in the Jacobian function determined by JAX.

        if initial_solution is None:
            initial_solution = InitialSolutionDict(species=self._species)
        assert initial_solution is not None

        initial_solution_guess: Array = initial_solution.get_log10_value(
            constraints,
            temperature=self.temperature(),
            pressure=1,
        )

        # TODO: Hacky to enclose the args in an extra tuple, but the arguments by root seem to
        # be passed differently to optimistix? To clarify and clean up with a consistent solver
        # interface.
        sol: OptimizeResult = root(
            self.objective_function,
            initial_solution_guess,
            args=((constraints,),),
            tol=tol,
            jac=self.jacobian_scipy,
        )

        solution: Solution = Solution.create(self._species, self._planet)
        solution.value = jnp.array(sol.x)

        residual = self.get_residual(solution, constraints)
        rmse = np.sqrt(np.sum(np.array(residual) ** 2))
        # Success is indicated by no message
        if sol.success:
            logger.info("Success")
            logger.info("Success. RMSE = %0.2e, steps = %d", rmse, sol["nfev"])
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))

        # It is useful to also return the jacobian of this system for testing
        jacobian: Callable = self.jacobian((constraints,))
        logger.info("Jacobian = %s", jacobian(solution.value))

        return sol, None, solution

    # TODO: Framework for batched calculations
    # def solve_optimistix_batched(
    #     self,
    #     initial_solution: InitialSolutionProtocol | None = None,
    #     *,
    #     constraints_list: list[SystemConstraints],
    #     tol: float = 1.0e-8,
    # ) -> tuple[list[Solution_optx], list[Callable], list[Solution]]:

    #     # Vectorize the solve_optimistix method over the constraints_list
    #     solve_optimistix_vmap = jax.vmap(
    #         lambda constraints: self.solve_optimistix(
    #             initial_solution=initial_solution, constraints=constraints, tol=tol
    #         ),
    #         in_axes=(0,),
    #     )

    #     # Apply vmap over the constraints_list
    #     solutions, jacobians, final_solutions =
    #       solve_optimistix_vmap(jnp.array(constraints_list))

    #     return solutions, jacobians, final_solutions
