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
import pprint
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array
from scipy.optimize import OptimizeResult, root

from atmodeller.constraints import SystemConstraints
from atmodeller.initial_solution import InitialSolutionDict, InitialSolutionProtocol
from atmodeller.reaction_network import ResidualProtocol
from atmodeller.solution import Solution

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class Solver(ABC):
    """A solver"""

    def get_initial_solution(
        self,
        solve_me: ResidualProtocol,
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol | None = None,
    ) -> Array:
        """Gets the initial solution

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial condition. Defaults to None.

        Returns:
            An array of the initial solution
        """
        if initial_solution is None:
            initial_solution = InitialSolutionDict(species=solve_me.species)
        assert initial_solution is not None

        initial_solution_guess: Array = initial_solution.get_log10_value(
            constraints,
            temperature=solve_me.temperature(),
            pressure=1,
        )

        return initial_solution_guess

    def jacobian(self, kwargs: dict[str, Any]) -> Callable:
        """Creates a Jacobian callable function.

        Args:
            kwargs: Required keyword arguments

        Returns:
            A callable function that computes the Jacobian with respect to `solution_array`.
        """

        # Partially apply `kwargs` to the `objective_function`
        def wrapped_objective(solution_array) -> Array:
            return self.objective_function(solution_array, kwargs)

        return jax.jacobian(wrapped_objective)

    def objective_function(self, solution_array: Array, kwargs: dict[str, Any]) -> Array:
        """Objective function

        Args:
            solution_array: Array of the solution
            kwargs: Required keyword arguments

        Returns:
            Residual array
        """
        logger.debug("solution_array passed into objective_functiom = %s", solution_array)

        solve_me: ResidualProtocol = kwargs["solve_me"]
        constraints: SystemConstraints = kwargs["constraints"]

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)
        solution.value = solution_array

        residual: Array = solve_me.get_residual(solution, constraints)

        # jax.debug.print("solution into objective = {solution}", solution=solution.value)
        # jax.debug.print("residual out of objective = {residual}", residual=residual)

        return residual

    @abstractmethod
    def solve(
        self,
        solve_me: ResidualProtocol,
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol | None = None,
        tol: float = 1.0e-8,
    ) -> Solution:
        """Solve

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial solution
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Solution
        """


class SolverOptimistix(Solver):
    """Optimistix solver"""

    @override
    def solve(
        self,
        solve_me: ResidualProtocol,
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol | None = None,
        tol: float = 1.0e-8,
    ) -> Solution:
        """Solve using Optimistix

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial condition for this solve only. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Solution
        """
        initial_solution_guess: Array = self.get_initial_solution(
            solve_me, constraints=constraints, initial_solution=initial_solution
        )
        kwargs: dict[str, Any] = {"solve_me": solve_me, "constraints": constraints}

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
            args=kwargs,
            throw=True,
        )

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)
        solution.value = jnp.array(sol.value)
        residual: Array = solve_me.get_residual(solution, constraints)
        rmse: npt.NDArray[np.float_] = np.sqrt(np.sum(np.array(residual) ** 2))

        # Success is indicated by no message
        if optx.RESULTS[sol.result] == "":
            logger.info(
                "Optimistix success. RMSE = %0.2e, steps = %d", rmse, sol.stats["num_steps"]
            )
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))

        return solution


class SolverScipy(Solver):
    """SciPy solver"""

    def _jacobian_scipy(self, solution_array: Array, kwargs: dict[str, Any]) -> Callable:
        """Jacobian for scipy root, which must accept the same arguments as the objective function.

        Args:
            solution_array: Array of the solution
            kwargs: Required keyword arguments

        Returns:
            The evaluated Jacobian at the given `solution_array`.
        """
        jacobian_func: Callable = self.jacobian(kwargs)

        return jacobian_func(solution_array)

    @override
    def solve(
        self,
        solve_me: ResidualProtocol,
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol | None = None,
        tol: float = 1.0e-8,
    ) -> Solution:
        """Solve using Scipy

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial condition for this solve only. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Solution
        """
        initial_solution_guess: Array = self.get_initial_solution(
            solve_me, constraints=constraints, initial_solution=initial_solution
        )
        kwargs: dict[str, Any] = {"solve_me": solve_me, "constraints": constraints}

        sol: OptimizeResult = root(
            self.objective_function,
            initial_solution_guess,
            args=kwargs,
            tol=tol,
            jac=self._jacobian_scipy,
        )

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)
        solution.value = jnp.array(sol.x)
        residual: Array = solve_me.get_residual(solution, constraints)
        rmse: npt.NDArray[np.float_] = np.sqrt(np.sum(np.array(residual) ** 2))

        if sol.success:
            logger.info("Scipy success. RMSE = %0.2e, steps = %d", rmse, sol["nfev"])
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))

        return solution


# TODO: Framework for batched calculations
# class solve_optimistix_batched(
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
