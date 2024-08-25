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
    """A solver

    Args:
        *args: Positional keyword arguments for child classes.
        **kwargs: Keyword arguments for child classes.
    """

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
    """Optimistix solver

    Args:
        method: Type of solver. Options are `bfgs`, `chord`, `dogleg`, `lm`, and `newton`. Defaults
            to `newton`.
    """

    @override
    def __init__(self, method: str = "newton"):
        super().__init__()
        self.method: str = method
        self.solver = self._get_solver()
        logger.debug("Creating %s with %s", self.__class__.__name__, method)

    def _get_solver(self):
        """Gets the Optimistic solver

        Returns:
            Solver class
        """
        if self.method == "bfgs":
            solver = optx.BFGS
        elif self.method == "chord":
            solver = optx.Chord
        elif self.method == "dogleg":
            solver = optx.Dogleg
        elif self.method == "lm":
            solver = optx.LevenbergMarquardt
        elif self.method == "newton":
            solver = optx.Newton

        return solver

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

        solver = self.solver(rtol=tol, atol=tol)

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
                "Optimistix success with %s. RMSE = %0.2e, steps = %d",
                self.method,
                rmse,
                sol.stats["num_steps"],
            )
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))

        return solution


class SolverScipy(Solver):
    """SciPy solver

    Args:
        method: Type of solver. Defaults to `hybr`.
        jac: Jacobian. If True uses the JAX autodiff derived Jacobian, otherwise False uses a
            numerical approximation. Note this differs from the definition of jac in the
            scipy.optimize.root documentation. Defaults to False.
        options: A dictionary of solver options. Defaults to None.
    """

    @override
    def __init__(self, method: str = "hybr", jac: bool = False, options: dict | None = None):
        super().__init__()
        self.method: str = method
        if jac:
            self.jac: Callable | bool = self._jacobian_scipy
        else:
            self.jac = False
        self.options: dict | None = options
        logger.debug("Creating %s with %s and jac = %s)", self.__class__.__name__, method, jac)

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
            method=self.method,
            jac=self._jacobian_scipy,
            tol=tol,
            options=self.options,
        )

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)
        solution.value = jnp.array(sol.x)
        residual: Array = solve_me.get_residual(solution, constraints)
        rmse: npt.NDArray[np.float_] = np.sqrt(np.sum(np.array(residual) ** 2))

        if sol.success:
            logger.info(
                "Scipy success with %s and jac=%s. RMSE=%0.2e, steps=%d",
                self.method,
                self.jac,
                rmse,
                sol["nfev"],
            )
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))

        return solution
