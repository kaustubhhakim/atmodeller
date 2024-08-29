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
from atmodeller.initial_solution import InitialSolutionProtocol
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
        initial_solution: InitialSolutionProtocol,
        perturb_log10_number_density: float = 0,
        pressure: float = 1.0,
    ) -> Array:
        """Gets the initial solution

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial solution.
            perturb_log10_number_density: Maximum log10 perturbation to apply to the number
                densities of the initial solution. Defaults to 0.
            pressure: Total pressure to evaluate the constraints. Defaults to 1 bar.

        Returns:
            An array of the initial solution
        """
        initial_solution_guess: Array = initial_solution.get_log10_value(
            constraints,
            temperature=solve_me.temperature(),
            pressure=pressure,
            perturb_log10_number_density=perturb_log10_number_density,
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
        logger.debug("solution_array passed into objective_function = %s", solution_array)

        solve_me: ResidualProtocol = kwargs["solve_me"]
        constraints: SystemConstraints = kwargs["constraints"]

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)
        solution.value = solution_array

        residual: Array = solve_me.get_residual(solution, constraints)

        return residual

    @abstractmethod
    def solve(
        self,
        solve_me: ResidualProtocol,
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol,
        tol: float = 1.0e-8,
        perturb_log10_number_density: float = 0,
    ) -> tuple[Solution, bool]:
        """Solve

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial solution
            tol: Tolerance. Defaults to 1.0e-8.
            perturb_log10_number_density: Maximum log10 perturbation to apply to the number
                densities of the initial solution. Defaults to 0.

        Returns:
            Solution and a bool to indicate success
        """


class SolverOptimistix(Solver):
    """Optimistix solver

    Args:
        method: Type of solver. Options are `bfgs`, `chord`, `dogleg`, `lm`, and `newton`. Defaults
            to `newton`.
        max_steps: Maximum number of steps. Defaults to 256
    """

    @override
    def __init__(self, method: str = "newton", max_steps: int = 256):
        super().__init__()
        self.method: str = method
        self.max_steps: int = max_steps
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
        initial_solution: InitialSolutionProtocol,
        tol: float = 1.0e-8,
        perturb_log10_number_density: float = 0,
    ) -> tuple[Solution, bool]:
        """Solve using Optimistix

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial solution
            tol: Tolerance. Defaults to 1.0e-8.
            perturb_log10_number_density: Maximum log10 perturbation to apply to the number
                densities of the initial solution. Defaults to 0.

        Returns:
            Solution and a bool to indicate success
        """
        initial_solution_guess: Array = self.get_initial_solution(
            solve_me,
            constraints=constraints,
            initial_solution=initial_solution,
            perturb_log10_number_density=perturb_log10_number_density,
        )
        kwargs: dict[str, Any] = {"solve_me": solve_me, "constraints": constraints}

        solver_optx = self.solver(rtol=tol, atol=tol)

        sol = optx.root_find(
            self.objective_function,
            solver_optx,
            initial_solution_guess,
            args=kwargs,
            throw=False,
            max_steps=self.max_steps,
        )

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)

        # Success is indicated by no message
        if optx.RESULTS[sol.result] == "":
            solution.value = jnp.array(sol.value)
            residual: Array = solve_me.get_residual(solution, constraints)
            rmse: npt.NDArray[np.float_] = np.sqrt(np.sum(np.array(residual) ** 2))
            logger.info(
                "Optimistix success with %s. RMSE = %0.2e, steps = %d",
                self.method,
                rmse,
                sol.stats["num_steps"],
            )
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))
            success: bool = True
        else:
            logger.warning("Optimistix solver failed (message=%s).", optx.RESULTS[sol.result])
            success = False

        return solution, success


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
        initial_solution: InitialSolutionProtocol,
        tol: float = 1.0e-8,
        perturb_log10_number_density: float = 0,
    ) -> tuple[Solution, bool]:
        """Solve using Scipy

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial solution
            tol: Tolerance. Defaults to 1.0e-8.
            perturb_log10_number_density: Maximum log10 perturbation to apply to the number
                densities. Defaults to 0.

        Returns:
            Solution and a bool to indicate success
        """
        initial_solution_guess: Array = self.get_initial_solution(
            solve_me,
            constraints=constraints,
            initial_solution=initial_solution,
            perturb_log10_number_density=perturb_log10_number_density,
        )
        kwargs: dict[str, Any] = {"solve_me": solve_me, "constraints": constraints}

        sol: OptimizeResult = root(
            self.objective_function,
            initial_solution_guess,
            args=kwargs,
            method=self.method,
            jac=self.jac,
            tol=tol,
            options=self.options,
        )

        solution: Solution = Solution.create(solve_me.species, solve_me.planet)

        if sol.success:
            solution.value = jnp.array(sol.x)
            residual: Array = solve_me.get_residual(solution, constraints)
            rmse: npt.NDArray[np.float_] = np.sqrt(np.sum(np.array(residual) ** 2))
            logger.info(
                "Scipy success with %s and jac = %s. RMSE = %0.2e, steps = %d",
                self.method,
                self.jac if self.jac is False else True,
                rmse,
                sol["nfev"],
            )
            logger.info("Solution = %s", pprint.pformat(solution.output_solution()))
            logger.info("Raw solution = %s", pprint.pformat(solution.output_raw_solution()))
            success: bool = True
        else:
            logger.warning("Scipy solver failed (message=%s).", sol.message)
            success = False

        return solution, success


class SolverTryAgain(Solver):
    """Solver that perturbs the initial solution on failure and tries again.

    Args:
        solver: Solver
        max_attempts: Maximum number of attempts to randomise the initial condition to find a
            solution if the initial guess fails. Defaults to 20.
        perturb_log10_number_density: Maximum log10 perturbation to apply to the number densities
            on failure. Defaults to 2.
    """

    @override
    def __init__(
        self,
        solver: Solver,
        max_attempts: int = 20,
        perturb_log10_number_density: float = 2.0,
        errors: str = "ignore",
    ):
        super().__init__()
        logger.debug("Creating %s with %s", self.__class__.__name__, solver.__class__.__name__)
        self._solver: Solver = solver
        self._max_attempts: int = max_attempts
        self._perturb_log10_number_density: float = perturb_log10_number_density
        self._errors: str = errors

    @override
    def solve(
        self,
        solve_me: ResidualProtocol,
        *,
        constraints: SystemConstraints,
        initial_solution: InitialSolutionProtocol,
        tol: float = 1.0e-8,
    ) -> tuple[Solution, bool]:
        """Solve

        Args:
            solve_me: System to solve
            constraints: Constraints for the system of equations
            initial_solution: Initial solution. Defaults to None.
            tol: Tolerance. Defaults to 1.0e-8.

        Returns:
            Solution and a bool to indicate success
        """
        # For the first solution we try without a perturbation
        perturb_log10_number_density: float = 0

        for attempt in range(self._max_attempts):
            logger.info("Attempt %d/%d", attempt + 1, self._max_attempts)

            solution, success = self._solver.solve(
                solve_me,
                constraints=constraints,
                initial_solution=initial_solution,
                tol=tol,
                perturb_log10_number_density=perturb_log10_number_density,
            )

            if success:
                return solution, success
            else:
                # Perturb solution and try again
                perturb_log10_number_density = self._perturb_log10_number_density

        logger.info("%s failed after %d attempts", __class__.__name__, self._max_attempts)

        return solution, False
