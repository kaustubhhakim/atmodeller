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


class SolverScipyTryAgain(Solver):
    """Scipy solver that perturbs the initial solution on failure and tries again.

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
        logger.debug("Creating %s with %s and jac = %s)", self.__class__.__name__, method, jac)
        solver = SolverScipy(method, jac, options)

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


#     @property
#     def failed_solves(self) -> int:
#         """Number of failed solves"""
#         percentage_failed: float = self._failed_solves * 100 / self.number_of_attempted_solves
#         logger.info(
#             "%d failed solves from a total attempted of %d (%.1f %%)",
#             self._failed_solves,
#             self.number_of_attempted_solves,
#             percentage_failed,
#         )

#         return self._failed_solves

#     @property
#     def number_of_attempted_solves(self) -> int:
#         """The total number of systems with attempted solves"""
#         return self._attempted_solves

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
