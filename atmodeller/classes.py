# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Equilibrium model API.

Provides the :class:`EquilibriumModel`, the primary entry point for constructing and solving
thermodynamic equilibrium problems.

This module coordinates:

- Phase definitions (gas, melt, solid, pure condensates),
- Reaction system construction,
- Nonlinear solver selection and execution (basic or robust),
- Batched solution handling via JAX,
- Post-processed results through :class:`Output`.

Helper utilities are included to broadcast and standardize initial conditions for batched solves.

Typical usage:

    model = EquilibriumModel(gas, melt=..., solid=..., condensates=...)
    model.solve(state=...)
    results = model.output
"""

import logging
from collections.abc import Callable
from typing import Self, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.solvers import make_solver_with_jit

logger: logging.Logger = logging.getLogger(__name__)


class EquilibriumModel(eqx.Module):
    """An equilibrium model

    This is the main class that the user interacts with to build equilibrium models, solve them,
    and retrieve the results.
    """

    _parameters: Parameters
    _solver: Callable
    _key: PRNGKeyArray

    def __init__(self, parameters: Parameters):
        self._parameters = parameters
        self._solver = make_solver_with_jit(parameters)
        self._key = jax.random.PRNGKey(0)

    # For testing and debugging
    # @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve(self, base_solution_array: Float[Array, "#n_batch twice_species"]) -> Output:
        """Runs the solver and returns the output state.

        Note:
            This method is intentionally a thin Python wrapper. The heavy numerical path is
            compiled inside ``self._solver``
            (created by :func:`~atmodeller.solvers.make_solver_with_jit`).

        Args:
            base_solution_array: Initial guess for the solver, typically a broadcasted array of
                log number moles and log stabilities.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        return self._solver(self._parameters, self._key, base_solution_array)

    # For testing and debugging
    # @eqx.filter_jit
    def solve_with_default(self) -> Output:
        """Runs the solver with a default initial guess.

        Note:
            Like :meth:`solve`, this method is a lightweight non-jitted wrapper around the
            already-jitted ``self._solver`` callable.

        Returns:
            An :class:`~atmodeller.output.Output` instance
        """
        base_solution_array: Float[Array, "#n_batch twice_species"] = jnp.full(
            (
                self._parameters.batch_size,
                self._parameters.reaction_system.species.number_species * 2,
            ),
            jnp.nan,
        )

        return self._solver(self._parameters, self._key, base_solution_array)

    def rebuild_solver(self) -> Self:
        """Rebuilds the compiled solver from the model's current parameters.

        Use this after parameter updates that may have changed array-leaf shapes or batching in a
        way that invalidates the vmapping axes captured by the existing solver closure, or that
        invalidates the original leaf-shape broadcasting assumptions preserved by the
        ``update`` methods on the parameter containers.

        Returns:
            A new instance of :class:`EquilibriumModel` with a rebuilt solver
        """
        solver_rebuilt: Callable = make_solver_with_jit(self._parameters)
        model_rebuilt: EquilibriumModel = eqx.tree_at(lambda m: m._solver, self, solver_rebuilt)

        return cast(Self, model_rebuilt)

    def update_constraints(self, *args, **kwargs) -> Self:
        """Updates the model's constraints.

        Args:
            *args: Positional arguments to update constraints
            **kwargs: Keyword arguments to update constraints

        Returns:
            A new instance of :class:`EquilibriumModel` with updated constraints
        """
        parameters_updated: Parameters = self._parameters.update_constraints(*args, **kwargs)
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m._parameters, self, parameters_updated
        )

        return cast(Self, model_updated)

    def update_state(self, *args, **kwargs) -> Self:
        """Updates the model's state.

        Args:
            *args: Positional arguments to update state
            **kwargs: Keyword arguments to update state

        Returns:
            A new instance of :class:`EquilibriumModel` with updated state
        """
        parameters_updated: Parameters = self._parameters.update_state(*args, **kwargs)
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m._parameters, self, parameters_updated
        )

        return cast(Self, model_updated)

    # TODO: To reinstate at some point, but needs to be adapted to new output structure and
    # parameters handling

    # def calculate_disequilibrium(
    #     self, *, state: ThermodynamicStateProtocol, log_number_moles: ArrayLike
    # ) -> None:
    #     """Computes the Gibbs free energy disequilibrium.

    #     This method calculates the Gibbs free energy difference (ΔG) for each considered reaction
    #     relative to equilibrium, based on the current state of the system. A value of zero
    #     indicates a reaction at equilibrium, while positive or negative values indicate departures
    #     from equilibrium in terms of energetic favourability.

    #     Args:
    #         state: Thermodynamic state
    #         log_number_moles: Log number of moles
    #     """
    #     parameters: Parameters = Parameters.from_reaction_system(self.reaction_system, state)
    #     solution_array: Array = broadcast_initial_solution(
    #         log_number_moles,
    #         None,
    #         self.reaction_system.species.number_species,
    #         parameters.batch_size,
    #     )
    #     # jax.debug.print("solution_array = {out}", out=solution_array)

    #     self._output = OutputDisequilibrium(parameters, solution_array)
