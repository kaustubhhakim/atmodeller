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
from jaxtyping import PRNGKeyArray

from atmodeller.jax_utils import FloatArray
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

    @eqx.filter_jit
    # For testing and debugging
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve(self, base_solution_array: FloatArray = jnp.array(jnp.nan)) -> Output:
        """Runs the nonlinear solver and initialises the output state.

        This method executes the compiled equilibrium solver produced by :meth:`_make_solver` and
        stores the resulting solution for downstream processing. It accepts updated planetary/
        environmental constraints and initial guesses for the nonlinear system. After successful
        convergence, an internal ``Output`` instance is created to expose number densities,
        activities, stabilities, and post-processed diagnostic quantities.

        Args:
            parameters: Parameters defining the equilibrium problem, including species, reactions,
                and environmental conditions.
            base_solution_array: Initial guess for the solver, typically a broadcasted array of
                log number moles and log stabilities.

        Returns:
            An :class:`~atmodeller.output.Output` instance containing the results
        """
        return self._solver(self._parameters, self._key, base_solution_array)

    def update_constraints(self, *args, **kwargs) -> Self:
        """Updates the model's constraints

        Args:
            *args: Positional arguments to update constraints
            **kwargs: Keyword arguments to update constraints

        Returns:
            A new instance of :class:`EquilibriumModel` with updated constraints"""
        parameters_updated: Parameters = self._parameters.update_constraints(*args, **kwargs)
        model_updated: EquilibriumModel = eqx.tree_at(
            lambda m: m._parameters, self, parameters_updated
        )

        return cast(Self, model_updated)

    def update_state(self, *args, **kwargs) -> Self:
        """Updates the model's state

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

    # def solve(
    #     self,
    #     *,
    #     initial_log_number_moles: Optional[ArrayLike] = None,
    #     initial_log_stability: Optional[ArrayLike] = None,
    #     state: Optional[BaseThermodynamicState] = None,
    #     activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
    #     mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
    #     solver_parameters: Optional[SolverParameters] = None,
    # ) -> Array:
    #     """Runs the nonlinear solver and initialises the output state.

    #     This method executes the compiled equilibrium solver produced by :meth:`set_solver` and
    #     stores the resulting solution for downstream processing. It optionally accepts updated
    #     planetary/environmental constraints and initial guesses for the nonlinear system. After
    #     successful convergence, an internal ``Output`` instance is created to expose number
    #     densities, activities, stabilities, and post-processed diagnostic quantities.

    #     If :meth:`set_solver` has not been called, a suitable solver will be constructed and
    #     JIT-compiled automatically. Repeated calls to :meth:`solve` with compatible shapes will be
    #     fast and will reuse cached compilation artifacts.

    #     Args:
    #         initial_log_number_moles: Initial log number of moles. Defaults to ``None``.
    #         initial_log_stability: Initial log stability. Defaults to ``None``.
    #         state: Thermodynamic state. Defaults to ``None``.
    #         activity_constraints: Activity constraints. Defaults to ``None``.
    #         mass_constraints: Mass constraints. Defaults to ``None``.
    #         solver_parameters: Solver parameters. Defaults to ``None``.
    #     """
    #     parameters: Parameters = Parameters.from_reaction_system(
    #         self.reaction_system,
    #         state,
    #         activity_constraints,
    #         mass_constraints,
    #         solver_parameters,
    #     )

    #     key: PRNGKeyArray = jax.random.PRNGKey(0)
    #     key, subkey = jax.random.split(key)  # Split the key for use in this function

    #     # Rebuild the solver only when the shape of any array leaf changes; this covers both
    #     # batch_size changes and structural changes (e.g. a scalar buffer becoming batched)
    #     # that affect vmap_axes_spec. filter_jit handles value-only changes automatically.
    #     solver_shapes: tuple = tuple(
    #         leaf.shape for leaf in jax.tree_util.tree_leaves(parameters) if hasattr(leaf, "shape")
    #     )
    #     if self._solver is None or self._solver_shapes != solver_shapes:
    #         self._solver = make_solver_with_jit(parameters)
    #         self._solver_shapes = solver_shapes

    #     # Allow the user to provide initial guesses for the solver, but if they are not provided,
    #     # apply a default auto guess (nan) that will trigger the solver's internal heuristic to
    #     # generate an initial guess. This is often more robust than a fixed initial guess, which
    #     # may be far from the solution for some models and lead to solver failure.
    #     if initial_log_number_moles is None and initial_log_stability is None:
    #         logger.info("Applying auto initial guess")
    #         # Any NaN value will trigger the solver's internal heuristic to generate an initial
    #         # guess.
    #         initial_solution: Float[Array, "n_batch twice_species"] = jnp.broadcast_to(
    #             jnp.nan, (parameters.batch_size, self.reaction_system.species.number_species * 2)
    #         )
    #     else:
    #         if initial_log_number_moles is None:
    #             initial_log_number_moles = INITIAL_LOG_NUMBER_MOLES
    #         initial_log_number_moles = jnp.broadcast_to(
    #             initial_log_number_moles,
    #             (parameters.batch_size, self.reaction_system.species.number_species),
    #         )
    #         logger.debug("initial_log_number_moles = %s", initial_log_number_moles)
    #         if initial_log_stability is None:
    #             initial_log_stability = INITIAL_LOG_STABILITY
    #         initial_log_stability = jnp.broadcast_to(
    #             initial_log_stability,
    #             (parameters.batch_size, self.reaction_system.species.number_species),
    #         )
    #         logger.debug("initial_log_stability = %s", initial_log_stability)
    #         initial_solution = jnp.concatenate(
    #             (initial_log_number_moles, initial_log_stability), axis=-1
    #         )
    #         logger.info("Initial solution = %s", initial_solution)
