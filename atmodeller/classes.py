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
from collections.abc import Callable, Iterable, Mapping
from pprint import pformat
from typing import Optional, cast

import jax
import jax.numpy as jnp
from jaxmod.solvers import MultiAttemptSolution
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

from atmodeller.containers import SolverParameters
from atmodeller.interfaces import FugacityConstraintProtocol
from atmodeller.output import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase
from atmodeller.reactions import ReactionSystem
from atmodeller.solvers import make_solve_with_jit
from atmodeller.state import ThermodynamicStateProtocol

logger: logging.Logger = logging.getLogger(__name__)


class EquilibriumModel:
    """An equilibrium model

    This is the main class that the user interacts with to build equilibrium models, solve them,
    and retrieve the results.

    Args:
        gas_phase: Gas phase
        melt_phase: Melt phase. Defaults to ``None``.
        solid_phase: Solid phase. Defaults to ``None``.
        condensate_phases: Pure condensate phases. Defaults to ``None``.
    """

    reaction_system: ReactionSystem
    _solver: Optional[Callable] = None
    _solver_shapes: Optional[tuple] = None
    _output: Optional[Output] = None

    def __init__(
        self,
        gas_phase: GasPhase,
        *,
        melt: Optional[MeltPhase] = None,
        solid_phase: Optional[SolidPhase] = None,
        condensate_phases: Optional[Iterable[PurePhase]] = None,
    ):
        self.reaction_system = ReactionSystem(
            gas_phase,
            melt_phase=melt,
            solid_phase=solid_phase,
            condensate_phases=condensate_phases,
        )
        self._solver: Optional[Callable] = None
        self._solver_shapes: Optional[tuple] = None
        self._output: Optional[Output] = None

    @property
    def output(self) -> Output:
        if self._output is None:
            raise AttributeError("Output has not been set.")

        return self._output

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

    def solve(
        self,
        *,
        initial_log_number_moles: Optional[ArrayLike] = None,
        initial_log_stability: Optional[ArrayLike] = None,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ) -> Array:
        """Runs the nonlinear solver and initialises the output state.

        This method executes the compiled equilibrium solver produced by :meth:`set_solver` and
        stores the resulting solution for downstream processing. It optionally accepts updated
        planetary/environmental constraints and initial guesses for the nonlinear system. After
        successful convergence, an internal ``Output`` instance is created to expose number
        densities, activities, stabilities, and post-processed diagnostic quantities.

        If :meth:`set_solver` has not been called, a suitable solver will be constructed and
        JIT-compiled automatically. Repeated calls to :meth:`solve` with compatible shapes will be
        fast and will reuse cached compilation artifacts.

        Args:
            initial_log_number_moles: Initial log number of moles. Defaults to ``None``.
            initial_log_stability: Initial log stability. Defaults to ``None``.
            state: Thermodynamic state. Defaults to ``None``.
            fugacity_constraints: Fugacity constraints. Defaults to ``None``.
            mass_constraints: Mass constraints. Defaults to ``None``.
            solver_parameters: Solver parameters. Defaults to ``None``.
        """
        parameters: Parameters = Parameters.from_reaction_system(
            self.reaction_system,
            state,
            fugacity_constraints,
            mass_constraints,
            solver_parameters,
        )

        key: PRNGKeyArray = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)  # Split the key for use in this function

        # Rebuild the solver only when the shape of any array leaf changes; this covers both
        # batch_size changes and structural changes (e.g. a scalar buffer becoming batched)
        # that affect vmap_axes_spec. filter_jit handles value-only changes automatically.
        solver_shapes: tuple = tuple(
            leaf.shape for leaf in jax.tree_util.tree_leaves(parameters) if hasattr(leaf, "shape")
        )
        if self._solver is None or self._solver_shapes != solver_shapes:
            self._solver = make_solve_with_jit(parameters)
            self._solver_shapes = solver_shapes

        # TODO: bit hacky, but trips the auto initial guess
        trigger_autogen: Float[Array, "batch_size twice_species"] = jnp.broadcast_to(
            jnp.nan, (parameters.batch_size, self.reaction_system.species.number_species * 2)
        )
        jax.debug.print("Trigger for auto initial guess = {out}", out=trigger_autogen)

        out: Output = self._solver(trigger_autogen, parameters, subkey)
        multi_sol: MultiAttemptSolution = out.multi_attempt_solution

        logger.info("asdict_split = \n%s", pformat(out.asdict_split()))

        # output: Output = self._solver(trigger_autogen, parameters, subkey)
        # multi_sol: MultiAttemptSolution = output.multi_attempt_solution
        # jax.debug.print("multi_sol.value = {out}", out=multi_sol.value)

        num_successful_models: int = jnp.count_nonzero(multi_sol.solver_success).item()
        num_failed_models: int = jnp.count_nonzero(~multi_sol.solver_success).item()

        logger.info(
            "Solve complete: %d (%0.2f%%) successful model(s)",
            num_successful_models,
            num_successful_models * 100 / parameters.batch_size,
        )
        if num_failed_models > 0:
            logger.warning(
                "%d (%0.2f%%) model(s) still failed",
                num_failed_models,
                num_failed_models * 100 / parameters.batch_size,
            )

        # Count unique values and their frequencies, ignoring failed models (attempts == 0)
        successful_attempts = multi_sol.attempts[multi_sol.attempts > 0]
        unique_vals, counts = jnp.unique(successful_attempts, return_counts=True)
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            logger.info(
                "Attempt summary (solved): %d (%0.2f%%) model(s) required %d attempt(s)",
                count,
                count * 100 / parameters.batch_size,
                val,
            )

        # Steps of 0 indicate no solution; replace with nan and report the max over solved models
        steps_float: Array = cast(
            Array, jnp.where(multi_sol.num_steps == 0, jnp.nan, multi_sol.num_steps.astype(float))
        )
        max_steps: Array = jnp.nanmax(steps_float)
        logger.info("Solver steps (max) = %s", int(max_steps.item()))

        self._output = Output(parameters, multi_sol)

        return multi_sol.value
