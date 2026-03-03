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
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxmod.solvers import MultiAttemptSolution
from jaxmod.type_aliases import NpFloat
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray

from atmodeller.constants import INITIAL_LOG_NUMBER_MOLES, INITIAL_LOG_STABILITY
from atmodeller.containers import SolverParameters
from atmodeller.interfaces import FugacityConstraintProtocol, ThermodynamicStateProtocol
from atmodeller.output import OutputDisequilibrium
from atmodeller.output_new import Output
from atmodeller.parameters import Parameters
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase
from atmodeller.reactions import ReactionSystem
from atmodeller.solvers import solve_with_jit

logger: logging.Logger = logging.getLogger(__name__)


class EquilibriumModel:
    """An equilibrium model

    This is the main class that the user interacts with to build equilibrium models, solve them,
    and retrieve the results.

    Args:
        gas: Gas phase
        melt: Melt phase. Defaults to an empty melt phase if not provided.
        solid: Solid phase. Defaults to an empty solid phase if not provided.
        condensates: Pure condensate phases. Defaults to an empty tuple if not provided.
    """

    reaction_system: ReactionSystem
    _solver: Optional[Callable] = None
    _output: Optional[Output] = None

    def __init__(
        self,
        gas: GasPhase,
        *,
        melt: Optional[MeltPhase] = None,
        solid: Optional[SolidPhase] = None,
        condensates: Optional[Iterable[PurePhase]] = None,
    ):
        if melt is None:
            melt = MeltPhase.empty()
        if solid is None:
            solid = SolidPhase.empty()
        if condensates is None:
            condensates = ()

        self.reaction_system = ReactionSystem(gas, melt=melt, solid=solid, condensates=condensates)

    @property
    def output(self) -> Output:
        if self._output is None:
            raise AttributeError("Output has not been set.")

        return self._output

    def calculate_disequilibrium(
        self, *, state: ThermodynamicStateProtocol, log_number_moles: ArrayLike
    ) -> None:
        """Computes the Gibbs free energy disequilibrium.

        This method calculates the Gibbs free energy difference (ΔG) for each considered reaction
        relative to equilibrium, based on the current state of the system. A value of zero
        indicates a reaction at equilibrium, while positive or negative values indicate departures
        from equilibrium in terms of energetic favourability.

        Args:
            state: Thermodynamic state
            log_number_moles: Log number of moles
        """
        parameters: Parameters = Parameters.create(self.reaction_system, state)
        solution_array: Array = broadcast_initial_solution(
            log_number_moles,
            None,
            self.reaction_system.species.number_species,
            parameters.batch_size,
        )
        # jax.debug.print("solution_array = {out}", out=solution_array)

        self._output = OutputDisequilibrium(parameters, solution_array)

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
        parameters: Parameters = Parameters.create(
            self.reaction_system,
            state,
            fugacity_constraints,
            mass_constraints,
            solver_parameters,
        )
        base_solution_array: Array = broadcast_initial_solution(
            initial_log_number_moles,
            initial_log_stability,
            self.reaction_system.species.number_species,
            parameters.batch_size,
        )
        # jax.debug.print("base_solution_array = {out}", out=base_solution_array)

        key: PRNGKeyArray = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)  # Split the key for use in this function

        # Previous
        # if self._solver is None or solver_recompile:
        #    if solver == "basic":
        #        self._solver = make_independent_solver(parameters)
        #        # Alternatively, could use the batch solver
        #        # self._solver = make_batch_solver(parameters)
        #    elif solver == "robust":
        #        self._solver = make_solver(parameters)
        #    else:
        #        raise ValueError(f"Unknown solver type: {solver}")
        #    self._selected_solver = solver

        multi_sol: MultiAttemptSolution = solve_with_jit(base_solution_array, parameters, subkey)

        # previous
        # multi_sol: MultiAttemptSolution = MultiAttemptSolution(
        #    sol
        # )  # self._solver(base_solution_array, parameters, subkey)

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

        # Count unique values and their frequencies
        unique_vals, counts = jnp.unique(multi_sol.attempts, return_counts=True)
        for val, count in zip(unique_vals.tolist(), counts.tolist()):
            logger.info(
                "Multistart summary: %d (%0.2f%%) models(s) required %d attempt(s)",
                count,
                count * 100 / parameters.batch_size,
                val,
            )

        # Want the maximum number of steps for cases that solved
        mask_num_steps: Bool[Array, "..."] = (
            multi_sol.num_steps < parameters.solver_parameters.max_steps
        )
        # Replace invalid values with -inf so they never win in the max
        max_less_than_max: Array = jnp.where(mask_num_steps, multi_sol.num_steps, -jnp.inf).max()
        logger.info("Solver steps (max) = %s", int(max_less_than_max.item()))

        self._output = Output(parameters, multi_sol)

        return multi_sol.value


# TODO: Make JAX compatible and can return a 1-D array
def _broadcast_component(
    component: Optional[ArrayLike], default_value: float, dim: int, batch_size: int, name: str
) -> NpFloat:
    """Broadcasts a scalar, 1D, or 2D input array to shape ``(batch_size, dim)``.

    This function standardizes inputs that may be:
        - ``None`` (in which case ``default_value`` is used),
        - a scalar (promoted to a 1D array of length ``dim``),
        - a 1D array of shape ``(dim,)`` (broadcast across the batch),
        - or a 2D array of shape ``(batch_size``, dim)`` (used as-is).

    Args:
        component: The input data (or ``None``), representing either a scalar, 1D array, or 2D array
        default_value: The default scalar value to use if ``component`` is ``None``
        dim: The number of features or dimensions per batch item
        batch_size: The number of batch items
        name: Name of the component (used for error messages)

    Returns:
        A numpy array of shape ``(batch_size, dim)``, with values broadcast as needed

    Raises:
        ValueError: If the input array has an unexpected shape or inconsistent dimensions
    """
    if component is None:
        base: NpFloat = np.full((dim,), default_value, dtype=np.float64)
    else:
        component = np.asarray(component, dtype=np.float64)
        if component.ndim == 0:
            base = np.full((dim,), component.item(), dtype=np.float64)
        elif component.ndim == 1:
            if component.shape[0] != dim:
                raise ValueError(f"{name} should have shape ({dim},), got {component.shape}")
            base = component
        elif component.ndim == 2:
            if component.shape[0] != batch_size or component.shape[1] != dim:
                raise ValueError(
                    f"{name} should have shape ({batch_size}, {dim}), got {component.shape}"
                )
            # Replace NaNs with default_value
            component = np.where(np.isnan(component), default_value, component)
            return component
        else:
            raise ValueError(
                f"{name} must be a scalar, 1D, or 2D array, got shape {component.shape}"
            )

    # Promote 1D base to (batch_size, dim)
    return np.broadcast_to(base[None, :], (batch_size, dim))


def broadcast_initial_solution(
    initial_log_number_moles: Optional[ArrayLike],
    initial_log_stability: Optional[ArrayLike],
    number_of_species: int,
    batch_size: int,
) -> Float[Array, "... solution"]:
    """Creates and broadcasts the initial solution to shape ``(batch_size, solution)``

    ``D = number_of_species + number_of_stability``, i.e. the total number of solution quantities

    Args:
        initial_log_number_moles: Initial log number moles or ``None``
        initial_log_stability: Initial log stability or ``None``
        number_of_species: Number of species
        batch_size: Batch size

    Returns:
        Initial solution with shape ``(batch_size, solution)`` or a 1-D array
    """
    number_moles: NpFloat = _broadcast_component(
        initial_log_number_moles,
        INITIAL_LOG_NUMBER_MOLES,
        number_of_species,
        batch_size,
        name="initial_log_number_moles",
    )
    stability: NpFloat = _broadcast_component(
        initial_log_stability,
        INITIAL_LOG_STABILITY,
        number_of_species,
        batch_size,
        name="initial_log_stability",
    )

    solution = jnp.concatenate((number_moles, stability), axis=-1)

    # TODO: Bit hacky. This is new since the objective supports broadcasting naturally now. To
    # clean up.
    if batch_size == 1:
        return solution.squeeze(axis=0)

    return solution
