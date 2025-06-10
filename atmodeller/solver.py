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
"""Solver wrappers"""

# TODO: Refresh
# def make_solve_tau_step(solver_fn: Callable, traced_parameters: TracedParameters) -> Callable:
#     """Wraps the repeat solver to call it for different tau values

#     Args:
#         solver_fn: Solver function with pre-bound fixed configuration
#         traced_parameters: Traced parameters

#     Returns:
#         Wrapped solver for a single tau value
#     """

#     @eqx.filter_jit
#     # @eqx.debug.assert_max_traces(max_traces=1)
#     def solve_tau_step(carry, tau):
#         # Unpack carry state
#         (
#             base_initial_solution,
#             active_indices,
#             multistart_perturbation,
#             max_attempts,
#             key,
#             initial_solution,
#             initial_status,
#             initial_steps,
#         ) = carry

#         # Call repeat_solver with current tau and previous solution as initial
#         new_solution, new_status, new_steps, success_attempt = repeat_solver(
#             solver_fn,
#             base_initial_solution,
#             active_indices,
#             tau,
#             traced_parameters,
#             initial_solution,
#             initial_status,
#             initial_steps,
#             multistart_perturbation,
#             max_attempts,
#             key,
#         )

#         # Update PRNG key for next iteration
#         key, _ = jax.random.split(key)

#         new_carry = (
#             base_initial_solution,
#             active_indices,
#             multistart_perturbation,
#             max_attempts,
#             key,
#             new_solution,
#             # The repeat solver should always run for each value of tau
#             jnp.zeros_like(initial_status, dtype=bool),
#             jnp.zeros_like(initial_steps, dtype=int),
#         )

#         # Output current solution etc for this tau step
#         out = (new_solution, new_status, new_steps, success_attempt)

#         return new_carry, out

#     return solve_tau_step
