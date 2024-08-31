#!/usr/bin/env python
"""Minimum working example (MWE) for comparing Scipy and Optimistix solvers.

Reproduces test_CHO_low_temperature in test_benchmark.py using some hard-coded parameters.
"""
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import optimistix as optx
from jax import Array, jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from scipy.constants import Avogadro, Boltzmann, gas_constant

# Scipy also fails if this is commented out. Evidently double precision is required regardless.
jax.config.update("jax_enable_x64", True)

MACHEPS: float = float(jnp.finfo(jnp.float_).eps)
"""Machine epsilon"""

# For scaling
GAS_CONSTANT_BAR: float = gas_constant * 1.0e-5
BOLTZMANN_CONSTANT_BAR: float = Boltzmann * 1.0e-5

# This allows for two scalings, which in practice can be thought of as two that can cancel
AVOGADRO: float = Avogadro
log10_AVOGADRO: float = np.log10(AVOGADRO)  # to convert to moles
log_AVOGADRO: float = np.log(AVOGADRO)
scaling: float = 1  # mole faction scaling
log10_scaling: float = np.log10(scaling)
log_scaling: float = np.log(scaling)

# For testing solvers, this is the known solution of the system
known_solution: Dict[str, float] = {
    "H2": 26.950804260065272,
    "H2O": 26.109794057030303,
    "CO2": 11.303173861822636,
    "O2": -27.890841758236377,
    "CH4": 26.411827244097612,
    "CO": 9.537726420793389,
}

known_solution_array: npt.NDArray[np.float_] = np.array([val for val in known_solution.values()])

# This should be kept at this temperature since the equilibrium constants for the reactions below
# are hard-coded for this temperature
temperature: float = 450
planet_surface_area: float = 510064471909788.25  # SI units
planet_surface_gravity: float = 9.819973426224687  # SI units

# MWE for reaction network / mass balance
# Species order is: H2, H2O, CO2, O2, CH4, CO

# Species molar masses in kg/mol
molar_masses_dict: Dict[str, float] = {
    "H2": 0.002015882,
    "H2O": 0.018015287,
    "CO2": 0.044009549999999995,
    "O2": 0.031998809999999996,
    "CH4": 0.016042504000000003,
    "CO": 0.028010145,
}


def dimensional_to_scaled_base10(dimensional_number_density):
    return dimensional_number_density - log10_AVOGADRO + log10_scaling


def scaled_to_dimensional_base10(scaled_number_density):
    return scaled_number_density - dimensional_to_scaled_base10(0)


# Element log10 number of total molecules constraints:
log10_oxygen_constraint: float = dimensional_to_scaled_base10(45.58848007858896)
log10_hydrogen_constraint: float = dimensional_to_scaled_base10(46.96664792007732)
log10_carbon_constraint: float = dimensional_to_scaled_base10(45.89051326565627)
# Initial solution guess number density (molecules/m^3)
# initial_solution_default: Array = jnp.array([26, 26, 12, -26, 26, 25], dtype=jnp.float_)
initial_solution_default: Array = jnp.array([26, 26, 26, 26, 26, 26], dtype=jnp.float_)

# If we start somewhere close to the solution then Optimistix is OK
# Parameters for the perturbation
mean = 10.0  # Mean of the perturbation (usually 0)
std_dev = 5.0  # Standard deviation of the perturbation
# Generate random perturbations
perturbation = np.random.normal(mean, std_dev, size=known_solution_array.shape)

# initial_solution: Array = known_solution_array  + perturbation
initial_solution = initial_solution_default
initial_solution = dimensional_to_scaled_base10(initial_solution)

# Reaction set is linearly independent (determined by Gaussian elimination in a previous step)
# log10 equilibrium constants
# 2 H2O = 2 H2 + 1 O2
reaction0_lnKp: float = -118.38862269303881
reaction0_delta_n: float = 1.0
# 4 H2 + 1 CO2 = 2 H2O + 1 CH4
reaction1_lnKp: float = 22.8840873584512
reaction1_delta_n: float = -2.0
# 1 H2 + 1 O2 = 1 H2O + 1 CO
reaction2_lnKp: float = -6.001590516742649
reaction2_delta_n: float = 0.0


def log10Kc_from_lnKp(lnKp: float, delta_n: float) -> float:
    # return (lnKp - delta_n * (np.log(GAS_CONSTANT_BAR) + np.log(temperature))) / np.log(10)
    log10Kc: float = lnKp - delta_n * (
        np.log(BOLTZMANN_CONSTANT_BAR) + log_AVOGADRO - log_scaling + np.log(temperature)
    )
    log10Kc = log10Kc / np.log(10)

    return log10Kc


reaction0_log10Kc: float = log10Kc_from_lnKp(reaction0_lnKp, reaction0_delta_n)
reaction1_log10Kc: float = log10Kc_from_lnKp(reaction1_lnKp, reaction1_delta_n)
reaction2_log10Kc: float = log10Kc_from_lnKp(reaction2_lnKp, reaction2_delta_n)

# Coefficient matrix (reaction stoichiometry)
# Columns correspond to species: H2, H2O, CO, CO2, CH4, O2
# Rows refer to reactions (three in total)
coefficient_matrix: Array = jnp.array(
    [
        [2.0, -2.0, 0.0, 1.0, 0.0, 0.0],
        [-4.0, 2.0, -1.0, 0.0, 1.0, 0.0],
        [-1.0, 1.0, -1.0, 0.0, 0.0, 1.0],
    ]
)

# rhs constraints are the equilibrium constants of the reaction
rhs: Array = jnp.array([reaction0_log10Kc, reaction1_log10Kc, reaction2_log10Kc])


# Register Pytree classes
@register_pytree_node_class
class SystemParams:
    def __init__(self, initial_solution):
        self.initial_solution = initial_solution

    def tree_flatten(self):
        children = (self.initial_solution,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class AdditionalParams:
    def __init__(self, coefficient_matrix, rhs, temperature):
        self.coefficient_matrix = coefficient_matrix
        self.rhs = rhs
        self.temperature = temperature

    def tree_flatten(self):
        children = (self.coefficient_matrix, self.rhs, self.temperature)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


system_params = SystemParams(initial_solution)
additional_params = AdditionalParams(coefficient_matrix, rhs, temperature)


# def solve_with_scipy(method="hybr", tol: float = 1.0e-8, jacobian: bool = True) -> None:
#     """Solve the system with Scipy"""

#     if jacobian:
#         jacobian_function: Callable | None = jax.jacobian(objective_function)
#     else:
#         jacobian_function = None

#     print("Solving with SciPy")
#     sol: OptimizeResult = root(
#         objective_function, initial_solution, method=method, jac=jacobian_function, tol=tol
#     )

#     # Shifted solution
#     solution: npt.NDArray[np.float_] = scaled_to_dimensional_base10(sol.x)

#     print(sol)

#     if sol.success and np.isclose(solution, known_solution_array).all():
#         print("SciPy success and agrees with known solution. Steps = %d" % sol["nfev"])

#     print(solution)


@jit
def solve_with_optimistix(system_params, additional_params) -> Array:
    # , method="Dogleg", tol: float = 1.0e-8) -> None:
    """Solve the system with Optimistix"""

    tol: float = 1.0e-8
    # if method == "Dogleg":
    solver = optx.Dogleg(atol=tol, rtol=tol)
    # elif method == "Newton":
    #    solver = optx.Newton(atol=tol, rtol=tol)
    # elif method == "LevenbergMarquardt":
    #    solver = optx.LevenbergMarquardt(atol=tol, rtol=tol)
    # else:
    #    raise ValueError(f"Unknown method: {method}")

    print("Solving with Optimistix")
    sol = optx.root_find(
        objective_function,
        solver,
        system_params.initial_solution,
        args=(additional_params),
        throw=True,
    )

    # Shifted solution
    solution: Array = scaled_to_dimensional_base10(sol.value)
    # print(solution)

    if optx.RESULTS[sol.result] == "" and np.isclose(solution, known_solution_array).all():
        print(
            "Optimistix success and agrees with known solution. Steps = %d"
            % sol.stats["num_steps"]
        )

    return solution


def atmosphere_log10_molar_mass(solution: Array) -> Array:
    """Log10 of the molar mass of the atmosphere"""
    molar_masses: Array = jnp.array([value for value in molar_masses_dict.values()])
    molar_mass: Array = logsumexp_base10(solution, molar_masses) - logsumexp_base10(solution)

    return molar_mass


def atmosphere_log10_volume(solution: Array) -> Array:
    """Log10 of the volume of the atmosphere"""
    return (
        jnp.log10(gas_constant)
        + jnp.log10(temperature)
        # Units of solution don't matter because it just weights (unless numerical problems)
        - atmosphere_log10_molar_mass(solution)
        + jnp.log10(planet_surface_area)
        - jnp.log10(planet_surface_gravity)
    )


@jit
def objective_function(solution: Array, additional_params) -> Array:
    """Residual of the reaction network and mass balance"""
    # Extract parameters from the pytree
    coefficient_matrix = additional_params.coefficient_matrix
    rhs = additional_params.rhs
    temperature = additional_params.temperature

    # Reaction network
    reaction_residual: Array = coefficient_matrix.dot(solution) - rhs

    log10_volume: Array = atmosphere_log10_volume(solution)

    # Mass balance residuals (stoichiometry coefficients are hard-coded for this MWE)
    oxygen_residual: Array = jnp.array(
        [
            solution[1],
            jnp.log10(2) + solution[2],
            jnp.log10(2) + solution[3],
            solution[5],
        ]
    )
    oxygen_residual = logsumexp_base10(oxygen_residual) - (log10_oxygen_constraint - log10_volume)

    hydrogen_residual: Array = jnp.array(
        [jnp.log10(2) + solution[0], jnp.log10(2) + solution[1], jnp.log10(4) + solution[4]]
    )
    hydrogen_residual = logsumexp_base10(hydrogen_residual) - (
        log10_hydrogen_constraint - log10_volume
    )

    carbon_residual: Array = jnp.array([solution[2], solution[4], solution[5]])
    carbon_residual = logsumexp_base10(carbon_residual) - (log10_carbon_constraint - log10_volume)

    residual: Array = jnp.concatenate(
        (
            reaction_residual,
            jnp.array([oxygen_residual]),
            jnp.array([hydrogen_residual]),
            jnp.array([carbon_residual]),
        )
    )

    return residual


@jit
def logsumexp_base10(log_values: Array, prefactors: ArrayLike = 1.0) -> Array:
    """Computes the log-sum-exp using base-10 exponentials in a numerically stable way.

    Args:
        log10_values: Array of log10 values to sum
        prefactors: Array of prefactors corresponding to each log10 value

    Returns:
        The log10 of the sum of prefactors multiplied by exponentials of the input values.
    """
    max_log: Array = jnp.max(log_values)
    prefactors_: Array = jnp.asarray(prefactors)

    value_sum: Array = jnp.sum(prefactors_ * jnp.power(10, log_values - max_log))

    return max_log + safe_log10(value_sum)


@jit
def safe_log10(x: ArrayLike) -> Array:
    """Computes log10 of x, safely adding machine epsilon to avoid log of zero."""

    return jnp.log10(x)  #  + MACHEPS)


def main():
    # Define the parameters
    system_params = SystemParams(initial_solution)
    additional_params = AdditionalParams(coefficient_matrix, rhs, temperature)

    # Solve using scipy and a JAX-provided Jacobian
    # solve_with_scipy(system_params, additional_params, jacobian=True)

    # Solve using Optimistix with Dogleg method
    out = solve_with_optimistix(system_params, additional_params)  # , method="Dogleg")

    print(out)

    # You can uncomment the following to test other methods
    # solve_with_optimistix(system_params, additional_params, method="LevenbergMarquardt")
    # solve_with_optimistix(system_params, additional_params, method="Newton")


if __name__ == "__main__":
    main()
