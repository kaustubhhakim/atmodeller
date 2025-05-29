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
"""JAX-related functionality for solving the system of equations"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array, lax
from jax.scipy.special import logsumexp
from jax.typing import ArrayLike

from atmodeller.constants import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.containers import (
    FixedParameters,
    FugacityConstraints,
    MassConstraints,
    Planet,
    SolverParameters,
    SpeciesCollection,
    TracedParameters,
)
from atmodeller.utilities import (
    get_log_number_density_from_log_pressure,
    safe_exp,
    to_hashable,
    unit_conversion,
)


@eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def solve(
    solution_array: Array,
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    solver_parameters: SolverParameters,
    options: dict[str, Any],
) -> tuple[Array, Array, Array]:
    """Solves the system of non-linear equations

    Args:
        solution_array: Solution array with species removed that have fugacity constraints
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        solver_parameters: Solver parameters
        options: Options for root find

    Returns:
        The solution array, the status of the solver
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        solver_parameters.solver_instance,
        solution_array,
        args={
            "traced_parameters": traced_parameters,
            "fixed_parameters": fixed_parameters,
            "solver_parameters": solver_parameters,
        },
        throw=solver_parameters.throw,
        max_steps=solver_parameters.max_steps,
        options=options,
    )

    # jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])
    solver_steps: Array = sol.stats["num_steps"]
    solver_status: Array = sol.result == optx.RESULTS.successful

    return sol.value, solver_status, solver_steps


@eqx.filter_jit
def repeat_solver(
    solver_fn: Callable,
    base_initial_solution: Array,
    initial_solution: Array,
    initial_status: Array,
    initial_steps: Array,
    traced_parameters: TracedParameters,
    multistart_perturbation: float,
    max_attempts: int,
    key: Array,
) -> tuple[int, Array, Array, Array]:
    """Repeat solver until a solution is found

    Args:
        solver_fn: Solver function with pre-bound fixed configuration
        base_initial_solution: Base initial solution to perturb if necessary
        initial_solution: Initial solution after first solve
        initial_status: Initial status after first solve
        initial_steps: Initial steps after first solve
        traced_parameters: Traced parameters
        multistart_perturbation: Multistart perturbation
        max_attempts: Maximum attempts
        key: Random key
    """

    def body_fn(state: tuple) -> tuple[int, Array, Array, Array, Array, Array]:
        """Perform one iteration of the solver retry loop

        Args:
            state: Tuple containing:
                i: Current attempt index
                key: PRNG key for random number generation
                solution: Current solution array
                status: Boolean array indicating successful solutions
                steps: Step count or similar solver output
                base_initial_solution: Unperturbed base initial solution

        Returns:
            Updated state tuple with incremented attempt index, potentially perturbed
            solutions, updated status, updated steps, and unmodified base initial solution
        """
        i, key, solution, status, _, base_initial_solution = state

        failed_mask: Array = ~status
        key, subkey = jax.random.split(key)

        # Implements a simple perturbation of the base initial solution, but something more
        # sophisticated could be implemented, such as training a network or using a regressor
        # to inform the next guess of the models that failed from the ones that succeeded.
        perturb_shape: tuple[int, int] = (solution.shape[0], solution.shape[1])
        raw_perturb: Array = jax.random.uniform(
            subkey, shape=perturb_shape, minval=-1.0, maxval=1.0
        )
        perturbations: Array = jnp.where(
            failed_mask[:, None],
            multistart_perturbation * raw_perturb,
            jnp.zeros_like(solution),
        )
        new_initial_solution = jnp.where(
            failed_mask[:, None], base_initial_solution + perturbations, solution
        )

        new_solution, new_status, new_steps = solver_fn(new_initial_solution, traced_parameters)

        return (i + 1, key, new_solution, new_status, new_steps, base_initial_solution)

    def cond_fn(state: tuple) -> Array:
        """Check if the solver should continue retrying

        Args:
            state: Tuple containing:
                i: Current attempt index
                _: Unused (PRNG key)
                _: Unused (solution)
                status: Boolean array indicating success of each solution
                _: Unused (steps)
                _: Unused (base initial solution)

        Returns:
            A boolean array indicating whether retries should continue (True if
            any solution failed and attempts are still available)
        """
        i, _, _, status, _, _ = state

        return jnp.logical_and(i < max_attempts, jnp.any(~status))

    initial_state: tuple = (
        1,  # A first solve has already been attempted before repeat_solver is called
        key,
        initial_solution,
        initial_status,
        initial_steps,
        base_initial_solution,
    )
    final_i, _, final_solution, final_status, final_steps, _ = lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    return final_i, final_solution, final_status, final_steps


@eqx.filter_jit
def get_min_log_elemental_abundance_per_species(
    formula_matrix: Array, mass_constraints: MassConstraints
) -> Array:
    """For each species, find the elemental mass constraint with the lowest abundance.

    If species are present for which there are no elemental mass constraints (for example,
    oxygen if a fugacity constraint is applied instead) the maximum of all elements will be
    returned for that particular species. However, the return array from this function is
    subsequently filtered depending on whether a stability calculation is required for each
    species.

    Args:
        formula_matrix: Formula matrix
        mass_constraints: Mass constraints

    Returns:
        A vector of the minimum log elemental abundance for each species
    """
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    # (n_elements, n_species)
    mask: Array = (formula_matrix != 0).astype(jnp.int_)
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    # Align for element-wise multiplication: (n_elements, 1)
    log_abundance: Array = mass_constraints.log_abundance.T
    # jax.debug.print("log_abundance = {out}", out=log_abundance)

    # Element-wise multiplication (broadcasted correctly): (n_elements, n_species)
    masked_abundance: Array = mask * log_abundance
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)
    masked_abundance = jnp.where(mask != 0, masked_abundance, jnp.nan)
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Find the minimum log abundance per species
    min_abundance_per_species: Array = jnp.nanmin(masked_abundance, axis=0)
    # jax.debug.print("min_abundance_per_species = {out}", out=min_abundance_per_species)

    return min_abundance_per_species


@eqx.filter_jit
def objective_function(solution: Array, kwargs: dict) -> Array:
    """Objective function

    Args:
        solution: Solution array for all species i.e. log number density and log stability
        kwargs: Dictionary of pytrees required to compute the residual

    Returns:
        Residual
    """
    # jax.debug.print("Starting new objective_function evaluation")
    traced_parameters: TracedParameters = kwargs["traced_parameters"]
    fixed_parameters: FixedParameters = kwargs["fixed_parameters"]
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.temperature

    mass_constraints: MassConstraints = traced_parameters.mass_constraints
    fugacity_constraints: FugacityConstraints = traced_parameters.fugacity_constraints
    gas_species_mask: Array = fixed_parameters.gas_species_mask
    # jax.debug.print("gas_species_mask = {out}", out=gas_species_mask)
    stability_species_mask: Array = fixed_parameters.stability_species_mask
    # jax.debug.print("stability_species_mask = {out}", out=stability_species_mask)

    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    reaction_stability_matrix: Array = jnp.array(fixed_parameters.reaction_stability_matrix)
    # jax.debug.print("reaction_stability_matrix = {out}", out=reaction_stability_matrix)
    formula_matrix: Array = jnp.array(fixed_parameters.formula_matrix)

    log_number_density, log_stability = jnp.split(solution, 2)
    # jax.debug.print("log_number_density = {out}", out=log_number_density)
    # jax.debug.print("log_stability = {out}", out=log_stability)

    log_activity: Array = get_log_activity(traced_parameters, fixed_parameters, log_number_density)
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # Atmosphere
    total_pressure: Array = get_total_pressure(fixed_parameters, log_number_density, temperature)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)
    log_volume: Array = get_atmosphere_log_volume(fixed_parameters, log_number_density, planet)
    # jax.debug.print("log_volume = {out}", out=log_volume)

    # Based on the definition of the reaction constant we need to convert gas activities
    # (fugacities) from bar to effective number density, whilst keeping condensate activities
    # unmodified.
    log_activity_number_density: Array = get_log_number_density_from_log_pressure(
        log_activity, temperature
    )
    log_activity_number_density = jnp.where(
        gas_species_mask, log_activity_number_density, log_activity
    )
    # jax.debug.print("log_activity_number_density = {out}", out=log_activity_number_density)

    residual: Array = jnp.array([])

    # Reaction network residual
    # TODO: Is it possible to remove this if statement?
    if reaction_matrix.size > 0:
        log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
            fixed_parameters, temperature
        )
        reaction_residual: Array = (
            reaction_matrix.dot(log_activity_number_density) - log_reaction_equilibrium_constant
        )
        # jax.debug.print("reaction_residual before stability = {out}", out=reaction_residual)
        reaction_residual = reaction_residual - reaction_stability_matrix.dot(
            safe_exp(log_stability)
        )
        # jax.debug.print("reaction_residual after stability = {out}", out=reaction_residual)
        residual = jnp.concatenate([residual, reaction_residual])

    # Fugacity constraints residual
    fugacity_residual = log_activity_number_density - fugacity_constraints.log_number_density(
        temperature, total_pressure
    )
    # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
    residual = jnp.concatenate([residual, fugacity_residual])

    # Elemental mass balance residual
    # Number density of elements in the gas or condensed phase
    element_density: Array = get_element_density(formula_matrix, log_number_density)
    # jax.debug.print("element_density = {out}", out=element_density)
    element_melt_density: Array = get_element_density_in_melt(
        traced_parameters,
        fixed_parameters,
        formula_matrix,
        log_number_density,
        log_activity,
        log_volume,
    )
    # jax.debug.print("element_melt_density = {out}", out=element_melt_density)

    # Relative mass error, computed in log-space for numerical stability
    element_density_total: Array = element_density + element_melt_density
    log_element_density_total: Array = jnp.log(element_density_total)
    # jax.debug.print("log_number_density_total = {out}", out=log_element_density_total)
    # Flattening since if only one set of abundances specified (no vmap) then the array will be
    # 2-D with only one row. Otherwise with vmap the abundances will be a 1-D array.
    log_target_density: Array = mass_constraints.log_number_density(log_volume).flatten()
    # jax.debug.print("log_target_density = {out}", out=log_target_density)
    mass_residual: Array = safe_exp(log_element_density_total - log_target_density) - 1
    # jax.debug.print("mass_residual = {out}", out=mass_residual)
    residual = jnp.concatenate([residual, mass_residual])

    # Stability residual
    log_min_number_density: Array = (
        get_min_log_elemental_abundance_per_species(formula_matrix, mass_constraints)
        - log_volume
        + jnp.log(fixed_parameters.tau)
    )
    # jax.debug.print("log_min_number_density = {out}", out=log_min_number_density)
    stability_residual: Array = log_number_density + log_stability - log_min_number_density
    # jax.debug.print("stability_residual = {out}", out=stability_residual)
    # Mask stability residuals for species that do not require stability calculations
    stability_residual = stability_residual * stability_species_mask
    # jax.debug.print("stability_residual = {out}", out=stability_residual)
    residual = jnp.concatenate([residual, stability_residual])
    # jax.debug.print("residual (with nans) = {out}", out=residual)
    # nans denote unused conditions
    residual = jnp.where(jnp.isnan(residual), 0.0, residual)
    # jax.debug.print("residual = {out}", out=residual)

    return residual


@eqx.filter_jit
def get_atmosphere_log_molar_mass(
    fixed_parameters: FixedParameters, log_number_density: Array
) -> Array:
    """Gets log molar mass of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log molar mass of the atmosphere
    """
    gas_log_number_density: Array = get_gas_species_data(fixed_parameters, log_number_density)
    gas_molar_mass: Array = get_gas_species_data(
        fixed_parameters, jnp.array(fixed_parameters.molar_masses)
    )
    molar_mass: Array = logsumexp(gas_log_number_density, b=gas_molar_mass) - logsumexp(
        gas_log_number_density, b=fixed_parameters.gas_species_mask
    )
    # jax.debug.print("molar_mass = {out}", out=molar_mass)

    return molar_mass


@eqx.filter_jit
def get_atmosphere_log_volume(
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    planet: Planet,
) -> Array:
    """Gets log volume of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """
    log_volume: Array = (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.temperature)
        - get_atmosphere_log_molar_mass(fixed_parameters, log_number_density)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )

    return log_volume


@eqx.filter_jit
def get_total_pressure(
    fixed_parameters: FixedParameters, log_number_density: Array, temperature: ArrayLike
) -> Array:
    """Gets total pressure

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Total pressure
    """
    gas_species_mask: Array = fixed_parameters.gas_species_mask
    pressure: Array = get_pressure_from_log_number_density(log_number_density, temperature)
    gas_pressure: Array = pressure * gas_species_mask
    # jax.debug.print("gas_pressure = {out}", out=gas_pressure)

    return jnp.sum(gas_pressure)


@eqx.filter_jit
def get_element_density(formula_matrix: Array, log_number_density: Array) -> Array:
    """Number density of elements in the gas or condensed phase

    Args:
        formula_matrix: Formula matrix
        log_number_density: Log number density

    Returns:
        Number density of elements in the gas or condensed phase
    """
    element_density: Array = formula_matrix.dot(safe_exp(log_number_density))

    return element_density


@eqx.filter_jit
def get_element_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    formula_matrix: Array,
    log_number_density: Array,
    log_activity: Array,
    log_volume: Array,
) -> Array:
    """Gets the number density of elements dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        formula_matrix: Formula matrix
        log_number_density: Log number density
        log_activity: Log activity
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of elements dissolved in melt
    """
    species_melt_density: Array = get_species_density_in_melt(
        traced_parameters, fixed_parameters, log_number_density, log_activity, log_volume
    )
    element_melt_density: Array = formula_matrix.dot(species_melt_density)

    return element_melt_density


@eqx.filter_jit
def get_gas_species_data(fixed_parameters: FixedParameters, some_array: ArrayLike) -> Array:
    """Masks the gas species data from an array

    Args:
        fixed_parameters: Fixed parameters
        some_array: Some array to mask the gas species data from

    Returns:
        An array with gas species data from `some_array` and condensate entries zeroed
    """
    gas_data: Array = fixed_parameters.gas_species_mask * some_array

    return gas_data


@eqx.filter_jit
def get_log_activity(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
) -> Array:
    """Gets the log activity

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log activity
    """
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.temperature
    species: SpeciesCollection = fixed_parameters.species
    total_pressure: Array = get_total_pressure(fixed_parameters, log_number_density, temperature)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)

    activity_funcs: list[Callable] = [
        to_hashable(species_.activity.log_activity) for species_ in species
    ]

    def apply_activity_function(index: ArrayLike) -> Array:
        return lax.switch(
            index,
            activity_funcs,
            temperature,
            total_pressure,
        )

    vmap_apply_function: Callable = eqx.filter_vmap(apply_activity_function, in_axes=(0,))
    indices: Array = jnp.arange(len(species))
    log_activity_pure_species: Array = vmap_apply_function(indices)
    # jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)
    log_activity: Array = get_log_activity_ideal_mixing(
        fixed_parameters, log_number_density, log_activity_pure_species
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    return log_activity


@eqx.filter_jit
def get_log_activity_ideal_mixing(
    fixed_parameters: FixedParameters, log_number_density: Array, log_activity_pure_species: Array
) -> Array:
    """Gets the log activity of species in the atmosphere assuming an ideal mixture

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity_pure_species: Log activity of the pure species

    Returns:
        Log activity of the species assuming ideal mixing in the atmosphere
    """
    gas_species_mask: Array = fixed_parameters.gas_species_mask
    number_density: Array = safe_exp(log_number_density)
    gas_species_number_density: Array = gas_species_mask * number_density
    atmosphere_log_number_density: Array = jnp.log(jnp.sum(gas_species_number_density))

    log_activity_gas_species: Array = (
        log_activity_pure_species + log_number_density - atmosphere_log_number_density
    )
    log_activity: Array = jnp.where(
        gas_species_mask, log_activity_gas_species, log_activity_pure_species
    )

    return log_activity


@eqx.filter_jit
def get_log_pressure_from_log_number_density(
    log_number_density: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets log pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Log pressure
    """
    log_pressure: Array = (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature) + log_number_density
    )

    return log_pressure


@eqx.filter_jit
def get_log_Kp(
    species: SpeciesCollection, reaction_matrix: Array, temperature: ArrayLike
) -> Array:
    """Gets log of the equilibrium constant in terms of partial pressures

    Args:
        species: Species
        reaction_matrix: Reaction matrix
        temperature: Temperature

    Returns:
        Log of the equilibrium constant in terms of partial pressures
    """
    gibbs_list: list[ArrayLike] = []
    for species_ in species:
        gibbs: ArrayLike = species_.data.get_gibbs_over_RT(temperature)
        gibbs_list.append(gibbs)

    gibbs_jnp: Array = jnp.array(gibbs_list)
    log_Kp: Array = -1.0 * reaction_matrix.dot(gibbs_jnp)

    return log_Kp


@eqx.filter_jit
def get_log_reaction_equilibrium_constant(
    fixed_parameters: FixedParameters,
    temperature: ArrayLike,
) -> Array:
    """Gets the log equilibrium constant of the reactions

    Args:
        fixed_parameters: Fixed parameters
        temperature: Temperature

    Returns:
        Log equilibrium constant of the reactions
    """
    species: SpeciesCollection = fixed_parameters.species
    reaction_matrix: Array = jnp.array(fixed_parameters.reaction_matrix)
    log_Kp: Array = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)
    delta_n: Array = jnp.sum(reaction_matrix * fixed_parameters.gas_species_mask, axis=1)
    # jax.debug.print("delta_n = {out}", out=delta_n)
    log_Kc: Array = log_Kp - delta_n * (jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature))
    # jax.debug.print("log10Kc = {out}", out=log_Kc)

    return log_Kc


@eqx.filter_jit
def get_pressure_from_log_number_density(
    log_number_density: ArrayLike, temperature: ArrayLike
) -> Array:
    """Gets pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Pressure
    """
    return safe_exp(get_log_pressure_from_log_number_density(log_number_density, temperature))


@eqx.filter_jit
def get_species_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    log_activity: Array,
    log_volume: Array,
) -> Array:
    """Gets the number density of species dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity: Log activity
        log_volume: Log volume of the atmosphere

    Returns:
        Number density of species dissolved in melt
    """
    molar_masses: Array = jnp.array(fixed_parameters.molar_masses)
    planet: Planet = traced_parameters.planet

    ppmw: Array = get_species_ppmw_in_melt(
        traced_parameters, fixed_parameters, log_number_density, log_activity
    )

    species_melt_density: Array = (
        ppmw
        * unit_conversion.ppm_to_fraction
        * AVOGADRO
        * planet.melt_mass
        / (molar_masses * safe_exp(log_volume))
    )
    # jax.debug.print("species_melt_density = {out}", out=species_melt_density)

    return species_melt_density


@eqx.filter_jit
def get_species_ppmw_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Array,
    log_activity: Array,
) -> Array:
    """Gets the ppmw of species dissolved in melt due to species solubility

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity: Log activity

    Returns:
        ppmw of species dissolved in melt
    """
    species: SpeciesCollection = fixed_parameters.species
    diatomic_oxygen_index: Array = jnp.array(fixed_parameters.diatomic_oxygen_index)
    planet: Planet = traced_parameters.planet
    temperature: ArrayLike = planet.temperature

    fugacity: Array = safe_exp(log_activity)
    total_pressure: Array = get_total_pressure(fixed_parameters, log_number_density, temperature)
    diatomic_oxygen_fugacity: Array = jnp.take(fugacity, diatomic_oxygen_index)

    # NOTE: All solubility formulations must return a JAX array to allow vmap
    solubility_funcs: list[Callable] = [
        to_hashable(species_.solubility.jax_concentration) for species_ in species
    ]

    def apply_solubility_function(index: ArrayLike, fugacity: ArrayLike):
        return lax.switch(
            index,
            solubility_funcs,
            fugacity,
            temperature,
            total_pressure,
            diatomic_oxygen_fugacity,
        )

    vmap_apply_function: Callable = eqx.filter_vmap(apply_solubility_function, in_axes=(0, 0))
    indices: ArrayLike = jnp.arange(len(species))
    species_ppmw: Array = vmap_apply_function(indices, fugacity)
    # jax.debug.print("ppmw = {out}", out=ppmw)

    return species_ppmw
