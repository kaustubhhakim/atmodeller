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

from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import lax, random
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Bool, Float, Integer, PRNGKeyArray, Shaped

from atmodeller.constants import AVOGADRO, BOLTZMANN_CONSTANT_BAR, GAS_CONSTANT
from atmodeller.containers import (
    FixedParameters,
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


# Since this is the core driver function for the solve it remains useful for debugging to see how
# many times recompilation is triggered
# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def solve(
    solution_array: Float[Array, " sol_dim"],
    active_indices: Integer[Array, " res_dim"],
    tau: Float[Array, ""],
    traced_parameters: TracedParameters,
    solver_parameters: SolverParameters,
    fixed_parameters: FixedParameters,
    options: dict[str, Any],
) -> tuple[Float[Array, " sol_dim"], Bool[Array, ""], Integer[Array, ""]]:
    """Solves the system of non-linear equations

    Args:
        solution_array: Solution array
        active_indices: Indices of the residual array that are active
        tau: Tau parameter for species' stability
        traced_parameters: Traced parameters
        solver_parameters: Solver parameters
        fixed_parameters: Fixed parameters
        options: Options for root find

    Returns:
        The solution array, the status of the solver, number of steps
    """
    sol: optx.Solution = optx.root_find(
        objective_function,
        solver_parameters.get_solver_instance(),
        solution_array,
        args={
            "traced_parameters": traced_parameters,
            "active_indices": active_indices,
            "tau": tau,
            "fixed_parameters": fixed_parameters,
            "solver_parameters": solver_parameters,
        },
        throw=solver_parameters.throw,
        max_steps=solver_parameters.max_steps,
        options=options,
    )

    # jax.debug.print("Optimistix success. Number of steps = {out}", out=sol.stats["num_steps"])
    solver_steps: Integer[Array, ""] = sol.stats["num_steps"]

    # TODO: sol.results contains more information about the solution process, but it's wrapped up
    # in an enum-like object
    solver_status: Bool[Array, ""] = sol.result == optx.RESULTS.successful

    return sol.value, solver_status, solver_steps


def get_min_log_elemental_abundance_per_species(
    formula_matrix: Integer[Array, "el_dim species_dim"], mass_constraints: MassConstraints
) -> Float[Array, " species_dim"]:
    """For each species, find the elemental mass constraint with the lowest abundance.

    Args:
        formula_matrix: Formula matrix
        mass_constraints: Mass constraints

    Returns:
        A vector of the minimum log elemental abundance for each species
    """
    # Create the binary mask where formula_matrix != 0 (1 where element is present in species)
    mask: Integer[Array, "el_dim species_dim"] = (formula_matrix != 0).astype(jnp.int_)
    # jax.debug.print("formula_matrix = {out}", out=formula_matrix)
    # jax.debug.print("mask = {out}", out=mask)

    # log_abundance is a 1-D array, which cannot be transposed, so make a 2-D array
    log_abundance: Float[Array, "el_dim 1"] = jnp.atleast_2d(mass_constraints.log_abundance).T
    # jax.debug.print("log_abundance = {out}", out=log_abundance)

    # Element-wise multiplication with broadcasting
    masked_abundance: Float[Array, "el_dim species_dim"] = mask * log_abundance
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)
    masked_abundance = jnp.where(mask != 0, masked_abundance, jnp.nan)
    # jax.debug.print("masked_abundance = {out}", out=masked_abundance)

    # Find the minimum log abundance per species
    min_abundance_per_species: Float[Array, " species_dim"] = jnp.nanmin(masked_abundance, axis=0)
    # jax.debug.print("min_abundance_per_species = {out}", out=min_abundance_per_species)

    return min_abundance_per_species


def objective_function(
    solution: Float[Array, " sol_dim"], kwargs: dict
) -> Float[Array, " res_dim"]:
    """Objective function

    The order of the residual does make a difference to the solution process. More investigations
    are necessary, but justification for the current ordering is as follows:

        1. Fugacity constraints - fixed target, well conditioned
        2. Reaction constraints - log-linear, physics-based coupling
        3. Mass balance constraints - stiffer, depends on solubility
        4. Stability constraints - stiffer still

    Args:
        solution: Solution array for all species i.e. log number density and log stability
        kwargs: Dictionary of pytrees required to compute the residual

    Returns:
        Residual
    """
    # jax.debug.print("Starting new objective_function evaluation")
    tp: TracedParameters = kwargs["traced_parameters"]
    active_indices: Integer[Array, " res_dim"] = kwargs["active_indices"]
    fp: FixedParameters = kwargs["fixed_parameters"]
    tau: Float[Array, ""] = kwargs["tau"]
    planet: Planet = tp.planet
    temperature: Float[Array, ""] = planet.temperature

    log_number_density, log_stability = jnp.split(solution, 2)
    # jax.debug.print("log_number_density = {out}", out=log_number_density)
    # jax.debug.print("log_stability = {out}", out=log_stability)

    log_activity: Float[Array, " species_dim"] = get_log_activity(tp, fp, log_number_density)
    # jax.debug.print("log_activity = {out}", out=log_activity)

    # Atmosphere
    total_pressure: Float[Array, ""] = get_total_pressure(fp, log_number_density, temperature)
    # jax.debug.print("total_pressure = {out}", out=total_pressure)
    log_volume: Float[Array, ""] = get_atmosphere_log_volume(fp, log_number_density, planet)
    # jax.debug.print("log_volume = {out}", out=log_volume)

    # Based on the definition of the reaction constant we need to convert gas activities
    # (fugacities) from bar to effective number density, whilst keeping condensate activities
    # unmodified.
    log_activity_number_density: Float[Array, " species_dim"] = (
        get_log_number_density_from_log_pressure(log_activity, temperature)
    )
    log_activity_number_density = jnp.where(
        fp.gas_species_mask, log_activity_number_density, log_activity
    )
    # jax.debug.print("log_activity_number_density = {out}", out=log_activity_number_density)

    # Here would be where fugacity constraints could be imposed as hard constraints. Although this
    # would reduce the degrees of freedom, previous preliminary testing identified two challenges:
    #   1. The solver performance appears to degrade rather than improve. This could be because
    #       soft constraints are better behaved with gradient-based solution approaches(?)
    #   2. Imposing fugacity/activity would require back-computing pressure/number density, which
    #       would involve solving non-linear real gas EOS, potentially increasing the solve
    #       complexity and time.

    # Fugacity constraints residual (dimensionless, log-ratio of number densities)
    fugacity_residual = log_activity_number_density - tp.fugacity_constraints.log_number_density(
        temperature, total_pressure
    )
    # jax.debug.print("fugacity_residual = {out}", out=fugacity_residual)
    # jax.debug.print(
    #     "fugacity_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(fugacity_residual),
    #     out2=jnp.nanmax(fugacity_residual),
    # )
    # jax.debug.print(
    #     "fugacity_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(fugacity_residual),
    #     out2=jnp.nanstd(fugacity_residual),
    # )

    # Reaction network residual
    # TODO: Is it possible to remove this if statement?
    if fp.reaction_matrix.size > 0:
        log_reaction_equilibrium_constant: Array = get_log_reaction_equilibrium_constant(
            fp, temperature
        )
        # jax.debug.print(
        #     "log_reaction_equilibrium_constant = {out}", out=log_reaction_equilibrium_constant
        # )
        reaction_residual: Array = (
            fp.reaction_matrix.dot(log_activity_number_density) - log_reaction_equilibrium_constant
        )
        # jax.debug.print("reaction_residual before stability = {out}", out=reaction_residual)
        reaction_stability_mask: Array = jnp.broadcast_to(
            fp.active_stability(), fp.reaction_matrix.shape
        )
        reaction_stability_matrix: Array = fp.reaction_matrix * reaction_stability_mask
        # jax.debug.print("reaction_stability_matrix = {out}", out=reaction_stability_matrix)

        # Dimensionless (log K residual)
        reaction_residual = reaction_residual - reaction_stability_matrix.dot(
            safe_exp(log_stability)
        )
        # jax.debug.print("reaction_residual after stability = {out}", out=reaction_residual)
        # jax.debug.print(
        #     "reaction_residual min/max: {out}/{out2}",
        #     out=jnp.nanmin(reaction_residual),
        #     out2=jnp.nanmax(reaction_residual),
        # )
        # jax.debug.print(
        #     "reaction_residual mean/std: {out}/{out2}",
        #     out=jnp.nanmean(reaction_residual),
        #     out2=jnp.nanstd(reaction_residual),
        # )

    else:
        reaction_residual = jnp.atleast_1d(jnp.array([]))

    # Elemental mass balance residual
    # Number density of elements in the gas or condensed phase
    element_density: Float[Array, " el_dim"] = get_element_density(
        fp.formula_matrix, log_number_density
    )
    # jax.debug.print("element_density = {out}", out=element_density)
    element_melt_density: Float[Array, " el_dim"] = get_element_density_in_melt(
        tp, fp, fp.formula_matrix, log_number_density, log_activity, log_volume
    )
    # jax.debug.print("element_melt_density = {out}", out=element_melt_density)

    # Relative mass error, computed in log-space for numerical stability
    element_density_total: Float[Array, " el_dim"] = element_density + element_melt_density
    log_element_density_total: Float[Array, " el_dim"] = jnp.log(element_density_total)
    # jax.debug.print("log_element_density_total = {out}", out=log_element_density_total)
    log_target_density: Float[Array, " el_dim"] = tp.mass_constraints.log_number_density(
        log_volume
    )
    # jax.debug.print("log_target_density = {out}", out=log_target_density)

    # Dimensionless (ratio error - 1)
    mass_residual: Float[Array, " el_dim"] = (
        safe_exp(log_element_density_total - log_target_density) - 1
    )
    # Log-space residual can perform better when close to the solution
    # mass_residual = log_element_density_total - log_target_density
    # jax.debug.print("mass_residual = {out}", out=mass_residual)
    # jax.debug.print(
    #     "mass_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(mass_residual),
    #     out2=jnp.nanmax(mass_residual),
    # )
    # jax.debug.print(
    #     "mass_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(mass_residual),
    #     out2=jnp.nanstd(mass_residual),
    # )

    # Stability residual
    log_min_number_density: Float[Array, " species_dim"] = (
        get_min_log_elemental_abundance_per_species(fp.formula_matrix, tp.mass_constraints)
        - log_volume
        + jnp.log(tau)
    )
    # jax.debug.print("log_min_number_density = {out}", out=log_min_number_density)
    # Dimensionless (log-ratio)
    stability_residual: Float[Array, " species_dim"] = (
        log_number_density + log_stability - log_min_number_density
    )
    # jax.debug.print("stability_residual = {out}", out=stability_residual)
    # jax.debug.print(
    #     "stability_residual min/max: {out}/{out2}",
    #     out=jnp.nanmin(stability_residual),
    #     out2=jnp.nanmax(stability_residual),
    # )
    # jax.debug.print(
    #     "stability_residual mean/std: {out}/{out2}",
    #     out=jnp.nanmean(stability_residual),
    #     out2=jnp.nanstd(stability_residual),
    # )

    # NOTE: Order must be compatible with active_indices
    residual = jnp.concatenate(
        [fugacity_residual, reaction_residual, mass_residual, stability_residual]
    )
    # jax.debug.print("residual (with nans) = {out}", out=residual)

    residual = jnp.take(residual, indices=active_indices)  # type: ignore
    # jax.debug.print("residual = {out}", out=residual)

    return residual


def get_atmosphere_log_molar_mass(
    fixed_parameters: FixedParameters, log_number_density: Float[Array, " species_dim"]
) -> Float[Array, ""]:
    """Gets log molar mass of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log molar mass of the atmosphere
    """
    gas_log_number_density: Float[Array, " species_dim"] = get_gas_species_data(
        fixed_parameters, log_number_density
    )
    gas_molar_mass: Float[Array, " species_dim"] = get_gas_species_data(
        fixed_parameters, jnp.array(fixed_parameters.molar_masses)
    )
    molar_mass: Float[Array, ""] = logsumexp(gas_log_number_density, b=gas_molar_mass) - logsumexp(
        gas_log_number_density, b=fixed_parameters.gas_species_mask
    )
    # jax.debug.print("molar_mass = {out}", out=molar_mass)

    return molar_mass


def get_atmosphere_log_volume(
    fixed_parameters: FixedParameters,
    log_number_density: Float[Array, " species_dim"],
    planet: Planet,
) -> Float[Array, ""]:
    """Gets log volume of the atmosphere

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        planet: Planet

    Returns:
        Log volume of the atmosphere
    """
    log_volume: Float[Array, ""] = (
        jnp.log(GAS_CONSTANT)
        + jnp.log(planet.temperature)
        - get_atmosphere_log_molar_mass(fixed_parameters, log_number_density)
        + jnp.log(planet.surface_area)
        - jnp.log(planet.surface_gravity)
    )

    return log_volume


def get_total_pressure(
    fixed_parameters: FixedParameters,
    log_number_density: Float[Array, " species_dim"],
    temperature: Float[Array, ""],
) -> Float[Array, ""]:
    """Gets total pressure

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Total pressure
    """
    gas_species_mask: Bool[Array, " species_dim"] = fixed_parameters.gas_species_mask
    pressure: Float[Array, " species_dim"] = get_pressure_from_log_number_density(
        log_number_density, temperature
    )
    gas_pressure: Float[Array, " species_dim"] = pressure * gas_species_mask
    # jax.debug.print("gas_pressure = {out}", out=gas_pressure)

    return jnp.sum(gas_pressure)


def get_element_density(
    formula_matrix: Integer[Array, "el_dim species_dim"],
    log_number_density: Float[Array, " species_dim"],
) -> Array:
    """Number density of elements in the gas or condensed phase

    Args:
        formula_matrix: Formula matrix
        log_number_density: Log number density

    Returns:
        Number density of elements in the gas or condensed phase
    """
    element_density: Float[Array, " el_dim"] = formula_matrix @ safe_exp(log_number_density)

    return element_density


def get_element_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    formula_matrix: Integer[Array, "el_dim species_dim"],
    log_number_density: Float[Array, " species_dim"],
    log_activity: Float[Array, " species_dim"],
    log_volume: Float[Array, ""],
) -> Float[Array, " species_dim"]:
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
    species_melt_density: Float[Array, " species_dim"] = get_species_density_in_melt(
        traced_parameters,
        fixed_parameters,
        log_number_density,
        log_activity,
        log_volume,
    )
    element_melt_density: Float[Array, " species_dim"] = formula_matrix.dot(species_melt_density)

    return element_melt_density


def get_gas_species_data(
    fixed_parameters: FixedParameters, some_array: ArrayLike
) -> Shaped[Array, " species_dim"]:
    """Masks the gas species data from an array

    Args:
        fixed_parameters: Fixed parameters
        some_array: Some array to mask the gas species data from

    Returns:
        An array with gas species data from `some_array` and condensate entries zeroed
    """
    gas_data: Shaped[Array, " species_dim"] = fixed_parameters.gas_species_mask * some_array

    return gas_data


def get_log_activity(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Float[Array, " species_dim"],
) -> Float[Array, " species_dim"]:
    """Gets the log activity

    Args:
        traced_parameters: Traced parameters
        fixed_parameters: Fixed parameters
        log_number_density: Log number density

    Returns:
        Log activity
    """
    planet: Planet = traced_parameters.planet
    temperature: Float[Array, ""] = planet.temperature
    species: SpeciesCollection = fixed_parameters.species
    total_pressure: Float[Array, ""] = get_total_pressure(
        fixed_parameters, log_number_density, temperature
    )
    # jax.debug.print("total_pressure = {out}", out=total_pressure)

    activity_funcs: list[Callable] = [
        to_hashable(species_.activity.log_activity) for species_ in species
    ]

    def apply_activity(index: ArrayLike) -> Float[Array, ""]:
        return lax.switch(
            index,
            activity_funcs,
            temperature,
            total_pressure,
        )

    indices: Integer[Array, " species_dim"] = jnp.arange(len(species))
    vmap_activity: Callable = eqx.filter_vmap(apply_activity, in_axes=(0,))
    log_activity_pure_species: Float[Array, " species_dim"] = vmap_activity(indices)
    # jax.debug.print("log_activity_pure_species = {out}", out=log_activity_pure_species)
    log_activity: Float[Array, " species_dim"] = get_log_activity_ideal_mixing(
        fixed_parameters, log_number_density, log_activity_pure_species
    )
    # jax.debug.print("log_activity = {out}", out=log_activity)

    return log_activity


def get_log_activity_ideal_mixing(
    fixed_parameters: FixedParameters,
    log_number_density: Float[Array, " species_dim"],
    log_activity_pure_species: Float[Array, " species_dim"],
) -> Float[Array, " species_dim"]:
    """Gets the log activity of species in the atmosphere assuming an ideal mixture

    Args:
        fixed_parameters: Fixed parameters
        log_number_density: Log number density
        log_activity_pure_species: Log activity of the pure species

    Returns:
        Log activity of the species assuming ideal mixing in the atmosphere
    """
    gas_species_mask: Bool[Array, " species_dim"] = fixed_parameters.gas_species_mask
    number_density: Float[Array, " species_dim"] = safe_exp(log_number_density)
    gas_species_number_density: Float[Array, " species_dim"] = gas_species_mask * number_density
    atmosphere_log_number_density: Float[Array, ""] = jnp.log(jnp.sum(gas_species_number_density))

    log_activity_gas_species: Float[Array, " species_dim"] = (
        log_activity_pure_species + log_number_density - atmosphere_log_number_density
    )
    log_activity: Float[Array, " species_dim"] = jnp.where(
        gas_species_mask, log_activity_gas_species, log_activity_pure_species
    )

    return log_activity


def get_log_pressure_from_log_number_density(
    log_number_density: Float[Array, " species_dim"], temperature: Float[Array, ""]
) -> Float[Array, " species_dim"]:
    """Gets log pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Log pressure
    """
    log_pressure: Float[Array, " species_dim"] = (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature) + log_number_density
    )

    return log_pressure


def get_log_Kp(
    species: SpeciesCollection,
    reaction_matrix: Float[Array, "react_dim species_dim"],
    temperature: Float[Array, ""],
) -> Float[Array, " react_dim"]:
    """Gets log of the equilibrium constant in terms of partial pressures

    Args:
        species: Species
        reaction_matrix: Reaction matrix
        temperature: Temperature

    Returns:
        Log of the equilibrium constant in terms of partial pressures
    """
    gibbs_funcs: list[Callable] = [
        to_hashable(species_.data.get_gibbs_over_RT) for species_ in species
    ]

    def apply_gibbs(
        index: Integer[Array, ""], temperature: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        return lax.switch(index, gibbs_funcs, temperature)

    indices: Integer[Array, " species_dim"] = jnp.arange(len(species))
    vmap_gibbs: Callable = eqx.filter_vmap(apply_gibbs, in_axes=(0, None))
    gibbs_values: Float[Array, "species_dim 1"] = vmap_gibbs(indices, temperature)
    # jax.debug.print("gibbs_values = {out}", out=gibbs_values)
    log_Kp: Float[Array, "react_dim 1"] = -1.0 * reaction_matrix @ gibbs_values

    return jnp.ravel(log_Kp)


def get_log_reaction_equilibrium_constant(
    fixed_parameters: FixedParameters, temperature: Float[Array, ""]
) -> Float[Array, " react_dim"]:
    """Gets the log equilibrium constant of the reactions

    Args:
        fixed_parameters: Fixed parameters
        temperature: Temperature

    Returns:
        Log equilibrium constant of the reactions
    """
    species: SpeciesCollection = fixed_parameters.species
    reaction_matrix: Float[Array, "react_dim species_dim"] = jnp.array(
        fixed_parameters.reaction_matrix
    )
    log_Kp: Float[Array, " react_dim"] = get_log_Kp(species, reaction_matrix, temperature)
    # jax.debug.print("lnKp = {out}", out=lnKp)
    delta_n: Float[Array, " react_dim"] = jnp.sum(
        reaction_matrix * fixed_parameters.gas_species_mask, axis=1
    )
    # jax.debug.print("delta_n = {out}", out=delta_n)
    log_Kc: Float[Array, " react_dim"] = log_Kp - delta_n * (
        jnp.log(BOLTZMANN_CONSTANT_BAR) + jnp.log(temperature)
    )
    # jax.debug.print("log10Kc = {out}", out=log_Kc)

    return log_Kc


def get_pressure_from_log_number_density(
    log_number_density: Float[Array, " species_dim"], temperature: Float[Array, ""]
) -> Float[Array, " species_dim"]:
    """Gets pressure from log number density

    Args:
        log_number_density: Log number density
        temperature: Temperature

    Returns:
        Pressure
    """
    return safe_exp(get_log_pressure_from_log_number_density(log_number_density, temperature))


def get_species_density_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Float[Array, " species_dim"],
    log_activity: Float[Array, " species_dim"],
    log_volume: Float[Array, ""],
) -> Float[Array, " species_dim"]:
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
    molar_masses: Float[Array, " species_dim"] = jnp.array(fixed_parameters.molar_masses)
    melt_mass: Float[Array, ""] = traced_parameters.planet.melt_mass

    ppmw: Float[Array, " species_dim"] = get_species_ppmw_in_melt(
        traced_parameters, fixed_parameters, log_number_density, log_activity
    )

    species_melt_density: Float[Array, " species_dim"] = (
        ppmw
        * unit_conversion.ppm_to_fraction
        * AVOGADRO
        * melt_mass
        / (molar_masses * safe_exp(log_volume))
    )
    # jax.debug.print("species_melt_density = {out}", out=species_melt_density)

    return species_melt_density


def get_species_ppmw_in_melt(
    traced_parameters: TracedParameters,
    fixed_parameters: FixedParameters,
    log_number_density: Float[Array, " species_dim"],
    log_activity: Float[Array, " species_dim"],
) -> Float[Array, " species_dim"]:
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
    diatomic_oxygen_index: Integer[Array, ""] = jnp.array(fixed_parameters.diatomic_oxygen_index)
    temperature: Float[Array, ""] = traced_parameters.planet.temperature

    fugacity: Float[Array, " species_dim"] = safe_exp(log_activity)
    total_pressure: Float[Array, ""] = get_total_pressure(
        fixed_parameters, log_number_density, temperature
    )
    diatomic_oxygen_fugacity: Float[Array, ""] = jnp.take(fugacity, diatomic_oxygen_index)

    # NOTE: All solubility formulations must return a JAX array to allow vmap
    solubility_funcs: list[Callable] = [
        to_hashable(species_.solubility.jax_concentration) for species_ in species
    ]

    def apply_solubility(
        index: Integer[Array, ""], fugacity: Float[Array, ""]
    ) -> Float[Array, ""]:
        return lax.switch(
            index,
            solubility_funcs,
            fugacity,
            temperature,
            total_pressure,
            diatomic_oxygen_fugacity,
        )

    indices: Integer[Array, " species_dim"] = jnp.arange(len(species))
    vmap_solubility: Callable = eqx.filter_vmap(apply_solubility, in_axes=(0, 0))
    species_ppmw: Float[Array, " species_dim"] = vmap_solubility(indices, fugacity)
    # jax.debug.print("ppmw = {out}", out=ppmw)

    return species_ppmw


@eqx.filter_jit
# Useful for optimising how many times JAX compiles the solve function
# @eqx.debug.assert_max_traces(max_traces=1)
def repeat_solver(
    solver_vmap_fn: Callable,
    active_indices: Integer[Array, " res_dim"],
    tau: Float[Array, "..."],
    solution: Float[Array, "batch_dim sol_dim"],
    traced_parameters: TracedParameters,
    solver_parameters: SolverParameters,
    key: PRNGKeyArray,
) -> tuple[
    Float[Array, "batch_dim sol_dim"],
    Bool[Array, " batch_dim"],
    Integer[Array, " batch_dim"],
    Integer[Array, " batch_dim"],
]:
    """Repeat solver that perturbs the initial solution for cases that fail and tries again

    Args:
        solver_vmap_fn: Vmapped solver function with pre-bound fixed configuration
        active_indices: Indices of the residual array that are active
        tau: Tau parameter for species' stability
        solution: Solution
        traced_parameters: Traced parameters
        solver_paramters: Solver parameters
        key: Random key

    Returns:
        A tuple with the state: (solution, solver_status, solver_steps, solver_attempts)
    """

    def body_fn(state: tuple[Array, ...]) -> tuple[Array, ...]:
        """Perform one iteration of the solver retry loop

        Args:
            state: Tuple containing:
                i: Current attempt index
                key: PRNG key for random number generation
                solution: Current solution array
                status: Boolean array indicating successful solutions
                steps: Step count
                success_attempt: Integer array recording iteration of success for each entry

        Returns:
            Updated state tuple
        """
        i, key, solution, status, steps, success_attempt = state

        failed_mask: Bool[Array, " batch_dim"] = ~status
        key, subkey = random.split(key)

        # Perturb the (initial) solution for cases that failed. Something more sophisticated could
        # be implemented, such as a regressor or neural network to inform failed cases based on
        # successful solves.
        perturb_shape: tuple[int, int] = (solution.shape[0], solution.shape[1])
        raw_perturb: Float[Array, "batch_dim sol_dim"] = random.uniform(
            subkey, shape=perturb_shape, minval=-1.0, maxval=1.0
        )
        perturbations: Float[Array, "batch_dim sol_dim"] = jnp.where(
            failed_mask[:, None],
            solver_parameters.multistart_perturbation * raw_perturb,
            jnp.zeros_like(solution),
        )
        new_initial_solution: Float[Array, "batch_dim sol_dim"] = solution + perturbations
        # jax.debug.print("new_initial_solution = {out}", out=new_initial_solution)

        new_solution, new_status, new_steps = solver_vmap_fn(
            new_initial_solution, active_indices, tau, traced_parameters, solver_parameters
        )

        # Determine which entries to update: previously failed, now succeeded
        update_mask: Bool[Array, " batch_dim"] = failed_mask & new_status
        updated_i: Integer[Array, "..."] = i + 1
        updated_solution: Float[Array, "batch_dim sol_dim"] = cast(
            Array, jnp.where(update_mask[:, None], new_solution, solution)
        )
        updated_status: Bool[Array, " batch_dim"] = status | new_status
        updated_steps: Integer[Array, " batch_dim"] = cast(
            Array, jnp.where(update_mask, new_steps, steps)
        )
        updated_success_attempt: Array = jnp.where(update_mask, updated_i, success_attempt)

        return (
            updated_i,
            key,
            updated_solution,
            updated_status,
            updated_steps,
            updated_success_attempt,
        )

    def cond_fn(state: tuple[Array, ...]) -> Bool[Array, "..."]:
        """Check if the solver should continue retrying

        Args:
            state: Tuple containing:
                i: Current attempt index
                _: Unused (PRNG key)
                _: Unused (solution)
                status: Boolean array indicating success of each solution
                _: Unused (steps)
                _: Unused (success_attempt)

        Returns:
            A boolean array indicating whether retries should continue (True if any solution
            failed and attempts are still available)
        """
        i, _, _, status, _, _ = state

        return jnp.logical_and(i < solver_parameters.multistart, jnp.any(~status))

    # Try first solution
    first_solution, first_solver_status, first_solver_steps = solver_vmap_fn(
        solution, active_indices, tau, traced_parameters, solver_parameters
    )
    # Failback solution
    solution = cast(Array, jnp.where(first_solver_status[:, None], first_solution, solution))

    initial_state: tuple[Array, ...] = (
        jnp.array(1),  # First attempt of the repeat_solver
        key,
        solution,
        first_solver_status,
        first_solver_steps,
        first_solver_status.astype(int),  # 1 for solved, otherwise 0
    )

    _, _, final_solution, final_status, final_steps, final_success_attempt = lax.while_loop(
        cond_fn, body_fn, initial_state
    )

    return final_solution, final_status, final_steps, final_success_attempt


def make_solve_tau_step(
    solver_vmap_fn: Callable,
    active_indices: Integer[Array, " res_dim"],
    traced_parameters: TracedParameters,
    solver_parameters: SolverParameters,
) -> Callable:
    """Wraps the repeat solver to call it for different tau values

    Args:
        solver_vmap_fn: Vmapped solver function with pre-bound fixed configuration
        active_indices: Indices of the residual array that are active
        traced_parameters: Traced parameters
        solver_parameters: Solver parameters

    Returns:
        Wrapped solver for a single tau value
    """

    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)
    def solve_tau_step(carry: tuple, tau: Float[Array, " batch_dim"]) -> tuple[tuple, tuple]:
        (key, solution) = carry
        key, subkey = jax.random.split(key)

        new_solution, new_status, new_steps, success_attempt = repeat_solver(
            solver_vmap_fn,
            active_indices,
            tau,
            solution,
            traced_parameters,
            solver_parameters,
            subkey,
        )

        new_carry: tuple[PRNGKeyArray, Float[Array, "batch_dim sol_dim"]] = (key, new_solution)

        # Output current solution etc for this tau step
        out: tuple[Array, ...] = (new_solution, new_status, new_steps, success_attempt)

        return new_carry, out

    return solve_tau_step
