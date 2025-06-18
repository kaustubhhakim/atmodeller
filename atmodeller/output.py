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
"""Output

This uses existing functions as much as possible to calculate desired output quantities, where some
must be vmapped to compute the output.
"""

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import Formula
from openpyxl.styles import PatternFill
from scipy.constants import mega

from atmodeller import TAU
from atmodeller.constants import AVOGADRO
from atmodeller.containers import (
    FixedParameters,
    Planet,
    SolverParameters,
    SpeciesCollection,
    TracedParameters,
)
from atmodeller.engine import (
    get_atmosphere_log_molar_mass,
    get_atmosphere_log_volume,
    get_element_density,
    get_element_density_in_melt,
    get_log_activity,
    get_log_number_density_from_log_pressure,
    get_pressure_from_log_number_density,
    get_species_density_in_melt,
    get_species_ppmw_in_melt,
    get_total_pressure,
    objective_function,
)
from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.mytypes import NpArray, NpBool, NpFloat, NpInt
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.utilities import unit_conversion, vmap_axes_spec

logger: logging.Logger = logging.getLogger(__name__)


class Output:
    """Output

    Args:
        species: Species
        solution: Array output from solve
        active_indices: Indices of the residual array that are active
        solver_status: Solver status
        solver_steps: Number of solver steps
        solver_attempts: Number of solver attempts (multistart)
        fixed_parameters: Fixed parameters
        traced_parameters: Traced parameters
        solver_parameters: Solver parameters
    """

    def __init__(
        self,
        species: SpeciesCollection,
        solution: Float[Array, " batch_dim sol_dim"],
        active_indices: Integer[Array, " res_dim"],
        solver_status: Bool[Array, " batch_dim"],
        solver_steps: Integer[Array, " batch_dim"],
        solver_attempts: Integer[Array, " batch_dim"],
        fixed_parameters: FixedParameters,
        traced_parameters: TracedParameters,
        solver_parameters: SolverParameters,
    ):
        logger.debug("Creating Output")
        self._species: SpeciesCollection = species
        self._solution: NpFloat = np.asarray(solution)
        self._active_indices: NpInt = np.asarray(active_indices)
        self._solver_status: NpBool = np.asarray(solver_status)
        self._solver_steps: NpInt = np.asarray(solver_steps)
        self._solver_attempts: NpInt = np.asarray(solver_attempts)
        self._fixed_parameters: FixedParameters = fixed_parameters
        self._traced_parameters: TracedParameters = traced_parameters
        self._solver_parameters: SolverParameters = solver_parameters

        log_number_density, log_stability = np.split(self._solution, 2, axis=1)
        self._log_number_density: NpFloat = log_number_density
        # Mask stabilities that are not solved
        self._log_stability: NpFloat = np.where(
            fixed_parameters.active_stability(), log_stability, np.nan
        )
        # Caching output to avoid recomputation
        self._cached_dict: dict[str, dict[str, NpArray]] | None = None
        self._cached_dataframes: dict[str, pd.DataFrame] | None = None

    @property
    def active_indices(self) -> NpInt:
        """Active indices"""
        return self._active_indices

    @property
    def formula_matrix(self) -> NpInt:
        """Formula matrix"""
        return np.asarray(self._fixed_parameters.formula_matrix)

    @property
    def condensed_species_mask(self) -> NpBool:
        """Mask of condensed species"""
        return np.invert(self._fixed_parameters.gas_species_mask)

    @property
    def gas_species_mask(self) -> NpBool:
        """Mask of gas species"""
        return np.asarray(self._fixed_parameters.gas_species_mask)

    @property
    def log_number_density(self) -> NpFloat:
        """Log number density"""
        return self._log_number_density

    @property
    def log_stability(self) -> NpFloat:
        """Log stability of relevant species"""
        return self._log_stability

    @property
    def molar_mass(self) -> NpFloat:
        """Molar mass of all species"""
        return np.asarray(self._fixed_parameters.molar_masses)

    @property
    def number_solutions(self) -> int:
        """Number of solutions"""
        return self.log_number_density.shape[0]

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self._traced_parameters.planet

    @property
    def temperature(self) -> NpFloat:
        """Temperature"""
        return np.asarray(self.planet.temperature)

    def activity(self) -> NpFloat:
        """Gets the activity of all species

        Returns:
            Activity of all species
        """
        return np.exp(self.log_activity())

    def activity_without_stability(self) -> NpFloat:
        """Gets the activity without stability of all species

        Returns:
            Activity without stability of all species
        """
        return np.exp(self.log_activity_without_stability())

    def asdict(self) -> dict[str, dict[str, NpArray]]:
        """All output in a dictionary, with caching.

        Returns:
            Dictionary of all output
        """
        if self._cached_dict is not None:
            logger.info("Returning cached asdict output")
            return self._cached_dict  # Return cached result

        logger.info("Computing asdict output")

        out: dict[str, dict[str, NpArray]] = {}

        # These are required for condensed and gas species
        molar_mass: NpFloat = self.species_molar_mass_expanded()
        number_density: NpFloat = self.number_density()
        activity: NpFloat = self.activity()

        out |= self.gas_species_asdict(molar_mass, number_density, activity)
        out |= self.condensed_species_asdict(molar_mass, number_density, activity)
        out |= self.elements_asdict()

        out["planet"] = broadcast_arrays_in_dict(self.planet.asdict(), self.number_solutions)
        out["atmosphere"] = self.atmosphere_asdict()
        # Temperature and pressure have already been expanded to the number of solutions
        temperature: NpFloat = out["planet"]["surface_temperature"]
        pressure: NpFloat = out["atmosphere"]["pressure"]
        # Convenient to also attach temperature to the atmosphere output
        out["atmosphere"]["temperature"] = temperature
        out["raw_solution"] = self.raw_solution_asdict()

        out["constraints"] = {}
        out["constraints"] |= broadcast_arrays_in_dict(
            self._traced_parameters.mass_constraints.asdict(), self.number_solutions
        )
        out["constraints"] |= broadcast_arrays_in_dict(
            self._traced_parameters.fugacity_constraints.asdict(temperature, pressure),
            self.number_solutions,
        )

        out["residual"] = self.residual_asdict()  # type: ignore since keys are int

        if "O2_g" in out:
            logger.debug("Found O2_g so back-computing log10 shift for fO2")
            log10_fugacity: NpFloat = np.log10(out["O2_g"]["fugacity"])
            buffer: RedoxBufferProtocol = IronWustiteBuffer()
            # Shift at 1 bar
            buffer_at_one_bar: NpFloat = np.asarray(buffer.log10_fugacity(temperature, 1.0))
            log10_shift_at_one_bar: NpFloat = log10_fugacity - buffer_at_one_bar
            # logger.debug("log10_shift_at_1bar = %s", log10_shift_at_one_bar)
            out["O2_g"]["log10dIW_1_bar"] = log10_shift_at_one_bar
            # Shift at actual pressure
            buffer_at_P: NpFloat = np.asarray(buffer.log10_fugacity(temperature, pressure))
            log10_shift_at_P: NpFloat = log10_fugacity - buffer_at_P
            # logger.debug("log10_shift_at_P = %s", log10_shift_at_P)
            out["O2_g"]["log10dIW_P"] = log10_shift_at_P

        out["solver"] = {
            "status": self._solver_status,
            "steps": self._solver_steps,
            "attempts": self._solver_attempts,
        }

        # For debugging to confirm all outputs are numpy arrays
        # def find_non_numpy(d) -> None:
        #     for key, value in d.items():
        #         if isinstance(value, dict):
        #             find_non_numpy(value)
        #         else:
        #             if not isinstance(value, np.ndarray):
        #                 logger.warning("Non numpy array type found")
        #                 logger.warning("key = %s, value = %s", key, value)
        #                 logger.warning("type = %s", type(value))

        # find_non_numpy(out)

        self._cached_dict = out  # Cache result for faster re-accessing

        return out

    def atmosphere_asdict(self) -> dict[str, NpArray]:
        """Gets the atmosphere properties

        Returns:
            Atmosphere properties
        """
        out: dict[str, NpArray] = {}

        log_number_density_from_log_pressure_func: Callable = eqx.filter_vmap(
            get_log_number_density_from_log_pressure, in_axes=(0, self.temperature_vmap_axes())
        )
        log_number_density: Array = log_number_density_from_log_pressure_func(
            jnp.log(self.total_pressure()), jnp.asarray(self.temperature)
        )
        # Must be 2-D to align arrays for computing number-density-related quantities
        number_density: NpArray = np.exp(log_number_density)[:, np.newaxis]
        molar_mass: NpArray = self.atmosphere_molar_mass()[:, np.newaxis]
        out: dict[str, NpArray] = self._get_number_density_output(
            number_density, molar_mass, "species_"
        )
        # Species mass is simply mass so rename for clarity
        out["mass"] = out.pop("species_mass")

        out["molar_mass"] = molar_mass
        # Ensure all arrays are 1-D, which is required for creating dataframes
        out = {key: value.ravel() for key, value in out.items()}

        out["pressure"] = self.total_pressure()
        out["volume"] = self.atmosphere_volume()
        out["element_number_density"] = np.sum(self.element_density_gas(), axis=1)
        out["element_number"] = out["element_number_density"] * out["volume"]
        out["element_moles"] = out["element_number"] / AVOGADRO

        return out

    def atmosphere_log_molar_mass(self) -> NpFloat:
        """Gets log molar mass of the atmosphere

        Returns:
            Log molar mass of the atmosphere
        """
        atmosphere_log_molar_mass_func: Callable = eqx.filter_vmap(
            get_atmosphere_log_molar_mass, in_axes=(None, 0)
        )
        atmosphere_log_molar_mass: Array = atmosphere_log_molar_mass_func(
            self._fixed_parameters, jnp.asarray(self.log_number_density)
        )

        return np.asarray(atmosphere_log_molar_mass)

    def atmosphere_molar_mass(self) -> NpArray:
        """Gets the molar mass of the atmosphere

        Returns:
            Molar mass of the atmosphere
        """
        return np.exp(self.atmosphere_log_molar_mass())

    def atmosphere_log_volume(self) -> NpFloat:
        """Gets the log volume of the atmosphere

        Returns:
            Log volume of the atmosphere
        """
        atmosphere_log_volume_func: Callable = eqx.filter_vmap(
            get_atmosphere_log_volume,
            in_axes=(
                None,
                0,
                vmap_axes_spec(self._traced_parameters.planet),
            ),
        )
        atmosphere_log_volume: Array = atmosphere_log_volume_func(
            self._fixed_parameters,
            jnp.asarray(self.log_number_density),
            self.planet,
        )

        return np.asarray(atmosphere_log_volume)

    def atmosphere_volume(self) -> NpFloat:
        """Gets the volume of the atmosphere

        Returns:
            Volume of the atmosphere
        """
        return np.exp(self.atmosphere_log_volume())

    def total_pressure(self) -> NpFloat:
        """Gets total pressure

        Returns:
            Total pressure
        """
        total_pressure_func: Callable = eqx.filter_vmap(
            get_total_pressure, in_axes=(None, 0, self.temperature_vmap_axes())
        )
        total_pressure: Array = total_pressure_func(
            self._fixed_parameters,
            jnp.asarray(self.log_number_density),
            jnp.asarray(self.temperature),
        )

        return np.asarray(total_pressure)

    def condensed_species_asdict(
        self,
        molar_mass: NpArray,
        number_density: NpArray,
        activity: NpArray,
    ) -> dict[str, dict[str, NpArray]]:
        """Gets the condensed species output as a dictionary

        Args:
            molar_mass: Molar mass of all species
            number_density: Number density of all species
            activity: Activity of all species

        Returns:
            Condensed species output as a dictionary
        """
        molar_mass = molar_mass[:, self.condensed_species_mask]
        number_density = number_density[:, self.condensed_species_mask]
        activity = activity[:, self.condensed_species_mask]

        condensed_species: tuple[str, ...] = self._species.get_condensed_species_names()

        out: dict[str, NpArray] = self._get_number_density_output(
            number_density, molar_mass, "total_"
        )
        out["molar_mass"] = molar_mass
        out["activity"] = activity

        split_dict: list[dict[str, NpArray]] = split_dict_by_columns(out)
        species_out: dict[str, dict[str, NpArray]] = {
            species_name: split_dict[ii] for ii, species_name in enumerate(condensed_species)
        }

        return species_out

    def elements_asdict(self) -> dict[str, dict[str, NpArray]]:
        """Gets the element properties as a dictionary

        Returns:
            Element outputs as a dictionary
        """
        molar_mass: NpArray = self.element_molar_mass_expanded()
        atmosphere: NpArray = self.element_density_gas()
        condensed: NpArray = self.element_density_condensed()
        dissolved: NpArray = self.element_density_dissolved()
        total: NpArray = atmosphere + condensed + dissolved

        out: dict[str, NpArray] = self._get_number_density_output(
            atmosphere, molar_mass, "atmosphere_"
        )
        out |= self._get_number_density_output(condensed, molar_mass, "condensed_")
        out |= self._get_number_density_output(dissolved, molar_mass, "dissolved_")
        out |= self._get_number_density_output(total, molar_mass, "total_")

        out["molar_mass"] = molar_mass
        out["degree_of_condensation"] = out["condensed_number"] / out["total_number"]
        out["volume_mixing_ratio"] = out["atmosphere_number"] / np.sum(
            out["atmosphere_number"], axis=1, keepdims=True
        )
        out["atmosphere_ppm"] = out["volume_mixing_ratio"] * mega
        out["atmosphere_ppmw"] = (
            out["atmosphere_mass"] / np.sum(out["atmosphere_mass"], axis=1, keepdims=True) * mega
        )

        unique_elements: tuple[str, ...] = self._species.get_unique_elements_in_species()
        if "H" in unique_elements:
            index: int = unique_elements.index("H")
            H_total_moles: NpArray = out["total_moles"][:, index]
            out["logarithmic_abundance"] = (
                np.log10(out["total_moles"] / H_total_moles[:, np.newaxis]) + 12
            )

        # logger.debug("out = %s", out)

        split_dict: list[dict[str, NpArray]] = split_dict_by_columns(out)
        # logger.debug("split_dict = %s", split_dict)

        elements_out: dict[str, dict[str, NpArray]] = {
            f"element_{element}": split_dict[ii] for ii, element in enumerate(unique_elements)
        }
        # logger.debug("elements_out = %s", elements_out)

        return elements_out

    def element_density_condensed(self) -> NpFloat:
        """Gets the number density of elements in the condensed phase

        Unlike for the objective function, we want the number density of all elements, regardless
        of whether they were used to impose a mass constraint on the system.

        Returns:
            Number density of elements in the condensed phase
        """
        element_density_func: Callable = eqx.filter_vmap(get_element_density, in_axes=(None, 0))
        element_density: Array = element_density_func(
            jnp.asarray(self.formula_matrix),
            jnp.asarray(self.log_number_density * self.condensed_species_mask),
        )

        return np.asarray(element_density)

    def element_density_dissolved(self) -> NpFloat:
        """Gets the number density of elements dissolved in melt due to species solubility

        Unlike for the objective function, we want the number density of all elements, regardless
        of whether they were used to impose a mass constraint on the system.

        Returns:
            Number density of elements dissolved in melt due to species solubility
        """
        element_density_dissolved_func: Callable = eqx.filter_vmap(
            get_element_density_in_melt,
            in_axes=(self.traced_parameters_vmap_axes(), None, None, 0, 0, 0),
        )
        element_density_dissolved: Array = element_density_dissolved_func(
            self._traced_parameters,
            self._fixed_parameters,
            jnp.asarray(self._fixed_parameters.formula_matrix),
            jnp.asarray(self.log_number_density),
            jnp.asarray(self.log_activity()),
            jnp.asarray(self.atmosphere_log_volume()),
        )

        return np.asarray(element_density_dissolved)

    def element_density_gas(self) -> NpFloat:
        """Gets the number density of elements in the gas phase

        Unlike for the objective function, we want the number density of all elements, regardless
        of whether they were used to impose a mass constraint on the system.

        Returns:
            Number density of elements in the gas phase
        """
        element_density_func: Callable = eqx.filter_vmap(get_element_density, in_axes=(None, 0))
        element_density: Array = element_density_func(
            jnp.asarray(self.formula_matrix),
            jnp.asarray(self.log_number_density * self.gas_species_mask),
        )

        return np.asarray(element_density)

    def element_molar_mass_expanded(self) -> NpFloat:
        """Gets molar mass of elements

        Returns:
            Molar mass of elements
        """
        unique_elements: tuple[str, ...] = self._species.get_unique_elements_in_species()
        molar_mass: npt.ArrayLike = np.array(
            [Formula(element).mass for element in unique_elements]
        )
        molar_mass = unit_conversion.g_to_kg * molar_mass

        return np.tile(molar_mass, (self.number_solutions, 1))

    def _get_number_density_output(
        self, number_density: NpArray, molar_mass_expanded: NpArray, prefix: str = ""
    ) -> dict[str, NpArray]:
        """Gets the outputs associated with a number density

        Args:
            number_density: Number density
            molar_mass_expanded: Molar mass associated with the number density
            prefix: Key prefix for the output. Defaults to an empty string.

        Returns
            Dictionary of output quantities
        """
        atmosphere_volume: NpArray = self.atmosphere_volume()
        # Volume must be a column vector because it multiples all elements in the row
        number: NpArray = number_density * atmosphere_volume[:, np.newaxis]
        moles: NpArray = number / AVOGADRO
        mass: NpArray = moles * molar_mass_expanded

        out: dict[str, NpArray] = {}
        out[f"{prefix}number_density"] = number_density
        out[f"{prefix}number"] = number
        out[f"{prefix}moles"] = moles
        out[f"{prefix}mass"] = mass

        return out

    def gas_species_asdict(
        self,
        molar_mass: NpArray,
        number_density: NpArray,
        activity: NpArray,
    ) -> dict[str, dict[str, NpArray]]:
        """Gets the gas species output as a dictionary

        Args:
            molar_mass: Molar mass of all species
            number_density: Number density of all species
            activity: Activity of all species

        Returns:
            Gas species output as a dictionary
        """
        # Below are all filtered to only include the data (columns) of gas species
        molar_mass = molar_mass[:, self.gas_species_mask]
        number_density = number_density[:, self.gas_species_mask]
        activity = activity[:, self.gas_species_mask]
        dissolved_number_density: NpArray = self.species_density_in_melt()[
            :, self.gas_species_mask
        ]
        total_number_density: NpArray = number_density + dissolved_number_density
        pressure: NpArray = self.pressure()[:, self.gas_species_mask]

        gas_species: tuple[str, ...] = self._species.get_gas_species_names()

        out: dict[str, NpArray] = {}
        out |= self._get_number_density_output(number_density, molar_mass, "atmosphere_")
        out |= self._get_number_density_output(dissolved_number_density, molar_mass, "dissolved_")
        out |= self._get_number_density_output(total_number_density, molar_mass, "total_")
        out["molar_mass"] = molar_mass
        out["volume_mixing_ratio"] = out["atmosphere_number"] / np.sum(
            out["atmosphere_number"], axis=1, keepdims=True
        )
        out["atmosphere_ppm"] = out["volume_mixing_ratio"] * mega
        out["atmosphere_ppmw"] = (
            out["atmosphere_mass"] / np.sum(out["atmosphere_mass"], axis=1, keepdims=True) * mega
        )
        out["pressure"] = pressure
        out["fugacity"] = activity
        out["fugacity_coefficient"] = activity / pressure
        out["dissolved_ppmw"] = self.species_ppmw_in_melt()

        split_dict: list[dict[str, NpArray]] = split_dict_by_columns(out)
        species_out: dict[str, dict[str, NpArray]] = {
            species_name: split_dict[ii] for ii, species_name in enumerate(gas_species)
        }

        return species_out

    def log_activity(self) -> NpFloat:
        """Gets log activity of all species.

        This is usually what the user wants when referring to activity because it includes a
        consideration of species stability

        Returns:
            Log activity of all species
        """
        log_activity_without_stability: NpFloat = self.log_activity_without_stability()
        log_activity_with_stability: NpFloat = log_activity_without_stability - np.exp(
            self.log_stability
        )
        # Now select the appropriate activity for each species, depending if stability is relevant.
        condition_broadcasted = np.broadcast_to(
            self._fixed_parameters.active_stability(), log_activity_without_stability.shape
        )
        # logger.debug("condition_broadcasted = %s", condition_broadcasted)

        log_activity: NpFloat = np.where(
            condition_broadcasted,
            log_activity_with_stability,
            log_activity_without_stability,
        )

        return log_activity

    def log_activity_without_stability(self) -> NpFloat:
        """Gets log activity without stability of all species

        Args:
            Log activity without stability of all species
        """
        log_activity_func: Callable = eqx.filter_vmap(
            get_log_activity,
            in_axes=(self.traced_parameters_vmap_axes(), None, 0),
        )
        log_activity: Array = log_activity_func(
            self._traced_parameters, self._fixed_parameters, jnp.asarray(self.log_number_density)
        )

        return np.asarray(log_activity)

    def number_density(self) -> NpFloat:
        r"""Gets number density of all species

        Returns:
            Number density in :math:`\mathrm{molecules}\, \mathrm{m}^{-3}`
        """
        return np.exp(self.log_number_density)

    def species_molar_mass_expanded(self) -> NpFloat:
        r"""Gets molar mass of all species in an expanded array.

        Returns:
            Molar mass of all species in an expanded array.
        """
        return np.tile(self.molar_mass, (self.number_solutions, 1))

    def pressure(self) -> NpFloat:
        """Gets pressure of species in bar

        This will compute pressure of all species, including condensates, for simplicity.

        Returns:
            Pressure of species in bar
        """
        pressure_func: Callable = eqx.filter_vmap(
            get_pressure_from_log_number_density, in_axes=(0, self.temperature_vmap_axes())
        )
        pressure: Array = pressure_func(
            jnp.asarray(self.log_number_density), jnp.asarray(self.temperature)
        )

        return np.asarray(pressure)

    def quick_look(self) -> dict[str, ArrayLike]:
        """Quick look at the solution

        Provides a quick first glance at the output with convenient units and to ease comparison
        with test or benchmark data.

        Returns:
            Dictionary of the solution
        """
        out: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self._species):
            pressure: NpArray = self.pressure()[:, nn]
            activity: NpArray = self.activity()[:, nn]
            out[species_.name] = pressure
            out[f"{species_.name}_activity"] = activity

        return {key: np.squeeze(value) for key, value in out.items()}

    def raw_solution_asdict(self) -> dict[str, NpArray]:
        """Gets the raw solution

        Returns:
            Dictionary of the raw solution
        """
        raw_solution: dict[str, NpArray] = {}

        species_names: tuple[str, ...] = self._species.get_species_names()

        for ii, species_name in enumerate(species_names):
            raw_solution[species_name] = self.log_number_density[:, ii]
            raw_solution[f"{species_name}_stability"] = self.log_stability[:, ii]

        # Remove keys where the array values are all nan
        for key in list(raw_solution.keys()):
            if np.all(np.isnan(raw_solution[key])):
                raw_solution.pop(key)

        return raw_solution

    def residual_asdict(self) -> dict[int, NpFloat]:
        """Gets the residual

        Returns:
            Dictionary of the residual
        """
        residual_func: Callable = eqx.filter_vmap(
            objective_function,
            in_axes=(
                0,
                {
                    "traced_parameters": self.traced_parameters_vmap_axes(),
                    "active_indices": None,
                    "tau": None,
                    "fixed_parameters": None,
                    "solver_parameters": None,
                },
            ),
        )
        residual: Array = residual_func(
            self._solution,
            {
                "traced_parameters": self._traced_parameters,
                "active_indices": jnp.asarray(self.active_indices),
                "tau": jnp.asarray(TAU),
                "fixed_parameters": self._fixed_parameters,
                "solver_parameters": self._solver_parameters,
            },
        )

        out: dict[int, NpArray] = {}
        for ii in range(residual.shape[1]):
            out[ii] = np.asarray(residual[:, ii])

        return out

    def species_density_in_melt(self) -> NpFloat:
        """Gets species number density in the melt

        Returns:
            Species number density in the melt
        """
        species_density_in_melt_func: Callable = eqx.filter_vmap(
            get_species_density_in_melt,
            in_axes=(self.traced_parameters_vmap_axes(), None, 0, 0, 0),
        )
        species_density_in_melt: Array = species_density_in_melt_func(
            self._traced_parameters,
            self._fixed_parameters,
            jnp.asarray(self.log_number_density),
            jnp.asarray(self.log_activity()),
            jnp.asarray(self.atmosphere_log_volume()),
        )

        return np.asarray(species_density_in_melt)

    def species_ppmw_in_melt(self) -> NpFloat:
        """Gets species ppmw in the melt

        Return:
            Species ppmw in the melt
        """
        species_ppmw_in_melt_func: Callable = eqx.filter_vmap(
            get_species_ppmw_in_melt, in_axes=(self.traced_parameters_vmap_axes(), None, 0, 0)
        )
        species_ppmw_in_melt: Array = species_ppmw_in_melt_func(
            self._traced_parameters,
            self._fixed_parameters,
            jnp.asarray(self.log_number_density),
            jnp.asarray(self.log_activity()),
        )

        return np.asarray(species_ppmw_in_melt)

    def stability(self) -> NpFloat:
        """Gets stability of relevant species

        Returns:
            Stability of relevant species
        """
        return np.exp(self.log_stability)

    def temperature_vmap_axes(self) -> Literal[0, None]:
        """Gets vmap axes for temperature"""
        return vmap_axes_spec(self._traced_parameters.planet.temperature)

    def traced_parameters_vmap_axes(self) -> TracedParameters:
        """Gets vmap axes for traced parameters"""
        return vmap_axes_spec(self._traced_parameters)

    def _drop_unsuccessful_solves(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Drops unsuccessful solves

        Args:
            dataframes: Dataframes from which to drop unsuccessful models

        Returns:
            Dictionary of dataframes without unsuccessful models
        """
        return {key: df.loc[self._solver_status] for key, df in dataframes.items()}

    def to_dataframes(self, drop_unsuccessful: bool = False) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Args:
            drop_unsuccessful: Drop models that did not solve. Defaults to False.

        Returns:
            Output in a dictionary of dataframes
        """
        if self._cached_dataframes is not None:
            logger.debug("Returning cached to_dataframes output")
            dataframes: dict[str, pd.DataFrame] = self._cached_dataframes  # Return cached result
        else:
            logger.info("Computing to_dataframes output")
            dataframes = nested_dict_to_dataframes(self.asdict())
            self._cached_dataframes = dataframes
            # logger.debug("to_dataframes = %s", self._cached_dataframes)

        if drop_unsuccessful:
            logger.info("Dropping models that did not solve")
            dataframes: dict[str, pd.DataFrame] = self._drop_unsuccessful_solves(dataframes)

        return dataframes

    def to_excel(
        self, file_prefix: Path | str = "new_atmodeller_out", drop_unsuccessful: bool = False
    ) -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to new_atmodeller_out.
            drop_unsuccessful: Drop models that did not solve. Defaults to False.
        """
        logger.info("Writing output to excel")
        out: dict[str, pd.DataFrame] = self.to_dataframes(drop_unsuccessful)
        output_file: Path = Path(f"{file_prefix}.xlsx")

        # Convenient to highlight rows where the solver failed to find a solution for follow-up
        # analysis. Define a fill color for highlighting rows (e.g., yellow)
        highlight_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

        # Get the indices where the successful_solves mask is False
        unsuccessful_indices: NpArray[np.int_] = np.where(
            np.array(self._solver_status) == False  # noqa: E712
        )[0]

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)
                sheet = writer.sheets[df_name]

                # Apply highlighting to the rows where the solver failed to find a solution
                for idx in unsuccessful_indices:
                    # Highlight the entire row (starting from index 2 to skip header row)
                    for col in range(1, len(df.columns) + 2):
                        # row=idx+2 because Excel is 1-indexed and row 1 is the header
                        cell = sheet.cell(row=idx + 2, column=col)
                        cell.fill = highlight_fill

        logger.info("Output written to %s", output_file)

    def to_pickle(
        self, file_prefix: Path | str = "new_atmodeller_out", drop_unsuccessful: bool = False
    ) -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to new_atmodeller_out.
            drop_unsuccessful: Drop models that did not solve. Defaults to False.
        """
        logger.info("Writing output to pickle")
        out: dict[str, pd.DataFrame] = self.to_dataframes(drop_unsuccessful)
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)


def broadcast_arrays_in_dict(some_dict: dict[str, NpArray], shape: int) -> dict[str, NpArray]:
    """Gets a dictionary of broadcasted arrays

    Args:
        some_dict: Some dictionary
        size: Shape (size) of the desired array

    Returns:
        A dictionary with broadcasted arrays
    """
    expanded_dict: dict[str, NpArray] = {}
    for key, value in some_dict.items():
        expanded_dict[key] = np.broadcast_to(value, shape)

    return expanded_dict


def split_dict_by_columns(dict_to_split: dict[str, NpArray]) -> list[dict[str, NpArray]]:
    """Splits a dictionary based on columns in the values.

    Args:
        dict_to_split: A dictionary to split

    Returns:
        A list of dictionaries split by column
    """
    # Assume all arrays have the same number of columns
    first_key: str = next(iter(dict_to_split))
    num_columns: int = dict_to_split[first_key].shape[1]

    # Preallocate list of dicts
    split_dicts: list[dict] = [{} for _ in range(num_columns)]

    for key, array in dict_to_split.items():
        for i in range(num_columns):
            split_dicts[i][key] = array[:, i]

    return split_dicts


def nested_dict_to_dataframes(nested_dict: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """Creates a dictionary of dataframes from a nested dictionary

    Args:
        nested_dict: A nested dictionary

    Returns:
        A dictionary of dataframes
    """
    dataframes: dict[str, pd.DataFrame] = {}

    for outer_key, inner_dict in nested_dict.items():
        # Convert inner dictionary to DataFrame
        df: pd.DataFrame = pd.DataFrame(inner_dict)
        dataframes[outer_key] = df

    return dataframes
