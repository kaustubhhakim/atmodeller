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

This uses existing functions as much as possible to calculate desired output quantities. Notably,
some of these functions must be vmapped to compute the output for batch calculations.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from molmass import Formula

from atmodeller import AVOGADRO
from atmodeller.containers import (
    FixedParameters,
    Planet,
    Solution,
    Species,
    TracedParameters,
)
from atmodeller.engine import (
    get_atmosphere_log_molar_mass,
    get_atmosphere_log_volume,
    get_atmosphere_pressure,
    get_element_density,
    get_element_density_in_melt,
    get_log_activity,
    get_pressure_from_log_number_density,
    get_species_density_in_melt,
)
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if TYPE_CHECKING:
    from atmodeller.classes import InteriorAtmosphere

logger: logging.Logger = logging.getLogger(__name__)


class Output(ABC):
    """Output

    Args:
        solution: Array output from solve
        interior_atmosphere: Interior atmosphere
        initial_solution: Initial solution
        traced_parameters: Traced parameters
    """

    def __init__(
        self,
        solution: Array,
        interior_atmosphere: InteriorAtmosphere,
        initial_solution: Solution,
        traced_parameters: TracedParameters,
    ):
        logger.info("Creating Output")
        self._solution: Array = jnp.atleast_2d(solution)
        self._interior_atmosphere: InteriorAtmosphere = interior_atmosphere
        self._initial_solution: Solution = initial_solution
        self._traced_parameters: TracedParameters = traced_parameters

        log_number_density, log_stability = jnp.split(self._solution, 2, axis=1)
        self._log_number_density: Array = log_number_density
        self._log_stability: Array = log_stability

    @abstractmethod
    def atmosphere_log_molar_mass(self) -> Array:
        """Gets log molar mass of the atmosphere

        Returns:
            Log molar mass of the atmosphere
        """

    @abstractmethod
    def atmosphere_pressure(self) -> Array:
        """Gets pressure of the atmosphere

        Returns:
            Pressure of the atmosphere
        """

    @abstractmethod
    def atmosphere_log_volume(self) -> Array:
        """Gets volume of the atmosphere

        Returns:
            Volume of the atmosphere
        """

    @abstractmethod
    def element_density_condensed(self) -> Array:
        """Gets the number density of elements in the condensed phase

        Unlike for the objective function, we want the number density of all elements, regardless
        of whether they were used to impose a mass constraint on the system.

        Returns:
            Number density of elements in the condensed phase
        """

    @abstractmethod
    def element_density_dissolved(self) -> Array:
        """Gets the number density of elements dissolved in melt due to species solubility

        Unlike for the objective function, we want the number density of all elements, regardless
        of whether they were used to impose a mass constraint on the system.

        Returns:
            Number density of elements dissolved in melt due to species solubility
        """

    @abstractmethod
    def element_density_gas(self) -> Array:
        """Gets the number density of elements in the gas phase

        Unlike for the objective function, we want the number density of all elements, regardless
        of whether they were used to impose a mass constraint on the system.

        Returns:
            Number density of elements in the gas phase
        """

    @abstractmethod
    def pressure(self) -> Array:
        """Gets pressure of species in bar

        This will compute pressure of all species, including condensates, for simplicity.

        Returns:
            Pressure of species in bar
        """

    @abstractmethod
    def species_density_in_melt(self) -> Array:
        """Gets number density of species dissolved in melt due to species solubility

        Returns:
            Number density of species dissolved in melt
        """

    @abstractmethod
    def _log_activity_without_stability(self) -> Array:
        """Gets log activity without stability of all species

        Args:
            Log activity without stability of all species
        """

    # TODO: Remove. I think not required because we zero the formula matrix instead
    # @property
    # def condensed_molar_mass(self) -> Array:
    #     """Molar mass of the condensed species as a 1-D array"""
    #     return jnp.take(self.molar_mass, self.condensed_species_indices)

    @property
    def condensed_species_indices(self) -> Array:
        """Condensed species indices as a 1-D array"""
        return jnp.array(self._interior_atmosphere.get_condensed_species_indices(), dtype=int)

    @property
    def fixed_parameters(self) -> FixedParameters:
        """Fixed parameters"""
        return self._interior_atmosphere.fixed_parameters

    # TODO: Remove. I think not required because we zero the formula matrix instead
    # @property
    # def gas_molar_mass(self) -> Array:
    #     """Molar mass of the gas species as a 1-D array"""
    #     return jnp.take(self.molar_mass, self.gas_species_indices)

    @property
    def gas_species_indices(self) -> Array:
        """Gas species indices as a 1-D array"""
        return jnp.array(self.fixed_parameters.gas_species_indices, dtype=int)

    @property
    def log_number_density(self) -> Array:
        """Log number density"""
        return self._log_number_density

    # TODO: Remove. I think not required because we zero the formula matrix instead
    # @property
    # def log_number_density_condensed_species(self) -> Array:
    #     """Log number density of condensed species"""
    #     return jnp.take(self.log_number_density, self.condensed_species_indices, axis=1)

    # TODO: Remove. I think not required because we zero the formula matrix instead
    # @property
    # def log_number_density_gas_species(self) -> Array:
    #     """Log number density of gas species"""
    #     return jnp.take(self.log_number_density, self.gas_species_indices, axis=1)

    @property
    def log_stability(self) -> Array:
        """Log stability of all species"""
        return self._log_stability

    @property
    def molar_mass(self) -> Array:
        """Gets molar mass of all species as 1-D array"""
        return jnp.array(self.fixed_parameters.molar_masses)

    @property
    def number_solutions(self) -> int:
        """Number of solutions"""
        return self.log_number_density.shape[0]

    @property
    def planet(self) -> Planet:
        """Planet"""
        return self.traced_parameters.planet

    @property
    def species(self) -> tuple[Species, ...]:
        """Species"""
        return self._interior_atmosphere.species

    @property
    def temperature(self) -> ArrayLike:
        """Temperature"""
        return self.planet.surface_temperature

    @property
    def traced_parameters(self) -> TracedParameters:
        """Traced parameters"""
        return self._traced_parameters

    def activity(self) -> Array:
        """Gets the activity of all species

        Returns:
            Activity of all species
        """
        return jnp.exp(self.log_activity())

    def atmosphere_asdict(self) -> dict[str, ArrayLike]:
        """Gets the atmosphere properties

        Returns:
            Atmosphere properties
        """
        out: dict[str, ArrayLike] = {}
        out["pressure"] = self.atmosphere_pressure()
        out["temperature"] = self.temperature
        out["volume"] = self.atmosphere_volume()

        return collapse_single_entry_values(out)

    def atmosphere_molar_mass(self) -> Array:
        """Gets the molar mass of the atmosphere

        Returns:
            Molar mass of the atmosphere
        """
        return jnp.exp(self.atmosphere_log_molar_mass())

    def atmosphere_volume(self) -> Array:
        """Gets the volume of the atmosphere

        Returns:
            Volume of the atmosphere
        """
        return jnp.exp(self.atmosphere_log_volume())

    def formula_matrix_condensed(self) -> Array:
        """Formula matrix for condensed species

        Only columns in the formula matrix referring to condensed species have non-zero values.

        Returns:
            Formula matrix for condensed species
        """
        formula_matrix: Array = jnp.array(self.fixed_parameters.formula_matrix)
        mask: Array = jnp.zeros_like(formula_matrix, dtype=bool)
        mask = mask.at[:, self.condensed_species_indices].set(True)
        logger.debug("formula_matrix_condensed mask = %s", mask)
        formula_matrix_condensed: Array = formula_matrix * mask

        return formula_matrix_condensed

    def formula_matrix_gas(self) -> Array:
        """Formula matrix for gas species

        Only columns in the formula matrix referring to gas species have non-zero values.

        Returns:
            Formula matrix for gas species
        """
        formula_matrix: Array = jnp.array(self.fixed_parameters.formula_matrix)
        mask: Array = jnp.zeros_like(formula_matrix, dtype=bool)
        mask = mask.at[:, self.gas_species_indices].set(True)
        logger.debug("formula_matrix_gas mask = %s", mask)
        formula_matrix_gas: Array = formula_matrix * mask

        return formula_matrix_gas

    def get_element_density_output(
        self, element_density: ArrayLike, prefix: str = ""
    ) -> dict[str, ArrayLike]:
        """Gets the outputs associated with an element density

        Args:
            element_density: Element density
            prefix: Key prefix for the output. Defaults to an empty string.

        Returns
            Dictionary of output quantities
        """
        atmosphere_volume: Array = self.atmosphere_volume()
        element_molar_mass_expanded: Array = self.element_molar_mass_expanded()
        molecules: ArrayLike = element_density * atmosphere_volume
        moles: ArrayLike = molecules / AVOGADRO
        mass: ArrayLike = moles * element_molar_mass_expanded

        out: dict[str, ArrayLike] = {}
        out[f"{prefix}_number_density"] = element_density
        out[f"{prefix}_molecules"] = molecules
        out[f"{prefix}_moles"] = moles
        out[f"{prefix}_mass"] = mass

        return out

    def elements_asdict(self) -> dict[str, dict[str, ArrayLike]]:
        """Gets the element properties as a dictionary

        Returns:
            Element outputs as a dictionary
        """
        atmosphere: Array = self.element_density_gas()
        condensed: Array = self.element_density_condensed()
        dissolved: Array = self.element_density_dissolved()
        total: Array = atmosphere + condensed + dissolved

        out: dict[str, ArrayLike] = self.get_element_density_output(atmosphere, "atmosphere")
        out |= self.get_element_density_output(condensed, "condensed")
        out |= self.get_element_density_output(dissolved, "dissolved")
        out |= self.get_element_density_output(total, "total")

        out["molar_mass"] = self.element_molar_mass_expanded()
        out["degree_of_condensation"] = out["condensed_molecules"] / out["total_molecules"]
        out["volume_mixing_ratio"] = out["atmosphere_molecules"] / jnp.sum(
            out["atmosphere_molecules"]
        )

        # TODO: Add logarithmic abundance, volume mixing ratio, degree of condensation

        logger.debug("out = %s", out)

        # TODO: generally seems like a bad idea.  Elements are in columns for element_density and
        # not rows;
        # for nn, element in enumerate(unique_elements):
        #     # logarithmic abundance
        #     # volume_mixing_ratio
        #     # molar_mass
        #     element_dict: dict[str, ArrayLike] = {}
        #     # TODO: Split between gas and condensed
        #     element_dict["number_density"] = element_density[nn]
        #     # element_dict["number"] = (
        #     #    element_dict["number_density"] * atmosphere_volume[:, jnp.newaxis]
        #     # )
        #     element_dict["dissolved_number_density"] = element_density_dissolved[nn]
        #     # element_dict["dissolved_number"] = (
        #     #    element_dict["dissolved_number_density"] * atmosphere_volume[:, jnp.newaxis]
        #     # )
        #     element_dict["molar_mass"] = Formula(element).mass
        #     out[element] = collapse_single_entry_values(element_dict)

        return out

    def element_molar_mass_expanded(self) -> Array:
        unique_elements: tuple[str, ...] = (
            self._interior_atmosphere.get_unique_elements_in_species()
        )
        molar_mass: Array = jnp.array([Formula(element).mass for element in unique_elements])
        molar_mass = unit_conversion.g_to_kg * molar_mass

        return jnp.tile(molar_mass, (self.number_solutions, 1))

    def log_activity(self) -> Array:
        """Gets log activity of all species.

        This is usually what the user wants when referring to activity because it includes a
        consideration of species stability

        Returns:
            Log activity of all species
        """
        log_activity_without_stability: Array = self._log_activity_without_stability()
        log_activity: Array = log_activity_without_stability - jnp.exp(self.log_stability)

        return log_activity

    def molar_mass_expanded(self) -> Array:
        r"""Gets molar mass of all species in an expanded array.

        Returns:
            Molar mass of all species in an expanded array.
        """
        return jnp.tile(self.molar_mass, (self.number_solutions, 1))

    def number_density(self) -> Array:
        r"""Gets number density of all species

        Returns:
            Number density in :math:`\mathrm{molecules}\, \mathrm{m}^{-3}`
        """
        return jnp.exp(self.log_number_density)

    def planet_asdict(self) -> dict[str, ArrayLike]:
        """Gets the planet properties as a dictionary

        Returns:
            Planet properties as a dictionary
        """
        return collapse_single_entry_values(self.planet.expanded_asdict())

    # TODO: Might need a general function to deal with difference in indexing between single and
    # batch cases
    # def planet_asdataframe(self) -> pd.DataFrame:
    #     """Gets the planet properties as a dataframe

    #     Returns:
    #         Planet properties as a dataframe
    #     """
    #     return pd.DataFrame(self.planet_asdict())

    def quick_look(self) -> dict[str, ArrayLike]:
        """Quick look at the solution

        Provides a quick first glance at the output with convenient units and to ease comparison
        with test or benchmark data.

        Returns:
            Dictionary of the solution
        """
        out: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self.species):
            pressure: Array = self.pressure()[:, nn]
            activity: Array = self.activity()[:, nn]
            out[species_.name] = pressure
            out[f"{species_.name}_activity"] = activity

        return collapse_single_entry_values(out)

    def stability(self) -> Array:
        """Gets stability of all species

        Returns:
            Stability of all the species
        """
        return jnp.exp(self.log_stability)

    def output_to_logger(self) -> None:
        """Writes output to the logger.

        Useful for debugging.
        """
        logger.info("log_number_density = %s", self.log_number_density)
        logger.info("number_density = %s", self.number_density())
        # logger.info("log_stability = %s", self.log_stability)
        # logger.info("stability = %s", self.stability())
        logger.info("pressure = %s", self.pressure())
        # logger.info("log_activity = %s", self.log_activity())
        logger.info("activity = %s", self.activity())
        logger.info("molar_mass = %s", self.molar_mass)
        logger.info("molar_mass_expanded = %s", self.molar_mass_expanded())
        logger.info("atmosphere_molar_mass = %s", self.atmosphere_molar_mass())
        logger.info("atmosphere_pressure = %s", self.atmosphere_pressure())
        logger.info("atmosphere_volume = %s", self.atmosphere_volume())
        logger.info("atmosphere_asdict = %s", self.atmosphere_asdict())
        logger.info("planet_asdict = %s", self.planet_asdict())
        # logger.info("planet_asdataframe = %s", self.planet_asdataframe())
        logger.info("species_density_in_melt = %s", self.species_density_in_melt())
        logger.info("element_density_dissolved = %s", self.element_density_dissolved())
        logger.info("element_asdict = %s", self.elements_asdict())
        # logger.info("jnp.ravel(self.log_number_density) = %s", jnp.ravel(self.log_number_density))
        # logger.info(
        #    "jnp.squeeze(self.log_number_density) = %s", jnp.squeeze(self.log_number_density)
        # )

    def _activity_without_stability(self) -> Array:
        """Gets activity without stability of all species

        Returns:
            Activity without stability of all species
        """
        return jnp.exp(self._log_activity_without_stability())


class OutputSingle(Output):
    """Converts single calculation output to user-friendly dimensional output"""

    @override
    def atmosphere_log_molar_mass(self) -> Array:
        return get_atmosphere_log_molar_mass(self.fixed_parameters, self.log_number_density)

    @override
    def atmosphere_log_volume(self) -> Array:
        return get_atmosphere_log_volume(
            self.fixed_parameters,
            self.log_number_density,
            self.planet,
        )

    @override
    def atmosphere_pressure(self) -> Array:
        return get_atmosphere_pressure(
            self.fixed_parameters, self.log_number_density, self.temperature
        )

    @override
    def element_density_condensed(self) -> Array:
        return get_element_density(
            self.formula_matrix_condensed(), jnp.ravel(self.log_number_density)
        )

    @override
    def element_density_dissolved(self) -> Array:
        return get_element_density_in_melt(
            self.traced_parameters,
            self.fixed_parameters,
            jnp.array(self.fixed_parameters.formula_matrix),
            # The function expects 1-D arrays
            jnp.ravel(self.log_number_density),
            jnp.ravel(self.log_activity()),
            jnp.ravel(self.atmosphere_log_volume()),
        )

    @override
    def element_density_gas(self) -> Array:
        return get_element_density(self.formula_matrix_gas(), jnp.ravel(self.log_number_density))

    @override
    def pressure(self) -> Array:
        return get_pressure_from_log_number_density(self.log_number_density, self.temperature)

    @override
    def species_density_in_melt(self) -> Array:
        return get_species_density_in_melt(
            self.traced_parameters,
            self.fixed_parameters,
            # The function expects 1-D arrays
            jnp.ravel(self.log_number_density),
            jnp.ravel(self.log_activity()),
            jnp.ravel(self.atmosphere_log_volume()),
        )

    @override
    def _log_activity_without_stability(self) -> Array:
        return get_log_activity(
            self.traced_parameters,
            self.fixed_parameters,
            self.log_number_density,
        )


class OutputBatch(Output):
    """Converts batch calculation output to user-friendly dimensional output"""

    @property
    def temperature_vmap(self) -> int | None:
        """Axis for temperature vmap"""
        return self._interior_atmosphere.traced_parameters_vmap.planet.surface_temperature  # type: ignore

    @property
    def traced_parameters_vmap(self) -> int | None:
        """Axis for traced parameters vmap"""
        return self._interior_atmosphere.traced_parameters_vmap  # type: ignore

    @override
    def atmosphere_log_molar_mass(self) -> Array:
        atmosphere_log_molar_mass_func: Callable = jax.vmap(
            get_atmosphere_log_molar_mass, in_axes=(None, 0)
        )
        atmosphere_log_molar_mass: Array = atmosphere_log_molar_mass_func(
            self.fixed_parameters, self.log_number_density
        )

        return atmosphere_log_molar_mass

    @override
    def atmosphere_log_volume(self) -> Array:
        atmosphere_log_volume_func: Callable = jax.vmap(
            get_atmosphere_log_volume,
            in_axes=(None, 0, self._interior_atmosphere.planet.vmap_axes()),
        )
        atmosphere_log_volume: Array = atmosphere_log_volume_func(
            self.fixed_parameters,
            self.log_number_density,
            self._interior_atmosphere.planet,
        )

        return atmosphere_log_volume[:, jnp.newaxis]  # Column vector for subsequent calculations

    @override
    def atmosphere_pressure(self) -> Array:
        atmosphere_pressure_func: Callable = jax.vmap(
            get_atmosphere_pressure, in_axes=(None, 0, self.temperature_vmap)
        )
        atmosphere_pressure: Array = atmosphere_pressure_func(
            self.fixed_parameters, self.log_number_density, self.temperature
        )

        return atmosphere_pressure

    @override
    def element_density_condensed(self) -> Array:
        element_density_func: Callable = jax.vmap(get_element_density, in_axes=(None, 0))
        element_density: Array = element_density_func(
            self.formula_matrix_condensed(), self.log_number_density
        )

        return element_density

    @override
    def element_density_dissolved(self) -> Array:
        element_density_dissolved_func: Callable = jax.vmap(
            get_element_density_in_melt,
            in_axes=(self.traced_parameters_vmap, None, None, 0, 0, 0),
        )
        element_density_dissolved: Array = element_density_dissolved_func(
            self.traced_parameters,
            self.fixed_parameters,
            jnp.array(self.fixed_parameters.formula_matrix),
            self.log_number_density,
            self.log_activity(),
            self.atmosphere_log_volume(),
        )

        return element_density_dissolved

    @override
    def element_density_gas(self) -> Array:
        element_density_func: Callable = jax.vmap(get_element_density, in_axes=(None, 0))
        element_density: Array = element_density_func(
            self.formula_matrix_gas(), self.log_number_density
        )

        return element_density

    @override
    def pressure(self) -> Array:
        pressure_func: Callable = jax.vmap(
            get_pressure_from_log_number_density, in_axes=(0, self.temperature_vmap)
        )
        pressure: Array = pressure_func(self.log_number_density, self.temperature)

        return pressure

    @override
    def species_density_in_melt(self) -> Array:
        species_density_in_melt_func: Callable = jax.vmap(
            get_species_density_in_melt,
            in_axes=(self.traced_parameters_vmap, None, 0, 0, 0),
        )
        species_density_in_melt: Array = species_density_in_melt_func(
            self.traced_parameters,
            self.fixed_parameters,
            self.log_number_density,
            self.pressure(),
            self.atmosphere_log_volume(),
        )

        return species_density_in_melt

    @override
    def _log_activity_without_stability(self) -> Array:
        log_activity_func: Callable = jax.vmap(
            get_log_activity, in_axes=(self._interior_atmosphere.traced_parameters_vmap, None, 0)
        )
        log_activity: Array = log_activity_func(
            self.traced_parameters, self.fixed_parameters, self.log_number_density
        )

        return log_activity


def collapse_single_entry_values(input_dict: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
    """Collapses single entry values in a dictionary

    Args:
        input_dict: Input dictionary

    Returns:
        Dictionary with collapsed values
    """
    out: dict[str, ArrayLike] = {}
    for key, value in input_dict.items():
        try:
            if value.size > 1:  # type: ignore because AttributeError dealt with
                out[key] = jnp.squeeze(value)
            else:
                out[key] = value.item()  # type:ignore because AttributeError dealt with
        except AttributeError:
            out[key] = value

    return out


# TODO: Removing this old class into single and batch subclasses
# class Output(OutputBatch):
#     """Converts the array output to user-friendly scaled output"""

# def molar_mass(self) -> Array:
#     r"""Gets molar mass of all species

#     Returns:
#         Molar mass of all species
#     """
#     return jnp.tile(
#         jnp.array(self.model.fixed_parameters.molar_masses), (self.number_solutions, 1)
#     )

# def to_dataframes(self) -> dict[str, pd.DataFrame]:
#     out: dict[str, pd.DataFrame] = {}

#     # TODO: Split loop over gas species and condensed to only output relevant quantities
#     for nn, species_ in enumerate(self.model.species):
#         species_out: dict[str, ArrayLike] = {}
#         species_out["atmosphere_number_density"] = self.number_density()[:, nn]
#         species_out["pressure"] = self.pressure()[:, nn]
#         species_out["activity"] = self.activity()[:, nn]
#         species_out["fugacity_coefficient"] = species_out["activity"] / species_out["pressure"]
#         out[species_.name] = pd.DataFrame(species_out)

#     print(out)

#     return out

# TODO: Old output class continues below. To keep until development of the new class complete.

# @classmethod
# def read_pickle(cls, pickle_file: Path | str) -> Output:
#     """Reads output data from a pickle file and creates an Output instance.

#     Args:
#         pickle_file: Pickle file of the output from a previous (or similar) model run.
#             Importantly, the reaction network must be the same (same number of species in the
#             same order) and the constraints must be the same (also in the same order).

#     Returns:
#         Output
#     """
#     with open(pickle_file, "rb") as handle:
#         output_data: dict[str, list[dict[str, float]]] = pickle.load(handle)

#     logger.info("%s: Reading data from %s", cls.__name__, pickle_file)

#     return cls(output_data)

# @classmethod
# def from_dataframes(cls, dataframes: dict[str, pd.DataFrame]) -> Output:
#     """Reads a dictionary of dataframes and creates an Output instance.

#     Args:
#         dataframes: A dictionary of dataframes.

#     Returns:
#         Output
#     """
#     output_data: dict[str, list[dict[Hashable, float]]] = {}
#     for key, dataframe in dataframes.items():
#         output_data[key] = dataframe.to_dict(orient="records")

#     return cls(output_data)

# def add(
#     self,
#     solution: Solution,
#     residual_dict: dict[str, float],
#     constraints_dict: dict[str, float],
#     extra_output: dict[str, float] | None = None,
# ) -> None:
#     """Adds all outputs.

#     Args:
#         solution: Solution
#         residual_dict: Dictionary of residuals
#         constraints_dict: Dictionary of constraints
#         extra_output: Extra data to write to the output. Defaults to None.
#     """
#     output_full: dict[str, dict[str, float]] = solution.output_full()

#     # Back-compute and add the log10 shift relative to the default iron-wustite buffer
#     if "O2_g" in output_full:
#         temperature: float = output_full["atmosphere"]["temperature"]
#         pressure: float = output_full["atmosphere"]["pressure"]
#         # pylint: disable=invalid-name
#         O2_g_output: dict[str, float] = output_full["O2_g"]
#         O2_g_fugacity: float = O2_g_output["fugacity"]
#         O2_g_shift_at_1bar: float = solve_for_log10_dIW(O2_g_fugacity, temperature)
#         O2_g_output["log10dIW_1_bar"] = O2_g_shift_at_1bar
#         O2_g_shift_at_P: float = solve_for_log10_dIW(O2_g_fugacity, temperature, pressure)
#         O2_g_output["log10dIW_P"] = O2_g_shift_at_P

#     for key, value in output_full.items():
#         data_list: list[dict[str, float]] = self.data.setdefault(key, [])
#         data_list.append(value)

#     constraints_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
#     constraints_list.append(constraints_dict)
#     residual_list: list[dict[str, float]] = self.data.setdefault("residual", [])
#     residual_list.append(residual_dict)

#     if extra_output is not None:
#         data_list: list[dict[str, float]] = self.data.setdefault("extra", [])
#         data_list.append(extra_output)

# def to_dataframes(self) -> dict[str, pd.DataFrame]:
#     """Output as a dictionary of dataframes

#     Returns:
#         The output as a dictionary of dataframes
#     """
#     out: dict[str, pd.DataFrame] = {
#         key: pd.DataFrame(value) for key, value in self.data.items()
#     }
#     return out

# def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
#     """Writes the output to an Excel file.

#     Args:
#         file_prefix: Prefix of the output file. Defaults to atmodeller_out.
#     """
#     out: dict[str, pd.DataFrame] = self.to_dataframes()
#     output_file: Path = Path(f"{file_prefix}.xlsx")

#     with pd.ExcelWriter(output_file, engine="openpyxl") as writer:  # pylint: disable=E0110
#         for df_name, df in out.items():
#             df.to_excel(writer, sheet_name=df_name, index=True)

#     logger.info("Output written to %s", output_file)

# def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
#     """Writes the output to a pickle file.

#     Args:
#         file_prefix: Prefix of the output file. Defaults to atmodeller_out.
#     """
#     output_file: Path = Path(f"{file_prefix}.pkl")

#     with open(output_file, "wb") as handle:
#         pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     logger.info("Output written to %s", output_file)

# def _check_keys_the_same(self, other: Output) -> None:
#     """Checks if the keys are the same in 'other' before combining output.

#     Args:
#         other: Other output to potentially combine (if keys are the same)
#     """
#     if not self.keys() == other.keys():
#         msg: str = "Keys for 'other' are not the same as 'self' so cannot combine them"
#         logger.error(msg)
#         raise KeyError(msg)

# def __add__(self, other: Output) -> Output:
#     """Addition

#     Args:
#         other: Other output to combine with self

#     Returns:
#         Combined output
#     """
#     self._check_keys_the_same(other)
#     output: Output = copy.deepcopy(self)
#     for key in self.keys():
#         output[key].extend(other[key])

#     return output

# def __iadd__(self, other: Output) -> Output:
#     """In-place addition

#     Args:
#         other: Other output to combine with self in-place

#     Returns:
#         self
#     """
#     self._check_keys_the_same(other)
#     for key in self:
#         self[key].extend(other[key])

#     return self

# def filter_by_index_notin(self, other: Output, index_key: str, index_name: str) -> Output:
#     """Filters out the entries in `self` that are not present in the index of `other`

#     Args:
#         other: Other output with the filtering index
#         index_key: Key of the index
#         index_name: Name of the index

#     Returns:
#         The filtered output
#     """
#     self_dataframes: dict[str, pd.DataFrame] = self.to_dataframes()
#     other_dataframes: dict[str, pd.DataFrame] = other.to_dataframes()
#     index: pd.Index = pd.Index(other_dataframes[index_key][index_name])

#     for key, dataframe in self_dataframes.items():
#         self_dataframes[key] = dataframe[~dataframe.index.isin(index)]

#     return self.from_dataframes(self_dataframes)

# def reorder(self, other: Output, index_key: str, index_name: str) -> Output:
#     """Reorders all the entries according to an index in `other`

#     Args:
#         other: Other output with the reordering index
#         index_key: Key of the index
#         index_name: Name of the index

#     Returns:
#         The reordered output
#     """
#     self_dataframes: dict[str, pd.DataFrame] = self.to_dataframes()
#     other_dataframes: dict[str, pd.DataFrame] = other.to_dataframes()
#     index: pd.Index = pd.Index(other_dataframes[index_key][index_name])

#     for key, dataframe in self_dataframes.items():
#         self_dataframes[key] = dataframe.reindex(index)

#     return self.from_dataframes(self_dataframes)

# def __call__(
#     self,
#     file_prefix: Path | str = "atmodeller_out",
#     to_dataframes: bool = True,
#     to_pickle: bool = False,
#     to_excel: bool = False,
# ) -> dict | None:
#     """Gets the output as a dict and/or optionally write it to a pickle or Excel file.

#     Args:
#         file_prefix: Prefix of the output file if writing to a pickle or Excel. Defaults to
#             atmodeller_out
#         to_dataframes: Returns the output data in a dictionary of dataframes. Defaults to
#             True.
#         to_pickle: Writes a pickle file. Defaults to False.
#         to_excel: Writes an Excel file. Defaults to False.

#     Returns:
#         A dictionary of the output or None if no data
#     """
#     if self.size == 0:
#         logger.warning("There is no data to export")
#         return None

#     if to_pickle:
#         self.to_pickle(file_prefix)

#     if to_excel:
#         self.to_excel(file_prefix)

#     if to_dataframes:
#         return self.to_dataframes()
#     else:
#         return self.data
