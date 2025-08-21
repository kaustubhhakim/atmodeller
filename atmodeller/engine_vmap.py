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
"""Vmapped wrappers for core engine functions.

This module provides a high-level container (:class:`VmappedFunctions`) that precompiles
vectorised versions of key thermodynamic and mass-balance functions. By wrapping each function with
:func:`equinox.filter_vmap`, the module ensures efficient batched evaluation of model properties.

Currently, these wrappers are used primarily as a convenience for generating and inspecting
outputs. They are not responsible for performing the actual equilibrium solution, which is instead
handled by the :mod:`~atmodeller.solvers` module.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
from jaxtyping import Array, ArrayLike

from atmodeller.containers import Parameters
from atmodeller.engine import (
    get_atmosphere_log_molar_mass,
    get_atmosphere_log_volume,
    get_element_density,
    get_element_density_in_melt,
    get_log_activity,
    get_pressure_from_log_number_density,
    get_reactions_only_mask,
    get_species_density_in_melt,
    get_species_ppmw_in_melt,
    get_total_pressure,
    objective_function,
)
from atmodeller.solvers import LOG_NUMBER_DENSITY_VMAP_AXES, vmap_axes_spec
from atmodeller.utilities import get_log_number_density_from_log_pressure


@dataclass
class VmappedFunctions:
    """Container for precompiled ``vmap``-ped model functions.

    This class wraps a set of model functions (e.g., thermodynamic property calculations, reaction
    masks, etc.) with :func:`equinox.filter_vmap` so they can be evaluated efficiently over batched
    inputs.

    The primary assumption is that ``log_number_density`` inputs are already batched along axis 0.
    The ``in_axes`` specifications for all ``vmap`` calls are precomputed at initialisation from
    the provided ``parameters`` object, ensuring consistent vectorisation behavior across all
    functions.

    Each wrapped function is stored as a bound method and internally calls a preconstructed
    ``vmap`` object. This minimizes tracing overhead and avoids recomputing ``in_axes`` specs for
    each call.

    Args:
        parameters: Parameters
    """

    parameters: Parameters

    # Precompiled vmapped functions
    _get_atmosphere_log_molar_mass: Callable
    _get_atmosphere_log_volume: Callable
    _get_element_density: Callable
    _get_element_density_in_melt: Callable
    _get_log_activity: Callable
    _get_log_number_density_from_log_pressure: Callable
    _get_pressure_from_log_number_density: Callable
    _get_reactions_only_mask: Callable
    _get_species_density_in_melt: Callable
    _get_species_ppmw_in_melt: Callable
    _get_total_pressure: Callable
    _objective_function_vmap: Callable

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        # Compute axes specs once
        parameters_vmap_axes: Parameters = vmap_axes_spec(parameters)
        temperature_vmap_axes: Literal[0, None] = vmap_axes_spec(parameters.planet.temperature)

        # Pre-build vmap wrappers
        self._get_atmosphere_log_molar_mass = eqx.filter_vmap(
            get_atmosphere_log_molar_mass,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_atmosphere_log_volume = eqx.filter_vmap(
            get_atmosphere_log_volume,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_element_density = eqx.filter_vmap(
            get_element_density,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_element_density_in_melt = eqx.filter_vmap(
            get_element_density_in_melt,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_log_activity = eqx.filter_vmap(
            get_log_activity,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_log_number_density_from_log_pressure = eqx.filter_vmap(
            get_log_number_density_from_log_pressure,
            in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, temperature_vmap_axes),
        )

        self._get_pressure_from_log_number_density = eqx.filter_vmap(
            get_pressure_from_log_number_density,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_reactions_only_mask = eqx.filter_vmap(
            get_reactions_only_mask,
            in_axes=(parameters_vmap_axes,),
        )

        self._get_species_density_in_melt = eqx.filter_vmap(
            get_species_density_in_melt,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_species_ppmw_in_melt = eqx.filter_vmap(
            get_species_ppmw_in_melt,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._get_total_pressure = eqx.filter_vmap(
            get_total_pressure,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_DENSITY_VMAP_AXES),
        )

        self._objective_function_vmap = eqx.filter_vmap(
            objective_function,
            in_axes=(LOG_NUMBER_DENSITY_VMAP_AXES, parameters_vmap_axes),
        )

    def get_atmosphere_log_molar_mass(self, log_number_density: Array) -> Array:
        return self._get_atmosphere_log_molar_mass(self.parameters, log_number_density)

    def get_atmosphere_log_volume(self, log_number_density: Array) -> Array:
        return self._get_atmosphere_log_volume(self.parameters, log_number_density)

    def get_element_density(self, log_number_density: Array) -> Array:
        return self._get_element_density(self.parameters, log_number_density)

    def get_element_density_in_melt(self, log_number_density: Array) -> Array:
        return self._get_element_density_in_melt(self.parameters, log_number_density)

    def get_log_activity(self, log_number_density: Array) -> Array:
        return self._get_log_activity(self.parameters, log_number_density)

    def get_log_number_density_from_log_pressure(
        self, log_pressure: ArrayLike, temperature: ArrayLike
    ) -> Array:
        return self._get_log_number_density_from_log_pressure(log_pressure, temperature)

    def get_pressure_from_log_number_density(self, log_number_density: Array) -> Array:
        return self._get_pressure_from_log_number_density(self.parameters, log_number_density)

    def get_reactions_only_mask(self) -> Array:
        return self._get_reactions_only_mask(self.parameters)

    def get_species_density_in_melt(self, log_number_density: Array) -> Array:
        return self._get_species_density_in_melt(self.parameters, log_number_density)

    def get_species_ppmw_in_melt(self, log_number_density: Array) -> Array:
        return self._get_species_ppmw_in_melt(self.parameters, log_number_density)

    def get_total_pressure(self, log_number_density: Array) -> Array:
        return self._get_total_pressure(self.parameters, log_number_density)

    def objective_function(self, solution: Array) -> Array:
        return self._objective_function_vmap(solution, self.parameters)
