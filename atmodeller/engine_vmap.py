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
"""`Vmapped engine for Atmodeller.`"""

from typing import Literal

import equinox as eqx
from jaxtyping import Array

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
from atmodeller.utilities import get_log_number_density_from_log_pressure, vmap_axes_spec


class VmappedFunctions:
    """Container for vmapped functions.

    Args:
        parameters: Parameters
    """

    def __init__(self, parameters: Parameters):
        self.parameters: Parameters = parameters

    @property
    def log_number_density_vmap_axes(self) -> int:
        """Vmap axes of the log number density"""
        return 0

    @property
    def parameters_vmap_axes(self) -> Parameters:
        """Vmap axes of the parameters."""
        return vmap_axes_spec(self.parameters)

    @property
    def temperature_vmap_axes(self) -> Literal[0, None]:
        """Vmap axes of the temperature."""
        return vmap_axes_spec(self.parameters.planet.temperature)

    def get_atmosphere_log_molar_mass(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_atmosphere_log_molar_mass,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_atmosphere_log_volume(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_atmosphere_log_volume,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_element_density(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_element_density,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_element_density_in_melt(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_element_density_in_melt,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_log_activity(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_log_activity,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_log_number_density_from_log_pressure(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_log_number_density_from_log_pressure,
            in_axes=(self.log_number_density_vmap_axes, self.temperature_vmap_axes),
        )(*args, **kwargs)

    def get_pressure_from_log_number_density(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_pressure_from_log_number_density,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_reactions_only_mask(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(get_reactions_only_mask, in_axes=(self.parameters_vmap_axes,))(
            *args, **kwargs
        )

    def get_species_density_in_melt(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_species_density_in_melt,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_species_ppmw_in_melt(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_species_ppmw_in_melt,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def get_total_pressure(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            get_total_pressure,
            in_axes=(self.parameters_vmap_axes, self.log_number_density_vmap_axes),
        )(*args, **kwargs)

    def objective_function(self, *args, **kwargs) -> Array:
        return eqx.filter_vmap(
            objective_function,
            in_axes=(self.log_number_density_vmap_axes, self.parameters_vmap_axes),
        )(*args, **kwargs)
