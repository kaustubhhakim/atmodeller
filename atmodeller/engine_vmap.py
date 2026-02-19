# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

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

import equinox as eqx
from jaxtyping import Array

from atmodeller.engine import (
    # get_element_moles,
    # get_element_moles_in_melt,
    get_log_activity,
    get_reactions_only_mask,
    # get_species_moles_in_melt,
    # get_species_ppmw_in_melt,
    get_total_pressure,
    objective_function,
)
from atmodeller.parameters import Parameters
from atmodeller.solvers import LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec


@dataclass
class VmappedFunctions:
    """Container for precompiled ``vmap``-ped model functions.

    This class wraps a set of model functions (e.g., thermodynamic property calculations, reaction
    masks, etc.) with :func:`equinox.filter_vmap` so they can be evaluated efficiently over batched
    inputs.

    The primary assumption is that ``log_number_moles`` inputs are already batched along axis 0.
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
    _get_element_moles: Callable
    _get_element_moles_in_melt: Callable
    _get_log_activity: Callable
    _get_reactions_only_mask: Callable
    _get_species_moles_in_melt: Callable
    _get_species_ppmw_in_melt: Callable
    _get_total_pressure: Callable
    _objective_function_vmap: Callable

    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        # Compute axes specs once
        parameters_vmap_axes: Parameters = vmap_axes_spec(parameters)

        # self._get_element_moles = eqx.filter_vmap(
        #     get_element_moles,
        #     in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        # )

        # self._get_element_moles_in_melt = eqx.filter_vmap(
        #     get_element_moles_in_melt,
        #     in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        # )

        self._get_log_activity = eqx.filter_vmap(
            get_log_activity,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        )

        self._get_reactions_only_mask = eqx.filter_vmap(
            get_reactions_only_mask,
            in_axes=(parameters_vmap_axes,),
        )

        # self._get_species_moles_in_melt = eqx.filter_vmap(
        #     get_species_moles_in_melt,
        #     in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        # )

        # self._get_species_ppmw_in_melt = eqx.filter_vmap(
        #     get_species_ppmw_in_melt,
        #     in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        # )

        self._get_total_pressure = eqx.filter_vmap(
            get_total_pressure,
            in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        )

        self._objective_function_vmap = eqx.filter_vmap(
            objective_function,
            in_axes=(LOG_NUMBER_MOLES_VMAP_AXES, parameters_vmap_axes),
        )

    def get_element_moles(self, log_number_moles: Array) -> Array:
        return self._get_element_moles(self.parameters, log_number_moles)

    def get_element_moles_in_melt(self, log_number_moles: Array) -> Array:
        return self._get_element_moles_in_melt(self.parameters, log_number_moles)

    def get_log_activity(self, log_number_moles: Array) -> Array:
        return self._get_log_activity(self.parameters, log_number_moles)

    def get_reactions_only_mask(self) -> Array:
        return self._get_reactions_only_mask(self.parameters)

    def get_species_moles_in_melt(self, log_number_moles: Array) -> Array:
        return self._get_species_moles_in_melt(self.parameters, log_number_moles)

    def get_species_ppmw_in_melt(self, log_number_moles: Array) -> Array:
        return self._get_species_ppmw_in_melt(self.parameters, log_number_moles)

    def get_total_pressure(self, log_number_moles: Array) -> Array:
        return self._get_total_pressure(self.parameters, log_number_moles)

    def objective_function(self, solution: Array) -> Array:
        return self._objective_function_vmap(solution, self.parameters)
