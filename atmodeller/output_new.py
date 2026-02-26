# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
from pprint import pformat
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxmod.solvers import MultiAttemptSolution
from jaxtyping import Array, ArrayLike, Float

from atmodeller.engine import get_total_pressure
from atmodeller.parameters import Parameters
from atmodeller.type_aliases import NpArray, NpBool, NpFloat

logger: logging.Logger = logging.getLogger(__name__)


@eqx.filter_jit
def get_gas_log_mole_fraction(
    parameters: Parameters, solution: Float[Array, "... solution"]
) -> Float[Array, "... n_species"]:
    """Gets gas log mole fraction.

    Args:
        parameters: Parameters
        solution: Solution array

    Returns:
        Gas log mole fraction
    """
    log_number_moles, _ = jnp.split(solution, 2, axis=-1)
    gas_slice: slice = parameters.reaction_system.gas_slice
    gas_log_mole_fraction: Float[Array, "... n_species"] = (
        parameters.reaction_system.gas.get_log_mole_fraction(log_number_moles[..., gas_slice])
    )

    return gas_log_mole_fraction


@eqx.filter_jit
def get_gas_log_partial_pressure(
    parameters: Parameters, solution: Float[Array, "... solution"]
) -> Float[Array, "... n_species"]:
    """Gets gas log partial pressure.

    Args:
        parameters: Parameters
        solution: Solution array

    Returns:
        Gas log partial pressure
    """
    log_number_moles, _ = jnp.split(solution, 2, axis=-1)
    gas_log_mole_fraction: Float[Array, "... n_species"] = get_gas_log_mole_fraction(
        parameters, solution
    )
    total_pressure_log: Float[Array, "..."] = jnp.log(
        get_total_pressure(parameters, log_number_moles)
    )
    gas_log_partial_pressure: Float[Array, "... n_species"] = (
        gas_log_mole_fraction + total_pressure_log[..., None]
    )

    return gas_log_partial_pressure


class Output:
    """Output

    Args:
        parameters: Parameters
        solution: Solution
    """

    def __init__(
        self,
        parameters: Parameters,
        solution: Float[Array, "... solution"],
        multi_attempt_solution: MultiAttemptSolution,
    ):
        logger.debug("Creating Output")
        self.parameters: Parameters = parameters
        self.solution: NpFloat = np.asarray(solution)

        # Compute axes specs once
        # parameters_vmap_axes: Parameters = vmap_axes_spec(parameters)

        # self.get_total_pressure = eqx.filter_vmap(
        #    get_total_pressure,
        #    in_axes=(parameters_vmap_axes, LOG_NUMBER_MOLES_VMAP_AXES),
        # )

        # np.split retains dimensions
        log_number_moles, log_stability = np.split(self.solution, 2, axis=-1)

        self.log_number_moles: NpFloat = log_number_moles  # 2-D
        # Mask stabilities that are not solved for by the model
        active_stability_mask: NpBool = parameters.species.active_stability
        self.log_stability = np.where(active_stability_mask, log_stability, np.nan)

        self.multi_attempt_solution: MultiAttemptSolution = multi_attempt_solution

        # Caching output to avoid recomputation
        self._cached_dict: Optional[dict[str, dict[str, NpArray]]] = None
        self._cached_dataframes: Optional[dict[str, pd.DataFrame]] = None

    def quick_look(self) -> dict[str, ArrayLike | dict[str, dict[str, ArrayLike]]]:
        """Quick look at the solution

        Returns:
            Dictionary of the solution
        """
        out: dict[str, dict[str, ArrayLike | dict[str, ArrayLike]]] = {}
        out["gas"] = {}
        out["melt"] = {}
        out["solid"] = {}

        gas_log_mole_fraction: NpFloat = np.asarray(
            self.parameters.reaction_system.gas.get_log_mole_fraction(
                jnp.asarray(self.log_number_moles[:, self.parameters.reaction_system.gas_slice])
            )
        )

        gas_slice = self.parameters.reaction_system.gas_slice

        out["gas"]["mole_fraction"] = dict(
            zip(
                self.parameters.reaction_system.gas.species.species_names,
                np.exp(gas_log_mole_fraction).T,
            )
        )
        out["gas"]["pressure_bar"] = np.asarray(
            get_total_pressure(self.parameters, jnp.asarray(self.log_number_moles))
        )
        out["gas"]["partial_pressure_bar"] = dict(
            zip(
                self.parameters.reaction_system.gas.species.species_names,
                np.exp(gas_log_mole_fraction).T * out["gas"]["pressure_bar"],
            ),
        )
        out["gas"]["mass"] = np.asarray(
            self.parameters.reaction_system.gas.get_log_phase_mass(
                jnp.asarray(self.log_number_moles[..., gas_slice])
            )
        )

        logger.info(f"\n{pformat(out)}")

        return out
