# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
from pprint import pformat
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxmod.solvers import MultiAttemptSolution
from jaxtyping import Array, Float

from atmodeller.engine import get_log_activity, get_total_pressure
from atmodeller.parameters import Parameters
from atmodeller.type_aliases import NpArray, NpBool

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO
# @eqx.filter_jit
# def get_melt_log_mole_fraction(
#     parameters: Parameters, solution: Float[Array, "... solution"]
# ) -> Float[Array, "... n_species"]:
#     """Gets melt log mole fraction.

#     Args:
#         parameters: Parameters
#         solution: Solution array

#     Returns:
#         Melt log mole fraction
#     """
#     log_number_moles, _ = jnp.split(solution, 2, axis=-1)
#     melt_slice: slice = parameters.reaction_system.melt_slice
#     melt_log_mole_fraction: Float[Array, "... n_species"] = (
#         parameters.reaction_system.melt.get_log_mole_fraction(log_number_moles[..., melt_slice])
#     )

#     return melt_log_mole_fraction


def _flatten_dict(d: dict, parent_keys: tuple = ()) -> dict[tuple, Any]:
    """Recursively flattens a nested dict to {path_tuple: leaf} mapping."""
    items: dict = {}
    for k, v in d.items():
        path = parent_keys + (k,)
        if isinstance(v, dict):
            items.update(_flatten_dict(v, path))
        else:
            items[path] = v

    return items


def _set_nested(d: dict, path: tuple, value: Any) -> None:
    """Sets a value in a nested dict given a path tuple, creating intermediate dicts."""
    for key in path[:-1]:
        d = d.setdefault(key, {})

    d[path[-1]] = value


class Output(eqx.Module):
    """Output

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    parameters: Parameters
    multi_attempt_solution: MultiAttemptSolution

    def __init__(self, parameters: Parameters, multi_attempt_solution: MultiAttemptSolution):
        self.parameters = parameters
        self.multi_attempt_solution = multi_attempt_solution

        # Caching output to avoid recomputation
        # self._cached_dict: Optional[dict[str, dict[str, NpArray]]] = None
        # self._cached_dataframes: Optional[dict[str, pd.DataFrame]] = None

    @property
    def log_number_moles(self) -> Float[Array, "... n_species"]:
        """Log number of moles for each species"""
        log_number_moles, _ = jnp.split(self.multi_attempt_solution.value, 2, axis=-1)

        return log_number_moles

    @property
    def log_stability(self) -> Float[Array, "... n_species"]:
        """Log stability for each species"""
        _, log_stability = jnp.split(self.multi_attempt_solution.value, 2, axis=-1)

        active_stability: NpBool = self.parameters.reaction_system.species.active_stability
        log_stability = jnp.where(active_stability, log_stability, -jnp.inf)
        logger.debug("Log stability = %s", log_stability)

        return log_stability

    @property
    def solution(self) -> Float[Array, "... twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.multi_attempt_solution.value

    def get_gas_log_mole_fraction(self) -> Float[Array, "... n_gas_species"]:
        """Gets gas log mole fraction.

        Returns:
            Gas log mole fraction
        """
        gas_slice: slice = self.parameters.reaction_system.gas_slice
        gas_log_mole_fraction: Float[Array, "... n_species"] = (
            self.parameters.reaction_system.gas.get_log_mole_fraction(
                self.log_number_moles[..., gas_slice]
            )
        )

        return gas_log_mole_fraction

    def get_gas_log_partial_pressure(self) -> Float[Array, "... n_gas_species"]:
        """Gets gas log partial pressure.

        Returns:
            Gas log partial pressure
        """
        total_pressure_log: Float[Array, "..."] = jnp.log(
            get_total_pressure(self.parameters, self.solution)
        )
        gas_log_partial_pressure: Float[Array, "... n_species"] = (
            self.get_gas_log_mole_fraction() + total_pressure_log[..., None]
        )

        return gas_log_partial_pressure

    def get_log_activity_with_stability(self) -> Float[Array, "... n_species"]:
        """Gets the log activity of each species, including stability effects.

        Returns:
            Log activity of each species, including stability effects
        """
        log_activity_with_stability: Float[Array, "... n_species"] = get_log_activity(
            self.parameters, self.solution
        ) - jnp.exp(self.log_stability)

        return log_activity_with_stability

    def get_species_mass(self) -> Float[Array, "... n_species"]:
        """Gets the mass of each species.

        Returns:
            Log mass of each species in kg
        """
        log_molar_mass: Float[Array, " n_species"] = jnp.log(
            self.parameters.reaction_system.species.molar_masses
        )
        log_species_mass: Float[Array, "... n_species"] = self.log_number_moles + log_molar_mass

        return jnp.exp(log_species_mass)

    def quick_look(self) -> dict[str, Any]:
        """Quick look at the solution

        Returns:
            Dictionary of the solution
        """
        gas_names: tuple[str, ...] = self.parameters.reaction_system.gas.species.species_names
        gas_slice: slice = self.parameters.reaction_system.gas_slice

        # All computation in JAX
        gas_log_partial_pressure = self.get_gas_log_partial_pressure()
        gas_log_mole_fraction = self.get_gas_log_mole_fraction()
        total_pressure = get_total_pressure(self.parameters, self.solution)
        gas_log_mass = self.parameters.reaction_system.gas.get_log_phase_mass(
            self.log_number_moles[..., gas_slice]
        )

        # condensates_dict: dict[str, Any] = {}
        condensate_names: list[str] = [
            condensate.name for condensate in self.parameters.reaction_system.condensates
        ]
        condensate_slice: slice = self.parameters.reaction_system.condensates_slice

        # Single conversion boundary: JAX -> NumPy -> dict
        out: dict[str, Any] = {
            "gas": {
                "partial_pressure_bar": dict(zip(gas_names, np.exp(gas_log_partial_pressure).T)),
                "number_moles": dict(
                    zip(gas_names, np.exp(self.log_number_moles[..., gas_slice]).T)
                ),
                "mole_fraction": dict(zip(gas_names, np.exp(gas_log_mole_fraction).T)),
                "pressure_bar": np.asarray(total_pressure),
                "mass_kg": np.squeeze(np.exp(gas_log_mass)),
                "fugacity_bar": dict(
                    zip(
                        gas_names,
                        np.exp(self.get_log_activity_with_stability()[..., gas_slice]).T,
                    )
                ),
            },
            "melt": {},
            "solid": {},
            "condensates": {
                "activity": dict(
                    zip(
                        condensate_names,
                        np.exp(self.get_log_activity_with_stability()[..., condensate_slice]).T,
                    )
                ),
                "number_moles": dict(
                    zip(
                        condensate_names,
                        np.exp(self.log_number_moles[..., condensate_slice]).T,
                    )
                ),
                "mass_kg": dict(
                    zip(
                        condensate_names,
                        self.get_species_mass()[..., condensate_slice].T,
                    )
                ),
            },
        }

        logger.info(f"Quick look output:\n{pformat(out)}")

        return out

    def compare(self, target: dict[str, Any], rtol: float, atol: float, log: bool = False) -> bool:
        """Compares matching keys in target to quick_look output.

        Only keys present in both target and quick_look are compared. Keys present in one but not
        the other are ignored.

        Args:
            target: Nested dictionary with the same structure as quick_look output
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            log: Compare closeness in log10-space. Defaults to ``False``.

        Returns:
            True if all matching keys agree within tolerance
        """
        current_map: dict[tuple, Any] = _flatten_dict(self.quick_look())
        target_map: dict[tuple, Any] = _flatten_dict(target)

        result: dict[str, Any] = {}
        all_match: bool = True

        for path, target_value in target_map.items():
            if path not in current_map:
                continue
            a: NpArray = np.atleast_1d(current_map[path])
            b: NpArray = np.atleast_1d(target_value)
            if log:
                a, b = np.log10(a), np.log10(b)

            match: bool = bool(np.allclose(a, b, rtol=rtol, atol=atol))
            _set_nested(result, path, match)
            all_match = all_match and match

        logger.info(f"\nComparison result:\n{pformat(result)}")
        logger.info(f"All matching keys agree within tolerance: {all_match}")

        return all_match
