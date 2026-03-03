# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
import pickle
from collections.abc import Iterable
from pathlib import Path
from pprint import pformat
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxmod.solvers import MultiAttemptSolution
from jaxmod.type_aliases import NpArray, NpBool
from jaxtyping import Array, Float

from atmodeller.engine import get_total_pressure, objective_function
from atmodeller.parameters import Parameters
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


_SUMMABLE_KEYS: frozenset[str] = frozenset({"mass_kg", "number_moles"})


def sum_phase_outputs(phase_outputs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Sums summable quantities across an iterable of phase output dicts.

    Only leaves under a ``"mass_kg"`` or ``"number_moles"`` sub-category are included, preserving
    the ``elements``, ``species``, and ``phase`` sub-structure. Values at the same path are summed;
    ``np.nan`` values are treated as zero so that a phase with unconstrained moles does not
    contaminate the total.

    Args:
        phase_outputs: Iterable of phase output dicts as returned by
            :meth:`~atmodeller.phases.BasePhase.output`.

    Returns:
        A nested dict with the same sub-structure as the inputs but restricted to the summable
        keys, holding the element-wise sum across all phases.
    """
    totals: dict[tuple, Any] = {}

    for phase_out in phase_outputs:
        for path, value in _flatten_dict(phase_out).items():
            if not any(k in _SUMMABLE_KEYS for k in path):
                continue
            scalar = np.asarray(value)
            addend = np.where(np.isnan(scalar), 0.0, scalar)
            totals[path] = totals.get(path, 0.0) + addend

    out: dict[str, Any] = {}
    for path, value in totals.items():
        _set_nested(out, path, value)

    return out


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
        # logger.debug("Log number moles = %s", log_number_moles)

        return log_number_moles

    @property
    def log_stability(self) -> Float[Array, "... n_species"]:
        """Log stability for each species"""
        _, log_stability = jnp.split(self.multi_attempt_solution.value, 2, axis=-1)

        active_stability: NpBool = self.parameters.reaction_system.species.active_stability
        log_stability = jnp.where(active_stability, log_stability, -jnp.inf)
        # logger.debug("Log stability = %s", log_stability)

        return log_stability

    @property
    def condensates(self) -> tuple[PurePhase, ...]:
        return self.parameters.reaction_system.condensates

    @property
    def gas(self) -> GasPhase:
        """Gas phase output"""
        return self.parameters.reaction_system.gas

    @property
    def melt(self) -> MeltPhase:
        """Melt phase output"""
        return self.parameters.reaction_system.melt

    @property
    def solid(self) -> SolidPhase:
        """Solid phase output"""
        return self.parameters.reaction_system.solid

    @property
    def solution(self) -> Float[Array, "... twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.multi_attempt_solution.value

    def quick_look(self) -> dict[str, Any]:
        """Quick look at the solution

        Returns:
            Dictionary of the solution
        """
        gas_slice: slice = self.parameters.reaction_system.gas_slice
        melt_slice: slice = self.parameters.reaction_system.melt_slice
        solid_slice: slice = self.parameters.reaction_system.solid_slice

        temperature = self.parameters.state.temperature
        total_pressure = get_total_pressure(self.parameters, self.solution)

        out: dict[str, Any] = {}

        # No background component for gas, so no need to pass log_background_molar_mass or
        # log_background_melt_mass
        out["gas"] = self.gas.output(
            self.log_number_moles[..., gas_slice],
            self.log_stability[..., gas_slice],
            temperature,
            total_pressure,
        )

        # Background component for melt
        log_background_molar_mass = jnp.log(self.parameters.state.molar_mass)
        log_background_melt_mass = jnp.log(self.parameters.state.melt_mass)

        out["melt"] = self.melt.output(
            self.log_number_moles[..., melt_slice],
            self.log_stability[..., melt_slice],
            temperature,
            total_pressure,
            log_background_molar_mass,
            log_background_melt_mass,
        )

        # Background component for solid
        log_background_molar_mass = jnp.log(self.parameters.state.molar_mass)
        log_background_solid_mass = jnp.log(self.parameters.state.solid_mass)

        out["solid"] = self.solid.output(
            self.log_number_moles[..., solid_slice],
            self.log_stability[..., solid_slice],
            temperature,
            total_pressure,
            log_background_molar_mass,
            log_background_solid_mass,
        )

        # Condensates
        condensate_names: list[str] = [
            condensate.name for condensate in self.parameters.reaction_system.condensates
        ]
        condensate_slice: slice = self.parameters.reaction_system.condensates_slice

        out_condensates: list[dict[str, Any]] = []

        for nn, condensate in enumerate(self.condensates):
            out_condensates.append(
                condensate.output(
                    jnp.atleast_1d(self.log_number_moles[..., condensate_slice][nn]),
                    jnp.atleast_1d(self.log_stability[..., condensate_slice][nn]),
                    temperature,
                    total_pressure,
                )
            )

        out["condensates"] = dict(zip(condensate_names, out_condensates))

        totals: dict[str, Any] = sum_phase_outputs(
            [out["gas"], out["melt"], out["solid"], *out["condensates"].values()]
        )
        out["totals"] = totals

        out["constraints"] = {}
        out["constraints"].update(self.parameters.mass_constraints.asdict())
        out["constraints"].update(
            self.parameters.fugacity_constraints.asdict(temperature, total_pressure)
        )

        out["residual"] = np.asarray(objective_function(self.solution, self.parameters))
        out["solution"] = np.asarray(self.solution)
        out["solver"] = self.multi_attempt_solution.asdict()

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


# TODO: Below are functions used by the previous output module, but not yet integrated into the new
# output module. They may be useful for future development, but are not currently used.


def broadcast_arrays_in_dict(some_dict: dict[str, NpArray], shape: int) -> dict[str, NpArray]:
    """Gets a dictionary of broadcasted arrays.

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
    """Creates a dictionary of dataframes from a nested dictionary.

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


# TODO: These were previously methods of the old output class, but are not yet integrated into the
# new output class. They may be useful for future development, but are not currently used.


def to_dataframes(self) -> dict[str, pd.DataFrame]:
    """Gets the output in a dictionary of dataframes.

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

    return dataframes


def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
    """Writes the output to an Excel file.

    Args:
        file_prefix: Prefix of the output file. Defaults to atmodeller_out.
    """
    logger.info("Writing output to excel")
    out: dict[str, pd.DataFrame] = self.to_dataframes()
    output_file: Path = Path(f"{file_prefix}.xlsx")

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for df_name, df in out.items():
            df.to_excel(writer, sheet_name=df_name, index=True)

    logger.info("Output written to %s", output_file)


def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
    """Writes the output to a pickle file.

    Args:
        file_prefix: Prefix of the output file. Defaults to atmodeller_out.
    """
    logger.info("Writing output to pickle")
    out: dict[str, pd.DataFrame] = self.to_dataframes()
    output_file: Path = Path(f"{file_prefix}.pkl")

    with open(output_file, "wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Output written to %s", output_file)
