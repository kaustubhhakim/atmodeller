# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
import pickle
from collections.abc import Iterable
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, Optional

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
    """Iteratively flattens a nested dict to {path_tuple: leaf} mapping."""
    result: dict[tuple, Any] = {}
    stack: list[tuple[tuple, dict]] = [(parent_keys, d)]
    while stack:
        prefix, current = stack.pop()
        for k, v in current.items():
            path = prefix + (k,)
            if isinstance(v, dict):
                stack.append((path, v))
            else:
                result[path] = v
    return result


def _set_nested(d: dict, path: tuple, value: Any) -> None:
    """Sets a value in a nested dict given a path tuple, creating intermediate dicts."""
    for key in path[:-1]:
        d = d.setdefault(key, {})

    d[path[-1]] = value


_SUMMABLE_KEYS: frozenset[str] = frozenset({"mass_kg", "number_moles"})


def _sum_phase_outputs(phase_outputs: Iterable[dict[str, Any]]) -> dict[str, Any]:
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

    # Logarithmic abundance of all elements relative to hydrogen (A(X) = log10(n_X/n_H) + 12)
    element_moles: dict[str, Any] = out.get("elements", {}).get("number_moles", {})
    if "H" in element_moles:
        h_moles: NpArray = np.asarray(element_moles["H"])
        out.setdefault("elements", {})["logarithmic_abundance"] = {
            element: np.log10(np.asarray(moles) / h_moles) + 12
            for element, moles in element_moles.items()
        }

    return out


def _expand_to_batch(nested_dict: dict[str, Any]) -> dict[str, Any]:
    """Expands all array leaves in a nested dict to a common batch length.

    The batch length is inferred from ``solution``: ``1`` if it is 1-D (single run), or
    ``solution.shape[0]`` if it is 2-D (batched run of ``N`` conditions).  Scalars and length-1
    arrays are broadcast to ``(batch_length,)``; arrays already of the correct length are passed
    through unchanged; multi-dimensional arrays such as ``solution`` and ``residual`` whose first
    axis is already ``batch_length`` are also left unchanged.

    Args:
        nested_dict: A nested dict as returned by
            :meth:`~atmodeller.output_new.Output.quick_look`. Must contain a ``"solution"`` key.

    Returns:
        A new nested dict with the same structure where every scalar leaf is expanded to a
        :class:`numpy.ndarray` of length ``batch_length``.
    """
    # Infer batch length from `solution`, which is always present:
    # shape (2*n_species,) for a single run → batch_length = 1
    # shape (N, 2*n_species) for a batched run → batch_length = N
    solution_arr = np.asarray(nested_dict["solution"])
    batch_length: int = 1 if solution_arr.ndim == 1 else solution_arr.shape[0]

    def _expand_leaf(value: Any) -> NpArray:
        arr = np.atleast_1d(np.asarray(value))
        if batch_length > 1 and arr.shape[0] == 1:
            # Scalar-promoted or length-1: broadcast first axis to batch_length.
            # np.array() materialises the broadcast view into a writable array.
            return np.array(np.broadcast_to(arr, (batch_length,) + arr.shape[1:]))

        return arr

    def _map(d: dict) -> dict:
        return {k: _map(v) if isinstance(v, dict) else _expand_leaf(v) for k, v in d.items()}

    return _map(nested_dict)


# Top-level keys in quick_look output that represent physically meaningful phases or aggregations
_PHASE_KEYS: frozenset[str] = frozenset({"gas", "melt", "solid", "condensates", "totals"})

_OutputKey = Literal["phases", "species", "elements", "other"]
"""Valid category selectors for :func:`_group_by_all` and :meth:`Output.to_dataframes`."""

_ALL_OUTPUT_KEYS: tuple[_OutputKey, ...] = ("phases", "species", "elements", "other")


def _group_by_all(
    nested_dict: dict[str, Any], keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS
) -> dict[str, dict[str, Any]]:
    """Groups output by phase, species, and/or element.

    Groups the requested views in a single pass.  Phase names, species names, and element symbols
    all share the same namespace of primary keys.  A gas-phase species ``H2O_g`` and the phase
    key ``gas`` will both appear as top-level entries when both ``"phases"`` and ``"species"``
    are selected::

        {
            "gas":   {"phase.mass_kg": ..., "species.activity.H2O_g": ...,
                      "phase.log10dIW_1_bar": ..., ...},
            "H2O_g": {"gas.activity": ..., "gas.mass_kg": ..., ...},
            "H":     {"gas.mass_kg": ..., "totals.logarithmic_abundance": ..., ...},
            ...
        }

    Args:
        nested_dict: A nested dict as returned by
            :func:`~atmodeller.output_new.expand_to_batch`.
        keys: Categories to include.  Any combination of ``"phases"``, ``"species"``, and
            ``"elements"``.  Defaults to all three.

    Returns:
        A combined dict with phase names, species names, and element symbols as primary keys.
    """
    result: dict[str, dict[str, Any]] = {}

    include_phases: bool = "phases" in keys
    include_species: bool = "species" in keys
    include_elements: bool = "elements" in keys

    for path, value in _flatten_dict(nested_dict).items():
        if include_phases and path[0] in _PHASE_KEYS:
            phase_name: str = str(path[0])
            phase_key: str = ".".join(str(k) for k in path[1:])
            result.setdefault(phase_name, {})[phase_key] = value

        if include_species and "species" in path:
            name: str = str(path[-1])
            species_key: str = ".".join(str(k) for k in path[:-1] if k != "species")
            result.setdefault(name, {})[species_key] = value

        if include_elements and "elements" in path:
            name = str(path[-1])
            element_key: str = ".".join(str(k) for k in path[:-1] if k != "elements")
            result.setdefault(name, {})[element_key] = value

    return result


def _group_other(nested_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Collects non-phase top-level keys (``solution``, ``residual``, ``constraints``,
    ``solver``, etc.) into a dict of flat column dicts, one per top-level key.

    Each non-phase top-level key becomes a primary key whose value is a flat
    ``{column_name: array}`` dict suitable for :class:`~pandas.DataFrame` construction:

    * **Dict value** — flattened recursively; the dotted sub-path forms the column name,
      e.g. ``constraints → {"H": ..., "O": ...}``.
    * **Array value** — if the last axis has more than one element each column is named
      ``"0"``, ``"1"``, …; if effectively scalar per row the column takes the key name.

    Args:
        nested_dict: A nested dict as returned by
            :func:`~atmodeller.output_new._expand_to_batch`.

    Returns:
        A dict mapping each non-phase top-level key to a flat column dict.
    """
    result: dict[str, dict[str, Any]] = {}

    for top_key, value in nested_dict.items():
        if top_key in _PHASE_KEYS:
            continue
        if isinstance(value, dict):
            cols: dict[str, Any] = {}
            for path, leaf in _flatten_dict(value).items():
                cols[".".join(str(k) for k in path)] = leaf
            result[str(top_key)] = cols
        else:
            arr = np.asarray(value)
            if arr.ndim <= 1 or arr.shape[-1] == 1:
                result[str(top_key)] = {str(top_key): arr.ravel()}
            else:
                result[str(top_key)] = {str(i): arr[..., i] for i in range(arr.shape[-1])}

    return result


class Output(eqx.Module):
    """Output

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    parameters: Parameters
    multi_attempt_solution: MultiAttemptSolution
    _cached_dict: Optional[dict[str, Any]]

    def __init__(self, parameters: Parameters, multi_attempt_solution: MultiAttemptSolution):
        self.parameters = parameters
        self.multi_attempt_solution = multi_attempt_solution

        # Caching output to avoid recomputation
        self._cached_dict = None
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

    def asdict(self) -> dict[str, Any]:
        """Solution as a nested dictionary.

        Returns:
            Dictionary of the solution
        """
        if self._cached_dict is not None:
            logger.info("Returning cached asdict output")
            return self._cached_dict

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

        totals: dict[str, Any] = _sum_phase_outputs(
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

        return out

    def quick_look(self) -> dict[str, Any]:
        """Quick look at the output.

        Returns:
            A nested dictionary of the output, suitable for quick inspection and comparison.
        """
        out: dict[str, Any] = self.asdict()
        logger.info(f"Quick look output:\n{pformat(out)}")

        return out

    def to_dataframes(
        self, keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS
    ) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Args:
            keys: Categories to include in the output.  Any combination of ``"phases"``,
                ``"species"``, ``"elements"``, and ``"other"``.  ``"other"`` collects the
                remaining top-level entries (``solution``, ``residual``, ``constraints``,
                ``solver``, etc.) into a single sheet.  Defaults to all four.

        Returns:
            Output in a dictionary of dataframes
        """
        expanded: dict[str, Any] = _expand_to_batch(self.asdict())
        result: dict[str, pd.DataFrame] = {
            name: pd.DataFrame(props) for name, props in _group_by_all(expanded, keys=keys).items()
        }
        if "other" in keys:
            for name, cols in _group_other(expanded).items():
                result[name] = pd.DataFrame(cols)

        return result

    def to_excel(
        self, file_prefix: str = "atmodeller_out", keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS
    ) -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            keys: Categories to include in the output.  Any combination of ``"phases"``,
                ``"species"``, ``"elements"``, and ``"other"``.  Defaults to all four.
        """
        logger.info("Writing output to excel")
        out: dict[str, pd.DataFrame] = self.to_dataframes(keys=keys)
        output_file: str = f"{file_prefix}.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output written to %s", output_file)

    def to_pickle(
        self,
        file_prefix: Path | str = "atmodeller_out",
        keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS,
    ) -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            keys: Categories to include in the output.  Any combination of ``"phases"``,
                ``"species"``, ``"elements"``, and ``"other"``.  Defaults to all four.
        """
        logger.info("Writing output to pickle")
        out: dict[str, pd.DataFrame] = self.to_dataframes(keys=keys)
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)

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
