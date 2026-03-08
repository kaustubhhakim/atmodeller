# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
import pickle
from collections.abc import Iterable
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxmod.solvers import MultiAttemptSolution
from jaxmod.type_aliases import NpArray, NpBool
from jaxmod.utils import vmap_axes_spec
from jaxtyping import Array, Float
from openpyxl.styles import PatternFill

from atmodeller.engine import get_total_pressure, objective_function
from atmodeller.parameters import Parameters
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase

logger: logging.Logger = logging.getLogger(__name__)


def _flatten_dict(d: dict, parent_keys: tuple = ()) -> dict[tuple, Any]:
    """Iteratively flattens a nested dict to {path_tuple: leaf} mapping.

    Args:
        d: The nested dictionary to flatten.
        parent_keys: A tuple of keys representing the path prefix to prepend to all
            entries.  Defaults to an empty tuple (top-level traversal).

    Returns:
        A flat dictionary mapping each leaf's full key path (as a tuple) to its value.
    """
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
    """Sets a value in a nested dict given a path tuple, creating intermediate dicts.

    Args:
        d: The nested dictionary to modify in place.
        path: A tuple of keys describing the location of the value.  Intermediate
            dicts are created as needed.
        value: The value to assign at the leaf position identified by ``path``.
    """
    for key in path[:-1]:
        d = d.setdefault(key, {})

    d[path[-1]] = value


_SUMMABLE_KEYS: frozenset[str] = frozenset({"mass_kg", "number_moles"})
"""Leaf-level keys whose values are summed across phases by :func:`_sum_phase_outputs`."""


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
            # For species, strip the trailing phase suffix (e.g. _g, _d, _s) so that
            # H2O_g and H2O_d both accumulate under the base formula H2O.
            if "species" in path:
                species_name = str(path[-1])
                base = species_name.rsplit("_", 1)[0] if "_" in species_name else species_name
                path = path[:-1] + (base,)
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
            :meth:`~atmodeller.output.Output.quick_look`. Must contain a ``"solution"`` key.

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


_PHASE_KEYS: frozenset[str] = frozenset({"gas", "melt", "solid", "condensates", "totals"})
"""Top-level keys in the :meth:`~atmodeller.output.Output.quick_look` output that represent
physically meaningful phases or phase aggregations."""

_OutputKey = Literal["phases", "species", "elements", "other"]
"""Valid category selectors for :func:`_group_by_all` and :meth:`Output.to_dataframes`."""

_ALL_OUTPUT_KEYS: tuple[_OutputKey, ...] = ("phases", "species", "elements", "other")
"""Default set of all output category selectors passed to 
:meth:`~atmodeller.output.Output.to_dataframes`."""


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
            :func:`~atmodeller.output.expand_to_batch`.
        keys: Categories to include.  Any combination of ``"phases"``, ``"species"``,
            ``"elements"``, and ``"other"``.  Defaults to all four.

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

        if include_species and "species" in path and path[0] != "totals":
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
            :func:`~atmodeller.output._expand_to_batch`.

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

    def __init__(self, parameters: Parameters, multi_attempt_solution: MultiAttemptSolution):
        self.parameters = parameters
        self.multi_attempt_solution = multi_attempt_solution

    @property
    def _split_solution(self) -> list[Float[Array, "... n_species"]]:
        """Log number of moles and log stability, split from the solution array in one pass."""
        return jnp.split(self.multi_attempt_solution.value, 2, axis=-1)

    @property
    def log_number_moles(self) -> Float[Array, "... n_species"]:
        """Log number of moles for each species"""
        log_number_moles, _ = self._split_solution

        return log_number_moles

    @property
    def log_stability(self) -> Float[Array, "... n_species"]:
        """Log stability for each species"""
        _, log_stability = self._split_solution

        active_stability: NpBool = self.parameters.reaction_system.species.active_stability
        log_stability = jnp.where(active_stability, log_stability, -jnp.inf)

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
                    jnp.atleast_1d(self.log_number_moles[..., condensate_slice][..., nn]),
                    jnp.atleast_1d(self.log_stability[..., condensate_slice][..., nn]),
                    temperature,
                    total_pressure,
                )
            )

        out["condensates"] = dict(zip(condensate_names, out_condensates))

        totals: dict[str, Any] = _sum_phase_outputs(
            [out["gas"], out["melt"], out["solid"], *out["condensates"].values()]
        )
        out["totals"] = totals

        out["state"] = self.parameters.state.asdict()

        out["constraints"] = {}
        out["constraints"].update(self.parameters.mass_constraints.asdict())
        out["constraints"].update(
            self.parameters.fugacity_constraints.asdict(temperature, total_pressure)
        )

        # Must vmap the residual evaluation to match what the solver did: parameters contains a
        # mix of scalar and batched leaves, so calling objective_function directly on the 2-D
        # solution gives incorrect results. vmap_axes_spec maps None/0 per leaf appropriately.
        objective_function_vmapped = eqx.filter_vmap(
            objective_function, in_axes=(0, vmap_axes_spec(self.parameters))
        )
        out["residual"] = np.asarray(objective_function_vmapped(self.solution, self.parameters))

        out["solution"] = np.asarray(self.solution)
        out["solver"] = self.multi_attempt_solution.asdict()

        return out

    def quick_look(self) -> dict[str, Any]:
        """Quick look at the output.

        Returns:
            A nested dictionary of the output, suitable for quick inspection and comparison.
        """
        out: dict[str, Any] = self.asdict()
        logger.info("Quick look output:\n%s", pformat(out))

        return out

    def to_dataframes(
        self,
        keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS,
        drop_unsuccessful_solves: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Args:
            keys: Categories to include in the output.  Any combination of ``"phases"``,
                ``"species"``, ``"elements"``, and ``"other"``.  ``"other"`` produces one
                DataFrame per non-phase top-level key (``solution``, ``residual``,
                ``constraints``, ``solver``, etc.).  Defaults to all four.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.

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

        if drop_unsuccessful_solves:
            logger.info("Dropping unsuccessful solves from output")
            result = self._drop_unsuccessful_solves(result)

        return result

    def to_excel(
        self,
        file_prefix: str = "atmodeller_out",
        keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS,
        drop_unsuccessful_solves: bool = False,
    ) -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            keys: Categories to include in the output.  Any combination of ``"phases"``,
                ``"species"``, ``"elements"``, and ``"other"``.  Defaults to all four.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
        """
        logger.info("Writing output to excel")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            keys=keys, drop_unsuccessful_solves=drop_unsuccessful_solves
        )
        output_file: str = f"{file_prefix}.xlsx"

        # Convenient to highlight rows where the solver failed to find a solution for follow-up
        # analysis. Define a fill colour for highlighting rows (e.g., yellow)
        highlight_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

        # Get the indices where the successful_solves mask is False
        unsuccessful_indices: NpArray = np.where(
            ~np.asarray(self.multi_attempt_solution.solver_success)
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

        # Without the consideration of highlighting
        # with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        #    for df_name, df in out.items():
        #        df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output written to %s", output_file)

    def to_pickle(
        self,
        file_prefix: Path | str = "atmodeller_out",
        keys: tuple[_OutputKey, ...] = _ALL_OUTPUT_KEYS,
        drop_unsuccessful_solves: bool = False,
    ) -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            keys: Categories to include in the output.  Any combination of ``"phases"``,
                ``"species"``, ``"elements"``, and ``"other"``.  Defaults to all four.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
        """
        logger.info("Writing output to pickle")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            keys=keys, drop_unsuccessful_solves=drop_unsuccessful_solves
        )
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

        logger.info("\nComparison result:\n%s", pformat(result))
        logger.info("All matching keys agree within tolerance: %s", all_match)

        return all_match

    def _drop_unsuccessful_solves(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Drops unsuccessful solves.

        Args:
            dataframes: Dataframes from which to drop unsuccessful models

        Returns:
            Dictionary of dataframes without unsuccessful models
        """
        return {
            key: df.loc[np.asarray(self.multi_attempt_solution.solver_success)]
            for key, df in dataframes.items()
        }


# TODO: Reinstate disequilibrium calculations and add to output

# class OutputDisequilibrium:
#     """Output disequilibrium calculations

#     Args:
#         parameters: Parameters
#         solution: Solution
#     """

#     @override
#     def asdict(self) -> dict[str, dict[str, NpArray]]:
#         """All outputs in a dictionary, with caching.

#         Additionally includes the disequilibrium group, compared to the base class.

#         Returns:
#             Dictionary of all output
#         """
#         out: dict[str, dict[str, NpArray]] = super().asdict()

#         out["disequilibrium"] = self.disequilibrium_asdict()

#         self._cached_dict = out  # Re-cache result for faster re-accessing

#         return out

#     def disequilibrium_asdict(self) -> dict[str, NpArray]:
#         """Gets the reaction disequilibrium as a dictionary.

#         Returns:
#             Reaction disequilibrium as a dictionary
#         """
#         reaction_mask: NpBool = self.reaction_mask()
#         residual: NpFloat = np.asarray(self.vmapf.objective_function(jnp.asarray(self.solution)))

#         # Number of True entries per row (must be same for all rows)
#         n_cols: NpInt = reaction_mask.sum(axis=1)[0]
#         # logger.debug("n_cols = %s", n_cols)
#         # Convert boolean mask to sorted column indices for each row
#         col_indices: NpInt = np.argsort(~reaction_mask, axis=1)[:, :n_cols]
#         # logger.debug("col_indices = %s", col_indices)
#         # Gather the True entries in order
#         compressed: NpFloat = np.take_along_axis(residual, col_indices, axis=1)
#         # logger.debug("compressed = %s", compressed)

#         # To compute the limiting reactant/product in each reaction we need to know the
#         # availability of each species. We will ignore condensates later because their stability
#         # criteria prevents a simple calculation of what is limiting the reaction.
#         number_fraction: NpFloat = self.number_moles / np.sum(
#             self.number_moles, axis=1, keepdims=True
#         )
#         # logger.debug("number_fraction = %s", number_fraction)
#         reaction_matrix: NpFloat = self.parameters.reaction_network.reaction_matrix
#         # logger.debug("reaction_matrix = %s", reaction_matrix)

#         out: dict[str, NpArray] = {}

#         for jj in range(n_cols):
#             # logger.debug("Working on reaction %d", jj)
#             per_mole_of_reaction: NpFloat = compressed[:, jj] * GAS_CONSTANT * self.temperature
#             stoich: NpFloat = reaction_matrix[jj]
#             # logger.debug("stoich = %s", stoich)

#             # Normalised ratios for limiting species (ignore divide-by-zero warnings)
#             with np.errstate(divide="ignore"):
#                 ratios: NpFloat = np.where(stoich != 0, number_fraction / stoich, np.nan)
#             # logger.debug("ratios = %s", ratios)
#             limiting: NpFloat = np.full_like(per_mole_of_reaction, np.nan)
#             # logger.debug("limiting (full_like) = %s", limiting)

#             # Initialise with None placeholders for every row
#             limiting_species_names: list[Optional[str]] = [None] * residual.shape[0]
#             limiting_species_type: list[Optional[str]] = [None] * residual.shape[0]

#             # Backward-favoured: products limit
#             mask_back: NpBool = per_mole_of_reaction > 0
#             # logger.debug("mask_back = %s", mask_back)
#             if np.any(mask_back):
#                 # Subarray of only product species for backward-favoured reactions
#                 sub_ratios: NpFloat = ratios[mask_back][:, stoich > 0]
#                 # Column indices of product species in the full array
#                 sub_cols: NpInt = np.where(stoich > 0)[0]
#                 # Value of limiting species
#                 limiting[mask_back] = np.min(sub_ratios, axis=1)
#                 # logger.debug("limiting[mask_back] = %s", limiting[mask_back])
#                 # Column index (within subarray) of limiting species
#                 min_idx_within: NpInt = np.argmin(sub_ratios, axis=1)
#                 # Map back to global indices in ratios / species_names
#                 min_idx_global: NpInt = sub_cols[min_idx_within]
#                 # Get the actual species names
#                 for row_idx, species_idx in zip(np.where(mask_back)[0], min_idx_global):
#                     limiting_species_names[row_idx] = self.species_collection.data.species_names[
#                         species_idx
#                     ]
#                     limiting_species_type[row_idx] = "Product"
#                 # logger.debug("limiting_species_names (back) = %s", limiting_species_names)

#             # Forward-favoured: reactants limit
#             mask_fwd: NpBool = ~mask_back
#             # logger.debug("mask_fwd = %s", mask_fwd)
#             if np.any(mask_fwd):
#                 sub_ratios: NpFloat = ratios[mask_fwd][:, stoich < 0]
#                 sub_cols: NpInt = np.where(stoich < 0)[0]
#                 # Limiting species is the largest negative ratio among reactants (closest to zero)
#                 limiting[mask_fwd] = np.max(sub_ratios, axis=1)
#                 # logger.debug("limiting[mask_fwd] = %s", limiting[mask_fwd])
#                 max_idx_within: NpInt = np.argmax(sub_ratios, axis=1)
#                 max_idx_global: NpInt = sub_cols[max_idx_within]
#                 # Get the actual species names
#                 for row_idx, species_idx in zip(np.where(mask_fwd)[0], max_idx_global):
#                     limiting_species_names[row_idx] = self.species_collection.data.species_names[
#                         species_idx
#                     ]
#                     limiting_species_type[row_idx] = "Reactant"
#                 # logger.debug("limiting_species_names (fwd) = %s", limiting_species_names)

#             # Compute the energy per mole of atmosphere
#             energy_per_mol_atmosphere: NpFloat = per_mole_of_reaction * limiting
#             logger.debug("energy_per_mol_atmosphere = %s", energy_per_mol_atmosphere)

#             out[f"Reaction_{jj}"] = per_mole_of_reaction

#             # TODO: To reinstate?
#             # if self.species.gas_only:
#             #     out[f"Reaction_{jj}_per_atmosphere"] = energy_per_mol_atmosphere
#             #     out[f"Reaction_{jj}_limiting_species"] = np.array(limiting_species_names)
#             #     out[f"Reaction_{jj}_limiting_species_role"] = np.array(limiting_species_type)

#         return out
