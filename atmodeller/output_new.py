# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
from collections.abc import Iterable
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

    def _expand_leaf(value: Any) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(value))
        if batch_length > 1 and arr.shape[0] == 1:
            # Scalar-promoted or length-1: broadcast first axis to batch_length.
            # np.array() materialises the broadcast view into a writable array.
            return np.array(np.broadcast_to(arr, (batch_length,) + arr.shape[1:]))
        return arr

    def _map(d: dict) -> dict:
        return {k: _map(v) if isinstance(v, dict) else _expand_leaf(v) for k, v in d.items()}

    return _map(nested_dict)


def _group_by_level(nested_dict: dict[str, Any], level: str) -> dict[str, dict[str, Any]]:
    """Groups leaves whose path passes through *level*, keyed by the name that follows it.

    The species/element name is always the last path component; the *level* marker and the name
    itself are both stripped from the inner key, and the remaining components are joined with
    ``"."`` to form a compact property label.
    """
    result: dict[str, dict[str, Any]] = {}
    for path, value in _flatten_dict(nested_dict).items():
        if level not in path:
            continue
        name: str = str(path[-1])  # species name or element symbol is always the leaf key
        key: str = ".".join(str(k) for k in path[:-1] if k != level)
        result.setdefault(name, {})[key] = value
    return result


def _group_by_species(nested_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Groups species-level output by species name.

    Walks every leaf whose path passes through a ``"species"`` sub-dict and re-indexes
    the data with species names as the outer key.  The inner key is formed by joining the
    remaining path components (excluding ``"species"`` and the name itself) with ``"."``::

        ("gas", "species", "activity", "H2O_g")  →  {"H2O_g": {"gas.activity": <value>}}

    Args:
        nested_dict: A nested dict, typically the output of
            :func:`~atmodeller.output_new.expand_to_batch`.

    Returns:
        A dict mapping each species name to a flat dict of ``"phase.property"`` keys and
        their corresponding values.
    """
    return _group_by_level(nested_dict, "species")


def _group_by_elements(nested_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Groups element-level output by element symbol.

    Walks every leaf whose path passes through an ``"elements"`` sub-dict and re-indexes
    the data with element symbols as the outer key.  The inner key is formed by joining the
    remaining path components (excluding ``"elements"`` and the symbol itself) with ``"."``::

        ("gas", "elements", "mass_kg", "H")  →  {"H": {"gas.mass_kg": <value>}}

    Args:
        nested_dict: A nested dict, typically the output of
            :func:`~atmodeller.output_new.expand_to_batch`.

    Returns:
        A dict mapping each element symbol to a flat dict of ``"phase.property"`` keys and
        their corresponding values.
    """
    return _group_by_level(nested_dict, "elements")


# Top-level keys in quick_look output that represent physically meaningful phases or aggregations
_PHASE_KEYS: frozenset[str] = frozenset({"gas", "melt", "solid", "condensates", "totals"})


def _group_by_phase(nested_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Groups all output by phase (top-level key).

    Each top-level key in *nested_dict* that is a member of :data:`_PHASE_KEYS` becomes a
    primary key.  All leaves under it are collected into a flat dict whose keys are the
    remaining path components joined with ``"."``::

        ("gas", "phase", "log10dIW_1_bar")  →  {"gas": {"phase.log10dIW_1_bar": <value>}}
        ("gas", "species", "activity", "H2O_g")  →  {"gas": {"species.activity.H2O_g": <value>}}

    Args:
        nested_dict: A nested dict as returned by
            :func:`~atmodeller.output_new.expand_to_batch`.

    Returns:
        A dict mapping each phase name to a flat dict of all its leaves.
    """
    result: dict[str, dict[str, Any]] = {}
    for path, value in _flatten_dict(nested_dict).items():
        if path[0] not in _PHASE_KEYS:
            continue
        phase_name: str = str(path[0])
        key: str = ".".join(str(k) for k in path[1:])
        result.setdefault(phase_name, {})[key] = value
    return result


def _group_by_all(nested_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Groups output by phase, species, and element simultaneously.

    Combines :func:`~atmodeller.output_new._group_by_phase`,
    :func:`~atmodeller.output_new._group_by_species`, and
    :func:`~atmodeller.output_new._group_by_elements` into a single dict.  All three views
    share the same namespace of primary keys, which means a gas-phase species ``H2O_g`` and
    the phase key ``gas`` will both appear as top-level entries::

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

    Returns:
        A combined dict with phase names, species names, and element symbols as primary keys.
    """
    result: dict[str, dict[str, Any]] = _group_by_phase(nested_dict)
    for name, props in _group_by_species(nested_dict).items():
        result.setdefault(name, {}).update(props)
    for name, props in _group_by_elements(nested_dict).items():
        result.setdefault(name, {}).update(props)
    return result


def _to_dataframes(
    grouped: dict[str, dict[str, Any]] | None = None,
    nested_dict: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Converts grouped output to a dictionary of :class:`pandas.DataFrame` objects.

    Each primary key (phase name, species name, or element symbol) maps to a
    :class:`~pandas.DataFrame` whose columns are the combined property keys (e.g.
    ``"phase.mass_kg"``, ``"gas.activity"``, ``"totals.mass_kg"``) and whose rows
    correspond to the batch dimension.

    Exactly one of *grouped* or *nested_dict* must be supplied.  If *nested_dict* is given,
    :func:`~atmodeller.output_new._group_by_all` is called automatically.

    Args:
        grouped: A pre-grouped dict, e.g. from
            :func:`~atmodeller.output_new._group_by_all`,
            :func:`~atmodeller.output_new._group_by_phase`,
            :func:`~atmodeller.output_new._group_by_species`, or
            :func:`~atmodeller.output_new._group_by_elements`.
        nested_dict: A nested dict as returned by
            :meth:`~atmodeller.output_new.Output.quick_look`.  Mutually exclusive with
            *grouped*.

    Returns:
        A dict mapping each primary key to a :class:`~pandas.DataFrame`.
    """
    if grouped is None and nested_dict is None:
        raise ValueError("Supply either 'grouped' or 'nested_dict'.")
    if grouped is not None and nested_dict is not None:
        raise ValueError("Supply only one of 'grouped' or 'nested_dict', not both.")

    source: dict[str, dict[str, Any]] = (
        grouped if grouped is not None else _group_by_all(nested_dict)  # type: ignore[arg-type]
    )

    return {name: pd.DataFrame(props) for name, props in source.items()}


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

        logger.info(f"Quick look output:\n{pformat(out)}")

        return out

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Returns:
            Output in a dictionary of dataframes
        """
        return _to_dataframes(nested_dict=_expand_to_batch(self.quick_look()))

    def to_excel(self, file_prefix: str = "atmodeller_out") -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
        """
        logger.info("Writing output to excel")
        out: dict[str, pd.DataFrame] = self.to_dataframes()
        output_file: str = f"{file_prefix}.xlsx"

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            for df_name, df in out.items():
                df.to_excel(writer, sheet_name=df_name, index=True)

        logger.info("Output written to %s", output_file)

    def group_by_species(self, to_dataframes: bool = False) -> dict[str, Any]:
        """Groups species-level output by species name.

        Args:
            to_dataframes: Whether to convert the inner dicts to pandas DataFrames. Defaults to
                ``False``.

        Returns:
            A dict mapping each species name to a flat dict of ``"phase.property"`` keys and
            their corresponding values.
        """
        group_by_species_ = _group_by_species(_expand_to_batch(self.quick_look()))

        if to_dataframes:
            return _to_dataframes(grouped=group_by_species_)

        else:
            return group_by_species_

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


# def broadcast_arrays_in_dict(some_dict: dict[str, NpArray], shape: int) -> dict[str, NpArray]:
#     """Gets a dictionary of broadcasted arrays.

#     Args:
#         some_dict: Some dictionary
#         size: Shape (size) of the desired array

#     Returns:
#         A dictionary with broadcasted arrays
#     """
#     expanded_dict: dict[str, NpArray] = {}
#     for key, value in some_dict.items():
#         expanded_dict[key] = np.broadcast_to(value, shape)

#     return expanded_dict


# def split_dict_by_columns(dict_to_split: dict[str, NpArray]) -> list[dict[str, NpArray]]:
#     """Splits a dictionary based on columns in the values.

#     Args:
#         dict_to_split: A dictionary to split

#     Returns:
#         A list of dictionaries split by column
#     """
#     # Assume all arrays have the same number of columns
#     first_key: str = next(iter(dict_to_split))
#     num_columns: int = dict_to_split[first_key].shape[1]

#     # Preallocate list of dicts
#     split_dicts: list[dict] = [{} for _ in range(num_columns)]

#     for key, array in dict_to_split.items():
#         for i in range(num_columns):
#             split_dicts[i][key] = array[:, i]

#     return split_dicts


# def nested_dict_to_dataframes(nested_dict: dict[str, dict[str, Any]]) -> dict[str, pd.DataFrame]:
#     """Creates a dictionary of dataframes from a nested dictionary.

#     Args:
#         nested_dict: A nested dictionary

#     Returns:
#         A dictionary of dataframes
#     """
#     dataframes: dict[str, pd.DataFrame] = {}

#     for outer_key, inner_dict in nested_dict.items():
#         # Convert inner dictionary to DataFrame
#         df: pd.DataFrame = pd.DataFrame(inner_dict)
#         dataframes[outer_key] = df

#     return dataframes


# TODO: These were previously methods of the old output class, but are not yet integrated into the
# new output class. They may be useful for future development, but are not currently used.


# def to_dataframes(self) -> dict[str, pd.DataFrame]:
#     """Gets the output in a dictionary of dataframes.

#     Returns:
#         Output in a dictionary of dataframes
#     """
#     if self._cached_dataframes is not None:
#         logger.debug("Returning cached to_dataframes output")
#         dataframes: dict[str, pd.DataFrame] = self._cached_dataframes  # Return cached result
#     else:
#         logger.info("Computing to_dataframes output")
#         dataframes = nested_dict_to_dataframes(self.asdict())
#         self._cached_dataframes = dataframes
#         # logger.debug("to_dataframes = %s", self._cached_dataframes)

#     return dataframes


# def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
#     """Writes the output to an Excel file.

#     Args:
#         file_prefix: Prefix of the output file. Defaults to atmodeller_out.
#     """
#     logger.info("Writing output to excel")
#     out: dict[str, pd.DataFrame] = self.to_dataframes()
#     output_file: Path = Path(f"{file_prefix}.xlsx")

#     with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
#         for df_name, df in out.items():
#             df.to_excel(writer, sheet_name=df_name, index=True)

#     logger.info("Output written to %s", output_file)


# def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
#     """Writes the output to a pickle file.

#     Args:
#         file_prefix: Prefix of the output file. Defaults to atmodeller_out.
#     """
#     logger.info("Writing output to pickle")
#     out: dict[str, pd.DataFrame] = self.to_dataframes()
#     output_file: Path = Path(f"{file_prefix}.pkl")

#     with open(output_file, "wb") as handle:
#         pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     logger.info("Output written to %s", output_file)
