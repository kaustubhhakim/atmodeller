# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
import pickle
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxmod.solvers import MultiAttemptSolution
from jaxmod.type_aliases import NpArray
from jaxtyping import Array, ArrayLike, Float, PyTree
from openpyxl.styles import PatternFill

from atmodeller.engine import get_total_pressure
from atmodeller.parameters import Parameters
from atmodeller.phases import GasPhaseOutput, MeltPhase, PhaseOutput, SolidPhase

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# These functions all feel awkward and clunky, and are ultimately not compatible within a jitted
# workflow

# _SUMMABLE_KEYS: frozenset[str] = frozenset({"mass_kg", "number_moles"})
# """Leaf-level keys whose values are summed across phases by :func:`_sum_phase_outputs`."""


# def _sum_phase_outputs(phase_outputs: Iterable[dict[str, Any]]) -> dict[str, Any]:
#     """Sums summable quantities across an iterable of phase output dicts.

#     Only leaves under a ``"mass_kg"`` or ``"number_moles"`` sub-category are included, preserving
#     the ``elements``, ``species``, and ``phase`` sub-structure. Values at the same path are summed;
#     ``np.nan`` values are treated as zero so that a phase with unconstrained moles does not
#     contaminate the total.

#     Args:
#         phase_outputs: Iterable of phase output dicts as returned by
#             :meth:`~atmodeller.phases.BasePhase.output`.

#     Returns:
#         A nested dict with the same sub-structure as the inputs but restricted to the summable
#         keys, holding the element-wise sum across all phases.
#     """
#     total: dict[tuple, Any] = {}

#     for phase_out in phase_outputs:
#         for path, value in _flatten_dict(phase_out).items():
#             if not any(k in _SUMMABLE_KEYS for k in path):
#                 continue
#             scalar = np.asarray(value)
#             addend = np.where(np.isnan(scalar), 0.0, scalar)
#             # For species, strip the trailing phase suffix (e.g. _g, _d, _s) so that
#             # H2O_g and H2O_d both accumulate under the base formula H2O.
#             if "species" in path:
#                 species_name = str(path[-1])
#                 base = species_name.rsplit("_", 1)[0] if "_" in species_name else species_name
#                 path = path[:-1] + (base,)
#             total[path] = total.get(path, 0.0) + addend

#     out: dict[str, Any] = {}
#     for path, value in total.items():
#         _set_nested(out, path, value)

#     # Logarithmic abundance of all elements relative to hydrogen (A(X) = log10(n_X/n_H) + 12)
#     element_moles: dict[str, Any] = out.get("elements", {}).get("number_moles", {})
#     if "H" in element_moles:
#         h_moles: NpArray = np.asarray(element_moles["H"])
#         out.setdefault("elements", {})["logarithmic_abundance"] = {
#             element: np.log10(np.asarray(moles) / h_moles) + 12
#             for element, moles in element_moles.items()
#         }

#     return out

# _OutputKey = Literal["phases", "species", "elements", "other"]
# """Valid category selectors for :func:`_group_by_all` and :meth:`Output.to_dataframes`."""

# _ALL_OUTPUT_KEYS: tuple[_OutputKey, ...] = ("phases", "species", "elements", "other")
# """Default set of all output category selectors passed to
# :meth:`~atmodeller.output.Output.to_dataframes`."""


def expand_jax_arrays_to_batch(pytree: PyTree, batch_size: int, *, ravel: bool = False) -> PyTree:
    """Expands all arrays in a PyTree to the batch size.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to expand
        batch_size: Batch size to expand to
        ravel: Whether to ravel arrays to 1-D after expanding. Can be used when the expanded
            arrays are intended for conversion to DataFrames (which also requires an
            additional step of converting the arrays to NumPy). Defaults to ``False``.

    Returns:
        PyTree with arrays expanded to batch size
    """

    def expand(x: Any) -> Any:
        if isinstance(x, jnp.ndarray):
            x = jnp.atleast_1d(x)
            # Always broadcast if shape[0] != batch_size
            if x.shape[0] != batch_size:
                x = jnp.broadcast_to(x, (batch_size,) + x.shape[1:])
            if ravel:
                return jnp.ravel(x)
            else:
                return x
        return x

    return jax.tree_util.tree_map(expand, pytree)


def convert_jax_arrays_to_numpy(pytree: PyTree) -> PyTree:
    """Converts all JAX arrays in a PyTree to NumPy arrays.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to convert

    Returns:
        PyTree with JAX arrays converted to NumPy arrays
    """
    return jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, pytree
    )


class Output(eqx.Module):
    """Output

    Properties can be called within a jitted context to access output quantites for downstream
    processing. Arrays are always broadcastable to avoid necessitating expanding all arrays to the
    batch size.

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
    def _split_solution(self) -> list[Float[Array, "#n_batch n_species"]]:
        """Log number of moles and log stability, split from the solution array in one pass."""
        return jnp.split(self.multi_attempt_solution.value, 2, axis=-1)

    @property
    def batch_size(self) -> int:
        """Batch size of the output"""
        return self.parameters.batch_size

    @property
    def log_number_moles(self) -> Float[Array, "#n_batch n_species"]:
        """Log number of moles for each species"""
        log_number_moles, _ = self._split_solution

        return log_number_moles

    @property
    def log_stability(self) -> Float[Array, "#n_batch n_species"]:
        """Log stability for each species"""
        _, log_stability = self._split_solution

        active_stability: ArrayLike = self.parameters.reaction_system.species.active_stability
        log_stability = jnp.where(active_stability, log_stability, -jnp.inf)

        return log_stability

    @property
    def condensates(self) -> tuple[PhaseOutput, ...]:
        """Pure phase condensates"""

        condensate_slice: slice = self.parameters.reaction_system.condensates_slice

        condensates_out = []

        for nn, condensate in enumerate(self.parameters.reaction_system.condensate_phases):
            condensate_out = condensate.output(
                jnp.atleast_2d(self.log_number_moles[..., condensate_slice][..., nn]),
                jnp.atleast_2d(self.log_stability[..., condensate_slice][..., nn]),
                self.temperature,
                self.pressure,
                jnp.atleast_2d(0.0),
                jnp.atleast_2d(-jnp.inf),
            )
            condensates_out.append(condensate_out)

        return tuple(condensates_out)

    @property
    def constraints_element_mass(self) -> Float[Array, "#n_batch n_elements"]:
        """Element mass constraints in kg"""
        return self.parameters.mass_constraints.abundance_mass()

    @property
    def constraints_element_moles(self) -> Float[Array, "#n_batch n_elements"]:
        """Element abundance constraints in moles"""
        return self.parameters.mass_constraints.abundance_mol()

    @property
    def constraints_fugacity(self) -> Float[Array, "#n_batch n_species"]:
        """Fugacity constraints in bar"""
        constraints: Float[Array, "#n_batch n_species"] = jnp.stack(
            [
                jnp.exp(constraint.log_fugacity(self.temperature, self.pressure))
                for constraint in self.parameters.fugacity_constraints.constraints
            ],
            axis=-1,
        )
        return constraints

    @property
    def gas(self) -> GasPhaseOutput:
        """Gas phase output"""

        gas_slice: slice = self.parameters.reaction_system.gas_slice

        gas_output: GasPhaseOutput = self.parameters.reaction_system.gas_phase.output(
            self.log_number_moles[..., gas_slice],
            self.log_stability[..., gas_slice],
            self.temperature,
            self.pressure,
            jnp.atleast_2d(0.0),
            jnp.atleast_2d(-jnp.inf),
        )

        return gas_output

    @property
    def melt(self) -> PhaseOutput[MeltPhase]:
        """Melt phase output"""

        melt_slice: slice = self.parameters.reaction_system.melt_slice

        melt_output: PhaseOutput[MeltPhase] = self.parameters.reaction_system.melt_phase.output(
            self.log_number_moles[..., melt_slice],
            self.log_stability[..., melt_slice],
            self.temperature,
            self.pressure,
            jnp.log(jnp.atleast_2d(self.parameters.state.molar_mass)),
            jnp.log(jnp.atleast_2d(self.parameters.state.melt_mass)),
        )

        return melt_output

    @property
    def solid(self) -> PhaseOutput[SolidPhase]:
        """Solid phase output"""

        solid_slice: slice = self.parameters.reaction_system.solid_slice

        solid_output: PhaseOutput[SolidPhase] = self.parameters.reaction_system.solid_phase.output(
            self.log_number_moles[..., solid_slice],
            self.log_stability[..., solid_slice],
            self.temperature,
            self.pressure,
            jnp.log(jnp.atleast_2d(self.parameters.state.molar_mass)),
            jnp.log(jnp.atleast_2d(self.parameters.state.solid_mass)),
        )

        return solid_output

    def state_asdict(self) -> dict[str, Any]:
        """Thermodynamic state of the system"""
        return self.parameters.state.asdict(jnp.squeeze(self.gas.phase_mass))

    @property
    def temperature(self) -> Float[Array, "#n_batch 1"]:
        """Temperature in K"""
        return jnp.atleast_2d(self.parameters.state.temperature).T

    @property
    def pressure(self) -> Float[Array, "#n_batch 1"]:
        """Pressure in bar"""
        return jnp.atleast_2d(get_total_pressure(self.parameters, self.solution)).T

    @property
    def solution(self) -> Float[Array, "#n_batch twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.multi_attempt_solution.value

    @property
    def solver(self) -> dict[str, ArrayLike]:
        """Solver information such as success flags and number of iterations"""
        return self.multi_attempt_solution.asdict()

    def asdict(self, *, to_numpy: bool = False) -> dict[str, Any]:
        """Complete output as a nested dictionary with JAX or NumPy arrays.

        Args:
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``.
                Must be ``False`` if used within a jitted context, as NumPy arrays are not
                compatible with JAX transformations (jit, vmap, etc.).

        Returns:
            Dictionary of the solution with JAX or NumPy arrays
        """
        out: dict[str, Any] = {}

        if not self.gas.is_empty:
            out["gas"] = self.gas.asdict()
        if not self.melt.is_empty:
            out["melt"] = self.melt.asdict()
        if not self.solid.is_empty:
            out["solid"] = self.solid.asdict()

        if len(self.condensates) > 0:
            # This retains symmetry with the output structure of the other phases (gas, melt,
            # solid), where condensates are ordered in a list and identified by their species name
            # within the species sub-category.
            condensate_out: list = []
            for condensate in self.condensates:
                condensate_out.append(condensate.asdict())
            out["condensates"] = condensate_out

        out["solver"] = self.multi_attempt_solution.asdict()
        out["state"] = self.state_asdict()

        temperature = self.parameters.state.temperature
        total_pressure = get_total_pressure(self.parameters, self.solution)
        out["constraints"] = {}
        out["constraints"].update(self.parameters.mass_constraints.asdict())

        # TODO
        # out["constraints"].update(
        #    self.parameters.fugacity_constraints.asdict(temperature, total_pressure)
        # )

        if to_numpy:
            out = convert_jax_arrays_to_numpy(out)

        return out

    def asdict_split(
        self, *, expand_to_batch: bool = False, to_numpy: bool = False, ravel: bool = False
    ) -> dict[str, Any]:
        """Output as a nested dictionary with JAX or NumPy arrays, split by elements and species

        Args:
            expand_to_batch: Whether to expand arrays to the batch size. Defaults to ``False``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``. Must be
                ``False`` if used within a jitted context, as NumPy arrays are not compatible with
                JAX transformations (jit, vmap, etc.).
            ravel: Whether to ravel arrays to 1-D after expanding. Can be used when the expanded
                arrays are intended for conversion to DataFrames (which also requires
                ``to_numpy=False``). Defaults to ``False``.

        Returns:
            Dictionary of the solution with JAX or NumPy arrays, split by elements and species
        """
        out: dict[str, Any] = {}

        if not self.gas.is_empty:
            out["gas"] = self.gas.asdict_split()
        if not self.melt.is_empty:
            out["melt"] = self.melt.asdict_split()
        if not self.solid.is_empty:
            out["solid"] = self.solid.asdict_split()

        if len(self.condensates) > 0:
            condensate_dict: dict = {}
            for condensate in self.condensates:
                condensate_dict[condensate.phase.species.species_names[0]] = condensate.asdict()
            out["condensates"] = condensate_dict

        out["solver"] = self.multi_attempt_solution.asdict()
        out["state"] = self.state_asdict()

        out["constraints"] = {}
        out["constraints"].update(self.parameters.mass_constraints.asdict_split())

        if expand_to_batch:
            out = expand_jax_arrays_to_batch(out, self.batch_size, ravel=ravel)

        if to_numpy:
            out = convert_jax_arrays_to_numpy(out)

        return out

        # out.update(self.condensates_asdict())

        # Must vmap the residual evaluation to match what the solver did: parameters contains a
        # mix of scalar and batched leaves, so calling objective_function directly on the 2-D
        # solution gives incorrect results. vmap_axes_spec maps None/0 per leaf appropriately.
        # FIXME: This is breaking because all arrays including numpy are seen as batchable under
        # jit
        # objective_function_vmapped = eqx.filter_vmap(
        #    objective_function, in_axes=(0, vmap_axes_spec(self.parameters))
        # )
        # out["residual"] = objective_function_vmapped(self.solution, self.parameters)
        # out["solution"] = self.solution

        # out["totals"] = self.totals_asdict()

    def quick_look(self) -> dict[str, Any]:
        """Quick look at the output.

        Returns:
            A nested dictionary of the output, suitable for quick inspection and comparison.
        """
        out: dict[str, Any] = self.asdict(to_numpy=True)
        logger.info("Quick look output:\n%s", pformat(out))

        return out

    def to_dataframes(self, drop_unsuccessful_solves: bool = False) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Each top-level key becomes a DataFrame, with columns formed by joining nested keys with "."

        Args:
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.

        Returns:
            Dictionary mapping top-level keys to pandas DataFrames
        """

        def flatten(d: dict, parent_key: str = ""):
            """Recursively flattens a nested dictionary, joining keys with "." to form column names.

            Args:
                d: Dictionary to flatten
                parent_key: Prefix for keys (used during recursion)

            Returns:
                Flat dictionary with dot-joined keys
            """
            items: dict = {}
            for k, v in d.items():
                new_key: str = f"{parent_key}.{k}" if parent_key else str(k)
                if isinstance(v, dict):
                    items.update(flatten(v, new_key))
                else:
                    items[new_key] = v
            return items

        split_nested_dict: dict[str, Any] = self.asdict_split(
            expand_to_batch=True, to_numpy=True, ravel=True
        )

        result: dict = {}

        for top_key, value in split_nested_dict.items():
            if isinstance(value, dict):
                flat: dict = flatten(value)
                result[top_key] = pd.DataFrame(flat)
            else:
                result[top_key] = pd.DataFrame({top_key: value})

        if drop_unsuccessful_solves:
            logger.info("Dropping unsuccessful solves from output")
            result = self._drop_unsuccessful_solves(result)

        return result

    def to_excel(
        self, file_prefix: str = "atmodeller_out", drop_unsuccessful_solves: bool = False
    ) -> None:
        """Writes the output to an Excel file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
        """
        logger.info("Writing output to excel")

        out: dict[str, pd.DataFrame] = self.to_dataframes(
            drop_unsuccessful_solves=drop_unsuccessful_solves
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
        drop_unsuccessful_solves: bool = False,
    ) -> None:
        """Writes the output to a pickle file.

        Args:
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
        """
        logger.info("Writing output to pickle")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            drop_unsuccessful_solves=drop_unsuccessful_solves
        )
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)

    def compare(
        self,
        d1: dict,
        rtol: float,
        atol: float,
        log: bool = False,
        d2: Optional[dict] = None,
        path: tuple = (),
        all_match: bool = True,
    ) -> bool:
        """Compares two nested dictionaries of output.

        Args:
            d1: Target dictionary
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            log: Whether to compare the base-10 logarithm of the values
            d2: Dictionary to compare against. If ``None``, compares against the current output.
                Defaults to ``None``.
            path: Internal parameter for tracking the current path in the nested structure during
                recursion. Should not be set by the user. Defaults to an empty tuple.
            all_match: Internal parameter for tracking whether all comparisons have matched so far
                during recursion. Should not be set by the user. Defaults to ``True``.

        Returns:
            ``True`` if all values match within the specified tolerances, else ``False``
        """
        if d2 is None:
            d2 = self.asdict_split(to_numpy=True, ravel=True)

        keys = d1.keys()

        for key in keys:
            v1 = d1.get(key)
            v2 = d2.get(key)
            current_path = path + (key,)

            if isinstance(v1, dict) and isinstance(v2, dict):
                all_match = self.compare(v1, rtol, atol, log, v2, current_path, all_match)
            else:
                if isinstance(v1, (np.ndarray, float, int)) and isinstance(
                    v2, (np.ndarray, float, int)
                ):
                    if log:
                        v1, v2 = np.log10(v1), np.log10(v2)
                    is_close: bool = np.allclose(v1, v2, rtol=rtol, atol=atol)
                    all_match = all_match and is_close
                    logger.info(
                        "Comparing %s: %s vs %s --> %s",
                        current_path,
                        v1,
                        v2,
                        "match" if is_close else "mismatch",
                    )

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


# TODO: To reinstate at some point, but needs to be adapted to new output structure and parameters
# handling

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
