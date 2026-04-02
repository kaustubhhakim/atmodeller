# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output handling and export module.

This module provides a unified interface for extracting, converting, comparing, and exporting model
results from atmodeller simulations. It supports multiple output formats and is designed for
compatibility with JAX-based scientific workflows, with explicit warnings for operations that are
not JAX-compiled safe.

Key features:

- **Multiple output formats:**
    - Natural (internal) format: closely matches the model's internal array structure
    - Named arrays: output split and labeled by element and species names
    - Grouped by element/species: output grouped for easy lookup by element or species
- **Conversion utilities:**
    - Convert outputs to nested dictionaries (with JAX or NumPy arrays)
    - Expand arrays to batch size, ravel arrays for DataFrame conversion
    - Convert JAX arrays to NumPy arrays (not JAX-compiled safe)
- **Export options:**
    - Output as pandas DataFrames (one per top-level key)
    - Write results to Excel (with highlighting for unsuccessful solves)
    - Write results to pickle files for later analysis
- **Comparison and validation:**
    - Compare outputs for regression testing or validation (with tolerance controls)
    - Quick inspection/logging of results
- **JAX compatibility:**
    - Most output methods are compatible with JAX, but some (e.g., to_dict(to_numpy=True), compare,
      export methods) are **not** compatible with JAX-compiled workflows (e.g., inside a
      :func:`jax.jit` context). Sphinx warnings are included in relevant docstrings.

Classes:

- :func:`Output`: Master output interface for all formats and exports
- :func:`OutputNaturalDict`: Output in the model's natural (internal) format
- :func:`OutputNamedArraysDict`: Output split and labeled by element/species names
- :func:`OutputElementsSpeciesDict`: Output grouped by element and species
- :func:`BaseOutputDict`: Abstract base class for output dictionary representations

Utility functions are provided for expanding, raveling, and converting arrays in PyTrees.
"""

import logging
import pickle
from abc import abstractmethod
from pathlib import Path
from pprint import pformat
from typing import Any, Literal, Optional, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.tree_util import tree_map
from jaxtyping import Array, ArrayLike, Float, PyTree
from openpyxl.styles import PatternFill

from atmodeller import override
from atmodeller.containers import FugacityConstraintSet, MassConstraintSet, MultiAttemptSolution
from atmodeller.jax_utils import FloatArray, NpArray
from atmodeller.parameters import Parameters
from atmodeller.phases import (
    GasPhaseOutput,
    MeltPhase,
    PhaseOutput,
    PurePhase,
    SolidPhase,
    TPhase_co,
)

logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def flatten_dictionary(d: dict, parent_key: str = "") -> dict:
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
            items.update(flatten_dictionary(v, new_key))
        else:
            items[new_key] = v

    return items


def recursively_merge_dictionaries(d1: dict, d2: dict) -> dict:
    """Recursively merges two dictionaries.

    Args:
        d1: The first dictionary
        d2: The second dictionary, which will overwrite values in the first dictionary if there are
            duplicate keys

    Returns:
        The merged dictionary
    """
    out: dict = dict(d1)

    for k, v in d2.items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = recursively_merge_dictionaries(out[k], v)
            else:
                out[k] = v
        else:
            out[k] = v

    return out


def expand_jax_arrays_in_pytree_to_batch(pytree: PyTree, batch_size: int) -> PyTree:
    """Expands all arrays in a PyTree to the batch size.

    Note:
        :func:`jax.tree_util.tree_map` does not preserve the insertion order of standard Python
        dictionaries. Dictionary keys are treated as an unordered set and are typically processed
        in a canonical (sorted) order. If preserving key order is important, consider using
        :class:`collections.OrderedDict`.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to expand
        batch_size: Batch size to expand to

    Returns:
        PyTree with arrays expanded to batch size
    """

    def expand(x: Any) -> Any:
        if isinstance(x, jnp.ndarray):
            x = jnp.atleast_1d(x)
            # Always broadcast if shape[0] != batch_size
            if x.shape[0] != batch_size:
                x = jnp.broadcast_to(x, (batch_size,) + x.shape[1:])
            return x
        return x

    return tree_map(expand, pytree)


def ravel_jax_arrays_in_pytree(pytree: PyTree) -> PyTree:
    """Ravels all JAX arrays in a PyTree to 1-D arrays.

    Note:
        :func:`jax.tree_util.tree_map` does not preserve the insertion order of standard Python
        dictionaries. Dictionary keys are treated as an unordered set and are typically processed
        in a canonical (sorted) order. If preserving key order is important, consider using
        :class:`collections.OrderedDict`.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to ravel

    Returns:
        PyTree with all JAX arrays raveled to 1-D
    """

    def ravel(x: Any) -> Any:
        if isinstance(x, jnp.ndarray):
            return jnp.ravel(x)
        return x

    return tree_map(ravel, pytree)


def convert_jax_arrays_in_pytree_to_numpy(pytree: PyTree) -> PyTree:
    """Converts all JAX arrays in a PyTree to NumPy arrays.

    Note:
        :func:`jax.tree_util.tree_map` does not preserve the insertion order of standard Python
        dictionaries. Dictionary keys are treated as an unordered set and are typically processed
        in a canonical (sorted) order. If preserving key order is important, consider using
        :class:`collections.OrderedDict`.

    Args:
        pytree: PyTree (nested dict, list, tuple, etc.) of arrays to convert

    Returns:
        PyTree with JAX arrays converted to NumPy arrays
    """
    return tree_map(lambda x: np.asarray(x) if isinstance(x, jnp.ndarray) else x, pytree)


class BaseOutputDict(eqx.Module):
    """Represents the output of a model as a nested dictionary.

    This is a base class specification (explicit interface) for output dictionary representations.

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    parameters: Parameters
    multi_attempt_solution: MultiAttemptSolution

    @abstractmethod
    def _phase_output_to_dict(self, phase_output: PhaseOutput[TPhase_co]) -> dict[str, Any]:
        """Dictionary representation of the phase output

        Returns:
            A dictionary
        """

    @abstractmethod
    def to_dict(
        self, *, expand_to_batch: bool = False, to_numpy: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Output as a nested dictionary with JAX or NumPy arrays.

        Args:
            expand_to_batch: Whether to expand arrays to the batch size. Defaults to ``False``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``.
                Must be ``False`` if used within a jitted context, as NumPy arrays are not
                compatible with JAX transformations (jit, vmap, etc.).
            **kwargs: Arbitrary keyword arguments for the output dictionary

        Returns:
            Dictionary of the solution with JAX or NumPy arrays
        """

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
    def solution(self) -> Float[Array, "#n_batch twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.multi_attempt_solution.value

    def solution_to_dict(self) -> dict[str, ArrayLike]:
        """Returns a dictionary of solution arrays for each species.

        Returns:
            Dictionary mapping species names and their stability keys to arrays of values
        """
        out: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self.parameters.species.species_names):
            out[species_] = self.solution[..., nn]
            out[f"{species_}_stability"] = self.solution[
                ..., nn + self.parameters.species.number_species
            ]

        return out

    @property
    def condensate_phases(self) -> tuple[PhaseOutput[PurePhase], ...]:
        """Pure phase condensates output"""

        condensate_slice: slice = self.parameters.reaction_system.phase_system.condensates_slice

        condensates_out = []

        for nn, condensate in enumerate(self.parameters.reaction_system.phase_system.condensates):
            condensate_out = condensate.output(
                self.log_number_moles[..., condensate_slice][..., nn],
                self.log_stability[..., condensate_slice][..., nn],
                self._temperature,
                self._pressure,
            )
            condensates_out.append(condensate_out)

        return tuple(condensates_out)

    @property
    def gas(self) -> GasPhaseOutput:
        """Gas phase output"""

        gas_slice: slice = self.parameters.reaction_system.phase_system.gas_slice

        gas_output: PhaseOutput = self.parameters.reaction_system.phase_system.gas.output(
            self.log_number_moles[..., gas_slice],
            self.log_stability[..., gas_slice],
            self._temperature,
            self._pressure,
        )

        return cast(GasPhaseOutput, gas_output)

    @property
    def melt(self) -> PhaseOutput[MeltPhase]:
        """Melt phase output"""

        melt_slice: slice = self.parameters.reaction_system.phase_system.melt_slice

        melt_output: PhaseOutput[MeltPhase] = (
            self.parameters.reaction_system.phase_system.melt.output(
                self.log_number_moles[..., melt_slice],
                self.log_stability[..., melt_slice],
                self._temperature,
                self._pressure,
            )
        )

        return melt_output

    @property
    def solid(self) -> PhaseOutput[SolidPhase]:
        """Solid phase output"""

        solid_slice: slice = self.parameters.reaction_system.phase_system.solid_slice

        solid_output: PhaseOutput[SolidPhase] = (
            self.parameters.reaction_system.phase_system.solid.output(
                self.log_number_moles[..., solid_slice],
                self.log_stability[..., solid_slice],
                self._temperature,
                self._pressure,
            )
        )

        return solid_output

    @property
    def _temperature(self) -> FloatArray:
        """Temperature in K"""
        return self.parameters.state.temperature

    @property
    def temperature(self) -> Float[Array, "#n_batch 1"]:
        """Temperature in K"""
        return jnp.atleast_1d(self._temperature)[:, None]

    @property
    def _pressure(self) -> FloatArray:
        """Pressure in bar"""
        return self.parameters.state.get_pressure(self.log_number_moles)

    @property
    def pressure(self) -> Float[Array, "#n_batch 1"]:
        """Pressure in bar"""
        return jnp.atleast_1d(self._pressure)[:, None]

    def phase_to_dict(self, phase_output: PhaseOutput[TPhase_co]) -> dict[str, Any]:
        """Phase-level properties such as total mass, number of moles, molar mass, etc.

        Returns:
            A dictionary of phase-level properties
        """
        return {
            "background_mass": phase_output.background_mass,
            "background_number_moles": phase_output.background_number_moles,
            "background_molar_mass": phase_output.background_molar_mass,
            "mass": phase_output.phase_mass,
            "species_number_moles": phase_output.phase_species_number_moles,
            "elements_number_moles": phase_output.phase_element_number_moles,
            "molar_mass": phase_output.phase_molar_mass,
            "species_to_phase_mass_ratio": phase_output.species_to_phase_mass_ratio,
        }

    def solver_to_dict(self) -> dict[str, ArrayLike]:
        """Solver information such as success flags and number of iterations"""
        return self.multi_attempt_solution.asdict()

    def state_to_dict(self) -> dict[str, Any]:
        """Thermodynamic state of the system"""
        return self.parameters.state.asdict(self.solution)


class OutputNaturalDict(BaseOutputDict):
    """Represents the natural output of a model based on the arrays used internally.

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    @override
    def _phase_output_to_dict(self, phase_output: PhaseOutput[TPhase_co]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        # Phase-level properties
        out["phase"] = self.phase_to_dict(phase_output)

        out["elements"] = {
            "names": phase_output.phase.species.unique_elements,
            "mass": phase_output.element_mass,
            "number_moles": phase_output.element_number_moles,
        }
        out["species"] = {
            "names": phase_output.phase.species_names,
            "activity": phase_output.species_activity,
            "mass": phase_output.species_mass,
            "mass_fraction": phase_output.species_mass_fraction,
            "number_moles": phase_output.species_number_moles,
            "mole_fraction": phase_output.species_mole_fraction,
            "include_in_phase_mass": phase_output.include_in_mass_phase,
        }

        return out

    @override
    def to_dict(
        self, *, expand_to_batch: bool = False, to_numpy: bool = False, **kwargs
    ) -> dict[str, Any]:
        del kwargs
        out: dict[str, Any] = {}

        if not self.gas.is_empty:
            phase_name: str = self.gas.phase.name
            out[phase_name] = self._phase_output_to_dict(self.gas)
            # Additional outputs for the gas phase only
            out[phase_name]["species"]["partial_pressure"] = self.gas.species_partial_pressure
            out[phase_name]["phase"]["volume"] = self.gas.volume
            out[phase_name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[phase_name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[phase_name]["phase"]["pressure"] = self.gas.pressure

        if not self.melt.is_empty:
            phase_name = self.melt.phase.name
            out[phase_name] = self._phase_output_to_dict(self.melt)

        if not self.solid.is_empty:
            phase_name = self.solid.phase.name
            out[phase_name] = self._phase_output_to_dict(self.solid)

        if len(self.condensate_phases) > 0:
            # This retains symmetry with the output structure of the other phases (gas, melt, and
            # solid), where condensates are ordered in a list and identified by their species name
            # within the species sub-category.
            condensate_out: list = []
            for condensate in self.condensate_phases:
                condensate_out.append(self._phase_output_to_dict(condensate))
            out["condensates"] = condensate_out

        out["constraints"] = {
            "elements": {
                "number_moles": self.parameters.mass_constraints.abundance_mol(self.batch_size),
                "names": self.parameters.mass_constraints.species.unique_elements,
                "mass": self.parameters.mass_constraints.abundance_mass(self.batch_size),
            }
        }
        out["constraints"]["species"] = {
            "activity": jnp.exp(
                self.parameters.fugacity_constraints.log_fugacity(
                    jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
                )
            ),
            "names": self.parameters.fugacity_constraints.species.species_names,
        }

        out["solution"] = self.solution_to_dict()
        out["solver"] = self.solver_to_dict()
        out["state"] = self.state_to_dict()

        # Order of operations matters here: expansion must be done before conversion to NumPy
        # arrays
        if expand_to_batch:
            out = expand_jax_arrays_in_pytree_to_batch(out, self.batch_size)

        if to_numpy:
            out = convert_jax_arrays_in_pytree_to_numpy(out)

        return out


class OutputNamedArraysDict(BaseOutputDict):
    """Output dictionary with arrays split and labeled by element and species names"""

    @staticmethod
    def _split_by_name_and_add(
        names: tuple[str, ...], inarray: Array, output: dict, keyname: str
    ) -> None:
        """Splits the species/element-level data by species/element and adds them to the output.

        Args:
            names: The species/elements corresponding to the columns of the input array
            inarray: The input array to split
            output: The output dictionary to which the split entries will be added
            keyname: The name of the property being split (e.g., "mass", "number_moles", etc.)
                to use in the output keys
        """
        split_data: list[Array] = jnp.split(inarray, max(len(names), 1), axis=-1)
        out_dict: dict = output.setdefault(keyname, {})

        for ii, name in enumerate(names):
            out_dict[name] = split_data[ii]

    def _split_by_elements_and_add(
        self, phase_output: PhaseOutput[TPhase_co], inarray: Array, output: dict, keyname: str
    ) -> None:
        """Splits the element-level data by element and adds them to the output.

        Args:
            phase_output: The phase output object containing the element information
            inarray: The input array to split, with shape (... n_elements)
            output: The output dictionary to which the split entries will be added
            keyname: The name of the property being split (e.g., "mass", "number_moles", etc.)
                to use in the output keys
        """
        self._split_by_name_and_add(
            phase_output.phase.species.unique_elements, inarray, output, keyname
        )

    def _split_by_species_and_add(
        self, phase_output: PhaseOutput[TPhase_co], inarray: Array, output: dict, keyname: str
    ) -> None:
        """Splits the species-level data by species' and adds them to the output.

        Args:
            phase_output: The phase output object containing the species information
            inarray: The input array to split, with shape (... n_species)
            output: The output dictionary to which the split entries will be added
            keyname: The name of the property being split (e.g., "mass", "activity", etc.) to
                use in the output keys
        """
        self._split_by_name_and_add(phase_output.phase.species_names, inarray, output, keyname)

    @override
    def _phase_output_to_dict(self, phase_output: PhaseOutput[TPhase_co]) -> dict[str, Any]:

        out: dict[str, Any] = {}

        # Phase-level properties
        out["phase"] = self.phase_to_dict(phase_output)

        # Element-level properties, split by element
        elements_out: dict = out.setdefault("elements", {})
        self._split_by_elements_and_add(
            phase_output, phase_output.element_mass, elements_out, "mass"
        )
        self._split_by_elements_and_add(
            phase_output, phase_output.element_number_moles, elements_out, "number_moles"
        )

        # Species-level properties, split by species
        species_out: dict = out.setdefault("species", {})
        self._split_by_species_and_add(
            phase_output, phase_output.species_activity, species_out, "activity"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_mass, species_out, "mass"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_mass_fraction, species_out, "mass_fraction"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_number_moles, species_out, "number_moles"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.species_mole_fraction, species_out, "mole_fraction"
        )
        self._split_by_species_and_add(
            phase_output, phase_output.include_in_mass_phase, species_out, "include_in_phase_mass"
        )

        return out

    @override
    def to_dict(
        self,
        *,
        expand_to_batch: bool = False,
        ravel: bool = False,
        to_numpy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Complete output as a nested dictionary with JAX or NumPy arrays, split by element and
        species names.

        Args:
            expand_to_batch: Whether to expand arrays to the batch size. Defaults to ``False``.
            ravel: Whether to ravel arrays to 1-D after expanding. Can be used when the expanded
                arrays are intended for conversion to DataFrames (which also requires
                ``to_numpy=False``). Defaults to ``False``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``. Must be
                ``False`` if used within a jitted context, as NumPy arrays are not compatible with
                JAX transformations (jit, vmap, etc.).
            **kwargs: Arbitrary keyword arguments for the output dictionary

        Returns:
            Dictionary of the solution with JAX or NumPy arrays, split by element and species names
        """
        del kwargs
        out: dict[str, Any] = {}

        if not self.gas.is_empty:
            out[self.gas.phase.name] = self._phase_output_to_dict(self.gas)
            self._split_by_species_and_add(
                self.gas,
                self.gas.species_partial_pressure,
                out[self.gas.phase.name]["species"],
                "partial_pressure",
            )
            out[self.gas.phase.name]["phase"]["volume"] = self.gas.volume
            out[self.gas.phase.name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[self.gas.phase.name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[self.gas.phase.name]["phase"]["pressure"] = self.gas.pressure

        if not self.melt.is_empty:
            out[self.melt.phase.name] = self._phase_output_to_dict(self.melt)

        if not self.solid.is_empty:
            out[self.solid.phase.name] = self._phase_output_to_dict(self.solid)

        if len(self.condensate_phases) > 0:
            condensate_dict: dict = {}
            for condensate in self.condensate_phases:
                condensate_dict[condensate.phase.species.species_names[0]] = (
                    self._phase_output_to_dict(condensate)
                )

            out["condensates"] = condensate_dict

        out["constraints"] = {}
        elements_out: dict = out["constraints"].setdefault("elements", {})
        self._split_by_name_and_add(
            self.parameters.mass_constraints.species.unique_elements,
            self.parameters.mass_constraints.abundance_mass(self.batch_size),
            elements_out,
            "mass",
        )
        self._split_by_name_and_add(
            self.parameters.mass_constraints.species.unique_elements,
            self.parameters.mass_constraints.abundance_mol(self.batch_size),
            elements_out,
            "number_moles",
        )
        species_out: dict = out["constraints"].setdefault("species", {})
        evaluated_fugacity_constraints: Float[Array, "... n_species"] = jnp.exp(
            jnp.stack(
                [
                    constraint.log_fugacity(
                        jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
                    )
                    for constraint in self.parameters.fugacity_constraints.constraints
                ],
                axis=-1,
            )
        )
        self._split_by_name_and_add(
            self.parameters.fugacity_constraints.species.species_names,
            evaluated_fugacity_constraints,
            species_out,
            "activity",
        )

        out["solution"] = self.solution_to_dict()
        out["solver"] = self.solver_to_dict()
        out["state"] = self.state_to_dict()

        # Must vmap the residual evaluation to match what the solver did: parameters contains a
        # mix of scalar and batched leaves, so calling objective_function directly on the 2-D
        # solution gives incorrect results. vmap_axes_spec maps None/0 per leaf appropriately.
        # FIXME: This is breaking because all arrays including numpy are seen as batchable under
        # jit
        # objective_function_vmapped = eqx.filter_vmap(
        #    objective_function, in_axes=(0, vmap_axes_spec(self.parameters))
        # )
        # out["residual"] = objective_function_vmapped(self.solution, self.parameters)

        # Order of operations matters here: expansion must be done before ravl, and both before
        # conversion to NumPy arrays
        if expand_to_batch:
            out = expand_jax_arrays_in_pytree_to_batch(out, self.batch_size)

        if ravel:
            out = ravel_jax_arrays_in_pytree(out)

        if to_numpy:
            out = convert_jax_arrays_in_pytree_to_numpy(out)

        return out

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
            log: Whether to compare the base-10 logarithm of the values. Defaults to ``False``.
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
            d2 = self.to_dict(expand_to_batch=True, ravel=True, to_numpy=True)

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


class OutputElementsSpeciesDict(BaseOutputDict):
    """Output dictionary grouped by element and species names"""

    @staticmethod
    def _split_array_by_names(names: tuple[str, ...], inarray: Array) -> list[Array]:
        """Splits the input array into a list of arrays corresponding to the input names.

        Args:
            names: The species/elements corresponding to the columns of the input array
            inarray: The input array to split

        Returns:
            A list of arrays corresponding to the input names
        """
        return jnp.split(inarray, max(len(names), 1), axis=-1)

    @override
    def _phase_output_to_dict(self, phase_output: PhaseOutput[TPhase_co]) -> dict[str, Any]:

        out: dict[str, Any] = {}

        unique_elements: tuple[str, ...] = phase_output.phase.species.unique_elements
        element_mass: list[Array] = self._split_array_by_names(
            unique_elements, phase_output.element_mass
        )
        element_number_moles: list[Array] = self._split_array_by_names(
            unique_elements, phase_output.element_number_moles
        )

        for nn, element in enumerate(unique_elements):
            element_dict: dict[str, Any] = out.setdefault(element, {})
            phase_dict: dict[str, Any] = element_dict.setdefault(phase_output.phase.name, {})
            phase_dict["mass"] = element_mass[nn]
            phase_dict["number_moles"] = element_number_moles[nn]

        species_names: tuple[str, ...] = phase_output.phase.species_names
        species_activity: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_activity
        )
        species_mass: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_mass
        )
        species_mass_fraction: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_mass_fraction
        )
        species_number_moles: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_number_moles
        )
        species_mole_fraction: list[Array] = self._split_array_by_names(
            species_names, phase_output.species_mole_fraction
        )
        include_in_mass_phase: list[Array] = self._split_array_by_names(
            species_names, phase_output.include_in_mass_phase
        )

        for nn, species in enumerate(species_names):
            species_dict: dict[str, Any] = out.setdefault(species, {})
            phase_dict: dict[str, Any] = species_dict.setdefault(phase_output.phase.name, {})
            phase_dict["activity"] = species_activity[nn]
            phase_dict["mass"] = species_mass[nn]
            phase_dict["mass_fraction"] = species_mass_fraction[nn]
            phase_dict["number_moles"] = species_number_moles[nn]
            phase_dict["mole_fraction"] = species_mole_fraction[nn]
            phase_dict["include_in_phase_mass"] = include_in_mass_phase[nn]

        return out

    @override
    def to_dict(
        self,
        *,
        expand_to_batch: bool = False,
        ravel: bool = False,
        to_numpy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        out: dict[str, Any] = {}

        if not self.gas.is_empty:
            out = recursively_merge_dictionaries(out, self._phase_output_to_dict(self.gas))

            # Add the partial pressure of each species in the gas phase to the output
            species_partial_pressure: list[Array] = self._split_array_by_names(
                self.gas.phase.species_names, self.gas.species_partial_pressure
            )
            for nn, species in enumerate(self.gas.phase.species_names):
                species_dict: dict[str, Any] = out.setdefault(species, {})
                phase_dict: dict[str, Any] = species_dict.setdefault(self.gas.phase.name, {})
                phase_dict["partial_pressure"] = species_partial_pressure[nn]

            out[self.gas.phase.name] = {}
            out[self.gas.phase.name]["phase"] = self.phase_to_dict(self.gas)
            out[self.gas.phase.name]["phase"]["volume"] = self.gas.volume
            out[self.gas.phase.name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[self.gas.phase.name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[self.gas.phase.name]["phase"]["pressure"] = self.gas.pressure

            # Metallicity
            z_by_moles = 0
            z_by_mass = 0
            for element in self.gas.phase.species.unique_elements:
                if element != "H" and element != "He":
                    z_by_moles = z_by_moles + out[element][self.gas.phase.name]["number_moles"]
                    z_by_mass = z_by_mass + out[element][self.gas.phase.name]["mass"]
            z_by_moles = z_by_moles / out[self.gas.phase.name]["phase"]["elements_number_moles"]
            z_by_mass = z_by_mass / out[self.gas.phase.name]["phase"]["mass"]
            out[self.gas.phase.name]["phase"]["metallicity_by_moles"] = z_by_moles
            out[self.gas.phase.name]["phase"]["metallicity_by_mass"] = z_by_mass

        if not self.melt.is_empty:
            out = recursively_merge_dictionaries(out, self._phase_output_to_dict(self.melt))
            out[self.melt.phase.name] = {}
            out[self.melt.phase.name]["phase"] = self.phase_to_dict(self.melt)

        if not self.solid.is_empty:
            out = recursively_merge_dictionaries(out, self._phase_output_to_dict(self.solid))
            out[self.solid.phase.name] = {}
            out[self.solid.phase.name]["phase"] = self.phase_to_dict(self.solid)

        # Mass constraints
        mass_constraints: MassConstraintSet = self.parameters.mass_constraints
        unique_elements: tuple[str, ...] = mass_constraints.species.unique_elements
        element_mass: list[Array] = self._split_array_by_names(
            unique_elements, mass_constraints.abundance_mass(self.batch_size)
        )
        element_number_moles: list[Array] = self._split_array_by_names(
            unique_elements, mass_constraints.abundance_mol(self.batch_size)
        )

        for nn, element in enumerate(unique_elements):
            element_dict: dict[str, Any] = out.setdefault(element, {})
            constraints_dict: dict[str, Any] = element_dict.setdefault("constraints", {})
            constraints_dict["mass"] = element_mass[nn]
            constraints_dict["number_moles"] = element_number_moles[nn]

        # Fugacity constraints
        fugacity_constraints: FugacityConstraintSet = self.parameters.fugacity_constraints
        unique_species: tuple[str, ...] = fugacity_constraints.species.species_names
        species_log_fugacity: list[Array] = self._split_array_by_names(
            unique_species,
            fugacity_constraints.log_fugacity(
                jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
            ),
        )

        for nn, species in enumerate(unique_species):
            species_dict: dict[str, Any] = out.setdefault(species, {})
            constraints_dict: dict[str, Any] = species_dict.setdefault("constraints", {})
            constraints_dict["fugacity"] = jnp.exp(species_log_fugacity[nn])

        out["solution"] = self.solution_to_dict()
        out["solver"] = self.solver_to_dict()
        out["state"] = self.state_to_dict()

        if expand_to_batch:
            out = expand_jax_arrays_in_pytree_to_batch(out, self.batch_size)

        if ravel:
            out = ravel_jax_arrays_in_pytree(out)

        if to_numpy:
            out = convert_jax_arrays_in_pytree_to_numpy(out)

        return out


class Output(eqx.Module):
    """Master output interface for model results.

    This class provides a unified entry point for all output formats, conversions, comparisons,
    and exports. It wraps the various output dictionary representations (natural, named arrays,
    grouped by element/species) and exposes methods for:

    - Outputting results in different formats (dict, DataFrame, Excel, pickle)
    - Comparing outputs for regression testing/validation
    - Quick inspection and logging

    .. warning::
        Some methods (such as ``to_dict(to_numpy=True)``, ``compare``, and export methods) are
        **not compatible** with JAX-compiled workflows (e.g., inside a ``jax.jit`` context), as
        they may use operations or objects that are not supported by JAX transformations.

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    parameters: Parameters
    multi_attempt_solution: MultiAttemptSolution

    def to_dict(
        self,
        format: Literal["natural", "named_arrays", "elements_species"] = "named_arrays",
        to_numpy: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Output as a nested dictionary with JAX or NumPy arrays.

        .. warning::
            ``to_numpy`` must be ``False`` if used within a jitted context, as NumPy arrays are not
            compatible with JAX transformations (jit, vmap, etc.).

        Args:
            format: The format of the output dictionary. Can be ``natural`` for the natural output
                format based on the arrays used internally, ``named_arrays`` for an alternative
                format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``.
                Must be ``False`` if used within a jitted context, as NumPy arrays are not
                compatible with JAX transformations (jit, vmap, etc.).
            **kwargs: Arbitrary keyword arguments for the output dictionary

        Returns:
            Dictionary of the solution with JAX or NumPy arrays in the specified format
        """
        if format == "natural":
            return OutputNaturalDict(self.parameters, self.multi_attempt_solution).to_dict(
                to_numpy=to_numpy, **kwargs
            )
        elif format == "named_arrays":
            return OutputNamedArraysDict(self.parameters, self.multi_attempt_solution).to_dict(
                to_numpy=to_numpy, **kwargs
            )
        elif format == "elements_species":
            return OutputElementsSpeciesDict(self.parameters, self.multi_attempt_solution).to_dict(
                to_numpy=to_numpy, **kwargs
            )
        else:
            raise ValueError(f"Invalid output format: {format}")

    def compare(self, d1: dict, rtol: float, atol: float, log: bool = False) -> bool:
        """Compares a target dictionary to the model output.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context)

        Args:
            d1: Target dictionary
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison
            log: Whether to compare the base-10 logarithm of the values. Defaults to ``False``.

        Returns:
            ``True`` if all values match within the specified tolerances, else ``False``
        """
        return OutputNamedArraysDict(self.parameters, self.multi_attempt_solution).compare(
            d1, rtol, atol, log
        )

    def quick_look(
        self, format: Literal["natural", "named_arrays", "elements_species"] = "named_arrays"
    ) -> None:
        """Quick look at the output.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context)

        Args:
            format: The format of the output dictionary. Can be ``natural`` for the natural output
                format based on the arrays used internally, ``named_arrays`` for an alternative
                format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.

        Returns:
            A nested dictionary of the output, suitable for quick inspection and comparison.
        """
        out: dict[str, Any] = self.to_dict(format=format, to_numpy=True)
        logger.info("Quick look output:\n%s", pformat(out, sort_dicts=False))

    def _drop_unsuccessful_solves(
        self, dataframes: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Drops unsuccessful solves.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context)

        Args:
            dataframes: Dataframes from which to drop unsuccessful models

        Returns:
            Dictionary of dataframes without unsuccessful models
        """
        return {
            key: df.loc[np.asarray(self.multi_attempt_solution.solver_success)]
            for key, df in dataframes.items()
        }

    def to_dataframes(
        self,
        format: Literal["named_arrays", "elements_species"] = "named_arrays",
        drop_unsuccessful_solves: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Gets the output in a dictionary of dataframes.

        Each top-level key becomes a DataFrame, with columns formed by joining nested keys with "."

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context)

        Args:
            format: The format of the output dictionary. Can be ``natural`` for the natural output
                format based on the arrays used internally, ``named_arrays`` for an alternative
                format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.

        Returns:
            Dictionary mapping top-level keys to pandas DataFrames
        """
        nested_dict: dict[str, Any] = self.to_dict(
            format=format, to_numpy=True, expand_to_batch=True, ravel=True
        )

        result: dict[str, pd.DataFrame] = {}

        for top_key, value in nested_dict.items():
            if isinstance(value, dict):
                flat: dict = flatten_dictionary(value)
                result[top_key] = pd.DataFrame(flat)
            else:
                result[top_key] = pd.DataFrame({top_key: value})

        if drop_unsuccessful_solves:
            logger.info("Dropping unsuccessful solves from output")
            result = self._drop_unsuccessful_solves(result)

        return result

    def to_excel(
        self,
        format: Literal["named_arrays", "elements_species"] = "named_arrays",
        file_prefix: str = "atmodeller_out",
        drop_unsuccessful_solves: bool = False,
    ) -> None:
        """Writes the output to an Excel file.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context)

        Args:
            format: The format of the output dictionary. Can be ``natural`` for the natural output
                format based on the arrays used internally, ``named_arrays`` for an alternative
                format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
        """
        logger.info("Writing output to excel")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            format=format, drop_unsuccessful_solves=drop_unsuccessful_solves
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
        format: Literal["named_arrays", "elements_species"] = "named_arrays",
        file_prefix: Path | str = "atmodeller_out",
        drop_unsuccessful_solves: bool = False,
    ) -> None:
        """Writes the output to a pickle file.

        .. warning::
            Not compatible with JAX-compiled workflows (e.g., inside a ``jax.jit`` context)

        Args:
            format: The format of the output dictionary. Can be ``natural`` for the natural output
                format based on the arrays used internally, ``named_arrays`` for an alternative
                format with named arrays, or ``elements_species`` for an alternative
                format grouped by element and species names. Defaults to ``named_arrays``.
            file_prefix: Prefix of the output file. Defaults to atmodeller_out.
            drop_unsuccessful_solves: Whether to drop unsuccessful solves from the output. Defaults
                to ``False``.
        """
        logger.info("Writing output to pickle")
        out: dict[str, pd.DataFrame] = self.to_dataframes(
            format=format, drop_unsuccessful_solves=drop_unsuccessful_solves
        )
        output_file: Path = Path(f"{file_prefix}.pkl")

        with open(output_file, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info("Output written to %s", output_file)


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
