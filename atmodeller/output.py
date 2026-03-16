# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""New core functionality for output"""

import logging
import pickle
from abc import abstractmethod
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

from atmodeller import override
from atmodeller.containers import MassConstraintSet
from atmodeller.engine import get_total_pressure
from atmodeller.parameters import Parameters
from atmodeller.phase_base import TPhase_co
from atmodeller.phases import GasPhaseOutput, MeltPhase, PhaseOutput, PurePhase, SolidPhase
from atmodeller.utilities import recursively_merge_dictionaries

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


class OutputDict(eqx.Module):
    """Represents the output of a model solution as a nested dictionary.

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
    def to_dict(self, *, to_numpy: bool = False, **kwargs) -> dict[str, Any]:
        """Output as a nested dictionary with JAX or NumPy arrays.

        Args:
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
    def main_phases(self) -> tuple[PhaseOutput, ...]:
        """Main phases (gas, melt, solid) output as a list"""
        return (self.gas, self.melt, self.solid)

    @property
    def solution(self) -> Float[Array, "#n_batch twice_species"]:
        """Solution array for all species i.e. log number of moles and log stability"""
        return self.multi_attempt_solution.value

    @property
    def condensates(self) -> tuple[PhaseOutput[PurePhase], ...]:
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

    @property
    def temperature(self) -> Float[Array, "#n_batch 1"]:
        """Temperature in K"""
        return jnp.atleast_2d(self.parameters.state.temperature).T

    @property
    def pressure(self) -> Float[Array, "#n_batch 1"]:
        """Pressure in bar"""
        return jnp.atleast_2d(get_total_pressure(self.parameters, self.solution)).T

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
            "number_moles": phase_output.phase_number_moles,
            "molar_mass": phase_output.phase_molar_mass,
            "species_to_phase_mass_ratio": phase_output.species_to_phase_mass_ratio,
        }

    def solver_to_dict(self) -> dict[str, ArrayLike]:
        """Solver information such as success flags and number of iterations"""
        return self.multi_attempt_solution.asdict()

    def state_to_dict(self) -> dict[str, Any]:
        """Thermodynamic state of the system"""
        return self.parameters.state.asdict(jnp.squeeze(self.gas.phase_mass))

    def quick_look(self) -> dict[str, Any]:
        """Quick look at the output.

        Returns:
            A nested dictionary of the output, suitable for quick inspection and comparison.
        """
        out: dict[str, Any] = self.to_dict(to_numpy=True)
        logger.info("Quick look output:\n%s", pformat(out))

        return out


class OutputNaturalDict(OutputDict):
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

        out["constraints"] = {
            # Mass constraints are currently only included at the total system level
            "elements": {
                "number_moles": self.parameters.mass_constraints.abundance_mol(),
                "names": self.parameters.mass_constraints.species.unique_elements,
                "mass": self.parameters.mass_constraints.abundance_mass(),
            }
        }

        return out

    @override
    def to_dict(self, *, to_numpy: bool = False, **kwargs) -> dict[str, Any]:
        del kwargs
        out: dict[str, Any] = {}

        if not self.gas.is_empty:
            phase_name: str = self.gas.phase.name
            out[phase_name] = self._phase_output_to_dict(self.gas)
            out[phase_name]["species"]["partial_pressure"] = self.gas.species_partial_pressure
            out[phase_name]["phase"]["volume"] = self.gas.volume
            out[phase_name]["phase"]["log10dIW_1_bar"] = self.gas.log10dIW_1_bar
            out[phase_name]["phase"]["log10dIW_P"] = self.gas.log10dIW_P
            out[self.gas.phase.name]["phase"]["pressure"] = self.gas.pressure

        if not self.melt.is_empty:
            phase_name = self.melt.phase.name
            out[phase_name] = self._phase_output_to_dict(self.melt)

        if not self.solid.is_empty:
            phase_name = self.solid.phase.name
            out[phase_name] = self._phase_output_to_dict(self.solid)

        if len(self.condensates) > 0:
            # This retains symmetry with the output structure of the other phases (gas, melt, and
            # solid), where condensates are ordered in a list and identified by their species name
            # within the species sub-category.
            condensate_out: list = []
            for condensate in self.condensates:
                condensate_out.append(self._phase_output_to_dict(condensate))
            out["condensates"] = condensate_out

        out["solver"] = self.multi_attempt_solution.asdict()
        out["state"] = self.state_to_dict()

        out["constraints"] = {}
        # FIXME
        # out["constraints"].update(self.parameters.mass_constraints.to_dict())
        # out["constraints"].update(
        #    self.parameters.fugacity_constraints.to_dict(
        #        jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
        #    )
        # )

        if to_numpy:
            out = convert_jax_arrays_to_numpy(out)

        return out


class OutputElementsSpeciesDict(OutputDict):
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
        """Dictionary representation of the phase output, grouped by element and species names

        This is an alternative output format that groups the output by element and species names,
        which can be more intuitive for certain types of analysis.

        Returns:
            A dictionary
        """
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
    def to_dict(self, to_numpy: bool = False, **kwargs) -> dict[str, Any]:
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

        if not self.melt.is_empty:
            out = recursively_merge_dictionaries(out, self._phase_output_to_dict(self.melt))

        if not self.solid.is_empty:
            out = recursively_merge_dictionaries(out, self._phase_output_to_dict(self.solid))

        # Mass constraints
        mass_constraints: MassConstraintSet = self.parameters.mass_constraints
        unique_elements: tuple[str, ...] = mass_constraints.species.unique_elements
        element_mass: list[Array] = self._split_array_by_names(
            unique_elements, mass_constraints.abundance_mass()
        )
        element_number_moles: list[Array] = self._split_array_by_names(
            unique_elements, mass_constraints.abundance_mol()
        )

        for nn, element in enumerate(unique_elements):
            element_dict: dict[str, Any] = out.setdefault(element, {})
            constraints_dict: dict[str, Any] = element_dict.setdefault("constraints", {})
            constraints_dict["mass"] = element_mass[nn]
            constraints_dict["number_moles"] = element_number_moles[nn]

        # TODO: Fugacity constraints

        if to_numpy:
            out = convert_jax_arrays_to_numpy(out)

        return out


class OutputSplit(OutputDict):
    """Output dictionary split by element and species names"""

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
            to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``. Must be
                ``False`` if used within a jitted context, as NumPy arrays are not compatible with
                JAX transformations (jit, vmap, etc.).
            ravel: Whether to ravel arrays to 1-D after expanding. Can be used when the expanded
                arrays are intended for conversion to DataFrames (which also requires
                ``to_numpy=False``). Defaults to ``False``.

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

        if len(self.condensates) > 0:
            condensate_dict: dict = {}
            for condensate in self.condensates:
                condensate_dict[condensate.phase.species.species_names[0]] = (
                    self._phase_output_to_dict(condensate)
                )

            out["condensates"] = condensate_dict

        out["solver"] = self.multi_attempt_solution.asdict()
        out["state"] = self.state_to_dict()

        out["constraints"] = {}
        elements_out: dict = out["constraints"].setdefault("elements", {})
        self._split_by_name_and_add(
            self.parameters.mass_constraints.species.unique_elements,
            self.parameters.mass_constraints.abundance_mass(),
            elements_out,
            "mass",
        )
        self._split_by_name_and_add(
            self.parameters.mass_constraints.species.unique_elements,
            self.parameters.mass_constraints.abundance_mol(),
            elements_out,
            "number_moles",
        )
        species_out: dict = out["constraints"].setdefault("species", {})
        evaluated_fugacity_constraints = jnp.exp(
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

        if expand_to_batch:
            out = expand_jax_arrays_to_batch(out, self.batch_size, ravel=ravel)

        if to_numpy:
            out = convert_jax_arrays_to_numpy(out)

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


# Hot swap output dictionary
Output = OutputElementsSpeciesDict


class OutputRefactoring(OutputDict):
    """Output

    Properties can be called within a jitted context to access output quantites for downstream
    processing. Arrays are always broadcastable to avoid necessitating expanding all arrays to the
    batch size.

    Args:
        parameters: Parameters
        multi_attempt_solution: Multiple attempt solution object
    """

    # @property
    # def constraints_element_mass(self) -> Float[Array, "#n_batch n_elements"]:
    #     """Element mass constraints in kg"""
    #     return self.parameters.mass_constraints.abundance_mass()

    # @property
    # def constraints_element_moles(self) -> Float[Array, "#n_batch n_elements"]:
    #     """Element abundance constraints in moles"""
    #     return self.parameters.mass_constraints.abundance_mol()

    # @property
    # def constraints_fugacity(self) -> Float[Array, "#n_batch n_species"]:
    #     """Fugacity constraints in bar"""
    #     constraints: Float[Array, "#n_batch n_species"] = jnp.stack(
    #         [
    #             jnp.exp(constraint.log_fugacity(self.temperature, self.pressure))
    #             for constraint in self.parameters.fugacity_constraints.constraints
    #         ],
    #         axis=-1,
    #     )
    #     return constraints

    # def asdict_split(
    #     self, *, expand_to_batch: bool = False, to_numpy: bool = False, ravel: bool = False
    # ) -> dict[str, Any]:
    #     """Output as a nested dictionary with JAX or NumPy arrays, split by elements and species

    #     Args:
    #         expand_to_batch: Whether to expand arrays to the batch size. Defaults to ``False``.
    #         to_numpy: Whether to convert JAX arrays to NumPy arrays. Defaults to ``False``. Must be
    #             ``False`` if used within a jitted context, as NumPy arrays are not compatible with
    #             JAX transformations (jit, vmap, etc.).
    #         ravel: Whether to ravel arrays to 1-D after expanding. Can be used when the expanded
    #             arrays are intended for conversion to DataFrames (which also requires
    #             ``to_numpy=False``). Defaults to ``False``.

    #     Returns:
    #         Dictionary of the solution with JAX or NumPy arrays, split by elements and species
    #     """
    #     out: dict[str, Any] = {}

    #     if not self.gas.is_empty:
    #         out["gas"] = self.gas.asdict_split()
    #     if not self.melt.is_empty:
    #         out["melt"] = self.melt.asdict_split()
    #     if not self.solid.is_empty:
    #         out["solid"] = self.solid.asdict_split()

    #     if len(self.condensates) > 0:
    #         condensate_dict: dict = {}
    #         for condensate in self.condensates:
    #             condensate_dict[condensate.phase.species.species_names[0]] = (
    #                 condensate.asdict_split()
    #             )
    #         out["condensates"] = condensate_dict

    #     out["solver"] = self.multi_attempt_solution.asdict()
    #     out["state"] = self.state_asdict()

    #     out["constraints"] = {}
    #     out["constraints"].update(self.parameters.mass_constraints.asdict_split())
    #     out["constraints"].update(
    #         self.parameters.fugacity_constraints.asdict_split(
    #             jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
    #         )
    #     )

    #     if expand_to_batch:
    #         out = expand_jax_arrays_to_batch(out, self.batch_size, ravel=ravel)

    #     if to_numpy:
    #         out = convert_jax_arrays_to_numpy(out)

    #     return out

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
