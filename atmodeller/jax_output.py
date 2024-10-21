#
# Copyright 2024 Dan J. Bower
#
# This file is part of Atmodeller.
#
# Atmodeller is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Atmodeller is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Atmodeller. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Output for JAX-based code

This uses existing functions as much as possible to calculate desired output quantities. Notably,
some of these functions must be vmapped to account for batch calculations
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller.containers import Solution, TracedParameters
from atmodeller.engine import get_log_activity
from atmodeller.utilities import (
    log_pressure_from_log_number_density,
    unscale_number_density,
)

if TYPE_CHECKING:
    from atmodeller.classes import InteriorAtmosphere

logger: logging.Logger = logging.getLogger(__name__)


class JaxOutput:
    """Converts the array output to user-friendly scaled output"""

    def __init__(
        self,
        solution: Array,
        interior_atmosphere: InteriorAtmosphere,
        initial_solution: Solution,
        traced_parameters: TracedParameters,
    ):
        logger.info("Creating Output")
        self._interior_atmosphere: InteriorAtmosphere = interior_atmosphere
        self._initial_solution: Solution = initial_solution
        self._traced_parameters: TracedParameters = traced_parameters

        if self.model.is_batch:
            self._axis: int = 1
        else:
            self._axis = 0

        # Scale the solution quantities
        log_number_density, log_stability = jnp.split(solution, 2, axis=self._axis)
        self._log_number_density: Array = unscale_number_density(
            log_number_density, self.model.log_scaling
        )
        # Stability is non-dimensional and therefore does not need scaling
        self._log_stability: Array = log_stability

        # self.test_print_output()

    @property
    def log_number_density(self) -> Array:
        r"""Log number density in :math:`\mathrm{molecules}\, \mathrm{m}^{-3}`"""
        return self._log_number_density

    @property
    def log_stability(self) -> Array:
        """Log stability of all species"""
        return self._log_stability

    @property
    def model(self) -> InteriorAtmosphere:
        """Interior atmosphere model"""
        return self._interior_atmosphere

    def _activity_without_stability(self) -> Array:
        """Gets activity without stability of all species

        Returns:
            Activity without stability of all species
        """
        return jnp.exp(self._log_activity_without_stability())

    def _log_activity_without_stability(self) -> Array:
        """Gets log activity without stability of all species

        Args:
            Log activity without stability of all species
        """
        if self.model.is_batch:
            log_activity_func: Callable = jax.vmap(
                get_log_activity, in_axes=(self.model.traced_parameters_vmap, None, 0)
            )
        else:
            log_activity_func = get_log_activity

        log_activity: Array = log_activity_func(
            self._traced_parameters, self.model.fixed_parameters, self.pressure()
        )

        return log_activity

    def log_activity(self) -> Array:
        """Gets log activity of all species.

        This is usually what the user wants when referring to activity because it includes a
        consideration of species stability

        Returns:
            Log activity of all species
        """
        log_activity_without_stability: Array = self._log_activity_without_stability()
        log_activity: Array = log_activity_without_stability - jnp.exp(self.log_stability)

        return log_activity

    def log_pressure(self) -> Array:
        """Gets log pressure of all species in bar

        This will compute log pressure of all species, including condensates, for simplicity.

        Returns:
            Log pressure of all species in bar
        """

        # For single calculations or calculations where temperature is not batched
        log_pressure_func: Callable = log_pressure_from_log_number_density

        # Must vectorise for batch calculations with temperature
        if self.model.is_batch:
            if self.model.traced_parameters_vmap.planet.surface_temperature == 0:
                log_pressure_func = jax.vmap(log_pressure_from_log_number_density, in_axes=(0, 0))

        log_pressure: Array = log_pressure_func(
            self.log_number_density, self._traced_parameters.planet.surface_temperature
        )

        return log_pressure

    def activity(self) -> Array:
        """Gets the activity of all species

        Returns:
            Activity of all species
        """
        return jnp.exp(self.log_activity())

    def number_density(self) -> Array:
        r"""Gets number density of all species

        Returns:
            Number density in :math:`\mathrm{molecules}\, \mathrm{m}^{-3}`
        """
        return jnp.exp(self.log_number_density)

    def pressure(self) -> Array:
        """Gets pressure of all species in bar

        Returns:
            Pressure in bar
        """
        pressure: Array = jnp.exp(self.log_pressure())

        return pressure

    def quick_look(self) -> dict[str, ArrayLike]:
        """Quick look at the solution

        Provides a quick first glance at the output with convenient units and to ease comparison
        with test or benchmark data.

        Returns:
            Dictionary of the solution
        """

        def collapse_single_entry_values(input_dict: dict[str, ArrayLike]) -> dict[str, ArrayLike]:
            output_dict: dict[str, ArrayLike] = {}
            for key, value in input_dict.items():
                if value.size == 1:  # type: ignore
                    output_dict[key] = float(value[0])  # type: ignore
                else:
                    output_dict[key] = value

            return output_dict

        output_dict: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self.model.species):
            pressure: Array = jnp.atleast_2d(self.pressure())[:, nn]
            activity: Array = jnp.atleast_2d(self.activity())[:, nn]
            output_dict[species_.name] = pressure
            output_dict[f"{species_.name}_activity"] = activity

        return collapse_single_entry_values(output_dict)

    def stability(self) -> Array:
        """Gets stability of all species

        Returns:
            Stability of all the species
        """
        return jnp.exp(self.log_stability)

    def test_print_output(self) -> None:

        print("log_number_density = ", self.log_number_density)
        print("log_stability = ", self.log_stability)
        print("log_pressure = ", self.log_pressure())
        print("pressure = ", self.pressure())
        print("log_activity = ", self.log_activity())
        print("activity = ", self.activity())

    # @property
    # def size(self) -> int:
    #     """Number of rows"""
    #     try:
    #         return len(self.data["solution"])
    #     except KeyError:
    #         return 0

    # @classmethod
    # def read_pickle(cls, pickle_file: Path | str) -> Output:
    #     """Reads output data from a pickle file and creates an Output instance.

    #     Args:
    #         pickle_file: Pickle file of the output from a previous (or similar) model run.
    #             Importantly, the reaction network must be the same (same number of species in the
    #             same order) and the constraints must be the same (also in the same order).

    #     Returns:
    #         Output
    #     """
    #     with open(pickle_file, "rb") as handle:
    #         output_data: dict[str, list[dict[str, float]]] = pickle.load(handle)

    #     logger.info("%s: Reading data from %s", cls.__name__, pickle_file)

    #     return cls(output_data)

    # @classmethod
    # def from_dataframes(cls, dataframes: dict[str, pd.DataFrame]) -> Output:
    #     """Reads a dictionary of dataframes and creates an Output instance.

    #     Args:
    #         dataframes: A dictionary of dataframes.

    #     Returns:
    #         Output
    #     """
    #     output_data: dict[str, list[dict[Hashable, float]]] = {}
    #     for key, dataframe in dataframes.items():
    #         output_data[key] = dataframe.to_dict(orient="records")

    #     return cls(output_data)

    # def add(
    #     self,
    #     solution: Solution,
    #     residual_dict: dict[str, float],
    #     constraints_dict: dict[str, float],
    #     extra_output: dict[str, float] | None = None,
    # ) -> None:
    #     """Adds all outputs.

    #     Args:
    #         solution: Solution
    #         residual_dict: Dictionary of residuals
    #         constraints_dict: Dictionary of constraints
    #         extra_output: Extra data to write to the output. Defaults to None.
    #     """
    #     output_full: dict[str, dict[str, float]] = solution.output_full()

    #     # Back-compute and add the log10 shift relative to the default iron-wustite buffer
    #     if "O2_g" in output_full:
    #         temperature: float = output_full["atmosphere"]["temperature"]
    #         pressure: float = output_full["atmosphere"]["pressure"]
    #         # pylint: disable=invalid-name
    #         O2_g_output: dict[str, float] = output_full["O2_g"]
    #         O2_g_fugacity: float = O2_g_output["fugacity"]
    #         O2_g_shift_at_1bar: float = solve_for_log10_dIW(O2_g_fugacity, temperature)
    #         O2_g_output["log10dIW_1_bar"] = O2_g_shift_at_1bar
    #         O2_g_shift_at_P: float = solve_for_log10_dIW(O2_g_fugacity, temperature, pressure)
    #         O2_g_output["log10dIW_P"] = O2_g_shift_at_P

    #     for key, value in output_full.items():
    #         data_list: list[dict[str, float]] = self.data.setdefault(key, [])
    #         data_list.append(value)

    #     constraints_list: list[dict[str, float]] = self.data.setdefault("constraints", [])
    #     constraints_list.append(constraints_dict)
    #     residual_list: list[dict[str, float]] = self.data.setdefault("residual", [])
    #     residual_list.append(residual_dict)

    #     if extra_output is not None:
    #         data_list: list[dict[str, float]] = self.data.setdefault("extra", [])
    #         data_list.append(extra_output)

    # def to_dataframes(self) -> dict[str, pd.DataFrame]:
    #     """Output as a dictionary of dataframes

    #     Returns:
    #         The output as a dictionary of dataframes
    #     """
    #     out: dict[str, pd.DataFrame] = {
    #         key: pd.DataFrame(value) for key, value in self.data.items()
    #     }
    #     return out

    # def to_excel(self, file_prefix: Path | str = "atmodeller_out") -> None:
    #     """Writes the output to an Excel file.

    #     Args:
    #         file_prefix: Prefix of the output file. Defaults to atmodeller_out.
    #     """
    #     out: dict[str, pd.DataFrame] = self.to_dataframes()
    #     output_file: Path = Path(f"{file_prefix}.xlsx")

    #     with pd.ExcelWriter(output_file, engine="openpyxl") as writer:  # pylint: disable=E0110
    #         for df_name, df in out.items():
    #             df.to_excel(writer, sheet_name=df_name, index=True)

    #     logger.info("Output written to %s", output_file)

    # def to_pickle(self, file_prefix: Path | str = "atmodeller_out") -> None:
    #     """Writes the output to a pickle file.

    #     Args:
    #         file_prefix: Prefix of the output file. Defaults to atmodeller_out.
    #     """
    #     output_file: Path = Path(f"{file_prefix}.pkl")

    #     with open(output_file, "wb") as handle:
    #         pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #     logger.info("Output written to %s", output_file)

    # def _check_keys_the_same(self, other: Output) -> None:
    #     """Checks if the keys are the same in 'other' before combining output.

    #     Args:
    #         other: Other output to potentially combine (if keys are the same)
    #     """
    #     if not self.keys() == other.keys():
    #         msg: str = "Keys for 'other' are not the same as 'self' so cannot combine them"
    #         logger.error(msg)
    #         raise KeyError(msg)

    # def __add__(self, other: Output) -> Output:
    #     """Addition

    #     Args:
    #         other: Other output to combine with self

    #     Returns:
    #         Combined output
    #     """
    #     self._check_keys_the_same(other)
    #     output: Output = copy.deepcopy(self)
    #     for key in self.keys():
    #         output[key].extend(other[key])

    #     return output

    # def __iadd__(self, other: Output) -> Output:
    #     """In-place addition

    #     Args:
    #         other: Other output to combine with self in-place

    #     Returns:
    #         self
    #     """
    #     self._check_keys_the_same(other)
    #     for key in self:
    #         self[key].extend(other[key])

    #     return self

    # def filter_by_index_notin(self, other: Output, index_key: str, index_name: str) -> Output:
    #     """Filters out the entries in `self` that are not present in the index of `other`

    #     Args:
    #         other: Other output with the filtering index
    #         index_key: Key of the index
    #         index_name: Name of the index

    #     Returns:
    #         The filtered output
    #     """
    #     self_dataframes: dict[str, pd.DataFrame] = self.to_dataframes()
    #     other_dataframes: dict[str, pd.DataFrame] = other.to_dataframes()
    #     index: pd.Index = pd.Index(other_dataframes[index_key][index_name])

    #     for key, dataframe in self_dataframes.items():
    #         self_dataframes[key] = dataframe[~dataframe.index.isin(index)]

    #     return self.from_dataframes(self_dataframes)

    # def reorder(self, other: Output, index_key: str, index_name: str) -> Output:
    #     """Reorders all the entries according to an index in `other`

    #     Args:
    #         other: Other output with the reordering index
    #         index_key: Key of the index
    #         index_name: Name of the index

    #     Returns:
    #         The reordered output
    #     """
    #     self_dataframes: dict[str, pd.DataFrame] = self.to_dataframes()
    #     other_dataframes: dict[str, pd.DataFrame] = other.to_dataframes()
    #     index: pd.Index = pd.Index(other_dataframes[index_key][index_name])

    #     for key, dataframe in self_dataframes.items():
    #         self_dataframes[key] = dataframe.reindex(index)

    #     return self.from_dataframes(self_dataframes)

    # def __call__(
    #     self,
    #     file_prefix: Path | str = "atmodeller_out",
    #     to_dataframes: bool = True,
    #     to_pickle: bool = False,
    #     to_excel: bool = False,
    # ) -> dict | None:
    #     """Gets the output as a dict and/or optionally write it to a pickle or Excel file.

    #     Args:
    #         file_prefix: Prefix of the output file if writing to a pickle or Excel. Defaults to
    #             atmodeller_out
    #         to_dataframes: Returns the output data in a dictionary of dataframes. Defaults to
    #             True.
    #         to_pickle: Writes a pickle file. Defaults to False.
    #         to_excel: Writes an Excel file. Defaults to False.

    #     Returns:
    #         A dictionary of the output or None if no data
    #     """
    #     if self.size == 0:
    #         logger.warning("There is no data to export")
    #         return None

    #     if to_pickle:
    #         self.to_pickle(file_prefix)

    #     if to_excel:
    #         self.to_excel(file_prefix)

    #     if to_dataframes:
    #         return self.to_dataframes()
    #     else:
    #         return self.data
