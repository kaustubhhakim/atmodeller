# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core functionality for output"""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxmod.constants import GAS_CONSTANT_BAR
from jaxmod.units import unit_conversion
from jaxtyping import Array, ArrayLike, Float
from molmass import Formula

from atmodeller.containers import SpeciesCollection
from atmodeller.engine_vmap import VmappedFunctions
from atmodeller.interfaces import RedoxBufferProtocol, ThermodynamicStateProtocol
from atmodeller.parameters import Parameters
from atmodeller.solvers import LOG_NUMBER_MOLES_VMAP_AXES, vmap_axes_spec
from atmodeller.thermodata import IronWustiteBuffer
from atmodeller.type_aliases import NpArray, NpBool, NpFloat

logger: logging.Logger = logging.getLogger(__name__)


class Output:
    """Output

    Args:
        parameters: Parameters
        solution: Solution
    """

    def __init__(self, parameters: Parameters, solution: Float[Array, " batch solution"]):
        logger.debug("Creating Output")
        self.parameters: Parameters = parameters
        self.solution: NpFloat = np.asarray(solution)
        self.vmapf: VmappedFunctions = VmappedFunctions(parameters)

        # np.split retains dimensions
        log_number_moles, log_stability = np.split(self.solution, 2, axis=1)

        self.log_number_moles: NpFloat = log_number_moles  # 2-D
        # Mask stabilities that are not solved for by the model
        active_stability_mask: NpBool = parameters.species.active_stability
        self.log_stability = np.where(active_stability_mask, log_stability, np.nan)

        # Caching output to avoid recomputation
        self._cached_dict: Optional[dict[str, dict[str, NpArray]]] = None
        self._cached_dataframes: Optional[dict[str, pd.DataFrame]] = None

    @property
    def condensed_species_mask(self) -> NpBool:  # 1-D
        """Mask of condensed species"""
        return np.invert(self.parameters.reaction_system.gas_species_mask)

    @property
    def gas_species_mask(self) -> NpBool:  # 1-D
        """Mask of gas species"""
        return self.parameters.reaction_system.gas_species_mask

    @property
    def molar_mass(self) -> NpFloat:  # 1-D
        """Molar mass of all species"""
        return self.species.molar_masses

    @property
    def number_moles(self) -> NpFloat:  # 2-D
        """Number of moles of all species"""
        return np.exp(self.log_number_moles)

    @property
    def number_solutions(self) -> int:
        """Number of solutions"""
        return self.parameters.batch_size

    @property
    def state(self) -> ThermodynamicStateProtocol:
        """Thermodynamic state"""
        return self.parameters.state

    @property
    def species(self) -> SpeciesCollection:
        """Species collection"""
        return self.parameters.species

    @property
    def temperature(self) -> NpFloat:  # Must return 1-D for shape consistency
        """Temperature"""
        return np.atleast_1d(self.state.temperature)

    def activity(self) -> NpFloat:  # 2-D
        """Gets the activity of all species.

        Returns:
            Activity of all species
        """
        return np.exp(self.log_activity())

    def activity_without_stability(self) -> NpFloat:  # 2-D
        """Gets the activity without stability of all species.

        Returns:
            Activity without stability of all species
        """
        return np.exp(self.log_activity_without_stability())

    def asdict(self) -> dict[str, dict[str, NpArray]]:
        """Gets all output in a dictionary, with caching.

        Returns:
            Dictionary of all output
        """
        if self._cached_dict is not None:
            logger.info("Returning cached asdict output")
            return self._cached_dict  # Return cached result

        logger.info("Computing asdict output")

        out: dict[str, dict[str, NpArray]] = {}

        # These are required for condensed and gas species
        molar_mass: NpFloat = self.species_molar_mass_expanded()  # 2-D
        activity: NpFloat = self.activity()  # 2-D

        gas_species_asdict = self.gas_species_asdict(molar_mass, self.number_moles, activity)
        out |= gas_species_asdict
        out |= self.condensed_species_asdict(molar_mass, self.number_moles, activity)
        out |= self.elements_asdict()

        out["state"] = broadcast_arrays_in_dict(self.state.asdict(), self.number_solutions)
        # Always add/overwrite the pressure with the evaluation from the model, which by-passes the
        # need to re-evaluate the get_pressure method of state.
        out["state"]["pressure"] = self.total_pressure()

        out["raw"] = self.raw_solution_asdict()

        out["gas"] = self.gas_asdict()

        # Temperature and pressure have already been expanded to the number of solutions
        temperature: NpFloat = out["state"]["temperature"]
        pressure: NpFloat = out["state"]["pressure"]

        if "O2_g" in out:
            logger.debug("Found O2_g so back-computing log10 shift for fO2")
            log10_fugacity: NpFloat = np.log10(out["O2_g"]["fugacity"])
            buffer: RedoxBufferProtocol = IronWustiteBuffer()
            # Shift at 1 bar
            buffer_at_one_bar: NpFloat = np.asarray(buffer.log10_fugacity(temperature, 1.0))
            log10_shift_at_one_bar: NpFloat = log10_fugacity - buffer_at_one_bar
            # logger.debug("log10_shift_at_1bar = %s", log10_shift_at_one_bar)
            out["O2_g"]["log10dIW_1_bar"] = log10_shift_at_one_bar
            # Shift at actual pressure
            buffer_at_P: NpFloat = np.asarray(buffer.log10_fugacity(temperature, pressure))
            log10_shift_at_P: NpFloat = log10_fugacity - buffer_at_P
            # logger.debug("log10_shift_at_P = %s", log10_shift_at_P)
            out["O2_g"]["log10dIW_P"] = log10_shift_at_P

        # For debugging to confirm all outputs are numpy arrays
        # def find_non_numpy(d) -> None:
        #     for key, value in d.items():
        #         if isinstance(value, dict):
        #             find_non_numpy(value)
        #         else:
        #             if not isinstance(value, np.ndarray):
        #                 logger.warning("Non numpy array type found")
        #                 logger.warning("key = %s, value = %s", key, value)
        #                 logger.warning("type = %s", type(value))

        # find_non_numpy(out)

        self._cached_dict = out  # Cache result for faster re-accessing

        return out

    def gas_asdict(self) -> dict[str, NpArray]:
        """Gets the gas properties.

        Returns:
            Gas properties
        """
        out: dict[str, NpArray] = {}

        gas_number_moles: NpFloat = np.sum(
            self.number_moles[:, self.gas_species_mask], axis=1, keepdims=True
        )  # 2-D
        gas_molar_mass: NpArray = self.gas_molar_mass()[:, np.newaxis]  # 2-D
        out: dict[str, NpArray] = self._get_number_moles_output(
            gas_number_moles, gas_molar_mass, "species_"
        )
        # Volume must be a column vector because it multiples all elements in the row
        out["species_number_density"] = gas_number_moles / self.ideal_gas_volume()[:, np.newaxis]
        # Species mass is simply mass so rename for clarity
        out["mass"] = out.pop("species_mass")

        out["molar_mass"] = gas_molar_mass
        # Ensure all arrays are 1-D, which is required for creating dataframes
        out = {key: value.ravel() for key, value in out.items()}

        # Below must all be 1-D so that dataframes can be created.
        out["element_number"] = np.sum(self.element_moles_gas(), axis=1)  # 1-D
        out["element_number_density"] = out["element_number"] / self.ideal_gas_volume()
        out["volume"] = self.ideal_gas_volume()

        return out

    # TODO: Switch to phase method
    # def gas_log_molar_mass(self) -> NpFloat:  # 2-D
    #     """Gets log molar mass of the gas.

    #     Returns:
    #         Log molar mass of the gas
    #     """
    #     gas_log_molar_mass: Array = self.vmapf.get_atmosphere_log_molar_mass(
    #         jnp.asarray(self.log_number_moles)
    #     )

    #     return np.asarray(gas_log_molar_mass)

    # TODO: Switch to phase method
    # def gas_molar_mass(self) -> NpArray:  # 2-D
    #     """Gets the molar mass of the gas.

    #     Returns:
    #         Molar mass of the gas
    #     """
    #     return np.exp(self.gas_log_molar_mass())

    def ideal_gas_volume(self) -> NpFloat:  # 1-D
        """Gets the volume of the gas assuming it is ideal.

        Returns:
            Volume of the gas
        """
        # Total number of moles in the gas
        n: NpFloat = np.sum(self.number_moles[:, self.gas_species_mask], axis=1)
        volume: NpFloat = (n * GAS_CONSTANT_BAR * self.temperature) / self.total_pressure()

        return volume

    def total_pressure(self) -> NpFloat:  # 1-D
        """Gets total pressure.

        Returns:
            Total pressure
        """
        total_pressure: Array = self.vmapf.get_total_pressure(
            jnp.asarray(self.log_number_moles), jnp.asarray(self.log_stability)
        )

        return np.asarray(total_pressure)

    def condensed_species_asdict(
        self, molar_mass: NpArray, number_moles: NpArray, activity: NpArray
    ) -> dict[str, dict[str, NpArray]]:
        """Gets the condensed species output as a dictionary.

        Args:
            molar_mass: Molar mass of all species
            number_moles: Number of moles of all species
            activity: Activity of all species

        Returns:
            Condensed species output as a dictionary
        """
        molar_mass = molar_mass[:, self.condensed_species_mask]  # 2-D
        number_moles = number_moles[:, self.condensed_species_mask]  # 2-D
        activity = activity[:, self.condensed_species_mask]  # 2-D

        condensed_species: tuple[str, ...] = self.species.condensed_species_names

        out: dict[str, NpArray] = self._get_number_moles_output(number_moles, molar_mass, "total_")
        out["molar_mass"] = molar_mass
        out["activity"] = activity

        split_dict: list[dict[str, NpArray]] = split_dict_by_columns(out)
        species_out: dict[str, dict[str, NpArray]] = {
            species_name: split_dict[ii] for ii, species_name in enumerate(condensed_species)
        }

        return species_out

    def elements_asdict(self) -> dict[str, dict[str, NpArray]]:
        """Gets the element properties as a dictionary.

        Returns:
            Element outputs as a dictionary
        """
        molar_mass: NpArray = self.element_molar_mass_expanded()
        gas: NpArray = self.element_moles_gas()
        condensed: NpArray = self.element_moles_condensed()
        dissolved: NpArray = self.element_moles_dissolved()
        total: NpArray = gas + condensed + dissolved

        out: dict[str, NpArray] = self._get_number_moles_output(gas, molar_mass, "gas_")
        # Volume must be a column vector because it multiples all elements in the row
        out["gas_number_density"] = gas / self.ideal_gas_volume()[:, np.newaxis]

        out |= self._get_number_moles_output(condensed, molar_mass, "condensed_")
        out |= self._get_number_moles_output(dissolved, molar_mass, "dissolved_")
        out |= self._get_number_moles_output(total, molar_mass, "total_")

        out["molar_mass"] = molar_mass
        out["degree_of_condensation"] = out["condensed_number"] / out["total_number"]
        out["volume_mixing_ratio"] = out["gas_number"] / np.sum(
            out["gas_number"], axis=1, keepdims=True
        )
        out["gas_mass_fraction"] = out["gas_mass"] / np.sum(out["gas_mass"], axis=1, keepdims=True)

        unique_elements: tuple[str, ...] = self.species.unique_elements
        if "H" in unique_elements:
            index: int = unique_elements.index("H")
            H_total_moles: NpArray = out["total_number"][:, index]
            out["logarithmic_abundance"] = (
                np.log10(out["total_number"] / H_total_moles[:, np.newaxis]) + 12
            )

        # logger.debug("out = %s", out)

        split_dict: list[dict[str, NpArray]] = split_dict_by_columns(out)
        # logger.debug("split_dict = %s", split_dict)

        elements_out: dict[str, dict[str, NpArray]] = {
            f"element_{element}": split_dict[ii] for ii, element in enumerate(unique_elements)
        }
        # logger.debug("elements_out = %s", elements_out)

        return elements_out

    def element_moles_condensed(self) -> NpFloat:  # 2-D
        """Gets the number of moles of elements in the condensed phase.

        Returns:
            Number of moles of elements in the condensed phase
        """
        condensed_species_mask: NpFloat = np.where(self.condensed_species_mask, 1.0, np.nan)
        element_moles_condensed: Array = self.vmapf.get_element_moles(
            jnp.asarray(self.log_number_moles) * condensed_species_mask
        )

        return np.asarray(element_moles_condensed)

    def element_moles_dissolved(self) -> NpFloat:  # 2-D
        """Gets the number of moles of elements dissolved in melt due to species solubility.

        Returns:
            Number of moles of elements dissolved in melt due to species solubility
        """
        element_moles_dissolved: Array = self.vmapf.get_element_moles_in_melt(
            jnp.asarray(self.log_number_moles)
        )

        return np.asarray(element_moles_dissolved)

    def element_moles_gas(self) -> NpFloat:  # 2-D
        """Gets the number of moles of elements in the gas phase.

        Returns:
            Number of moles of elements in the gas phase
        """
        gas_species_mask: NpFloat = np.where(self.gas_species_mask, 1.0, np.nan)
        element_moles_gas: Array = self.vmapf.get_element_moles(
            jnp.asarray(self.log_number_moles) * gas_species_mask,
        )

        return np.asarray(element_moles_gas)

    def element_molar_mass_expanded(self) -> NpFloat:  # 2-D
        """Gets molar mass of elements.

        Returns:
            Molar mass of elements
        """
        unique_elements: tuple[str, ...] = self.species.unique_elements
        molar_mass: NpFloat = np.array([Formula(element).mass for element in unique_elements])
        molar_mass = unit_conversion.g_to_kg * molar_mass

        return np.tile(molar_mass, (self.number_solutions, 1))

    def _get_number_moles_output(
        self, number_moles: NpArray, molar_mass_expanded: NpArray, prefix: str = ""
    ) -> dict[str, NpArray]:
        """Gets the outputs associated with a given number of moles.

        Args:
            number_moles: Number of moles. Shape must be 2-D.
            molar_mass_expanded: Molar mass associated with the number of moles. Shape must be 2-D.
            prefix: Key prefix for the output. Defaults to an empty string.

        Returns
            Dictionary of output quantities
        """
        out: dict[str, NpArray] = {}
        out[f"{prefix}number"] = number_moles
        out[f"{prefix}mass"] = number_moles * molar_mass_expanded

        return out

    def gas_species_asdict(
        self, molar_mass: NpArray, number_moles: NpArray, activity: NpArray
    ) -> dict[str, dict[str, NpArray]]:
        """Gets the gas species output as a dictionary.

        Args:
            molar_mass: Molar mass of all species
            number_moles: Number of moles of all species
            activity: Activity of all species

        Returns:
            Gas species output as a dictionary
        """
        # Below are all filtered to only include the data (columns) of gas species
        molar_mass = molar_mass[:, self.gas_species_mask]  # 2-D
        number_moles = number_moles[:, self.gas_species_mask]  # 2-D
        activity = activity[:, self.gas_species_mask]  # 2-D
        dissolved_number_moles: NpArray = self.species_number_moles_in_melt()[
            :, self.gas_species_mask
        ]  # 2-D
        total_number_moles: NpArray = number_moles + dissolved_number_moles  # 2-D
        pressure: NpArray = self.pressure()[:, self.gas_species_mask]  # 2-D

        gas_species: tuple[str, ...] = self.species.gas_species_names

        out: dict[str, NpArray] = {}
        out |= self._get_number_moles_output(number_moles, molar_mass, "gas_")
        # Volume must be a column vector because it multiples all elements in the row
        out["gas_number_density"] = number_moles / self.ideal_gas_volume()[:, np.newaxis]
        out |= self._get_number_moles_output(dissolved_number_moles, molar_mass, "dissolved_")
        out |= self._get_number_moles_output(total_number_moles, molar_mass, "total_")
        out["molar_mass"] = molar_mass
        out["volume_mixing_ratio"] = number_moles / np.sum(number_moles, axis=1, keepdims=True)
        out["gas_mass_fraction"] = out["gas_mass"] / np.sum(out["gas_mass"], axis=1, keepdims=True)
        out["pressure"] = pressure
        out["fugacity"] = activity
        out["fugacity_coefficient"] = activity / pressure
        out["dissolved_ppmw"] = self.species_ppmw_in_melt()

        split_dict: list[dict[str, NpArray]] = split_dict_by_columns(out)
        species_out: dict[str, dict[str, NpArray]] = {
            species_name: split_dict[ii] for ii, species_name in enumerate(gas_species)
        }

        return species_out

    def log_activity(self) -> NpFloat:  # 2-D
        """Gets log activity of all species.

        This is usually what the user wants when referring to activity because it includes a
        consideration of species stability

        Returns:
            Log activity of all species
        """
        log_activity_without_stability: NpFloat = self.log_activity_without_stability()
        log_activity_with_stability: NpFloat = log_activity_without_stability - np.exp(
            self.log_stability
        )
        # Now select the appropriate activity for each species, depending if stability is relevant.
        condition_broadcasted = np.broadcast_to(
            self.species.active_stability, log_activity_without_stability.shape
        )
        # logger.debug("condition_broadcasted = %s", condition_broadcasted)

        log_activity: NpFloat = np.where(
            condition_broadcasted, log_activity_with_stability, log_activity_without_stability
        )

        return log_activity

    def log_activity_without_stability(self) -> NpFloat:  # 2-D
        """Gets log activity without stability of all species.

        Returns:
            Log activity without stability
        """
        # This function no longer exists
        # log_activity: Array = self.vmapf.get_log_activity(jnp.asarray(self.log_number_moles))

        vmap_temperature = vmap_axes_spec(self.state.temperature)
        pressure = self.vmapf.get_total_pressure(
            jnp.asarray(self.log_number_moles), jnp.asarray(self.log_stability)
        )
        vmap_pressure = vmap_axes_spec(pressure)
        vmap_background_melt_mass = vmap_axes_spec(self.state.melt_mass)
        vmap_background_solid_mass = vmap_axes_spec(self.state.solid_mass)
        vmap_molar_mass = vmap_axes_spec(self.state.molar_mass)

        log_activity_vmap = eqx.filter_vmap(
            self.parameters.reaction_system.get_log_activity,
            in_axes=(
                LOG_NUMBER_MOLES_VMAP_AXES,
                LOG_NUMBER_MOLES_VMAP_AXES,  # log_stability has the same shape as log_number_moles
                vmap_temperature,
                vmap_pressure,
                vmap_molar_mass,
                vmap_background_melt_mass,
                vmap_background_solid_mass,
            ),
        )  # type: ignore

        return np.asarray(
            log_activity_vmap(
                jnp.asarray(self.log_number_moles),
                jnp.asarray(self.log_stability),
                self.state.temperature,
                pressure,
                jnp.log(self.state.molar_mass),
                jnp.log(self.state.melt_mass),
                jnp.log(self.state.solid_mass),
            )
        )  # type: ignore

    # TODO: remove
    # def reaction_mask(self) -> NpBool:  # 2-D
    #     """Gets the reaction mask of the residual array.

    #     Returns:
    #         Reaction mask of the residual array
    #     """
    #     reaction_mask: Bool[Array, "..."] = self.vmapf.get_reactions_only_mask()

    #     return np.asarray(reaction_mask, dtype=bool)

    def species_molar_mass_expanded(self) -> NpFloat:  # 2-D
        """Gets molar mass of all species in an expanded array.

        Returns:
            Molar mass of all species in an expanded array.
        """
        return np.tile(self.molar_mass, (self.number_solutions, 1))

    def pressure(self) -> NpFloat:  # 2-D
        """Gets pressure of species in bar.

        This will compute pressure of all species, including condensates, for simplicity.

        Returns:
            Pressure of species in bar
        """
        pressure: NpFloat = (
            self.number_moles
            * GAS_CONSTANT_BAR
            * self.temperature[:, np.newaxis]
            / self.ideal_gas_volume()[:, np.newaxis]
        )

        return pressure

    def quick_look(self) -> dict[str, ArrayLike]:
        """Quick look at the solution

        Provides a quick first glance at the output with convenient units and to ease comparison
        with test or benchmark data.

        Returns:
            Dictionary of the solution
        """
        out: dict[str, ArrayLike] = {}

        for nn, species_ in enumerate(self.species):
            pressure: NpArray = self.pressure()[:, nn]
            activity: NpArray = self.activity()[:, nn]
            out[species_.data.name] = pressure
            out[f"{species_.data.name}_activity"] = activity

        return {key: np.squeeze(value) for key, value in out.items()}

    def raw_solution_asdict(self) -> dict[str, NpArray]:
        """Gets the raw solution.

        Returns:
            Dictionary of the raw solution
        """
        raw_solution: dict[str, NpArray] = {}

        species_names: tuple[str, ...] = self.species.species_names

        for ii, species_name in enumerate(species_names):
            raw_solution[species_name] = self.log_number_moles[:, ii]
            raw_solution[f"{species_name}_stability"] = self.log_stability[:, ii]

        # Remove keys where the array values are all nan
        for key in list(raw_solution.keys()):
            if np.all(np.isnan(raw_solution[key])):
                raw_solution.pop(key)

        return raw_solution

    def residual_asdict(self) -> dict[int, NpFloat]:
        """Gets the residual.

        Returns:
            Dictionary of the residual
        """
        residual: Array = self.vmapf.objective_function(jnp.asarray(self.solution))  # 2-D

        out: dict[int, NpArray] = {}
        for ii in range(residual.shape[1]):
            out[ii] = np.asarray(residual[:, ii])

        return out

    def species_number_moles_in_melt(self) -> NpFloat:  # 2-D
        """Gets species number of moles in the melt.

        Returns:
            Species number of moles in the melt
        """
        species_number_moles_in_melt: Array = self.vmapf.get_species_moles_in_melt(
            jnp.asarray(self.log_number_moles)
        )

        return np.asarray(species_number_moles_in_melt)

    def species_ppmw_in_melt(self) -> NpFloat:  # 2-D
        """Gets species ppmw in the melt.

        Return:
            Species ppmw in the melt
        """
        species_ppmw_in_melt: Array = self.vmapf.get_species_ppmw_in_melt(
            jnp.asarray(self.log_number_moles)
        )

        return np.asarray(species_ppmw_in_melt)

    def stability(self) -> NpFloat:  # 2-D
        """Gets stability of relevant species.

        Returns:
            Stability of relevant species
        """
        return np.exp(self.log_stability)

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
