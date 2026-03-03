# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core functionality for output"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from jaxmod.type_aliases import NpArray, NpBool, NpFloat
from jaxtyping import Array, Float

from atmodeller.interfaces import RedoxBufferProtocol
from atmodeller.parameters import Parameters
from atmodeller.thermodata import IronWustiteBuffer

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

        # np.split retains dimensions
        log_number_moles, log_stability = np.split(self.solution, 2, axis=1)

        self.log_number_moles: NpFloat = log_number_moles  # 2-D
        # Mask stabilities that are not solved for by the model
        active_stability_mask: NpBool = parameters.species.active_stability
        self.log_stability = np.where(active_stability_mask, log_stability, np.nan)

        # Caching output to avoid recomputation
        self._cached_dict: Optional[dict[str, dict[str, NpArray]]] = None
        self._cached_dataframes: Optional[dict[str, pd.DataFrame]] = None

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
