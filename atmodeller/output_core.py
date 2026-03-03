# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Core functionality for output"""

import logging

import numpy as np
from jaxmod.type_aliases import NpArray

logger: logging.Logger = logging.getLogger(__name__)


class Output:
    """Output

    Args:
        parameters: Parameters
        solution: Solution
    """

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
