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
"""Species data"""

import logging

from atmodeller.thermodata._condensates import (
    C_cr,
    H2O_cr,
    H2O_l,
    S_alpha,
    S_beta,
    S_l,
    Si_cr,
    Si_l,
    SiO2_l,
)
from atmodeller.thermodata._gases import (
    C_g,
    CH4_g,
    Cl2_g,
    CO2_g,
    CO_g,
    H2_g,
    H2O_g,
    H2S_g,
    He_g,
    N2_g,
    NH3_g,
    O2_g,
    S2_g,
    SiH4_g,
    SiO_g,
    SO2_g,
    SO_g,
)
from atmodeller.thermodata.core import SpeciesData

logger: logging.Logger = logging.getLogger(__name__)


def get_species_data() -> dict[str, SpeciesData]:
    """Gets a dictionary of species data

    Returns:
        Dictionary of species data
    """
    species_data: dict[str, SpeciesData] = {
        "C_g": C_g,
        "CH4_g": CH4_g,
        "Cl2_g": Cl2_g,
        "CO_g": CO_g,
        "CO2_g": CO2_g,
        "C_cr": C_cr,
        "H2_g": H2_g,
        "H2O_cr": H2O_cr,
        "H2O_g": H2O_g,
        "H2O_l": H2O_l,
        "H2S_g": H2S_g,
        "He_g": He_g,
        "N2_g": N2_g,
        "NH3_g": NH3_g,
        "O2_g": O2_g,
        "S_alpha": S_alpha,
        "S_beta": S_beta,
        "S_l": S_l,
        "S2_g": S2_g,
        "Si_cr": Si_cr,
        "Si_l": Si_l,
        "SiO_g": SiO_g,
        "SiH4_g": SiH4_g,
        "SiO2_l": SiO2_l,
        "SO_g": SO_g,
        "SO2_g": SO2_g,
    }

    return species_data


def select_species_data(species_name: str) -> SpeciesData:
    """Selects species data

    Args:
        species_name: Name of the species

    Returns:
        Species data
    """
    species_data: dict[str, SpeciesData] = get_species_data()

    try:
        data: SpeciesData = species_data[species_name]
    except KeyError as exc:
        msg: str = f"Species data for '{species_name}' is not available"
        logger.warning(msg)
        logger.warning("Available options are: %s", list(species_data.keys()))
        raise ValueError(msg) from exc

    return data
