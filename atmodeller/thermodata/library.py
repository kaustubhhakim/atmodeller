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
    C_g_thermodata,
    CH4_g_thermodata,
    Cl2_g_thermodata,
    CO2_g_thermodata,
    CO_g_thermodata,
    H2_g_thermodata,
    H2O_g_thermodata,
    H2S_g_thermodata,
    He_g_thermodata,
    N2_g_thermodata,
    NH3_g_thermodata,
    O2_g_thermodata,
    S2_g_thermodata,
    SiH4_g_thermodata,
    SiO_g_thermodata,
    SO2_g_thermodata,
    SO_g_thermodata,
)
from atmodeller.thermodata.core import SpeciesData

logger: logging.Logger = logging.getLogger(__name__)

C_g_data: SpeciesData = SpeciesData.create(
    "C",
    "g",
    C_g_thermodata,
)
"Species data for C_g"

CH4_g_data: SpeciesData = SpeciesData.create(
    "CH4",
    "g",
    CH4_g_thermodata,
)
"Species data for CH4_g"

Cl2_g_data: SpeciesData = SpeciesData.create("Cl2", "g", Cl2_g_thermodata)
"Species data for Cl2_g"

CO_g_data: SpeciesData = SpeciesData.create(
    "CO",
    "g",
    CO_g_thermodata,
)
"Species data for CO_g"

CO2_g_data: SpeciesData = SpeciesData.create(
    "CO2",
    "g",
    CO2_g_thermodata,
)
"Species data for CO2_g"


H2_g_data: SpeciesData = SpeciesData.create("H2", "g", H2_g_thermodata)
"Species data for H2_g"

H2O_g_data: SpeciesData = SpeciesData.create(
    "H2O",
    "g",
    H2O_g_thermodata,
)
"Species data for H2O_g"

H2S_g_data: SpeciesData = SpeciesData.create(
    "H2S",
    "g",
    H2S_g_thermodata,
)
"Species data for H2S_g"

He_g_data: SpeciesData = SpeciesData.create(
    "He",
    "g",
    He_g_thermodata,
)
"Species data for He_g"

N2_g_data: SpeciesData = SpeciesData.create("N2", "g", N2_g_thermodata)
"Species data for N2_g"

NH3_g_data: SpeciesData = SpeciesData.create(
    "NH3",
    "g",
    NH3_g_thermodata,
)
"Species data for NH3_g"

O2_g_data: SpeciesData = SpeciesData.create("O2", "g", O2_g_thermodata)
"Species data for O2_g"


S2_g_data: SpeciesData = SpeciesData.create("S2", "g", S2_g_thermodata)
"Species data for S2_g"


SiH4_g_data: SpeciesData = SpeciesData.create("SiH4", "g", SiH4_g_thermodata)
"Species data for SiH4_g"

SiO_g_data: SpeciesData = SpeciesData.create("SiO", "g", SiO_g_thermodata)
"Species data for SiO_g"


SO_g_data: SpeciesData = SpeciesData.create("SO", "g", SO_g_thermodata)
"Species data for SO_g"

SO2_g_data: SpeciesData = SpeciesData.create("SO2", "g", SO2_g_thermodata)
"Species data for SO2_g"

species_data: dict[str, SpeciesData] = {
    "C_g": C_g_data,
    "CH4_g": CH4_g_data,
    "Cl2_g": Cl2_g_data,
    "CO_g": CO_g_data,
    "CO2_g": CO2_g_data,
    "C_cr": C_cr,
    "H2_g": H2_g_data,
    "H2O_cr": H2O_cr,
    "H2O_g": H2O_g_data,
    "H2O_l": H2O_l,
    "H2S_g": H2S_g_data,
    "He_g": He_g_data,
    "N2_g": N2_g_data,
    "NH3_g": NH3_g_data,
    "O2_g": O2_g_data,
    "S_alpha": S_alpha,
    "S_beta": S_beta,
    "S_l": S_l,
    "S2_g": S2_g_data,
    "Si_cr": Si_cr,
    "Si_l": Si_l,
    "SiO_g": SiO_g_data,
    "SiH4_g": SiH4_g_data,
    "SiO2_l": SiO2_l,
    "SO_g": SO_g_data,
    "SO2_g": SO2_g_data,
}
"""Species data"""


def get_species_data(species_name: str) -> SpeciesData:
    """Gets the species data.

    Args:
        species_name: Name of the species

    Returns:
        Species data
    """
    try:
        data: SpeciesData = species_data[species_name]

    except KeyError as exc:
        msg: str = f"Species data for '{species_name}' is not available"
        logger.warning(msg)
        logger.warning("Must choose from: %s", list(species_data.keys()))
        raise ValueError(msg) from exc

    return data
