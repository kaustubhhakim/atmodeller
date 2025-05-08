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
"""Thermodynamic data for species"""

import logging

from atmodeller.thermodata import CriticalData, SpeciesData
from atmodeller.thermodata._condensates import (
    C_cr,
    ClH4N_cr,
    H2O4S_l,
    H2O_cr,
    H2O_l,
    O2Si_l,
    S_cr,
    S_l,
    Si_cr,
    Si_l,
)
from atmodeller.thermodata._gases import (
    Ar_g,
    C2H2_g,
    C_g,
    CH4_g,
    CHN_g,
    Cl2_g,
    ClH_g,
    CO2_g,
    CO_g,
    COS_g,
    Fe_g,
    FeO_g,
    H2_g,
    H2O4S_g,
    H2O_g,
    H2S_g,
    H3N_g,
    H4Si_g,
    He_g,
    HO_g,
    HS_g,
    Kr_g,
    Mg_g,
    MgO_g,
    N2_g,
    Ne_g,
    NO_g,
    O2_g,
    O2S_g,
    O2Si_g,
    O3S_g,
    OS_g,
    OSi_g,
    S2_g,
    Si_g,
    Xe_g,
    critical_data,
)

logger: logging.Logger = logging.getLogger(__name__)


def get_thermodata() -> dict[str, SpeciesData]:
    """Gets a dictionary of thermodynamic data for species

    Dictionary keys adhere to the Hill system with the JANAF state of aggregation as a suffix.

    Returns:
        Dictionary of thermodynamic data for species
    """
    species_data: dict[str, SpeciesData] = {
        "Ar_g": Ar_g,
        "C2H2_g": C2H2_g,
        "C_cr": C_cr,
        "C_g": C_g,
        "CH4_g": CH4_g,
        "CHN_g": CHN_g,
        "Cl2_g": Cl2_g,
        "ClH4N_cr": ClH4N_cr,
        "ClH_g": ClH_g,
        "CO2_g": CO2_g,
        "CO_g": CO_g,
        "COS_g": COS_g,
        "FeO_g": FeO_g,
        "Fe_g": Fe_g,
        "H2O4S_g": H2O4S_g,
        "H2O4S_l": H2O4S_l,
        "H2O_cr": H2O_cr,
        "H2O_g": H2O_g,
        "H2O_l": H2O_l,
        "H2S_g": H2S_g,
        "H2_g": H2_g,
        "H3N_g": H3N_g,
        "H4Si_g": H4Si_g,
        "He_g": He_g,
        "HO_g": HO_g,
        "HS_g": HS_g,
        "Kr_g": Kr_g,
        "MgO_g": MgO_g,
        "Mg_g": Mg_g,
        "N2_g": N2_g,
        "Ne_g": Ne_g,
        "NO_g": NO_g,
        "O2S_g": O2S_g,
        "O2Si_g": O2Si_g,
        "O2Si_l": O2Si_l,
        "O2_g": O2_g,
        "O3S_g": O3S_g,
        "OS_g": OS_g,
        "OSi_g": OSi_g,
        "S2_g": S2_g,
        "S_cr": S_cr,
        "S_l": S_l,
        "Si_cr": Si_cr,
        "Si_g": Si_g,
        "Si_l": Si_l,
        "Xe_g": Xe_g,
    }

    return species_data


def select_thermodata(species_name: str) -> SpeciesData:
    """Selects thermodynamic data for species

    Args:
        species_name: Name of the species

    Returns:
        Thermodynamic data for species
    """
    species_data: dict[str, SpeciesData] = get_thermodata()

    try:
        data: SpeciesData = species_data[species_name]
    except KeyError as exc:
        msg: str = f"Thermodynamic data for '{species_name}' is not available"
        logger.warning(msg)
        logger.warning("Available options are: %s", list(species_data.keys()))
        raise ValueError(msg) from exc

    return data


def select_critical_data(species_name: str) -> CriticalData:
    """Selects critical data for species

    Args:
        species_name: Name of the species

    Returns:
        Critical data for species
    """
    try:
        data: CriticalData = critical_data[species_name]
    except KeyError as exc:
        msg: str = f"Critical for '{species_name}' is not available"
        logger.warning(msg)
        logger.warning("Available options are: %s", list(critical_data.keys()))
        raise ValueError(msg) from exc

    return data
