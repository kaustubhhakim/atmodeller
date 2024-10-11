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

from atmodeller.thermodata.condensates import (
    C_cr_thermodata,
    H2O_l_thermodata,
    S_alpha_thermodata,
    S_beta_thermodata,
    S_l_thermodata,
    Si_cr_thermodata,
    Si_l_thermodata,
    SiO2_l_thermodata,
)
from atmodeller.thermodata.core import SpeciesData
from atmodeller.thermodata.gases import (
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
    SO2_g_thermodata,
    SO_g_thermodata,
)

C_g: SpeciesData = SpeciesData.create(
    "C",
    "g",
    C_g_thermodata,
)
CH4_g_data: SpeciesData = SpeciesData.create(
    "CH4",
    "g",
    CH4_g_thermodata,
)
Cl2_g_data: SpeciesData = SpeciesData.create("Cl2", "g", Cl2_g_thermodata)
CO_g_data: SpeciesData = SpeciesData.create(
    "CO",
    "g",
    CO_g_thermodata,
)
CO2_g_data: SpeciesData = SpeciesData.create(
    "CO2",
    "g",
    CO2_g_thermodata,
)
C_cr_data: SpeciesData = SpeciesData.create("C", "cr", C_cr_thermodata)
H2_g_data: SpeciesData = SpeciesData.create("H2", "g", H2_g_thermodata)
H2O_g_data: SpeciesData = SpeciesData.create(
    "H2O",
    "g",
    H2O_g_thermodata,
)
H2O_l_data: SpeciesData = SpeciesData.create(
    "H2O",
    "l",
    H2O_l_thermodata,
)
H2S_g_data: SpeciesData = SpeciesData.create(
    "H2S",
    "g",
    H2S_g_thermodata,
)
He_g_data: SpeciesData = SpeciesData.create(
    "He",
    "g",
    He_g_thermodata,
)
N2_g_data: SpeciesData = SpeciesData.create("N2", "g", N2_g_thermodata)
NH3_g_data: SpeciesData = SpeciesData.create(
    "NH3",
    "g",
    NH3_g_thermodata,
)
O2_g_data: SpeciesData = SpeciesData.create("O2", "g", O2_g_thermodata)
S_alpha_data: SpeciesData = SpeciesData.create("S", "alpha", S_alpha_thermodata)
S_beta_data: SpeciesData = SpeciesData.create("S", "beta", S_beta_thermodata)
S_l_data: SpeciesData = SpeciesData.create(
    "S",
    "l",
    S_l_thermodata,
)
S2_g_data: SpeciesData = SpeciesData.create("S2", "g", S2_g_thermodata)
Si_cr_data: SpeciesData = SpeciesData.create(
    "Si",
    "cr",
    Si_cr_thermodata,
)
Si_l_data: SpeciesData = SpeciesData.create(
    "Si",
    "l",
    Si_l_thermodata,
)
SiO2_l_data: SpeciesData = SpeciesData.create(
    "SiO2",
    "l",
    SiO2_l_thermodata,
)
SO_g_data: SpeciesData = SpeciesData.create("SO", "g", SO_g_thermodata)
SO2_g_data: SpeciesData = SpeciesData.create("SO2", "g", SO2_g_thermodata)
