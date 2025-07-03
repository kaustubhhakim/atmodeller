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
"""Thermochemical data for gases from :cite:t:`MZG02`

https://ntrs.nasa.gov/citations/20020085330
"""

from atmodeller.thermodata import CriticalData, SpeciesData

Ar_g: SpeciesData = SpeciesData("Ar", "g")
"Species data for Ar_g"
C_g: SpeciesData = SpeciesData("C", "g")
"Species data for C_g"
C2H2_g: SpeciesData = SpeciesData("C2H2", "g")
"Species data for C2H2_g"
CH4_g: SpeciesData = SpeciesData("CH4", "g")
"Species data for CH4_g"
CHN_g: SpeciesData = SpeciesData("CHN", "g")
"Species data for CHN_g"
CO_g: SpeciesData = SpeciesData("CO", "g")
"Species data for CO_g"
CO2_g: SpeciesData = SpeciesData("CO2", "g")
"Species data for CO2_g"
COS_g: SpeciesData = SpeciesData("COS", "g")
"Species data for COS_g"
Cl2_g: SpeciesData = SpeciesData("Cl2", "g")
"Species data for Cl2_g"
ClH_g: SpeciesData = SpeciesData("ClH", "g")
"Species data for ClH_g"
Fe_g: SpeciesData = SpeciesData("Fe", "g")
"Species data for Fe_g"
FeO_g: SpeciesData = SpeciesData("FeO", "g")
"Species data for FeO_g"
H2_g: SpeciesData = SpeciesData("H2", "g")
"Species data for H2_g"
He_g: SpeciesData = SpeciesData("He", "g")
"Species data for He_g"
H3N_g: SpeciesData = SpeciesData("H3N", "g")
"Species data for H3N_g"
HO_g: SpeciesData = SpeciesData("HO", "g")
"Species data for HO_g"
H2O_g: SpeciesData = SpeciesData("H2O", "g")
"Species data for H2O_g"
H2O4S_g: SpeciesData = SpeciesData("H2O4S", "g")
"Species data for H2O4S_g"
HS_g: SpeciesData = SpeciesData("HS", "g")
"Species data for HS_g"
H2S_g: SpeciesData = SpeciesData("H2S", "g")
"Species data for H2S_g"
H4Si_g: SpeciesData = SpeciesData("H4Si", "g")
"Species data for H4Si_g"
Kr_g: SpeciesData = SpeciesData("Kr", "g")
"Species data for Kr_g"
Mg_g: SpeciesData = SpeciesData("Mg", "g")
"Species data for Mg_g"
MgO_g: SpeciesData = SpeciesData("MgO", "g")
"Species data for MgO_g"
N2_g: SpeciesData = SpeciesData("N2", "g")
"Species data for N2_g"
NO_g: SpeciesData = SpeciesData("NO", "g")
"Species data for NO_g"
Ne_g: SpeciesData = SpeciesData("Ne", "g")
"Species data for Ne_g"
O2_g: SpeciesData = SpeciesData("O2", "g")
"Species data for O2_g"
OS_g: SpeciesData = SpeciesData("OS", "g")
"Species data for OS_g"
O2S_g: SpeciesData = SpeciesData("O2S", "g")
"Species data for O2S_g"
O3S_g: SpeciesData = SpeciesData("O3S", "g")
"Species data for O3S_g"
OSi_g: SpeciesData = SpeciesData("OSi", "g")
"Species data for OSi_g"
O2Si_g: SpeciesData = SpeciesData("O2Si", "g")
"Species data for O2Si_g"
S2_g: SpeciesData = SpeciesData("S2", "g")
"Species data for S2_g"
Si_g: SpeciesData = SpeciesData("Si", "g")
"Species data for Si_g"
Xe_g: SpeciesData = SpeciesData("Xe", "g")
"Species data for Xe_g"

_critical_data_H2O_g: CriticalData = CriticalData(647.25, 221.1925)
"""Critical parameters for H2O_g :cite:p:`SS92{Table 2}`"""
_critical_data_CO2_g: CriticalData = CriticalData(304.15, 73.8659)
"""Critical parameters for CO2_g :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 304.2 K and 73.8 bar
"""
_critical_data_CH4_g: CriticalData = CriticalData(191.05, 46.4069)
"""Critical parameters for CH4_g :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 190.6 K and 46 bar
"""
_critical_data_CO_g: CriticalData = CriticalData(133.15, 34.9571)
"""Critical parameters for CO :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 132.9 K and 35 bar
"""
_critical_data_O2_g: CriticalData = CriticalData(154.75, 50.7638)
"""Critical parameters for O2 :cite:p:`SS92{Table 2}`"""
_critical_data_H2_g: CriticalData = CriticalData(33.25, 12.9696)
"""Critical parameters for H2 :cite:p:`SS92{Table 2}`"""
_critical_data_H2_g_holland: CriticalData = CriticalData(41.2, 21.1)
"""Critical parameters for H2 :cite:p:`HP91`"""
_critical_data_S2_g: CriticalData = CriticalData(208.15, 72.954)
"""Critical parameters for S2 :cite:p:`SS92{Table 2}`

http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

:cite:p:`HP11` state that the critical parameters are from :cite:t:`RPS77`. However, in the fifth
edition of this book (:cite:t:`PPO00`) S2 is not given (only S is).
"""
_critical_data_SO2_g: CriticalData = CriticalData(430.95, 78.7295)
"""Critical parameters for SO2 :cite:p:`SS92{Table 2}`"""
_critical_data_COS_g: CriticalData = CriticalData(377.55, 65.8612)
"""Critical parameters for COS :cite:p:`SS92{Table 2}`"""
_critical_data_H2S_g: CriticalData = CriticalData(373.55, 90.0779)
"""Critical parameters for H2S :cite:p:`SS92{Table 2}`

Alternative values from :cite:t:`HP91` are 373.4 K and 0.08963 bar
"""
_critical_data_N2_g: CriticalData = CriticalData(126.2, 33.9)
"""Critical parameters for N2 :cite:p:`SF87{Table 1}`"""
_critical_data_Ar_g: CriticalData = CriticalData(151.0, 48.6)
"""Critical parameters for Ar :cite:p:`SF87{Table 1}`"""
_critical_data_He_g: CriticalData = CriticalData(5.2, 2.274)
"""Critical parameters for He :cite:p:`ADM77`"""
_critical_data_Ne_g: CriticalData = CriticalData(44.49, 26.8)
"""Critical paramters for Ne :cite:p:`KJS86{Table 4}`"""
_critical_data_Kr_g: CriticalData = CriticalData(209.46, 55.2019)
"""Critical parameters for Kr :cite:p:`TB70`"""
_critical_data_Xe_g: CriticalData = CriticalData(289.765, 5.8415)
"""Critical parameters for Xe :cite:p:`SK94`"""

critical_data: dict[str, CriticalData] = {
    "Ar_g": _critical_data_Ar_g,
    "CH4_g": _critical_data_CH4_g,
    "CO_g": _critical_data_CO_g,
    "CO2_g": _critical_data_CO2_g,
    "COS_g": _critical_data_COS_g,
    "H2_g": _critical_data_H2_g,
    "H2_g_Holland": _critical_data_H2_g_holland,
    "H2O_g": _critical_data_H2O_g,
    "H2S_g": _critical_data_H2S_g,
    "N2_g": _critical_data_N2_g,
    "O2_g": _critical_data_O2_g,
    "S2_g": _critical_data_S2_g,
    "SO2_g": _critical_data_SO2_g,
    "He_g": _critical_data_He_g,
    "Ne_g": _critical_data_Ne_g,
    "Kr_g": _critical_data_Kr_g,
    "Xe_g": _critical_data_Xe_g,
}
"""Critical parameters for gases

These critical data could be extended to more species using :cite:t:`PPO00{Appendix A.19}`
"""
