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

import numpy as np

from atmodeller.thermodata import CriticalData, SpeciesData, ThermoCoefficients

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
Cl2_g: SpeciesData = SpeciesData("Cl2", "g")
"Species data for Cl2_g"
ClH_g: SpeciesData = SpeciesData("ClH", "g")
"Species data for ClH_g"
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
Mg_g: SpeciesData = SpeciesData("Mg", "g")
"Species data for Mg_g"
MgO_g: SpeciesData = SpeciesData("MgO", "g")
"Species data for MgO_g"
N2_g: SpeciesData = SpeciesData("N2", "g")
"Species data for N2_g"
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


FeO_g: SpeciesData = SpeciesData("FeO", "g")
"Species data for FeO_g"

_Fe_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(5.466995940e04, 7.137370060e03, 4.847648290e06),
    b2=(-3.383946260e01, 6.504979860e01, -8.697289770e02),
    cp_coeffs=(
        (
            6.790822660e04,
            -1.197218407e03,
            9.843393310e00,
            -1.652324828e-02,
            1.917939959e-05,
            -1.149825371e-08,
            2.832773807e-12,
        ),
        (
            -1.954923682e06,
            6.737161100e03,
            -5.486410970e00,
            4.378803450e-03,
            -1.116286672e-06,
            1.544348856e-10,
            -8.023578182e-15,
        ),
        (
            1.216352511e09,
            -5.828563930e05,
            9.789634510e01,
            -5.370704430e-03,
            3.192037920e-08,
            6.267671430e-12,
            -1.480574914e-16,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
Fe_g: SpeciesData = SpeciesData("Fe", "g", _Fe_g_coeffs)
"Species data for Fe_g"

_NO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(9.098214410e03, 1.750317656e04, -4.677501240e06),
    b2=(6.728725490e00, -8.501669090e00, 1.242081216e03),
    cp_coeffs=(
        (
            -1.143916503e04,
            1.536467592e02,
            3.431468730e00,
            -2.668592368e-03,
            8.481399120e-06,
            -7.685111050e-09,
            2.386797655e-12,
        ),
        (
            2.239018716e05,
            -1.289651623e03,
            5.433936030e00,
            -3.656034900e-04,
            9.880966450e-08,
            -1.416076856e-11,
            9.380184620e-16,
        ),
        (
            -9.575303540e08,
            5.912434480e05,
            -1.384566826e02,
            1.694339403e-02,
            -1.007351096e-06,
            2.912584076e-11,
            -3.295109350e-16,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
NO_g: SpeciesData = SpeciesData("NO", "g", _NO_g_coeffs)
"Species data for NO_g"

_COS_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-1.191657685e04, -8.927096690e03),
    b2=(-2.991988593e01, -2.636328016e01),
    cp_coeffs=(
        (
            8.547876430e04,
            -1.319464821e03,
            9.735257240e00,
            -6.870830960e-03,
            1.082331416e-05,
            -7.705597340e-09,
            2.078570344e-12,
        ),
        (
            1.959098567e05,
            -1.756167688e03,
            8.710430340e00,
            -4.139424960e-04,
            1.015243648e-07,
            -1.159609663e-11,
            5.691053860e-16,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
COS_g: SpeciesData = SpeciesData("COS", "g", _COS_g_coeffs)
"Species data for COS_g"

Ar_g: SpeciesData = SpeciesData("Ar", "g")
"Species data for Ar_g"
Ne_g: SpeciesData = SpeciesData("Ne", "g")
"Species data for Ne_g"
Kr_g: SpeciesData = SpeciesData("Kr", "g")
"Species data for Kr_g"
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
