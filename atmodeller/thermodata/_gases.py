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
CH4_g: SpeciesData = SpeciesData("CH4", "g")
"Species data for CH4_g"
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
OSi_g: SpeciesData = SpeciesData("OSi", "g")
"Species data for OSi_g"
O2Si_g: SpeciesData = SpeciesData("O2Si", "g")
"Species data for O2Si_g"
S2_g: SpeciesData = SpeciesData("S2", "g")
"Species data for S2_g"
Si_g: SpeciesData = SpeciesData("Si", "g")
"Species data for Si_g"


_HS_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(1.742902395e04, 4.899214490e04),
    b2=(-1.760761843e01, -3.770400275e01),
    cp_coeffs=(
        (
            6.389434680e03,
            -3.747960920e02,
            7.548145770e00,
            -1.288875477e-02,
            1.907786343e-05,
            -1.265033728e-08,
            3.235158690e-12,
        ),
        (
            1.682631601e06,
            -5.177152210e03,
            9.198168520e00,
            -2.323550224e-03,
            6.543914780e-07,
            -8.468470420e-11,
            3.864741550e-15,
        ),
    ),
    T_min=np.array([200, 1000, 6000]),
    T_max=np.array([1000, 6000, 20000]),
)
HS_g: SpeciesData = SpeciesData("HS", "g", _HS_g_coeffs)
"Species data for HS_g"

_C2H2_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(3.712619060e04, 6.266578970e04),
    b2=(-5.244338900e01, -5.818960590e01),
    cp_coeffs=(
        (
            1.598112089e05,
            -2.216644118e03,
            1.265707813e01,
            -7.979651080e-03,
            8.054992750e-06,
            -2.433307673e-09,
            -7.529233180e-14,
        ),
        (
            1.713847410e06,
            -5.929106660e03,
            1.236127943e01,
            1.314186993e-04,
            -1.362764431e-07,
            2.712655786e-11,
            -1.302066204e-15,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
C2H2_g: SpeciesData = SpeciesData("C2H2", "g", _C2H2_g_coeffs)
"Species data for C2H2_g"

_CHN_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(2.098915450e04, 4.221513770e04),
    b2=(-2.746678076e01, -4.005774072e01),
    cp_coeffs=(
        (
            9.098286930e04,
            -1.238657512e03,
            8.721307870e00,
            -6.528242940e-03,
            8.872700830e-06,
            -4.808886670e-09,
            9.317898500e-13,
        ),
        (
            1.236889278e06,
            -4.446732410e03,
            9.738874850e00,
            -5.855182640e-04,
            1.072791440e-07,
            -1.013313244e-11,
            3.348247980e-16,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
CHN_g: SpeciesData = SpeciesData("CHN", "g", _CHN_g_coeffs)
"Species data for CHN_g"

_O3S_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-5.184106170e04, -4.398283990e04),
    b2=(3.391331216e01, -3.655217314e01),
    cp_coeffs=(
        (
            -3.952855290e04,
            6.208572570e02,
            -1.437731716e00,
            2.764126467e-02,
            -3.144958662e-05,
            1.792798000e-08,
            -4.126386660e-12,
        ),
        (
            -2.166923781e05,
            -1.301022399e03,
            1.096287985e01,
            -3.837100020e-04,
            8.466889040e-08,
            -9.705399290e-12,
            4.498397540e-16,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
O3S_g: SpeciesData = SpeciesData("O3S", "g", _O3S_g_coeffs)
"Species data for O3S_g"

_H2O4S_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-9.315660120e4, -5.259092950e4),
    b2=(3.961096201e1, -1.023603724e2),
    cp_coeffs=(
        (
            -4.129150050e4,
            6.681589890e2,
            -2.632753507,
            5.415382480e-2,
            -7.067502230e-5,
            4.684611420e-8,
            -1.236791238e-11,
        ),
        (
            1.437877914e6,
            -6.614902530e3,
            2.157662058e1,
            -4.806255970e-4,
            3.010775121e-8,
            2.334842469e-12,
            -2.946330375e-16,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
H2O4S_g: SpeciesData = SpeciesData("H2O4S", "g", _H2O4S_g_coeffs)
"Species data for H2O4S_g"

_FeO_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(2.964572665e04, 3.037985806e04),
    b2=(1.326115545e01, -3.633655420e00),
    cp_coeffs=(
        (
            1.569282213e04,
            -6.460188880e01,
            2.458925470e00,
            7.016047360e-03,
            -1.021405947e-05,
            7.179297870e-09,
            -1.978966365e-12,
        ),
        (
            -1.195971480e05,
            -3.624864780e02,
            5.518880750e00,
            -9.978856890e-04,
            4.376913830e-07,
            -6.790629460e-11,
            3.639292680e-15,
        ),
    ),
    T_min=np.array([200, 1000]),
    T_max=np.array([1000, 6000]),
)
FeO_g: SpeciesData = SpeciesData("FeO", "g", _FeO_g_coeffs)
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

_Ar_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.449939610e02, -5.078300340e06),
    b2=(4.379674910e00, 4.379180110e00, 1.465298484e03),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.010538475e01,
            -5.992661070e-02,
            2.500069401e00,
            -3.992141160e-08,
            1.205272140e-11,
            -1.819015576e-15,
            1.078576636e-19,
        ),
        (
            -9.951265080e08,
            6.458887260e05,
            -1.675894697e02,
            2.319933363e-02,
            -1.721080911e-06,
            6.531938460e-11,
            -9.740147729e-16,
        ),
    ),
    T_min=np.array([200, 1000, 6000]),
    T_max=np.array([1000, 6000, 20000]),
)
Ar_g: SpeciesData = SpeciesData("Ar", "g", _Ar_g_coeffs)
"Species data for Ar_g"

_He_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.453750000e02, 1.650518960e04),
    b2=(9.287239740e-01, 9.287239740e-01, -4.048814390e00),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            3.396845420e06,
            -2.194037652e03,
            3.080231878e00,
            -8.068957550e-05,
            6.252784910e-09,
            -2.574990067e-13,
            4.429960218e-18,
        ),
    ),
    T_min=np.array([200, 1000, 6000]),
    T_max=np.array([1000, 6000, 20000]),
)
He_g: SpeciesData = SpeciesData("He", "g", _He_g_coeffs)
"Species data for He_g"

_Ne_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.453750000e02, -5.663933630e04),
    b2=(3.355322720e00, 3.355322720e00, 1.648438697e01),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            -1.238252746e07,
            6.958579580e03,
            1.016709287e00,
            1.424664555e-04,
            -4.803933930e-09,
            -1.170213183e-13,
            8.415153652e-18,
        ),
    ),
    T_min=np.array([200, 1000, 6000]),
    T_max=np.array([1000, 6000, 20000]),
)
Ne_g: SpeciesData = SpeciesData("Ne", "g", _Ne_g_coeffs)
"Species data for Ne_g"

_Kr_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -7.403488940e02, -7.111667370e06),
    b2=(5.490956510e00, 5.484398150e00, 2.086866326e03),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            2.643639057e02,
            -7.910050820e-01,
            2.500920585e00,
            -5.328164110e-07,
            1.620730161e-10,
            -2.467898017e-14,
            1.478585040e-18,
        ),
        (
            -1.375531087e09,
            9.064030530e05,
            -2.403481435e02,
            3.378312030e-02,
            -2.563103877e-06,
            9.969787790e-11,
            -1.521249677e-15,
        ),
    ),
    T_min=np.array([200, 1000, 6000]),
    T_max=np.array([1000, 6000, 20000]),
)
Kr_g: SpeciesData = SpeciesData("Kr", "g", _Kr_g_coeffs)
"Species data for Kr_g"

_Xe_g_coeffs: ThermoCoefficients = ThermoCoefficients(
    b1=(-7.453750000e02, -6.685800730e02, 9.285443830e05),
    b2=(6.164454205e00, 6.063710715e00, -1.109834556e02),
    cp_coeffs=(
        (
            0.0,
            0.0,
            2.5,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            4.025226680e03,
            -1.209507521e01,
            2.514153347e00,
            -8.248102080e-06,
            2.530232618e-09,
            -3.892333230e-13,
            2.360439138e-17,
        ),
        (
            2.540397456e08,
            -1.105373774e05,
            1.382644099e01,
            1.500614606e-03,
            -3.935359030e-07,
            2.765790584e-11,
            -5.943990574e-16,
        ),
    ),
    T_min=np.array([200, 1000, 6000]),
    T_max=np.array([1000, 6000, 20000]),
)
Xe_g: SpeciesData = SpeciesData("Xe", "g", _Xe_g_coeffs)
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
