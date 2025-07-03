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
"""Thermochemical data for condensates from :cite:t:`MZG02`.

https://ntrs.nasa.gov/citations/20020085330
"""

import numpy as np

from atmodeller.thermodata import SpeciesData, ThermoCoefficients

C_cr: SpeciesData = SpeciesData("C", "cr")
"Species data for C_cr"
ClH4N_cr: SpeciesData = SpeciesData("ClH4N", "cr")
"Species data for ClH4N_cr"
H2O_cr: SpeciesData = SpeciesData("H2O", "cr")
"Species data for H2O_cr"
H2O_l: SpeciesData = SpeciesData("H2O", "l")
"Species data for H2O_l"
H2O4S_l: SpeciesData = SpeciesData("H2O4S", "l")
"Species data for H2O4S_l"


_S_cr_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-7.516389580e2, -6.852714730e2),
    (
        -7.961066980,
        -8.607846750,
    ),
    (
        (-1.035710779e4, 0.0, 1.866766938, 4.256140250e-3, -3.265252270e-06, 0.0, 0.0),
        (0.0, 0.0, 2.080514131, 2.440879557e-3, 0.0, 0.0, 0.0),
    ),
    np.array([200, 368.3]),
    np.array([368.3, 388.36]),
)
S_cr: SpeciesData = SpeciesData("S", "cr", _S_cr_coeffs)
"Species data for S_alpha and S_beta"

_S_l_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-6.356594920e5, -9.832222680e5, -2.638846929e4, 1.113013440e4, -8.284589830e2),
    (-1.186929589e4, -3.154806751e4, -7.681730097e2, 1.363174183e2, -1.736128237e1),
    (
        (-6.366550765e7, 0.0, 2.376860693e3, -7.888076026, 7.376076522e-3, 0.0, 0.0),
        (0.0, 0.0, 6.928522306e3, -3.254655981e1, 3.824448176e-2, 0.0, 0.0),
        (0.0, 0.0, 1.649945697e2, -6.843534977e-1, 7.315907973e-4, 0.0, 0.0),
        (1.972984578e6, 0.0, -2.441009753e1, 6.090352889e-2, -3.744069103e-5, 0.0, 0.0),
        (0.0, 0.0, 3.848693429, 0.0, 0.0, 0.0, 0.0),
    ),
    np.array([388.36, 428.15, 432.25, 453.15, 717]),
    np.array([428.15, 432.25, 453.15, 717, 6000]),
)
S_l: SpeciesData = SpeciesData(
    "S",
    "l",
    _S_l_coeffs,
)
"Species data for S_l"

_Si_cr_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-7.850635210e2, -1.042947234e3),
    (-1.038427318e1, -1.438964187e1),
    (
        (-2.323538208e4, 0.0, 2.102021680, 1.809220552e-3, 0.0, 0.0, 0.0),
        (-5.232559740e4, 0.0, 2.850169415, 3.975166970e-4, 0.0, 0.0, 0.0),
    ),
    np.array([200, 298.15]),
    np.array([298.15, 1690]),
)
Si_cr: SpeciesData = SpeciesData(
    "Si",
    "cr",
    _Si_cr_coeffs,
)
"Species data for Si_cr"

_Si_l_coeffs: ThermoCoefficients = ThermoCoefficients(
    (4.882667110e3,),
    (-1.326611073e1,),
    ((0.0, 0.0, 3.271389414, 0.0, 0.0, 0.0, 0.0),),
    np.array([1690]),
    np.array([6000]),
)
Si_l: SpeciesData = SpeciesData(
    "Si",
    "l",
    _Si_l_coeffs,
)
"Species data for Si_l"

_O2Si_l_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-1.140002976e5,),
    (-5.554279592e1,),
    ((0.0, 0.0, 1.004268442e1, 0.0, 0.0, 0.0, 0.0),),
    np.array([1996]),
    np.array([6000]),
)
O2Si_l: SpeciesData = SpeciesData(
    "O2Si",
    "l",
    _O2Si_l_coeffs,
)
"Species data for O2Si_l"
