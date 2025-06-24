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

_C_cr_coeffs: ThermoCoefficients = ThermoCoefficients(
    (8.943859760e3, 1.398412456e4, 5.848134850e3),
    (-7.295824740e1, -4.477183040e1, -2.350925275e1),
    (
        (
            1.132856760e5,
            -1.980421677e3,
            1.365384188e1,
            -4.636096440e-2,
            1.021333011e-4,
            -1.082893179e-7,
            4.472258860e-11,
        ),
        (
            3.356004410e5,
            -2.596528368e3,
            6.948841910,
            -3.484836090e-3,
            1.844192445e-6,
            -5.055205960e-10,
            5.750639010e-14,
        ),
        (
            2.023105106e5,
            -1.138235908e3,
            3.700279500,
            -1.833807727e-4,
            6.343683250e-8,
            -7.068589480e-12,
            3.335435980e-16,
        ),
    ),
    np.array([200, 600, 2000]),
    np.array([600, 2000, 6000]),
)
C_cr: SpeciesData = SpeciesData("C", "cr", _C_cr_coeffs)
"Species data for C_cr"

_H2O_cr_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-5.530314990e4,),
    (-1.902572063e2,),
    (
        (
            -4.026777480e5,
            2.747887946e3,
            5.738336630e1,
            -8.267915240e-1,
            4.413087980e-3,
            -1.054251164e-5,
            9.694495970e-9,
        ),
    ),
    np.array([200]),
    np.array([273.1507]),
)
H2O_cr: SpeciesData = SpeciesData("H2O", "cr", _H2O_cr_coeffs)
"Species data for H2O_cr"

_H2O_l_coeffs: ThermoCoefficients = ThermoCoefficients(
    (1.101760476e8, 8.113176880e7),
    (-9.779700970e5, -5.134418080e5),
    (
        (
            1.326371304e9,
            -2.448295388e7,
            1.879428776e5,
            -7.678995050e2,
            1.761556813,
            -2.151167128e-3,
            1.092570813e-6,
        ),
        (
            1.263631001e9,
            -1.680380249e7,
            9.278234790e4,
            -2.722373950e2,
            4.479243760e-1,
            -3.919397430e-4,
            1.425743266e-7,
        ),
    ),
    np.array([273.150, 373.150]),
    np.array([373.150, 600]),
)
H2O_l: SpeciesData = SpeciesData(
    "H2O",
    "l",
    _H2O_l_coeffs,
)
"Species data for H2O_l"

_H2O4S_l_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-1.068997367e5,),
    (-1.353966639e1,),
    (
        (
            -7.749933850e4,
            1.040538662e3,
            4.433804910,
            3.648845480e-2,
            -1.743440132e-5,
            1.175631937e-8,
            -3.170091690e-12,
        ),
    ),
    np.array([283.456]),
    np.array([1000]),
)
H2O4S_l: SpeciesData = SpeciesData(
    "H2O4S",
    "l",
    _H2O4S_l_coeffs,
)
"Species data for H2O4S_l"

_ClH4N_cr_coeffs: ThermoCoefficients = ThermoCoefficients(
    (-3.800615346677232e04, -3.792881500e4, -8.787975900e5, 2.063363145e7),
    (-4.372885943063882e01, 2.933013040e1, 3.897936900e3, -3.765976360e4),
    (
        (
            0,
            -5.172495931126793e02,
            8.781198882486828e00,
            1.129216705734911e-02,
            0,
            0,
            0,
        ),
        (1.593389657e5, 0, -5.965854940, 6.494193170e-2, -5.391456890e-5, 0, 0),
        (
            -1.641820157e7,
            1.639262758e5,
            -6.654529630e2,
            1.447086973,
            -1.698254160e-3,
            1.056255415e-6,
            -2.697574651e-10,
        ),
        (
            8.409592660e8,
            -3.426760780e6,
            5.548091470e3,
            -4.435674300,
            1.772486202e-3,
            -2.817160120e-7,
            0,
        ),
    ),
    np.array([100, 298.15, 457.7, 1000]),
    # First entry must be just less than 298.15 so reference enthalpy is computed using the
    # coefficients associated with the second entry to ensure consistency with McBride et al.
    # (2002).
    np.array([298.1499, 457.7, 1000, 1500]),
)
"""Lowermost bound was fit to JANAF data to extend the range below 298.15 K, and is reasonably
continuous across the 298.15 K join temperature."""
ClH4N_cr: SpeciesData = SpeciesData(
    "ClH4N",
    "cr",
    _ClH4N_cr_coeffs,
)
"Species data for ClH4N_cr"

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
