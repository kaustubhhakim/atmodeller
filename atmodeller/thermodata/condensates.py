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
"""Thermochemical data for condensates"""

from atmodeller.thermodata.core import ThermoData

C_cr_thermodata: ThermoData = ThermoData(
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
    (200, 600, 2000),
    (600, 2000, 6000),
)

H2O_cr_thermodata: ThermoData = ThermoData(
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
    (200,),
    (273.1507,),
)

H2O_l_thermodata: ThermoData = ThermoData(
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
    (273.150, 373.150),
    (373.150, 600),
)

S_alpha_thermodata: ThermoData = ThermoData(
    (-7.516389580e2,),
    (-7.961066980,),
    ((-1.035710779e4, 0, 1.866766938, 4.256140250e-3, -3.265252270e-06, 0, 0),),
    (200,),
    (368.3,),
)

S_beta_thermodata: ThermoData = ThermoData(
    (-6.852714730e2,),
    (-8.607846750,),
    ((0, 0, 2.080514131, 2.440879557e-3, 0, 0, 0),),
    (368.3,),
    (388.36,),
)

S_l_thermodata: ThermoData = ThermoData(
    (-6.356594920e5, -9.832222680e5, -2.638846929e4, 1.113013440e4, -8.284589830e2),
    (-1.186929589e4, -3.154806751e4, -7.681730097e2, 1.363174183e2, -1.736128237e1),
    (
        (-6.366550765e7, 0, 2.376860693e3, -7.888076026, 7.376076522e-3, 0, 0),
        (0, 0, 6.928522306e3, -3.254655981e1, 3.824448176e-2, 0, 0),
        (0, 0, 1.649945697e2, -6.843534977e-1, 7.315907973e-4, 0, 0),
        (1.972984578e6, 0, -2.441009753e1, 6.090352889e-2, -3.744069103e-5, 0, 0),
        (0, 0, 3.848693429, 0, 0, 0, 0),
    ),
    (388.36, 428.15, 432.25, 453.15, 717),
    (428.15, 432.25, 453.15, 717, 6000),
)

Si_cr_thermodata: ThermoData = ThermoData(
    (-7.850635210e2, -1.042947234e3),
    (-1.038427318e1, -1.438964187e1),
    (
        (-2.323538208e4, 0, 2.102021680, 1.809220552e-3, 0, 0, 0),
        (-5.232559740e4, 0, 2.850169415, 3.975166970e-4, 0, 0, 0),
    ),
    (200, 298.15),
    (298.15, 1690),
)

Si_l_thermodata: ThermoData = ThermoData(
    (4.882667110e3,),
    (-1.326611073e1,),
    ((0, 0, 3.271389414, 0, 0, 0, 0),),
    (1690,),
    (6000,),
)

SiO2_l_thermodata: ThermoData = ThermoData(
    (-1.140002976e5,),
    (-5.554279592e1,),
    ((0, 0, 1.004268442e1, 0, 0, 0, 0),),
    (1996,),
    (6000,),
)
