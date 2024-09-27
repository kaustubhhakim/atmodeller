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
"""Thermochemical data library"""

import jax.numpy as jnp
from jax import Array

from atmodeller.thermodata.jax_thermo import ThermoData, get_gibbs_over_RT

CO: ThermoData = ThermoData(
    [-1.303131878e4, -2.466261084e3, 5.701421130e6],
    [
        -7.859241350,
        -1.387413108e1,
        -2.060704786e3,
    ],
    [
        [
            1.489045326e4,
            -2.922285939e2,
            5.724527170,
            -8.176235030e-3,
            1.456903469e-5,
            -1.087746302e-8,
            3.027941827e-12,
        ],
        [
            4.619197250e5,
            -1.944704863e3,
            5.916714180,
            -5.664282830e-4,
            1.398814540e-7,
            -1.787680361e-11,
            9.620935570e-16,
        ],
        [
            8.868662960e8,
            -7.500377840e5,
            2.495474979e2,
            -3.956351100e-2,
            3.297772080e-6,
            -1.318409933e-10,
            1.998937948e-15,
        ],
    ],
    [200, 1000, 6000],
    [1000, 6000, 20000],
)

if __name__ == "__main__":

    # Example usage
    temperature: Array = jnp.array(20000.0)
    gibbs: Array = get_gibbs_over_RT(CO, temperature)
    print(gibbs)
