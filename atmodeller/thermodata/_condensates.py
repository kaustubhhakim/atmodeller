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

from atmodeller.thermodata import SpeciesData

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
O2Si_l: SpeciesData = SpeciesData("O2Si", "l")
"Species data for O2Si_l"
S_cr: SpeciesData = SpeciesData("S", "cr")
"Species data for S_alpha and S_beta"
S_l: SpeciesData = SpeciesData("S", "l")
"Species data for S_l"
Si_cr: SpeciesData = SpeciesData("Si", "cr")
"Species data for Si_cr"
Si_l: SpeciesData = SpeciesData("Si", "l")
"Species data for Si_l"
