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
"""Solubility laws for hydrogen species

For every law there should be a test in the test suite.
"""

from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility.classes import SolubilityPowerLaw, SolubilityPowerLawLog10

H2_andesite_hirschmann12: SolubilityProtocol = SolubilityPowerLawLog10(1.01058631, 0.60128868)
"""H2 in synthetic andesite :cite:p:`HWA12`

Log-scale linear fit to fH2 vs H2 concentration for andesite in :cite:t:`HWA12{Table 2}`.
Experiments conducted from 0.7-3 GPa at 1400 C.
"""

H2_basalt_hirschmann12: SolubilityProtocol = SolubilityPowerLawLog10(1.10083602, 0.52413928)
"""H2 in synthetic basalt :cite:p:`HWA12`

Log-scale linear fit to fH2 vs. H2 concentration for basalt in :cite:t:`HWA12{Table 2}`.
Experiments conducted from 0.7-3 GPa, 1400 C.
"""

H2_silicic_melts_gaillard03: SolubilityProtocol = SolubilityPowerLaw(0.163, 1.252)
"""Fe-H redox exchange in silicate glasses :cite:p:`GSM03`

Power law fit for fH2 vs. H2 (ppm-wt) from :cite:t:`GSM03{Table 4}` data. Experiments at
pressures from 0.02-70 bar, temperatures from 300-1000C.
"""

H2O_ano_dio_newcombe17: SolubilityProtocol = SolubilityPowerLaw(727, 0.5)
"""H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`

Power law from :cite:t:`NBB17{Figure 5(A)}` for anorthite-diopside glass. Experiments conducted
at 1 atm and 1350 C. Melts equilibrated in 1 atm furnace with H2/CO2 gas mixtures that spanned
fO2 from IW-3 to IW+4.8 and pH2/pH2O from 0.003-24.
"""

H2O_basalt_dixon95: SolubilityProtocol = SolubilityPowerLaw(965, 0.5)
"""H2O in MORB liquids :cite:p:`DSH95`

Refitted data to a power law by Paolo Sossi (fitting :cite:t:`DSH95{Figure 4}`, TODO: CHECK).
Experiments conducted at 1200 C, 200-717 bars with pure H2O.
"""

H2O_basalt_mitchell17: SolubilityProtocol = SolubilityPowerLaw(258.946, 0.669)
"""H2O in basaltic melt :cite:p:`MGO17`

Refitted the H2O wt. % vs. fH2O fitted line from :cite:t:`MGO17{Figure 8}` to a power-law.
Experiments conducted at 1200 C and 1000 MPa total pressure. This fit includes data from
their experiments and prior studies on H2O solubility in basaltic melt at 1200 C and P at or
below 600 MPa.
"""

H2O_basalt_wilson81: SolubilityProtocol = SolubilityPowerLaw(215, 0.7)
"""H2O in basalt :cite:p:`WH81,HBO64`

:cite:t:`WH81{Equation 30}`, and converting from weight % to ppmw. Not clear what all
experimental data is used to derive this fit, but it fits data at 1100 C and 1000-6000 bars H2O
from :cite:t:`HBO64` decently well (their Table 3).
"""

H2O_lunar_glass_newcombe17: SolubilityProtocol = SolubilityPowerLaw(683, 0.5)
"""H2O in lunar basalt :cite:p:`NBB17`

Power law from :cite:t:`NBB17{Figure 5(A)}` for Lunar glass. Experiments conducted at 1 atm and
1350 C. Melts equilibrated in 1-atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3
to IW+4.8.
"""

H2O_peridotite_sossi23: SolubilityProtocol = SolubilityPowerLaw(647, 0.5)
"""H2O in peridotite liquids :cite:p:`STB23`

Power law parameters in the abstract for peridotitic glasses. Experiments conducted at 2173 K
and 1 bar and range of fO2 from IW-1.9 to IW+6.0.
"""
