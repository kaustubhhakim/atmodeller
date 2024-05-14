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
"""Solubility laws for hydrogen species"""

# Convenient to use chemical formulas so pylint: disable=C0103

from __future__ import annotations

import logging
import sys

import numpy as np

from atmodeller.solubility.interfaces import Solubility, SolubilityPowerLaw

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class H2_andesite_hirschmann(Solubility):
    """H2 in synthetic andesite :cite:p:`HWA12`

    Log-scale linear fit to fH2 vs H2 concentration for andesite in :cite:t:`HWA12{Table 2}`. Experiments conducted
    from 0.7-3 GPa at 1400 C.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        ppmw: float = 10 ** (0.60128868 * np.log10(fugacity) + 1.01058631)

        return ppmw


class H2_basalt_hirschmann(Solubility):
    """H2 in synthetic basalt :cite:p:`HWA12`

    Log-scale linear fit to fH2 vs. H2 concentration for basalt in :cite:t:`HWA12{Table 2}`. Experiments conducted
    from 0.7-3 GPa, 1400 C.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        ppmw: float = 10 ** (0.52413928 * np.log10(fugacity) + 1.10083602)

        return ppmw


class H2_silicic_melts_gaillard(SolubilityPowerLaw):
    """Fe-H redox exchange in silicate glasses :cite:p:`GSM03`

    Power law fit for fH2 vs. H2 (ppm-wt) from :cite:t:`GSM03{Table 4}` data. Experiments at pressures from 0.02-70
    bar, temperatures from 300-1000C.
    """

    @override
    def __init__(self, constant: float = 0.163, exponent: float = 1.252):
        super().__init__(constant, exponent)


class H2O_ano_dio_newcombe(SolubilityPowerLaw):
    """H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`

    Power law from :cite:t:`NBB17{Figure 5(A)}` for anorthite-diopside glass. Experiments conducted at 1 atm and
    1350 C. Melts equilibrated in 1 atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3
    to IW+4.8 and pH2/pH2O from 0.003-24.
    """

    @override
    def __init__(self, constant: float = 727, exponent: float = 0.5):
        super().__init__(constant, exponent)


class H2O_basalt_dixon(SolubilityPowerLaw):
    """H2O solubilities in MORB liquids :cite:p:`DSH95`

    Refitted data to a power law by Paolo Sossi (fitting :cite:t:`DSH95{Figure 4}`, TODO: CHECK). Experiments
    conducted at 1200 C, 200-717 bars with pure H2O.
    """

    @override
    def __init__(self, constant: float = 965, exponent: float = 0.5):
        super().__init__(constant, exponent)


class H2O_basalt_mitchell(SolubilityPowerLaw):
    """H2O solubility in basaltic melt :cite:p:`MGO17`

    Refitted the H2O wt. % vs. fH2O fitted line from :cite:t:`MGO17{Figure 8}` to a power-law. Experiments
    conducted at 1200 C and 1000 MPa total pressure. This fit includes data from
    their experiments and prior studies on H2O solubility in basaltic melt at 1200 C and P at or
    below 600 MPa.
    """

    @override
    def __init__(self, constant: float = 258.946, exponent: float = 0.669):
        super().__init__(constant, exponent)


class H2O_basalt_wilson(SolubilityPowerLaw):
    """H2O in basalt :cite:p:`WH81,HBO64`

    :cite:t:`WH81{Equation 30}`, and converting from weight % to ppmw. Not clear what all experimental data is used
    to derive this fit, but it fits data at 1100 C and 1000-6000 bars H2O from :cite:t:`HBO64`
    decently well (their Table 3).
    """

    @override
    def __init__(self, constant: float = 215, exponent: float = 0.7):
        super().__init__(constant, exponent)


class H2O_lunar_glass_newcombe(SolubilityPowerLaw):
    """H2O in lunar basalt :cite:p:`NBB17`

    Power law from :cite:t:`NBB17{Figure 5(A)}` for Lunar glass. Experiments conducted at 1 atm and 1350 C. Melts
    equilibrated in 1-atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3 to IW+4.8.
    """

    @override
    def __init__(self, constant: float = 683, exponent: float = 0.5):
        super().__init__(constant, exponent)


class H2O_peridotite_sossi(SolubilityPowerLaw):
    """Solubility of water in peridotite liquids :cite:p:`STB23`

    Power law parameters in the abstract for peridotitic glasses. Experiments conducted at 2173 K
    and 1 bar and range of fO2 from IW-1.9 to IW+6.0.
    """

    @override
    def __init__(self, constant: float = 647, exponent: float = 0.5):
        super().__init__(constant, exponent)
