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
"""Solubility laws for sulfur species"""

# Convenient to use chemical formulas so pylint: disable=C0103

from __future__ import annotations

import logging
import sys

import numpy as np

from atmodeller.solubility.interfaces import Solubility
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class S2_sulfate_andesite_boulliung(Solubility):
    """Sulfur as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for andesite from Table 1.
    Experiments conducted at 1 atm, 1473-1773 K for silicate melts equilibrated with Air/SO2
    mixtures.
    """

    @override
    def concentration(self, fugacity: float, *, temperature: float, O2: float, **kwargs) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = -12.948 + (31586.2393 / temperature)
        logs_wtp: float = logcs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(O2))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class S2_sulfide_andesite_boulliung(Solubility):
    """Sulfur as sulfide (S^2-) in andesite :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for andesite from Table 1. Experiments conducted at 1 atm, 1473-1773 K in a controlled
    CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    def concentration(self, fugacity: float, *, temperature: float, O2: float, **kwargs) -> float:
        del kwargs
        logcs: float = 0.225 - (8921.0927 / temperature)
        logs_wtp: float = logcs - (0.5 * (np.log10(O2) - np.log10(fugacity)))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class S2_andesite_boulliung(Solubility):
    """S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    def __init__(self):
        self.sulfide: Solubility = S2_sulfide_andesite_boulliung()
        self.sulfate: Solubility = S2_sulfate_andesite_boulliung()

    @override
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        O2: float,
        **kwargs,
    ) -> float:
        del kwargs
        concentration: float = self.sulfide.concentration(fugacity, temperature=temperature, O2=O2)
        concentration += self.sulfate.concentration(fugacity, temperature=temperature, O2=O2)

        return concentration


class S2_sulfate_basalt_boulliung(Solubility):
    """Sulfur in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract and the corrected expression for sulfate capacity
    (C_S6+) in :cite:t:`BW23corr`. Composition for Basalt from Table 1. Experiments conducted at 1
    atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated with Air/SO2
    mixtures.
    """

    @override
    def concentration(self, fugacity: float, *, temperature: float, O2: float, **kwargs) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = -12.948 + (32333.5635 / temperature)
        logso4_wtp: float = logcs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(O2))
        so4_wtp: float = 10**logso4_wtp
        s_wtp: float = so4_wtp * (32.065 / 96.06)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class S2_sulfide_basalt_boulliung(Solubility):
    """Sulfur in basalt as sulfide (S^2-) :cite:p:`BW23`

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition for
    basalt from Table 1. Experiments conducted at 1 atm pressure and temperatures from 1473-1773 K
    in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    def concentration(self, fugacity: float, *, temperature: float, O2: float, **kwargs) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = 0.225 - (8045.7465 / temperature)
        logs_wtp: float = logcs - (0.5 * (np.log10(O2) - np.log10(fugacity)))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class S2_basalt_boulliung(Solubility):
    """Total sulfur in basalt due to both sulfide and sulfate dissolution
    :cite:p:`BW22,BW23corr,BW23`
    """

    def __init__(self):
        self.sulfide_solubility: Solubility = S2_sulfide_basalt_boulliung()
        self.sulfate_solubility: Solubility = S2_sulfate_basalt_boulliung()

    @override
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        O2: float,
        **kwargs,
    ) -> float:
        del kwargs
        concentration: float = self.sulfide_solubility.concentration(
            fugacity, temperature=temperature, O2=O2
        )
        concentration += self.sulfate_solubility.concentration(
            fugacity, temperature=temperature, O2=O2
        )

        return concentration


class S2_sulfate_trachybasalt_boulliung(Solubility):
    """Sulfur as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for trachybasalt from Table 1.
    Experiments conducted at 1 atm, 1473-1773 K for silicate melts equilibrated with Air/SO2
    mixtures.
    """

    @override
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        O2: float,
        **kwargs,
    ) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = -12.948 + (32446.366 / temperature)
        logs_wtp: float = logcs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(O2))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class S2_sulfide_trachybasalt_boulliung(Solubility):
    """Sulfur as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for trachybasalt from Table 1. Experiments conducted at 1 atm, 1473-1773 K in a controlled
    CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        O2: float,
        **kwargs,
    ) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = 0.225 - (7842.5 / temperature)
        logs_wtp: float = logcs - (0.5 * (np.log10(O2) - np.log10(fugacity)))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class S_mercury_magma_namur(Solubility):
    """Namur et al. 2016. Sulfur solubility in reduced mafic silicate melts relevant for Mercury

    https://ui.adsabs.harvard.edu/abs/2016E%26PSL.448..102N/abstract

    Dissolved S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like
    magmas Equation 10, with coefficients from Table 2, assumed composition is Northern Volcanic
    Plains (NVP). Experiments on Mercurian lavas and enstatite chondrites at 1200-1750 C and
    pressures from 1 bar to 4 GPa. Equilibrated silicate melts with sulfide and metallic melts at
    reducing conditions (fO2 at IW-1.5 to IW-9.4).
    """

    def __init__(self):
        self.coefficients: tuple[float, ...] = (7.25, -2.54e4, 0.04, -0.551)

    @override
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        O2: float,
        **kwargs,
    ) -> float:
        del kwargs
        wt_perc: float = np.exp(
            self.coefficients[0]
            + (self.coefficients[1] / temperature)
            + ((self.coefficients[2] * fugacity) / temperature)
            + (self.coefficients[3] * np.log10(O2))
            - 0.136
        )
        ppmw: float = UnitConversion.weight_percent_to_ppmw(wt_perc)

        return ppmw
