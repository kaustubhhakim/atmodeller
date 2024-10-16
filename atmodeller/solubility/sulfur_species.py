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
"""Solubility laws for sulfur species

For every law there should be a test in the test suite.
"""

# Convenient to use chemical formulas so pylint: disable=C0103

import logging
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller.solubility.core import SolubilityProtocol
from atmodeller.utilities import unit_conversion

logger: logging.Logger = logging.getLogger(__name__)


class _S2_sulfate_andesite_boulliung(NamedTuple):
    """Sulfur as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for andesite from
    :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
    equilibrated with Air/SO2 mixtures.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        logcs: Array = -12.948 + (31586.2393 / jnp.asarray(temperature))
        logs_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_andesite_boulliung: SolubilityProtocol = _S2_sulfate_andesite_boulliung()
"""Sulfur as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for andesite from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
equilibrated with Air/SO2 mixtures.
"""


class _S2_sulfide_andesite_boulliung(NamedTuple):
    """Sulfur as sulfide (S^2-) in andesite :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for andesite from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
    controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        logcs: Array = 0.225 - (8921.0927 / jnp.asarray(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_andesite_boulliung: SolubilityProtocol = _S2_sulfide_andesite_boulliung()
"""Sulfur as sulfide (S^2-) in andesite :cite:p:`BW23`

Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
for andesite from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
"""


class _S2_andesite_boulliung(NamedTuple):
    """S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    sulfide: SolubilityProtocol = S2_sulfide_andesite_boulliung
    sulfate: SolubilityProtocol = S2_sulfate_andesite_boulliung

    def concentration(self, *args, **kwargs) -> ArrayLike:
        concentration: ArrayLike = self.sulfide.concentration(*args, **kwargs)
        concentration = concentration + self.sulfate.concentration(*args, **kwargs)

        return concentration


S2_andesite_boulliung: SolubilityProtocol = _S2_andesite_boulliung()
"""S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""


class _S2_sulfate_basalt_boulliung(NamedTuple):
    """Sulfur in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract and the corrected expression for sulfate capacity
    (C_S6+) in :cite:t:`BW23corr`. Composition for Basalt from :cite:t:`BW22{Table 1}`. Experiments
    conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated with
    Air/SO2 mixtures.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        logcs: Array = -12.948 + (32333.5635 / jnp.asarray(temperature))
        logso4_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        so4_wtp: Array = jnp.power(10, logso4_wtp)
        s_wtp: Array = so4_wtp * (32.065 / 96.06)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_basalt_boulliung: SolubilityProtocol = _S2_sulfate_basalt_boulliung()
"""Sulfur in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`

Using the first equation in the abstract and the corrected expression for sulfate capacity
(C_S6+) in :cite:t:`BW23corr`. Composition for Basalt from :cite:t:`BW22{Table 1}`. Experiments
conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated with
Air/SO2 mixtures.
"""


class _S2_sulfide_basalt_boulliung(NamedTuple):
    """Sulfur in basalt as sulfide (S^2-) :cite:p:`BW23`

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition for
    basalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure and temperatures
    from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log
    unit below FMQ.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        logcs: Array = 0.225 - (8045.7465 / jnp.asarray(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_basalt_boulliung: SolubilityProtocol = _S2_sulfide_basalt_boulliung()
"""Sulfur in basalt as sulfide (S^2-) :cite:p:`BW23`

Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition for
basalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure and temperatures
from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log
unit below FMQ.
"""


class _S2_basalt_boulliung(NamedTuple):
    """Sulfur in basalt due to sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    sulfide: SolubilityProtocol = S2_sulfide_basalt_boulliung
    sulfate: SolubilityProtocol = S2_sulfate_basalt_boulliung

    def concentration(self, *args, **kwargs) -> ArrayLike:
        concentration: ArrayLike = self.sulfide.concentration(*args, **kwargs)
        concentration = concentration + self.sulfate.concentration(*args, **kwargs)

        return concentration


S2_basalt_boulliung: SolubilityProtocol = _S2_basalt_boulliung()
"""Sulfur in basalt due to sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""


class _S2_sulfate_trachybasalt_boulliung(NamedTuple):
    """Sulfur as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for trachybasalt from
    :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
    equilibrated with Air/SO2 mixtures.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        logcs: Array = -12.948 + (32446.366 / jnp.asarray(temperature))
        logs_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_trachybasalt_boulliung: SolubilityProtocol = _S2_sulfate_trachybasalt_boulliung()
"""Sulfur as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for trachybasalt from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
equilibrated with Air/SO2 mixtures.
"""


class _S2_sulfide_trachybasalt_boulliung(NamedTuple):
    """Sulfur as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for trachybasalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
    controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        logcs: Array = 0.225 - (7842.5 / jnp.asarray(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_trachybasalt_boulliung: SolubilityProtocol = _S2_sulfide_trachybasalt_boulliung()
"""Sulfur as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
for trachybasalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
"""


class _S2_trachybasalt_boulliung(NamedTuple):
    """Sulfur in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    sulfide: SolubilityProtocol = S2_sulfide_trachybasalt_boulliung
    sulfate: SolubilityProtocol = S2_sulfate_trachybasalt_boulliung

    def concentration(self, *args, **kwargs) -> ArrayLike:
        concentration: ArrayLike = self.sulfide.concentration(*args, **kwargs)
        concentration = concentration + self.sulfate.concentration(*args, **kwargs)

        return concentration


S2_trachybasalt_boulliung: SolubilityProtocol = _S2_trachybasalt_boulliung()
"""Sulfur in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""


# FIXME: input fugacity there should actually be the total pressure. unfortunately when they did
# their fitting to include the melt composition, they got rid of the fS2 dependency and have
# instead the dependency of T, P and fO2. This class needs correcting somehow.
class _S2_mercury_magma_namur(NamedTuple):
    """S in reduced mafic silicate melts relevant for Mercury :cite:p:`NCH16`

    Dissolved S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like
    magmas :cite:t:`NCH16{Equation 10}`, with coefficients from :cite:t:`NCH16{Table 2}`, assumed
    composition is Northern Volcanic Plains (NVP). Experiments on Mercurian lavas and enstatite
    chondrites at 1200-1750 C and pressures from 1 bar to 4 GPa. Equilibrated silicate melts with
    sulfide and metallic melts at reducing conditions (fO2 at IW-1.5 to IW-9.4).
    """

    coefficients: tuple[float, float, float, float] = (7.25, -2.54e4, 0.04, -0.551)

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del pressure
        wt_perc: Array = jnp.exp(
            self.coefficients[0]
            + (self.coefficients[1] / temperature)
            + ((self.coefficients[2] * fugacity) / temperature)
            + (self.coefficients[3] * jnp.log10(fO2))
            - 0.136
        )
        ppmw: Array = wt_perc * unit_conversion.percent_to_ppm

        return ppmw


S2_mercury_magma_namur: SolubilityProtocol = _S2_mercury_magma_namur()
"""S in reduced mafic silicate melts relevant for Mercury :cite:p:`NCH16`

Dissolved S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like
magmas :cite:t:`NCH16{Equation 10}`, with coefficients from :cite:t:`NCH16{Table 2}`, assumed
composition is Northern Volcanic Plains (NVP). Experiments on Mercurian lavas and enstatite
chondrites at 1200-1750 C and pressures from 1 bar to 4 GPa. Equilibrated silicate melts with
sulfide and metallic melts at reducing conditions (fO2 at IW-1.5 to IW-9.4).
"""
