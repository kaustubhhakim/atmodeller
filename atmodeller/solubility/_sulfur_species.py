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

import logging

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from atmodeller.solubility.core import Solubility
from atmodeller.utilities import unit_conversion

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier

logger: logging.Logger = logging.getLogger(__name__)


class _S2_sulfate_andesite_boulliung23(Solubility):
    """Sulfur as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for andesite from
    :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
    equilibrated with Air/SO2 mixtures.
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        logcs: Array = -12.948 + (31586.2393 / jnp.asarray(temperature))
        logso4_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        so4_wtp: Array = jnp.power(10, logso4_wtp)
        s_wtp: Array = so4_wtp * (32.065 / 96.06)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_andesite_boulliung23: Solubility = _S2_sulfate_andesite_boulliung23()
"""Sulfur as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for andesite from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
equilibrated with Air/SO2 mixtures.
"""


class _S2_sulfide_andesite_boulliung23(Solubility):
    """Sulfur as sulfide (S^2-) in andesite :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for andesite from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
    controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        logcs: Array = 0.225 - (8921.0927 / jnp.asarray(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_andesite_boulliung23: Solubility = _S2_sulfide_andesite_boulliung23()
"""Sulfur as sulfide (S^2-) in andesite :cite:p:`BW23`

Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
for andesite from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
"""


class _S2_andesite_boulliung23(Solubility):
    """S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    _sulfide: Solubility = eqx.field(init=False)
    _sulfate: Solubility = eqx.field(init=False)

    def __post_init__(self):
        self._sulfide = S2_sulfide_andesite_boulliung23
        self._sulfate = S2_sulfate_andesite_boulliung23

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        concentration: ArrayLike = self._sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )
        concentration = concentration + self._sulfate.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )

        return concentration


S2_andesite_boulliung23: Solubility = _S2_andesite_boulliung23()
"""S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""


class _S2_sulfate_basalt_boulliung23(Solubility):
    """Sulfur in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract and the corrected expression for sulfate capacity
    (C_S6+) in :cite:t:`BW23corr`. Composition for Basalt from :cite:t:`BW22{Table 1}`. Experiments
    conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated with
    Air/SO2 mixtures.
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        logcs: Array = -12.948 + (32333.5635 / jnp.asarray(temperature))
        logso4_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        so4_wtp: Array = jnp.power(10, logso4_wtp)
        s_wtp: Array = so4_wtp * (32.065 / 96.06)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_basalt_boulliung23: Solubility = _S2_sulfate_basalt_boulliung23()
"""Sulfur in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`

Using the first equation in the abstract and the corrected expression for sulfate capacity
(C_S6+) in :cite:t:`BW23corr`. Composition for Basalt from :cite:t:`BW22{Table 1}`. Experiments
conducted at 1 atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated with
Air/SO2 mixtures.
"""


class _S2_sulfide_basalt_boulliung23(Solubility):
    """Sulfur in basalt as sulfide (S^2-) :cite:p:`BW23`

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition for
    basalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure and temperatures
    from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log
    unit below FMQ.
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        logcs: Array = 0.225 - (8045.7465 / jnp.asarray(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_basalt_boulliung23: Solubility = _S2_sulfide_basalt_boulliung23()
"""Sulfur in basalt as sulfide (S^2-) :cite:p:`BW23`

Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition for
basalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm pressure and temperatures
from 1473-1773 K in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log
unit below FMQ.
"""


class _S2_basalt_boulliung23(Solubility):
    """Sulfur in basalt due to sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    _sulfide: Solubility = eqx.field(init=False)
    _sulfate: Solubility = eqx.field(init=False)

    def __post_init__(self):
        self._sulfide = S2_sulfide_basalt_boulliung23
        self._sulfate = S2_sulfate_basalt_boulliung23

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        concentration: ArrayLike = self._sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )
        concentration = concentration + self._sulfate.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )

        return concentration


S2_basalt_boulliung23: Solubility = _S2_basalt_boulliung23()
"""Sulfur in basalt due to sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""


class _S2_sulfate_trachybasalt_boulliung23(Solubility):
    """Sulfur as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for trachybasalt from
    :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
    equilibrated with Air/SO2 mixtures.
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        logcs: Array = -12.948 + (32446.366 / jnp.asarray(temperature))
        logso4_wtp: Array = logcs + (0.5 * jnp.log10(fugacity)) + (1.5 * jnp.log10(fO2))
        so4_wtp: Array = jnp.power(10, logso4_wtp)
        s_wtp: Array = so4_wtp * (32.065 / 96.06)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfate_trachybasalt_boulliung23: Solubility = _S2_sulfate_trachybasalt_boulliung23()
"""Sulfur as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for trachybasalt from
:cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K for silicate melts
equilibrated with Air/SO2 mixtures.
"""


class _S2_sulfide_trachybasalt_boulliung23(Solubility):
    """Sulfur as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for trachybasalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
    controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        del kwargs
        logcs: Array = 0.225 - (7842.5 / jnp.asarray(temperature))
        logs_wtp: Array = logcs - (0.5 * (jnp.log10(fO2) - jnp.log10(fugacity)))
        s_wtp: Array = jnp.power(10, logs_wtp)
        ppmw: Array = s_wtp * unit_conversion.percent_to_ppm

        return ppmw


S2_sulfide_trachybasalt_boulliung23: Solubility = _S2_sulfide_trachybasalt_boulliung23()
"""Sulfur as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
for trachybasalt from :cite:t:`BW22{Table 1}`. Experiments conducted at 1 atm, 1473-1773 K in a
controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
"""


class _S2_trachybasalt_boulliung23(Solubility):
    """Sulfur in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""

    _sulfide: Solubility = eqx.field(init=False)
    _sulfate: Solubility = eqx.field(init=False)

    def __post_init__(self):
        self._sulfide = S2_sulfide_trachybasalt_boulliung23
        self._sulfate = S2_sulfate_trachybasalt_boulliung23

    @override
    def concentration(
        self, fugacity: ArrayLike, *, temperature: ArrayLike, fO2: ArrayLike, **kwargs
    ) -> Array:
        concentration: ArrayLike = self._sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )
        concentration = concentration + self._sulfate.concentration(
            fugacity, temperature=temperature, fO2=fO2, **kwargs
        )

        return concentration


S2_trachybasalt_boulliung23: Solubility = _S2_trachybasalt_boulliung23()
"""Sulfur in trachybasalt by sulfide and sulfate dissolution :cite:p:`BW22,BW23corr,BW23`"""
