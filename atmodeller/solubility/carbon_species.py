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
"""Solubility laws for carbon species

For every law there should be a test in the test suite.
"""

# Convenient to use chemical formulas so pylint: disable=invalid-name

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller.solubility.core import SolubilityProtocol
from atmodeller.utilities import unit_conversion


class _CH4_basalt_ardia(NamedTuple):
    """CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`

    Experiments conducted at 0.7-3 GPa and 1400-1450 C. :cite:t:`AHW13{Equations 7a, 8}`, values
    for lnK0 and deltaV from the text.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> ArrayLike:
        del temperature
        del fO2
        pressure_gpa: ArrayLike = pressure * unit_conversion.bar_to_GPa
        one_bar_in_gpa: ArrayLike = unit_conversion.bar_to_GPa
        k: Array = jnp.exp(4.93 - (1.93 * (pressure_gpa - one_bar_in_gpa)))
        ppmw: Array = k * fugacity * unit_conversion.bar_to_GPa

        return ppmw


CH4_basalt_ardia: SolubilityProtocol = _CH4_basalt_ardia()
"""CH4 in haplobasalt (Fe-free) silicate melt :cite:p:`AHW13`

Experiments conducted at 0.7-3 GPa and 1400-1450 C. :cite:t:`AHW13{Equations 7a, 8}`, values
for lnK0 and deltaV from the text.
"""


class _CO_basalt_armstrong(NamedTuple):
    """Volatiles in mafic melts under reduced conditions :cite:p:`AHS15`

    Experiments on Martian and terrestrial basalts at 1.2 GPa and 1400 C with variable fO2 from
    IW-3.65 to IW+1.46. :cite:t:`AHS15{Equation 10}`, log-scale linear fit for CO and includes
    dependence on total pressure. The fitting coefficients also use data from :cite:p:`SHW14`
    (experiments from 1-1.2 GPa).
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        del temperature
        del fO2
        logco_ppm: Array = -0.738 + (0.876 * jnp.log10(fugacity)) - (5.44e-5 * pressure)
        ppmw: Array = jnp.power(10, logco_ppm)

        return ppmw


CO_basalt_armstrong: SolubilityProtocol = _CO_basalt_armstrong()
"""Volatiles in mafic melts under reduced conditions :cite:p:`AHS15`

Experiments on Martian and terrestrial basalts at 1.2 GPa and 1400 C with variable fO2 from
IW-3.65 to IW+1.46. :cite:t:`AHS15{Equation 10}`, log-scale linear fit for CO and includes
dependence on total pressure. The fitting coefficients also use data from :cite:p:`SHW14`
(experiments from 1-1.2 GPa).
"""


class _CO_basalt_yoshioka(NamedTuple):
    """Carbon in silicate melts :cite:p:`YNN19`

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C. Log-scale linear expression for solubility of CO in
    MORB in the abstract.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        del temperature
        del pressure
        del fO2
        co_wtp: Array = jnp.power(10, (-5.20 + (0.8 * jnp.log10(fugacity))))
        ppmw: Array = co_wtp * unit_conversion.percent_to_ppm

        return ppmw


CO_basalt_yoshioka: SolubilityProtocol = _CO_basalt_yoshioka()
"""Carbon in silicate melts :cite:p:`YNN19`

Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
CO-CO2 fluid phase at 3 GPa and 1500 C. Log-scale linear expression for solubility of CO in
MORB in the abstract.
"""


class _CO_rhyolite_yoshioka(NamedTuple):
    """Carbon in silicate melts :cite:p:`YNN19`

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C. Henry's Law, their expression for solubility of CO in
    rhyolite in the abstract.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        del temperature
        del pressure
        del fO2
        co_wtp: Array = jnp.power(10, (-4.08 + (0.52 * jnp.log10(fugacity))))
        ppmw: Array = co_wtp * unit_conversion.percent_to_ppm

        return ppmw


CO_rhyolite_yoshioka: SolubilityProtocol = _CO_rhyolite_yoshioka()
"""Carbon in silicate melts :cite:p:`YNN19`

Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
CO-CO2 fluid phase at 3 GPa and 1500 C. Henry's Law, their expression for solubility of CO in
rhyolite in the abstract.
"""


class _CO2_basalt_dixon(NamedTuple):
    """CO2 in MORB liquids :cite:p:`DSH95`

    :cite:t:`DSH95{Equation 6}` for mole fraction of dissolved carbonate (CO3^2-) and then
    converting to ppmw for CO2 experiments conducted at 1200 C, 210-980 bars with mixed H2O-CO2
    vapor phase (CO2 vapor mole fraction varied from 0.42-0.97).
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        fO2: ArrayLike,
    ) -> Array:
        del fO2
        arg: Array = jnp.exp(-23 * (pressure - 1) / (83.15 * temperature)) * fugacity * 3.8e-7
        ppmw: Array = 1.0e4 * (4400 * arg) / (36.6 - 44 * arg)

        return ppmw


CO2_basalt_dixon: SolubilityProtocol = _CO2_basalt_dixon()
"""CO2 in MORB liquids :cite:p:`DSH95`

:cite:t:`DSH95{Equation 6}` for mole fraction of dissolved carbonate (CO3^2-) and then
converting to ppmw for CO2 experiments conducted at 1200 C, 210-980 bars with mixed H2O-CO2
vapor phase (CO2 vapor mole fraction varied from 0.42-0.97).
"""
