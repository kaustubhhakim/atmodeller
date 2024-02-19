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
"""Solubility laws"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

# Solubility limiters
# Applied universally
MAXIMUM_PPMW: float = UnitConversion.weight_percent_to_ppmw(10)  # 10% by weight
# Applied to sulfur
SULFUR_MAXIMUM_PPMW: float = UnitConversion.weight_percent_to_ppmw(1)  # 1% by weight


def limit_concentration(bound: float = MAXIMUM_PPMW) -> Callable:
    """A decorator to limit the concentration in ppmw

    Args:
        bound: The maximum limit of the concentration in ppmw. Defaults to ``MAXIMUM_PPMW``.

    Returns:
        The decorator
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Solubility, *args, **kwargs):
            result: float = func(self, *args, **kwargs)
            if result > bound:
                logger.warning(
                    "%s concentration (%d ppmw) will be limited to %d ppmw",
                    self.__class__.__name__,
                    result,
                    bound,
                )

            return np.clip(result, 0, bound)

        return wrapper

    return decorator


class Solubility(ABC):
    """A solubility law"""

    @abstractmethod
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
        **kwargs,
    ) -> float:
        """Dissolved volatile concentration in the melt in ppmw

        Args:
            fugacity: Fugacity of the species in bar
            temperature: Temperature in kelvin. Defaults to None.
            pressure: Total pressure in bar. Defaults to None.
            **kwargs: Arbitrary keyword arguments. Keyword arguments that are fugacities must
                adhere to the naming convention: fO2, fH2, fH2O, etc.

        Returns:
            Dissolved volatile concentration in the melt in ppmw
        """
        raise NotImplementedError

    @limit_concentration()
    def clipped_concentration(self, *args, **kwargs) -> float:
        """Dissolved volatile concentration in the melt in ppmw limited below a maximum bound"""
        return self.concentration(*args, **kwargs)


class SolubilityPowerLaw(Solubility):
    """A solubility power law

    Args:
        constant: Constant
        exponent: Exponent
    """

    def __init__(self, constant: float, exponent: float):
        self.constant: float = constant
        self.exponent: float = exponent

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs

        return self.constant * fugacity**self.exponent


class NoSolubility(Solubility):
    """No solubility"""

    @override
    def concentration(self, *args, **kwargs) -> float:
        del args
        del kwargs
        return 0.0


class AndesiteH2(Solubility):
    """H2 in synthetic andesite :cite:p:`HWA12`

    Log-scale linear fit to fH2 vs H2 concentration for andesite in Table 2. Experiments conducted
    from 0.7-3 GPa at 1400 C.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        ppmw: float = 10 ** (0.60128868 * np.log10(fugacity) + 1.01058631)

        return ppmw


class AndesiteS2Sulfate(Solubility):
    """Sulfur as sulfate SO4^2-/S^6+ in andesite :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for andesite from Table 1.
    Experiments conducted at 1 atm, 1473-1773 K for silicate melts equilibrated with Air/SO2
    mixtures.
    """

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(self, fugacity: float, *, temperature: float, fO2: float, **kwargs) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = -12.948 + (31586.2393 / temperature)
        logs_wtp: float = logcs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fO2))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class AndesiteS2Sulfide(Solubility):
    """Sulfur as sulfide (S^2-) in andesite :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for andesite from Table 1. Experiments conducted at 1 atm, 1473-1773 K in a controlled
    CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(self, fugacity: float, *, temperature: float, fO2: float, **kwargs) -> float:
        del kwargs
        logcs: float = 0.225 - (8921.0927 / temperature)
        logs_wtp: float = logcs - (0.5 * (np.log10(fO2) - np.log10(fugacity)))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class AndesiteS2(Solubility):
    """S2 in andesite accounting for both sulfide and sulfate :cite:p:`BW22,BW23corr,BW23`"""

    def __init__(self):
        self.sulfide: Solubility = AndesiteS2Sulfide()
        self.sulfate: Solubility = AndesiteS2Sulfate()

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        fO2: float,
        **kwargs,
    ) -> float:
        del kwargs
        concentration: float = self.sulfide.concentration(
            fugacity, temperature=temperature, fO2=fO2
        )
        concentration += self.sulfate.concentration(fugacity, temperature=temperature, fO2=fO2)

        return concentration


class AnorthiteDiopsideH2O(SolubilityPowerLaw):
    """H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`

    Power law from Figure 5(A) for anorthite-diopside glass. Experiments conducted at 1 atm and
    1350 C. Melts equilibrated in 1 atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3
    to IW+4.8 and pH2/pH2O from 0.003-24.
    """

    @override
    def __init__(self, constant: float = 727, exponent: float = 0.5):
        super().__init__(constant, exponent)


class BasaltCO2(Solubility):
    """CO2 solubilities in MORB liquids :cite:p:`DSH95`

    Equation 6 for mole fraction of dissolved carbonate (CO3^2-) and then converting to ppmw for
    CO2 experiments conducted at 1200 C, 210-980 bars with mixed H2O-CO2 vapor phase (CO2 vapor
    mole fraction varied from 0.42-0.97).
    """

    @override
    def concentration(
        self, fugacity: float, *, temperature: float, pressure: float, **kwargs
    ) -> float:
        del kwargs
        ppmw: float = (3.8e-7) * fugacity * np.exp(-23 * (pressure - 1) / (83.15 * temperature))
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)

        return ppmw


class BasaltH2O(SolubilityPowerLaw):
    """H2O solubilities in MORB liquids :cite:p:`DSH95`

    Refitted data to a power law by Paolo Sossi (fitting Figure 4, TODO: CHECK). Experiments
    conducted at 1200 C, 200-717 bars with pure H2O.
    """

    @override
    def __init__(self, constant: float = 965, exponent: float = 0.5):
        super().__init__(constant, exponent)


class BasaltH2OMitchell(SolubilityPowerLaw):
    """H2O solubility in basaltic melt :cite:p:`MGO17`

    Refitted the H2O wt. % vs. fH2O fitted line from their Figure 8 to a power-law. Experiments
    conducted at 1200 C and 1000 MPa total pressure. Mitchell's Figure 8 fit includes data from their
    experiments and prior studies on H2O solubility in basaltic melt at 1200 C and P at or below
    600 MPa.
    """

    @override
    def __init__(self, constant: float = 258.946, exponent: float = 0.669):
        super().__init__(constant, exponent)


class BasaltH2(Solubility):
    """H2 in synthetic basalt :cite:p:`HWA12`

    Log-scale linear fit to fH2 vs. H2 concentration for basalt in Table 2. Experiments conducted
    from 0.7-3 GPa, 1400 C.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        ppmw: float = 10 ** (0.52413928 * np.log10(fugacity) + 1.10083602)

        return ppmw


class BasaltN2Libourel(Solubility):
    """N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`

    Equation 23, includes dependencies on fN2 and fO2. Experiments conducted at 1 atm and 1425 C
    (two experiments at 1400 C), fO2 from IW-8.3 to IW+8.7 using mixtures of CO, CO2 and N2 gases.
    """

    def __init__(self):
        self._power_law: SolubilityPowerLaw = SolubilityPowerLaw(constant=0.0611, exponent=1)

    @override
    def concentration(self, fugacity: float, *, fO2: float, **kwargs) -> float:
        del kwargs
        ppmw: float = self._power_law.concentration(fugacity)
        constant: float = (fO2**-0.75) * 5.97e-10
        power_law: SolubilityPowerLaw = SolubilityPowerLaw(constant=constant, exponent=0.5)
        ppmw += power_law.concentration(fugacity)

        return ppmw


class BasaltN2Dasgupta(Solubility):
    """N2 in silicate melts :cite:p:`DFP22`

    Using Equation 10, composition parameters from :cite:t:`DFP22{Figure 8}`, and
    Iron-wustite buffer (logIW_fugacity) from :cite:t:`OP93,HGD08`.

    Performed experiments on 80:20 synthetic basalt-Si3N4 mixture at 1.5-3.0 GPa and 1300-1600 C
    fO2 from ~IW-3 to IW-4. Combined this high pressure data with lower pressure studies to derive
    their N solubility law.

    Args:
        xsio2: Mole fraction of SiO2
        xal2o3: Mole fraction of Al2O3
        xtio2: Mole fraction of TiO2
    """

    def __init__(self, xsio2: float = 0.56, xal2o3: float = 0.11, xtio2: float = 0.01):
        self.xsio2: float = xsio2
        self.xal2o3: float = xal2o3
        self.xtio2: float = xtio2

    @override
    def concentration(
        self, fugacity: float, *, temperature: float, pressure: float, fO2: float, **kwargs
    ) -> float:
        del kwargs
        fugacity_gpa: float = UnitConversion.bar_to_GPa(fugacity)
        pressure_gpa: float = UnitConversion.bar_to_GPa(pressure)
        logiw_fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )
        fo2_shift = np.log10(fO2) - logiw_fugacity
        ppmw: float = (fugacity_gpa**0.5) * np.exp(
            (5908.0 * (pressure_gpa**0.5) / temperature) - (1.6 * fo2_shift)
        )
        ppmw += fugacity_gpa * np.exp(
            4.67 + (7.11 * self.xsio2) - (13.06 * self.xal2o3) - (120.67 * self.xtio2)
        )

        return ppmw


class BasaltN2Bernadou(Solubility):
    """N2 in basaltic silicate melt :cite:p:`BGF21`

    Equation 18 and using Equations 19-20 and the values for the thermodynamic constants from Table
    6. Experiments on basaltic samples at fluid saturation in C-H-O-N system, pressure range:
    0.8-10 kbar, temperature range: 1200-1300 C; fO2 range: IW+4.9 to IW-4.7. Using their
    experimental results and a database for N concentrations at fluid saturation from 1 bar to 10
    kbar, calibrated their solubility law.
    """

    @override
    def concentration(
        self, fugacity: float, *, temperature: float, pressure: float, fO2: float, **kwargs
    ) -> float:
        del kwargs
        k13: float = np.exp(
            -(29344 + 121 * temperature + 4 * pressure) / (GAS_CONSTANT * temperature)
        )
        k14: float = np.exp(
            -(183733 + 172 * temperature - 5 * pressure) / (GAS_CONSTANT * temperature)
        )
        molfrac: float = (k13 * fugacity) + ((fO2 ** (-3 / 4)) * k14 * (fugacity**0.5))
        ppmw: float = UnitConversion.fraction_to_ppm(molfrac)

        return ppmw


class BasaltS2Sulfate(Solubility):
    """Sulfur in basalt as sulfate, SO4^2-/S^6+ :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract and the corrected expression for sulfate capacity
    (C_S6+) in :cite:t:`BW23corr`. Composition for Basalt from Table 1. Experiments conducted at 1
    atm pressure, temperatures from 1473-1773 K for silicate melts equilibrated with Air/SO2
    mixtures.
    """

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(self, fugacity: float, *, temperature: float, fO2: float, **kwargs) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = -12.948 + (32333.5635 / temperature)
        logso4_wtp: float = logcs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fO2))
        so4_wtp: float = 10**logso4_wtp
        s_wtp: float = so4_wtp * (32.065 / 96.06)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class BasaltS2Sulfide(Solubility):
    """Sulfur in basalt as sulfide (S^2-) :cite:p:`BW23`

    Using expressions in the abstract for S wt% and sulfide capacity (C_S2-). Composition for
    basalt from Table 1. Experiments conducted at 1 atm pressure and temperatures from 1473-1773 K
    in a controlled CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(self, fugacity: float, *, temperature: float, fO2: float, **kwargs) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = 0.225 - (8045.7465 / temperature)
        logs_wtp: float = logcs - (0.5 * (np.log10(fO2) - np.log10(fugacity)))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class BasaltS2(Solubility):
    """Total sulfur in basalt due to both sulfide and sulfate dissolution
    :cite:p:`BW22,BW23corr,BW23`
    """

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltS2Sulfide()
        self.sulfate_solubility: Solubility = BasaltS2Sulfate()

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        fO2: float,
        **kwargs,
    ) -> float:
        del kwargs
        concentration: float = self.sulfide_solubility.concentration(
            fugacity, temperature=temperature, fO2=fO2
        )
        concentration += self.sulfate_solubility.concentration(
            fugacity, temperature=temperature, fO2=fO2
        )

        return concentration


class BasaltH2OWilson(SolubilityPowerLaw):
    """H2O in basalt :cite:p:`WH81,HBO64`

    Equation 30, and converting from weight % to ppmw. Not clear what all experimental data is used
    to derive this fit, but it fits data at 1100 C and 1000-6000 bars H2O from :cite:t:`HBO64`
    decently well (their Table 3).
    """

    @override
    def __init__(self, constant: float = 215, exponent: float = 0.7):
        super().__init__(constant, exponent)


class TBasaltS2Sulfate(Solubility):
    """Sulfur as sulfate SO4^2-/S^6+ in trachybasalt :cite:p:`BW22,BW23corr`

    Using the first equation in the abstract of :cite:t:`BW22` and the corrected expression for
    sulfate capacity (C_S6+) in :cite:t:`BW23corr`. Composition for trachybasalt from Table 1.
    Experiments conducted at 1 atm, 1473-1773 K for silicate melts equilibrated with Air/SO2
    mixtures.
    """

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        fO2: float,
        **kwargs,
    ) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = -12.948 + (32446.366 / temperature)
        logs_wtp: float = logcs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fO2))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class TBasaltS2Sulfide(Solubility):
    """Sulfur as sulfide (S^2-) in trachybasalt :cite:p:`BW23`

    Using expressions in the abstract for S wt.% and sulfide capacity (C_S2-). Composition
    for trachybasalt from Table 1. Experiments conducted at 1 atm, 1473-1773 K in a controlled
    CO-CO2-SO2 atmosphere fO2 conditions were greater than 1 log unit below FMQ.
    """

    @override
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        fO2: float,
        **kwargs,
    ) -> float:
        # Fugacity is fS2
        del kwargs
        logcs: float = 0.225 - (7842.5 / temperature)
        logs_wtp: float = logcs - (0.5 * (np.log10(fO2) - np.log10(fugacity)))
        s_wtp: float = 10**logs_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(s_wtp)

        return ppmw


class LunarGlassH2O(SolubilityPowerLaw):
    """H2O in lunar basalt :cite:p:`NBB17`

    Power law from Figure 5(A) for Lunar glass. Experiments conducted at 1 atm and 1350 C. Melts
    equilibrated in 1-atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3 to IW+4.8.
    """

    @override
    def __init__(self, constant: float = 683, exponent: float = 0.5):
        super().__init__(constant, exponent)


class MercuryMagmaS(Solubility):
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
    # @limit_concentration(SULFUR_MAXIMUM_PPMW)
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        fO2: float,
        **kwargs,
    ) -> float:
        del kwargs
        wt_perc: float = np.exp(
            self.coefficients[0]
            + (self.coefficients[1] / temperature)
            + ((self.coefficients[2] * fugacity) / temperature)
            + (self.coefficients[3] * np.log10(fO2))
            - 0.136
        )
        ppmw: float = UnitConversion.weight_percent_to_ppmw(wt_perc)

        return ppmw


class PeridotiteH2O(SolubilityPowerLaw):
    """Sossi et al. (2023). Solubility of water in peridotite liquids

    https://ui.adsabs.harvard.edu/abs/2023E%26PSL.60117894S/abstract

    Power law parameters in the abstract for peridotitic glasses. Experiments conducted at 2173 K
    and 1 bar and range of fO2 from IW-1.9 to IW+6.0.
    """

    @override
    def __init__(self, constant: float = 647, exponent: float = 0.5):
        super().__init__(constant, exponent)


class SilicicMeltsH2(SolubilityPowerLaw):
    """Gaillard et al. 2003. Fe-H redox exchange in silicate glasses

    https://ui.adsabs.harvard.edu/abs/2003GeCoA..67.2427G/abstract

    Power law fit for fH2 vs. H2 (ppm-wt) from Table 4 data. Experiments at pressures from 0.02-70
    bar, temperatures from 300-1000C.
    """

    @override
    def __init__(self, constant: float = 0.163, exponent: float = 1.252):
        super().__init__(constant, exponent)


class BasaltCO(Solubility):
    """Yoshioka et al. 2019. Carbon solubility in silicate melts

    https://ui.adsabs.harvard.edu/abs/2019GeCoA.259..129Y/abstract

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C. Log-scale linear expression for solubility of CO in
    MORB in the abstract.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        co_wtp: float = 10 ** (-5.20 + (0.8 * np.log10(fugacity)))
        ppmw: float = UnitConversion.weight_percent_to_ppmw(co_wtp)

        return ppmw


class RhyoliteCO(Solubility):
    """Yoshioka et al. 2019. Carbon solubility in silicate melts

    https://ui.adsabs.harvard.edu/abs/2019GeCoA.259..129Y/abstract

    Experiments on carbon solubility in silicate melts (Fe-free) coexisting with graphite and
    CO-CO2 fluid phase at 3 GPa and 1500 C. Henry's Law, their expression for solubility of CO in
    rhyolite in the abstract.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        co_wtp: float = 10 ** (-4.08 + (0.52 * np.log10(fugacity)))
        ppmw: float = UnitConversion.weight_percent_to_ppmw(co_wtp)

        return ppmw


class BasaltCOArmstrong(Solubility):
    """Armstrong et al. 2015. Solubility of volatiles in mafic melts under reduced conditions

    https://ui.adsabs.harvard.edu/abs/2015GeCoA.171..283A/abstract

    Experiments on Martian and terrestrial basalts at 1.2 GPa and 1400 C with variable fO2 from
    IW-3.65 to IW+1.46. Equation 10, log-scale linear fit for CO and includes dependence on total
    pressure. The fitting coefficients also use data from Stanley et al. 2014 (experiments from
    1-1.2 GPa).
    """

    @override
    def concentration(self, fugacity: float, *, pressure: float, **kwargs) -> float:
        del kwargs
        logco_ppm: float = -0.738 + (0.876 * np.log10(fugacity)) - (5.44e-5 * pressure)
        ppmw: float = 10**logco_ppm

        return ppmw


class BasaltCH4(Solubility):
    """Ardia et al. 2013, CH4 solubility in haplobasalt (Fe-free) silicate melt.

    https://ui.adsabs.harvard.edu/abs/2013GeCoA.114...52A/abstract

    Experiments conducted at 0.7-3 GPa and 1400-1450 C. Equations 7a and 8, values for lnK0 and
    deltaV from the text.
    """

    @override
    def concentration(self, fugacity: float, *, pressure: float, **kwargs) -> float:
        del kwargs
        pressure_gpa: float = UnitConversion.bar_to_GPa(pressure)
        one_bar_in_gpa: float = UnitConversion.bar_to_GPa(1)
        k: float = np.exp(4.93 - (1.93 * (pressure_gpa - one_bar_in_gpa)))
        ppmw: float = k * UnitConversion.bar_to_GPa(fugacity)

        return ppmw


class BasaltHe(Solubility):
    """Jambon et al. 1986, Solubility of He in tholeittic basalt melt

    https://ui.adsabs.harvard.edu/abs/1986GeCoA..50..401J/abstract

    Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
    1250-1600 C. Using Henry's Law solubility constant for He from the abstract, convert from STP
    units to mol/g*bar.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        henry_sol_constant: float = 56e-5  # cm3*STP/g*bar
        # Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
        he_conc: float = (henry_sol_constant / 2.24e4) * fugacity
        # Convert He conc from mol/g to g H2/g total and then to ppmw
        ppmw: float = he_conc * 4.0026 * 1e6

        return ppmw


class BasaltCl2(Solubility):
    """Thomas & Wood 2021. Solubility of chlorine in silicate melts

    https://ui.adsabs.harvard.edu/abs/2021GeCoA.294...28T/abstract

    Solubility law from Figure 4 showing relation between dissolved Cl concentration and Cl
    fugacity for Icelandic basalt at 1400 C and 1.5 GPa. Experiments from 0.5-2 GPa and 1200-1500 C
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        cl_wtp: float = 78.56 * np.sqrt(fugacity)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(cl_wtp)

        return ppmw


class AnorthiteDiopsideForsteriteCl2(Solubility):
    """Thomas & Wood 2021. Solubility of chlorine in silicate melts

    https://ui.adsabs.harvard.edu/abs/2021GeCoA.294...28T/abstract

    Solubility law from Figure 4 showing relation between dissolved Cl concentration and Cl
    fugacity for CMAS composition (An50Di28Fo22 (anorthite-diopside-forsterite), Fe-free low-degree
    mantle melt) at 1400 C and 1.5 GPa. Experiments from 0.5-2 GPa and 1200-1500 C
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        cl_wtp: float = 140.52 * np.sqrt(fugacity)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(cl_wtp)

        return ppmw


# Dictionaries of solubility laws for a given composition
andesite_solubilities: dict[str, Solubility] = {
    "H2": AndesiteH2(),
    "S2": AndesiteS2(),
}
anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltH2OMitchell(),
    "CO2": BasaltCO2(),
    "H2": BasaltH2(),
    "N2": BasaltN2Libourel(),
    "S2": BasaltS2(),
    "CO": BasaltCO(),
    "He": BasaltHe(),
    "Cl2": BasaltCl2(),
    "CH4": BasaltCH4(),
}
rhyolite_solubilities: dict[str, Solubility] = {
    "CO": RhyoliteCO(),
}
peridotite_solubilities: dict[str, Solubility] = {"H2O": PeridotiteH2O()}
reducedmagma_solubilities: dict[str, Solubility] = {"H2S": MercuryMagmaS()}

# Dictionary of all the composition solubilities. All of the dictionaries with solubility laws for
# a given composition (above) should be included in this dictionary.
composition_solubilities: dict[str, dict[str, Solubility]] = {
    "basalt": basalt_solubilities,
    "andesite": andesite_solubilities,
    "peridotite": peridotite_solubilities,
    "anorthiteDiopsideEuctectic": anorthdiop_solubilities,
    "reducedmagma": reducedmagma_solubilities,
    "rhyolite": rhyolite_solubilities,
}
