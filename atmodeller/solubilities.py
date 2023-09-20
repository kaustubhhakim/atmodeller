"""Solubility laws.

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Callable

import numpy as np

from atmodeller.interfaces import Solubility
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)

# Solubility limiters.
MAXIMUM_PPMW: float = UnitConversion.weight_percent_to_ppmw(10)  # 10% by weight.
# Maximum sulfur solubility.
SULFUR_MAXIMUM_PPMW: float = 1000


def limit_solubility(bound: float = MAXIMUM_PPMW) -> Callable:
    """A decorator to limit the solubility in ppmw.

    Args:
        bound: The maximum limit of the solubility in ppmw. Defaults to MAXIMUM_PPMW.

    Returns:
        The decorator.
    """

    def decorator(func) -> Callable:
        @wraps(func)
        def wrapper(self: Solubility, *args, **kwargs):
            result: float = func(self, *args, **kwargs)
            if result > bound:
                msg: str = "%s solubility (%d ppmw) will be limited to %d ppmw" % (
                    self.__class__.__name__,
                    result,
                    bound,
                )
                logger.warning(msg)

            return min(result, bound)  # Limit the result to 'bound'

        return wrapper

    return decorator


# region Andesite solubility


class AndesiteH2(Solubility):
    """Hirschmann et al. 2012.

    Fit to fH2 vs. H2 concentration from Table 2.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        ppmw: float = 10 ** (0.60128868 * np.log10(fugacity) + 1.01058631)
        return ppmw


class AndesiteS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Andesite from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (31586.2393 / temperature)
        logS_wtp: float = (
            logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        )
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class AndesiteS2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-).

    Using expression in abstract for S wt% and the expression for sulfide capacity. Composition
    for Andesite from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """fugacity is fS2."""
        logCs: float = 0.225 - (8921.0927 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class AndesiteS2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteS2_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteS2_Sulfate()

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)

        return solubility


# endregion


class AnorthiteDiopsideH2O(Solubility):
    """Newcombe et al. (2017).

    https://ui.adsabs.harvard.edu/abs/2017GeCoA.200..330N/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict

        return self.power_law(fugacity, 727, 0.5)


# region Basalt solubility


class BasaltDixonCO2(Solubility):
    """Dixon et al. (1995)."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del fugacities_dict
        ppmw: float = (3.8e-7) * fugacity * np.exp(-23 * (fugacity - 1) / (83.15 * temperature))
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)

        return ppmw


class BasaltDixonH2O(Solubility):
    """Dixon et al. (1995) refit by Paolo Sossi."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict

        return self.power_law(fugacity, 965, 0.5)


class BasaltH2(Solubility):
    """Hirschmann et al. 2012 for Basalt.

    Fit to fH2 vs. H2 concentration from Table 2.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        ppmw: float = 10 ** (0.52413928 * np.log10(fugacity) + 1.10083602)

        return ppmw


class BasaltLibourelN2(Solubility):
    """Libourel et al. (2003), basalt (tholeiitic) magmas.

    Eq. 23, includes dependence on pressure and fO2.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        ppmw: float = self.power_law(fugacity, 0.0611, 1.0)
        constant: float = (fugacities_dict["O2"] ** -0.75) * 5.97e-10
        ppmw += self.power_law(fugacity, constant, 0.5)

        return ppmw


class BasaltS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (32333.5635 / temperature)
        logSO4_wtp: float = (
            logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        )
        SO4_wtp: float = 10**logSO4_wtp
        S_wtp: float = SO4_wtp * (32.065 / 96.06)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class BasaltS2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = 0.225 - (8045.7465 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class BasaltS2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltS2_Sulfide()
        self.sulfate_solubility: Solubility = BasaltS2_Sulfate()

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)

        return solubility


class BasaltWilsonH2O(Solubility):
    """Hamilton (1964) and Wilson and Head (1981)."""

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 215, 0.7)


class TBasaltS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Trachy-Basalt from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (32446.366 / temperature)
        logS_wtp: float = (
            logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        )
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


class TBasaltS2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for Trachy-basalt from Table 1.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = 0.225 - (7842.5 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)

        return ppmw


# endregion


class LunarGlassH2O(Solubility):
    """Newcombe et al. (2017).

    https://ui.adsabs.harvard.edu/abs/2017GeCoA.200..330N/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 683, 0.5)


class MercuryMagmaS(Solubility):
    """Namur et al. 2016.

    S concentration at sulfide (S^2-) saturation conditions, relevant for Mercury-like magmas.
    """

    @limit_solubility(SULFUR_MAXIMUM_PPMW)
    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        a, b, c, d = [7.25, -2.54e4, 0.04, -0.551]  # Coeffs from eq. 10 (Namur et al., 2016).
        wt_perc: float = np.exp(
            a
            + (b / temperature)
            + ((c * fugacity) / temperature)
            + (d * np.log10(fugacities_dict["O2"]))
        )
        ppmw: float = UnitConversion.weight_percent_to_ppmw(wt_perc)

        return ppmw


class PeridotiteH2O(Solubility):
    """Sossi et al. (2023).

    Power law parameters are in the abstract:
    https://ui.adsabs.harvard.edu/abs/2023E%26PSL.60117894S/abstract
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        return self.power_law(fugacity, 524, 0.5)


class SilicicMeltsH2(Solubility):
    """Gaillard et al. 2003.

    Valid for pressures from 0.02-70 bar; power law fit to Table 4 data.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        del temperature
        del fugacities_dict
        ppmw: float = self.power_law(fugacity, 0.163, 1.252)
        return ppmw


# Dictionaries of self-consistent solubility laws for a given composition.
andesite_solubilities: dict[str, Solubility] = {
    "H2": AndesiteH2(),
    "S2": AndesiteS2(),
}

anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltDixonH2O(),
    "CO2": BasaltDixonCO2(),
    "H2": BasaltH2(),
    "N2": BasaltLibourelN2(),
    "S2": BasaltS2(),
}
peridotite_solubilities: dict[str, Solubility] = {"H2O": PeridotiteH2O()}
reducedmagma_solubilities: dict[str, Solubility] = {"H2S": MercuryMagmaS()}

# Dictionary of all the composition solubilities. Lowercase key name by convention. All of the
# dictionaries with self-consistent solubility laws for a given composition (above) should be
# included in this dictionary.
composition_solubilities: dict[str, dict[str, Solubility]] = {
    "basalt": basalt_solubilities,
    "andesite": andesite_solubilities,
    "peridotite": peridotite_solubilities,
    "anorthiteDiopsideEuctectic": anorthdiop_solubilities,
    "reducedmagma": reducedmagma_solubilities,
}
