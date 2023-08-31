"""Solubility laws.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of the 
    GNU General Public License as published by the Free Software Foundation, either version 3 of 
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
    the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with this program. If 
    not, see <https://www.gnu.org/licenses/>.   
"""

from __future__ import annotations

import logging

import numpy as np

from atmodeller.interfaces import GasSpecies, Solubility
from atmodeller.reaction_network import ReactionNetwork
from atmodeller.utilities import UnitConversion

logger: logging.Logger = logging.getLogger(__name__)


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
        # TODO: Maggie to add comment what the next line is (or remove if no longer required).
        # ppmw: float = self.power_law(fugacity, 34.43369241, 0.49459427)
        ppmw: float = 10 ** (1.20257736 * np.log10(np.sqrt(fugacity)) + 1.01058631)
        return ppmw


class AndesiteS_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (31984.243 / temperature)
        logS_wtp: float = (
            logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        )
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteS_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-).

    Using expression in abstract for S wt% and the expression for sulfide capacity. Composition
    for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """fugacity is fS2."""
        logCs: float = 0.225 - (8876.5 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteS(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteS_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteS_Sulfate()

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
        # TODO: Maggie to add comment what the next line is (or remove if no longer required).
        # ppmw: float = self.power_law(fugacity, 53.65376426, 0.38365457)
        ppmw: float = 10 ** (1.04827856 * np.log10(np.sqrt(fugacity)) + 1.10083602)
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
        # TODO: Could add fO2 lower and upper bounds.
        if "O2" in fugacities_dict:
            constant: float = (fugacities_dict["O2"] ** -0.75) * 5.97e-10
            ppmw += self.power_law(fugacity, constant, 0.5)
        return ppmw


class BasaltS_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (31532.862 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class BasaltS_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = 0.225 - (7817.134 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class BasaltS(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltS_Sulfide()
        self.sulfate_solubility: Solubility = BasaltS_Sulfate()

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


class TBasaltS_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Trachy-Basalt from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (32446.366 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class TBasaltS_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for Trachy-basalt from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = 0.225 - (7842.5 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
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

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        a, b, c, d = [7.25, -2.54e4, 0.04, -0.551]  # Coeffs from eq. 10 (Namur et al., 2016).
        # FIXME: How to deal if fO2 not available?  Drop last term?
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
    "O2S": AndesiteS(),
    "OS": AndesiteS(),
    "S2": AndesiteS(),
}
anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltDixonH2O(),
    "CO2": BasaltDixonCO2(),
    "H2": BasaltH2(),
    "N2": BasaltLibourelN2(),
    "O2S": BasaltS(),
    "OS": BasaltS(),
    "S2": BasaltS(),
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
