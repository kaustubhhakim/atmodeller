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

from atmodeller.constraints import (
    ConstantSystemConstraint,
    SystemConstraint,
    SystemConstraints,
)
from atmodeller.core import Species
from atmodeller.interfaces import GasSpecies, Solubility
from atmodeller.interior_atmosphere import InteriorAtmosphereSystem
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
        ppmw: float = 10 ** (0.60128868 * np.log10(fugacity) + 1.01058631)
        return ppmw


class AndesiteS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Andesite from Table 1.
    """

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


class AndesiteSO_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2. First, we need to convert the input, fSO, to fS2"""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species)

        SO_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO_fugacity])

        interior_atmosphere.solve(constraints)

        output = interior_atmosphere.output

        fS2 = output["S2"].fugacity

        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )

        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = (fugacity**2) / (K_rxn * fugacities_dict["O2"])

        logCs: float = -12.948 + (31586.2393 / temperature)
        logS_wtp: float = logCs + (0.5 * np.log10(fS2)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteSO2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2. First, we need to convert the input, fSO, to fS2"""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO2"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species)

        SO2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO2", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO2_fugacity])

        interior_atmosphere.solve(constraints)

        output = interior_atmosphere.output

        fS2 = output["S2"].fugacity
        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )

        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = ((fugacity) / (K_rxn * fugacities_dict["O2"])) ** 2

        logCs: float = -12.948 + (31586.2393 / temperature)
        logS_wtp: float = logCs + (0.5 * np.log10(fS2)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteS2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-).

    Using expression in abstract for S wt% and the expression for sulfide capacity. Composition
    for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """fugacity is fS2."""
        logCs: float = 0.225 - (8921.0927 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteSO_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-).

    Using expression in abstract for S wt% and the expression for sulfide capacity. Composition
    for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """fugacity is fS2."""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species)

        SO_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO_fugacity])

        interior_atmosphere.solve(constraints)

        output = interior_atmosphere.output

        fS2 = output["S2"].fugacity
        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )
        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = (fugacity**2) / (K_rxn * fugacities_dict["O2"])

        logCs: float = 0.225 - (8921.0927 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fS2)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteSO2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-).

    Using expression in abstract for S wt% and the expression for sulfide capacity. Composition
    for Andesite from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """fugacity is fS2."""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO2"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere: InteriorAtmosphereSystem = InteriorAtmosphereSystem(species=species)

        SO2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO2", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO2_fugacity])

        interior_atmosphere.solve(constraints)

        output = interior_atmosphere.output

        fS2 = output["S2"].fugacity

        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )
        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = ((fugacity) / (K_rxn * fugacities_dict["O2"])) ** 2

        logCs: float = 0.225 - (8921.0927 / temperature)
        logS_wtp: float = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fS2)))
        S_wtp: float = 10**logS_wtp
        ppmw: float = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class AndesiteS2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteS2_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteS2_Sulfate()

    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)
        return solubility


class AndesiteSO(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteSO_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteSO_Sulfate()

    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)
        return solubility


class AndesiteSO2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = AndesiteSO2_Sulfide()
        self.sulfate_solubility: Solubility = AndesiteSO2_Sulfate()

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
        # TODO: Could add fO2 lower and upper bounds.
        if "O2" in fugacities_dict:
            constant: float = (fugacities_dict["O2"] ** -0.75) * 5.97e-10
            ppmw += self.power_law(fugacity, constant, 0.5)
        return ppmw


class BasaltS2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logger.debug("S2 Fugacity for S2 Sulfate Solubility Law = \n%s", fugacity)

        logCs: float = -12.948 + (32333.5635 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        if ppmw >= 1000:
            logger.debug("WARNING: S2 Sulfate Solubility is getting too high: \n%s", ppmw)
            ppmw = 1000.0
        return ppmw


class BasaltSO_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere_SOSulfate: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
            species=species
        )

        SO_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO_fugacity])

        interior_atmosphere_SOSulfate.solve(constraints)

        output = interior_atmosphere_SOSulfate.output

        fS2 = output["S2"].fugacity
        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )
        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = (fugacity**2) / (K_rxn * fugacities_dict["O2"])
        logger.debug("Calculated S2 Fugacity for SO Sulfate Solubility Law = \n%s", fS2)

        logCs: float = -12.948 + (32333.5635 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fS2)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        if ppmw >= 1000:
            logger.debug("WARNING: SO Sulfate Solubility is getting too high: \n%s", ppmw)
            ppmw = 1000.0
        return ppmw


class BasaltSO2_Sulfate(Solubility):
    """Boulliung & Wood 2022. Solubility of sulfur as sulfate, SO4^2-/S^6+

    Using expression in the abstract and the corrected expression for sulfate capacity in
    corrigendum. Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO2"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere_SO2Sulfate: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
            species=species
        )

        SO2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO2", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO2_fugacity])

        interior_atmosphere_SO2Sulfate.solve(constraints)

        output = interior_atmosphere_SO2Sulfate.output

        fS2 = output["S2"].fugacity
        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )
        # logger.debug("SO2 Sulfate Reaction Network is \n%s", network.reactions)
        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = ((fugacity) / (K_rxn * fugacities_dict["O2"])) ** 2
        logger.debug("Calculated S2 Fugacity for SO2 Sulfate Solubility Law = \n%s", fS2)

        logCs: float = -12.948 + (32333.5635 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fS2)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        if ppmw >= 1000:
            logger.debug("WARNING: SO2 Sulfate Solubility is getting too high: \n%s", ppmw)
            ppmw = 1000.0
        return ppmw


class BasaltS2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logger.debug("S2 Fugacity for S2 Sulfide Solubility Law = \n%s", fugacity)
        logCs: float = 0.225 - (8045.7465 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fugacity)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        if ppmw >= 1000:
            logger.debug("WARNING: S2 Sulfide Solubility is getting too high: \n%s", ppmw)
            ppmw = 1000.0
        return ppmw


class BasaltSO_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere_SOSulfide: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
            species=species
        )

        SO_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO_fugacity])

        interior_atmosphere_SOSulfide.solve(constraints)

        output = interior_atmosphere_SOSulfide.output

        fS2 = output["S2"].fugacity
        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )
        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = (fugacity**2) / (K_rxn * fugacities_dict["O2"])
        logger.debug("Calculated S2 Fugacity for SO Sulfide Solubility Law = \n%s", fS2)
        logCs: float = 0.225 - (8045.7465 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fS2)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        if ppmw >= 1000:
            logger.debug("WARNING: SO Sulfide Solubility is getting too high: \n%s", ppmw)
            ppmw = 1000.0
        return ppmw


class BasaltSO2_Sulfide(Solubility):
    """Boulliung & Wood 2023 (preprint). Solubility of sulfur as sulfide (S^2-)

    Using expression in abstract for S wt% and the expression for sulfide capacity
    Composition for NIB (natural Icelandic basalt) from Table 1.
    """

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        species: Species = Species()
        species.append(GasSpecies(chemical_formula="SO2"))
        species.append(GasSpecies(chemical_formula="S2"))
        species.append(GasSpecies(chemical_formula="O2"))

        interior_atmosphere_SO2Sulfide: InteriorAtmosphereSystem = InteriorAtmosphereSystem(
            species=species
        )

        SO2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="SO2", value=fugacity
        )
        O2_fugacity: SystemConstraint = ConstantSystemConstraint(
            name="fugacity", species="O2", value=fugacities_dict["O2"]
        )
        constraints: SystemConstraints = SystemConstraints([O2_fugacity, SO2_fugacity])

        interior_atmosphere_SO2Sulfide.solve(constraints)

        output = interior_atmosphere_SO2Sulfide.output

        fS2 = output["S2"].fugacity
        # network = ReactionNetwork(
        #    species=species, gibbs_data=StandardGibbsFreeEnergyOfFormationJANAF()
        # )
        # logger.debug("SO2 Sulfide Reaction Network is \n%s", network.reactions)

        # K_rxn = network.get_reaction_equilibrium_constant(
        #    reaction_index=0, temperature=temperature, pressure=1
        # )

        # fS2 = ((fugacity) / (K_rxn * fugacities_dict["O2"])) ** 2
        logger.debug("Calculated S2 Fugacity for SO2 Sulfide Solubility Law = \n%s", fS2)

        logCs: float = 0.225 - (8045.7465 / temperature)
        logS_wtp = logCs - (0.5 * (np.log10(fugacities_dict["O2"]) - np.log10(fS2)))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        if ppmw >= 1000:
            logger.debug("WARNING: SO2 Sulfide Solubility is getting too high: \n%s", ppmw)
            ppmw = 1000.0
        return ppmw


class BasaltS2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltS2_Sulfide()
        self.sulfate_solubility: Solubility = BasaltS2_Sulfate()

    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)
        return solubility


class BasaltSO(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltSO_Sulfide()
        self.sulfate_solubility: Solubility = BasaltSO_Sulfate()

    def _solubility(self, *args, **kwargs) -> float:
        solubility: float = self.sulfide_solubility._solubility(*args, **kwargs)
        solubility += self.sulfate_solubility._solubility(*args, **kwargs)
        return solubility


class BasaltSO2(Solubility):
    """Total S solubility accounting for both sulfide and sulfate dissolution."""

    def __init__(self):
        self.sulfide_solubility: Solubility = BasaltSO2_Sulfide()
        self.sulfate_solubility: Solubility = BasaltSO2_Sulfate()

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

    def _solubility(
        self, fugacity: float, temperature: float, fugacities_dict: dict[str, float]
    ) -> float:
        """Fugacity is fS2."""
        logCs: float = -12.948 + (32446.366 / temperature)
        logS_wtp = logCs + (0.5 * np.log10(fugacity)) + (1.5 * np.log10(fugacities_dict["O2"]))
        S_wtp = 10**logS_wtp
        ppmw = UnitConversion.weight_percent_to_ppmw(S_wtp)
        return ppmw


class TBasaltS2_Sulfide(Solubility):
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
    "O2S": AndesiteSO2(),
    "OS": AndesiteSO(),
    "S2": AndesiteS2(),
}
anorthdiop_solubilities: dict[str, Solubility] = {"H2O": AnorthiteDiopsideH2O()}
basalt_solubilities: dict[str, Solubility] = {
    "H2O": BasaltDixonH2O(),
    "CO2": BasaltDixonCO2(),
    "H2": BasaltH2(),
    "N2": BasaltLibourelN2(),
    "O2S": BasaltSO2(),
    "OS": BasaltSO(),
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
