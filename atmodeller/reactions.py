# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reaction network API.

Defines reaction network components used to assemble the full thermodynamic equilibrium system.

This module provides:

- Construction of element-species formula matrices,
- Core chemical reaction networks derived from elemental constraints,
- Dissolution reaction networks coupling gas and condensed species,
- Aggregation of reaction blocks into a unified :class:`ReactionSystem`,
- Evaluation of equilibrium constants and nonlinear residuals.

The reaction system combines stoichiometric matrices, thermodynamic data, and phase-specific
constraints into a form suitable for JAX-compiled nonlinear solvers. Both core reactions and
dissolution reactions are represented in a unified matrix form with associated stability handling.

Primary entry point:
    :class:`ReactionSystem`
"""

import logging
import pprint
from abc import abstractmethod
from collections.abc import Callable, Iterable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from jaxmod.utils import partial_rref, safe_exp, to_hashable
from jaxtyping import Array, ArrayLike, Float, Integer
from xmmutablemap import ImmutableMap

from atmodeller.constants import GAS_STATE, STANDARD_CONCENTRATION
from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase
from atmodeller.thermodata import thermodynamic_data_source
from atmodeller.type_aliases import NpBool, NpFloat, NpInt
from atmodeller.utilities import get_reaction_dictionary

logger: logging.Logger = logging.getLogger(__name__)


def get_formula_matrix(species: SpeciesCollection[SpeciesProtocol]) -> NpInt:
    """Gets the formula matrix.

    Elements are given in rows and species in columns following the convention in :cite:t:`LKS17`.

    Args:
        species: Species collection

    Returns:
        Formula matrix
    """
    formula_matrix: NpInt = np.zeros(
        (len(species.unique_elements), species.number_species), dtype=int
    )

    for element_index, element in enumerate(species.unique_elements):
        for species_index, species_ in enumerate(species):
            count: int = 0
            try:
                count = species_.data.composition[element][0]
            except KeyError:
                count = 0
            formula_matrix[element_index, species_index] = count

    # logger.debug("formula_matrix = %s", formula_matrix)

    return formula_matrix


class BaseReactionBlock(eqx.Module):
    """Base reaction block"""

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""

    @property
    def number_reactions(self) -> int:  # pyright: ignore
        """Number of reactions in the reaction block"""

    @property
    def active_reactions(self) -> NpBool:
        """Boolean mask of active reactions"""
        return np.ones(self.number_reactions, dtype=bool)

    @abstractmethod
    def get_log_Kp(self, temperature: Float[Array, "..."], *args, **kwargs) -> Float[Array, "..."]:
        """Gets log of the equilibrium constant of each reaction in the reaction block"""

    @abstractmethod
    def get_matrix(self) -> NpFloat:
        """Gets the full reaction matrix of the reaction block"""

    @abstractmethod
    def get_stability_matrix(self) -> NpFloat:
        """Gets the full stability matrix of the reaction block"""

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.get_matrix(), self.species.species_names)

    def output_to_logger(self):
        """Outputs the reaction block to the logger"""
        logger.debug(f"{self.__class__.__name__} matrix = %s", self.get_matrix())
        logger.info(
            f"{self.__class__.__name__} network = %s",
            pprint.pformat(self.get_reaction_dictionary()),
        )


class ReactionNetwork(BaseReactionBlock):
    """Handles core chemical reactions.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    reaction_matrix: NpFloat
    """Reaction matrix"""
    reaction_matrix_full: NpFloat
    """Reaction matrix expanded to full species space"""
    reaction_stability_mask_full: NpBool
    """Stability mask for reaction matrix expanded to full species space"""
    reaction_stability_matrix_full: NpFloat
    """Reaction stability matrix expanded to full species space"""
    vmap_gibbs: Callable
    """Vectorised Gibbs free energy function for reaction species"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)

        # Reaction matrix of linearly independent reactions
        transpose_formula_matrix: NpInt = get_formula_matrix(self.species.reaction_species).T
        self.reaction_matrix: NpFloat = partial_rref(transpose_formula_matrix)

        # Reaction matrix expanded to full species space
        self.reaction_matrix_full: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )
        # Insert reduced matrix into correct columns
        self.reaction_matrix_full[:, self.species.reaction_species_mask] = self.reaction_matrix

        self.reaction_stability_mask_full = np.broadcast_to(
            self.species.active_stability, self.reaction_matrix_full.shape
        )
        self.reaction_stability_matrix_full = (
            self.reaction_matrix_full * self.reaction_stability_mask_full
        )

        gibbs_funcs: list[Callable] = [
            to_hashable(species_.get_gibbs_over_RT) for species_ in self.species.reaction_species
        ]

        def apply_gibbs(
            index: Integer[Array, ""], temperature: Float[Array, "..."]
        ) -> Float[Array, "..."]:
            return lax.switch(index, gibbs_funcs, temperature)

        self.vmap_gibbs = eqx.filter_vmap(apply_gibbs, in_axes=(0, None))

        self.output_to_logger()

        temperature_min, temperature_max = self.get_temperature_range()
        logger.info(
            "Thermodynamic data requires temperatures between %d K and %d K",
            np.ceil(temperature_min),
            np.floor(temperature_max),
        )

    @classmethod
    def available_species(cls) -> tuple[str, ...]:
        return thermodynamic_data_source.available_species()

    @property
    def number_reactions(self) -> int:
        """Number of core reactions"""
        return max(
            0,
            self.species.reaction_species.number_species
            - len(self.species.reaction_species.unique_elements),
        )

    def get_log_Kp(self, temperature: Float[Array, "..."]) -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Args:
            temperature: Temperature in K

        Returns:
            Log of the equilibrium constant of each reaction
        """
        gibbs_values: Float[Array, "reaction_species 1"] = self.vmap_gibbs(
            jnp.arange(self.species.reaction_species.number_species), temperature
        )
        # jax.debug.print("gibbs_values = {out}", out=gibbs_values)
        reaction_matrix: Float[Array, "reactions reaction_species"] = jnp.asarray(
            self.reaction_matrix
        )
        log_Kp: Float[Array, "reactions 1"] = -1.0 * reaction_matrix @ gibbs_values

        return jnp.ravel(log_Kp)

    def get_matrix(self) -> NpFloat:
        return self.reaction_matrix_full

    def get_stability_matrix(self) -> NpFloat:
        return self.reaction_stability_matrix_full

    def get_temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range of the thermodynamic data for the species

        Returns:
            Minimum and maximum temperature that is valid for the species
        """
        temperature_min: list[float] = [
            min(species.thermo.T_min) for species in self.species.reaction_species
        ]
        temperature_max: list[float] = [
            max(species.thermo.T_max) for species in self.species.reaction_species
        ]

        return max(temperature_min), min(temperature_max)


class DissolutionNetwork(BaseReactionBlock):
    """Handles all reactions where a species dissolves into or exchanges with a phase.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    reaction_indices_map: NpInt
    """Mapping of dissolution species to corresponding reaction species"""
    dissolution_matrix: NpFloat
    """Dissolution reaction matrix"""
    vmap_solubility: Callable
    """Vectorised solubility function for dissolution reactions"""
    dilute_limit: bool = True
    """Whether to assume dilute limit for all dissolution reactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)

        # Most direct to construct the dissolution matrix in full species space
        dissolution_matrix: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )

        # Gas indices in the full species collection corresponding to each dissolution species
        reaction_indices_map: list[int] = []

        for reaction_index, dissolution_species_ in enumerate(self.species.reservoir_species):
            # Reservoir mask
            reservoir_mask: NpBool = np.array(
                [s is dissolution_species_ for s in self.species],
                dtype=bool,
            )
            # Gas species index
            name: str = f"{dissolution_species_.data.hill_formula}_{GAS_STATE}"
            gas_idx: int = self.species.species_names.index(name)

            reaction_indices_map.append(gas_idx)

            dissolution_matrix[reaction_index, reservoir_mask] = 1.0
            dissolution_matrix[reaction_index, gas_idx] = -1.0

        self.reaction_indices_map = np.array(reaction_indices_map, dtype=int)
        self.dissolution_matrix = dissolution_matrix

        solubility_funcs: list[Callable] = [
            to_hashable(species_.solubility.jax_concentration)
            for species_ in self.species.reservoir_species
        ]

        def apply_solubility(
            index: Integer[Array, ""],
            fugacity_val: Float[Array, ""],
            temp: Float[Array, ""],
            press: Float[Array, ""],
            o2_fug: Float[Array, ""],
        ) -> Float[Array, ""]:
            return lax.switch(index, solubility_funcs, fugacity_val, temp, press, o2_fug)

        self.vmap_solubility: Callable = eqx.filter_vmap(
            apply_solubility, in_axes=(0, 0, None, None, None)
        )

        self.output_to_logger()

    @property
    def number_reactions(self) -> int:
        """Number of dissolution reactions"""
        return self.species.reservoir_species.number_species

    def get_log_Kp(
        self,
        temperature: Float[Array, "..."],
        gas_species_activity: Float[Array, "..."],
        pressure: Float[Array, ""],
        fO2: Float[Array, ""],
    ) -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Args:
            temperature: Temperature in K
            gas_species_activity: Gas species activity regulating dissolution reactions
            pressure: Pressure in bar
            fO2: Oxygen fugacity in bar

        Returns:
            Log of the equilibrium constant of each reaction
        """
        # Return empty array if no dissolution species
        if self.number_reactions == 0:
            return jnp.array([], dtype=float)

        species_ppmw: Float[Array, " num_dissolution_species"] = self.vmap_solubility(
            jnp.arange(self.number_reactions), gas_species_activity, temperature, pressure, fO2
        )
        # jax.debug.print("species_ppmw = {out}", out=species_ppmw)

        log_Kp: Float[Array, " num_reactions"] = (
            jnp.log(species_ppmw) - jnp.log(STANDARD_CONCENTRATION) - jnp.log(gas_species_activity)
        )

        return log_Kp

    def get_matrix(self) -> NpFloat:
        return self.dissolution_matrix

    def get_stability_matrix(self) -> NpFloat:
        """Dissolution reactions do not directly affect stability, so return zero"""
        return np.zeros_like(self.dissolution_matrix, dtype=float)


class PhaseIndex:
    """Stores start and stop indices of a phase in the full species collection.

    Args:
        start: Starting index of the phase in the full species collection
        stop: Stopping index of the phase in the full species collection
    """

    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop

    @property
    def slice(self) -> slice:
        """Slice object for indexing arrays."""
        return slice(self.start, self.stop)

    def mask(self, n_total: int) -> np.ndarray:
        """Boolean mask for this phase."""
        mask = np.zeros(n_total, dtype=bool)
        mask[self.start : self.stop] = True
        return mask

    def __len__(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"PhaseIndex(start={self.start}, stop={self.stop})"


class ReactionSystem(BaseReactionBlock):
    """Unified reaction system for core chemical reactions and dissolution reactions.

    Handles phase indexing, assembly of reaction matrices, stability, and evaluation of equilibrium
    constants and residuals in a JAX-compatible way.

    Phases supported:
        - gas
        - melt
        - solid
        - condensates

    Args:
        gas: GasPhase
        melt: MeltPhase
        solid: SolidPhase
        condensates: Iterable of PurePhase
    """

    species: SpeciesCollection[SpeciesProtocol]
    """All species"""
    gas: GasPhase
    """Gas"""
    melt: MeltPhase
    """Melt"""
    solid: SolidPhase
    """Solid"""
    condensates: tuple[PurePhase, ...]
    """Pure condensates"""
    formula_matrix: NpInt
    """Formula matrix of the full species collection"""
    reaction: ReactionNetwork
    """Core chemical reaction network"""
    dissolution: DissolutionNetwork
    """Dissolution reaction network"""
    matrix: NpFloat
    """Reaction matrix of the full reaction system"""
    stability_matrix: NpFloat
    """Stability matrix of the full reaction system"""
    _O2_index: NpInt
    _has_O2: NpBool
    _log_stoich_matrix: NpFloat
    phase_indices: ImmutableMap[str, PhaseIndex]

    def __init__(
        self,
        gas: GasPhase,
        *,
        melt: MeltPhase,
        solid: SolidPhase,
        condensates: Iterable[PurePhase],
    ):
        self.gas = gas
        self.melt = melt
        self.solid = solid
        self.condensates = tuple(condensates)

        # Flatten all species
        condensate_species: tuple[ChemicalSpecies, ...] = tuple(
            species_ for condensate in condensates for species_ in condensate.species
        )
        all_species: tuple[SpeciesProtocol, ...] = (
            gas.species.species + melt.species.species + solid.species.species + condensate_species
        )
        self.species = SpeciesCollection(all_species)

        # Phase indexing
        start = 0
        indices: dict[str, PhaseIndex] = {}
        phase_order: tuple[str, ...] = ("gas", "melt", "solid", "condensates")
        for phase_name, phase_collection in zip(
            phase_order, [gas, melt, solid, condensate_species]
        ):
            n: int = len(phase_collection)
            indices[phase_name] = PhaseIndex(start, start + n)
            start += n
        self.phase_indices = ImmutableMap(indices)

        self.formula_matrix = get_formula_matrix(self.species)
        self._log_stoich_matrix = np.where(
            self.formula_matrix > 0, np.log(self.formula_matrix), -np.inf
        )
        self.reaction = ReactionNetwork(self.species)
        self.dissolution = DissolutionNetwork(self.species)
        self.matrix = np.vstack([block.get_matrix() for block in self.blocks])
        self.stability_matrix = np.vstack([block.get_stability_matrix() for block in self.blocks])

        # Could be an integer (but represented as a float) or np.nan
        self._O2_index = np.nan_to_num(gas.O2_index, nan=0).astype(int)
        self._has_O2 = ~np.isnan(gas.O2_index)

        self.output_to_logger()

    @property
    def gas_slice(self) -> slice:
        return self.phase_slice("gas")

    @property
    def gas_species_mask(self) -> NpBool:
        return self.phase_mask("gas")

    @property
    def melt_slice(self) -> slice:
        return self.phase_slice("melt")

    @property
    def melt_species_mask(self) -> NpBool:
        return self.phase_mask("melt")

    @property
    def solid_slice(self) -> slice:
        return self.phase_slice("solid")

    @property
    def solid_species_mask(self) -> NpBool:
        return self.phase_mask("solid")

    @property
    def condensates_slice(self) -> slice:
        return self.phase_slice("condensates")

    @property
    def condensates_species_mask(self) -> NpBool:
        return self.phase_mask("condensates")

    @property
    def blocks(self) -> tuple[BaseReactionBlock, ...]:
        """Reaction blocks"""
        return self.reaction, self.dissolution

    @property
    def number_reactions(self):
        """Number of reactions"""
        return sum(block.number_reactions for block in self.blocks)

    def get_log_Kp(
        self,
        log_activity: Float[Array, " num_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, ""],
    ) -> Float[Array, " num_reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Assembles the log Kps from the reaction and dissolution networks, which may require
        different inputs.

        Args:
            log_activity: Log activity of each species
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log of the equilibrium constant of each reaction
        """
        log_Kp_reaction: Float[Array, " num_core_reactions"] = self.reaction.get_log_Kp(
            temperature
        )

        # Log activity of chemical species regulating dissolution
        log_activity_dissolution: Float[Array, " num_dissolution_species"] = jnp.take(
            log_activity,
            indices=self.dissolution.reaction_indices_map,
            unique_indices=True,
            indices_are_sorted=True,
        )
        # jax.debug.print("log_activity_dissolution = {out}", out=log_activity_dissolution)

        activity_dissolution: Float[Array, " num_dissolution_species"] = jnp.exp(
            log_activity_dissolution
        )

        # Get fO2 or nan if not present
        fO2: Float[Array, ""] = jnp.where(
            self._has_O2, jnp.take(jnp.exp(log_activity), self._O2_index), jnp.nan
        )

        log_Kp_dissolution: Float[Array, " num_dissolution_reactions"] = (
            self.dissolution.get_log_Kp(temperature, activity_dissolution, pressure, fO2)
        )

        return jnp.concatenate([log_Kp_reaction, log_Kp_dissolution])

    def get_matrix(self) -> NpFloat:
        return self.matrix

    def get_stability_matrix(self) -> ArrayLike:
        return self.stability_matrix

    def get_log_element_moles(
        self, log_number_moles: Float[Array, " num_species"]
    ) -> Float[Array, " num_elements"]:
        """Gets log number of moles of each element.

        Args:
            log_number_moles: Log number of moles of each species

        Returns:
            Log number of moles of each element
        """
        log_terms: Float[Array, "element species"] = log_number_moles + self._log_stoich_matrix

        return logsumexp(log_terms, axis=1)

    def apply_stability(
        self, residual: Float[Array, " num_reactions"], log_stability: Float[Array, " num_species"]
    ) -> Float[Array, " num_reactions"]:
        """Subtract stability contribution from residual.

        Args:
            residual: Residual of the reaction network before applying stability
            log_stability: Log stability of each species

        Returns:
            Residual of the reaction network after applying stability
        """
        return residual - jnp.dot(self.get_stability_matrix(), safe_exp(log_stability))

    def get_residual(
        self,
        log_activity: Float[Array, " num_species"],
        log_stability: Float[Array, " num_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, ""],
    ) -> Float[Array, " num_reactions"]:
        """Gets the residual of the reaction network.

        Args:
            log_activity: Log activity of each species
            log_stability: Log stability of each species
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Residual of the reaction network
        """
        log_Kp: Float[Array, " num_reactions"] = self.get_log_Kp(
            log_activity, temperature, pressure
        )
        residual: Float[Array, " num_reactions"] = jnp.dot(self.matrix, log_activity) - log_Kp
        # jax.debug.print("reaction residual before stability = {out}", out=residual)

        return self.apply_stability(residual, log_stability)

    def phase_slice(self, phase_name: str) -> slice:
        """Slice object for a given phase.

        Args:
            phase_name: Name of the phase

        Returns:
                Slice object for the phase
        """
        return self.phase_indices[phase_name].slice

    def phase_mask(self, phase_name: str) -> NpBool:
        """Boolean mask for a given phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Boolean mask for the phase
        """
        return self.phase_indices[phase_name].mask(len(self.species))
