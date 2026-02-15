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
"""Reactions"""

import logging
import pprint
from abc import abstractmethod
from collections.abc import Callable, Iterable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmod.utils import partial_rref, safe_exp, to_hashable
from jaxtyping import Array, ArrayLike, Float, Integer

from atmodeller.constants import GAS_STATE, STANDARD_CONCENTRATION
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.thermodata import thermodynamic_data_source
from atmodeller.type_aliases import NpBool, NpFloat, NpInt
from atmodeller.utilities import get_reaction_dictionary

logger: logging.Logger = logging.getLogger(__name__)


class BaseReactionBlock(eqx.Module):
    """Base reaction block

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""

    @property
    def number_reactions(self) -> int:  # pyright: ignore
        """Number of reactions in the reaction block"""

    @abstractmethod
    def __init__(self, species: Iterable[SpeciesProtocol]):
        """Initializes the reaction block with the species collection"""

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
    reaction_species: SpeciesCollection[ChemicalSpecies]
    """Reaction species collection"""
    reaction_species_indices: NpInt
    """Indices of reaction species in the full species collection"""
    reaction_mask: ArrayLike
    """Boolean mask of reaction species in the full species collection"""
    reaction_matrix: NpFloat
    """Reaction matrix"""
    reaction_matrix_full: NpFloat
    """Reaction matrix expanded to full species space"""
    reaction_stability_mask_full: NpBool
    """Stability mask for reaction matrix expanded to full species space"""
    reaction_stability_matrix_full: NpFloat
    """Reaction stability matrix expanded to full species space"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        reaction_species, reaction_species_indices = self.species.extract_reaction_species()
        self.reaction_species = reaction_species
        self.reaction_species_indices = reaction_species_indices

        # Boolean mask of reaction species in the full species collection
        self.reaction_mask = np.zeros(self.species.number_species, dtype=bool)
        self.reaction_mask[self.reaction_species_indices] = True

        # Reaction matrix of linearly independent reactions
        transpose_formula_matrix: NpInt = reaction_species.get_formula_matrix().T
        self.reaction_matrix: NpFloat = partial_rref(transpose_formula_matrix)

        # Reaction matrix expanded to full species space
        self.reaction_matrix_full: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )
        # Insert reduced matrix into correct columns
        self.reaction_matrix_full[:, self.reaction_species_indices] = self.reaction_matrix

        self.reaction_stability_mask_full = np.broadcast_to(
            self.species.active_stability, self.reaction_matrix_full.shape
        )
        self.reaction_stability_matrix_full = (
            self.reaction_matrix_full * self.reaction_stability_mask_full
        )

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
            0, self.reaction_species.number_species - len(self.reaction_species.unique_elements)
        )

    # TODO: Only used to determine output. Relevant for gas only or should be gas only AND no
    # solubility?
    # @property
    # def gas_only(self) -> bool:
    #     """Checks if a gas-only network"""
    #     return len(self.data) == len(self.gas_species_mask)

    def get_log_Kp(self, temperature: Float[Array, "..."]) -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Args:
            temperature: Temperature in K

        Returns:
            Log of the equilibrium constant of each reaction
        """
        gibbs_funcs: list[Callable] = [
            to_hashable(species_.get_gibbs_over_RT) for species_ in self.reaction_species
        ]

        def apply_gibbs(
            index: Integer[Array, ""], temperature: Float[Array, "..."]
        ) -> Float[Array, "..."]:
            return lax.switch(index, gibbs_funcs, temperature)

        indices: Integer[Array, " reaction_species"] = jnp.arange(
            self.reaction_species.number_species
        )
        vmap_gibbs: Callable = eqx.filter_vmap(apply_gibbs, in_axes=(0, None))
        gibbs_values: Float[Array, "reaction_species 1"] = vmap_gibbs(indices, temperature)
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
            min(species.thermo.T_min) for species in self.reaction_species
        ]
        temperature_max: list[float] = [
            max(species.thermo.T_max) for species in self.reaction_species
        ]

        return max(temperature_min), min(temperature_max)


class DissolutionNetwork(BaseReactionBlock):
    """Handles all reactions where a species dissolves into or exchanges with a phase.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    dissolution_species: SpeciesCollection[ReservoirSpecies]
    """Dissolution species collection"""
    dissolution_species_indices: NpInt
    """Indices of dissolution species in the full species collection"""
    dissolution_mask: ArrayLike
    """Boolean mask of dissolution species in the full species collection"""
    reaction_indices_map: NpInt
    """Mapping of dissolution species to corresponding reaction species"""
    dissolution_matrix: NpFloat
    """Dissolution reaction matrix"""
    dilute_limit: bool = True
    """Whether to assume dilute limit for all dissolution reactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        dissolution_species, dissolution_species_indices = (
            self.species.extract_dissolution_species()
        )
        self.dissolution_species = dissolution_species
        self.dissolution_species_indices = dissolution_species_indices

        # Boolean mask of dissolution species in the full species collection
        self.dissolution_mask = np.zeros(self.species.number_species, dtype=bool)
        self.dissolution_mask[self.dissolution_species_indices] = True

        # Construct dissolution reaction matrix
        # For each dissolution species, get the index of the corresponding reaction (gas) species
        reaction_species, reaction_indices = self.species.extract_reaction_species()
        reaction_indices_map: list[int] = []
        for dissolution_species_ in dissolution_species:
            name: str = f"{dissolution_species_.data.hill_formula}_{GAS_STATE}"
            idx: int = reaction_species.species_names.index(name)
            reaction_indices_map.append(reaction_indices[idx])

        self.reaction_indices_map = np.array(reaction_indices_map, dtype=int)
        # Most direct to construct the dissolution matrix in full species space
        dissolution_matrix: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )
        for reaction_index in range(self.number_reactions):
            dissolution_matrix[
                reaction_index, self.dissolution_species_indices[reaction_index]
            ] = 1.0
            dissolution_matrix[reaction_index, self.reaction_indices_map[reaction_index]] = -1.0

        self.dissolution_matrix = dissolution_matrix

        self.output_to_logger()

    @property
    def number_reactions(self) -> int:
        """Number of dissolution reactions"""
        return self.dissolution_species.number_species

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

        # NOTE: All solubility formulations must return a JAX array to allow vmap
        solubility_funcs: list[Callable] = [
            to_hashable(species_.solubility.jax_concentration)
            for species_ in self.dissolution_species
        ]

        def apply_solubility(
            index: Integer[Array, ""],
            fugacity_val: Float[Array, ""],
            temp: Float[Array, ""],
            press: Float[Array, ""],
            o2_fug: Float[Array, ""],
        ) -> Float[Array, ""]:
            return lax.switch(index, solubility_funcs, fugacity_val, temp, press, o2_fug)

        indices: Integer[Array, " num_dissolution_species"] = jnp.arange(self.number_reactions)

        vmap_solubility: Callable = eqx.filter_vmap(
            apply_solubility, in_axes=(0, 0, None, None, None)
        )
        species_ppmw: Float[Array, " num_dissolution_species"] = vmap_solubility(
            indices, gas_species_activity, temperature, pressure, fO2
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


class ReactionSystem(BaseReactionBlock):
    """Reaction system that includes both core chemical reactions and dissolution reactions.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    blocks: tuple[BaseReactionBlock, ...]
    matrix: NpFloat
    stability_matrix: NpFloat
    _O2_index: NpInt
    _has_O2: NpBool

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        reaction = ReactionNetwork(self.species)
        dissolution = DissolutionNetwork(self.species)
        self.blocks = (reaction, dissolution)
        self.matrix = np.vstack([block.get_matrix() for block in self.blocks])
        self.stability_matrix = np.vstack([block.get_stability_matrix() for block in self.blocks])

        # Could be an integer (but represented as a float) or np.nan
        self._O2_index = np.nan_to_num(self.species.O2_index, nan=0).astype(int)
        self._has_O2 = ~np.isnan(self.species.O2_index)

        self.output_to_logger()

    @property
    def reaction(self) -> ReactionNetwork:
        """Reaction network block"""
        return self.blocks[0]  # pyright: ignore

    @property
    def dissolution(self) -> DissolutionNetwork:
        """Dissolution network block"""
        return self.blocks[1]  # pyright: ignore

    @property
    def active_reactions(self) -> NpBool:
        """Boolean mask of active reactions in the full reaction network"""
        return np.ones(self.number_reactions, dtype=bool)

    @property
    def number_reactions(self):
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

        # log_Kp_funcs: list[Callable] = [to_hashable(block.get_log_Kp) for block in self.blocks]

        # def apply_log_Kp(
        #     index: Integer[Array, ""],
        #     fugacity_val: Float[Array, ""],
        #     temp: Float[Array, ""],
        #     press: Float[Array, ""],
        #     o2_fug: Float[Array, ""],
        # ) -> Float[Array, ""]:
        #     return lax.switch(index, log_Kp_funcs, fugacity_val, temp, press, o2_fug)

        # indices: Integer[Array, " num"] = jnp.arange(len(self.blocks))

        # vmap_log_Kp: Callable = eqx.filter_vmap(apply_log_Kp, in_axes=(0, 0, None, None, None))
        # species_ppmw: Float[Array, " num_dissolution_species"] = vmap_log_Kp(
        #     indices, gas_species_activity, temperature, pressure, fO2
        # )
        # # jax.debug.print("species_ppmw = {out}", out=species_ppmw)

        # log_Kp: Float[Array, " num_reactions"] = (
        #     jnp.log(species_ppmw) - jnp.log(STANDARD_CONCENTRATION) - jnp.log(gas_species_activity)
        # )

        # log_Kp_reaction: Float[Array, " num_core_reactions"] = self.reaction.get_log_Kp(
        #     temperature
        # )

        log_Kp_dissolution: Float[Array, " num_dissolution_reactions"] = (
            self.dissolution.get_log_Kp(temperature, activity_dissolution, pressure, fO2)
        )

        return jnp.concatenate([log_Kp_reaction, log_Kp_dissolution])

    def get_matrix(self) -> NpFloat:
        return self.matrix

    def get_stability_matrix(self) -> NpFloat:
        return self.stability_matrix

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
            log_stability_reaction: Log stability of each species for reactions
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Residual of the reaction network
        """
        # Reaction residual
        log_Kp: Float[Array, " num_reactions"] = self.get_log_Kp(
            log_activity=log_activity, temperature=temperature, pressure=pressure
        )
        residual: Float[Array, " num_reactions"] = jnp.dot(self.matrix, log_activity) - log_Kp

        # Account for species stability
        residual = residual - jnp.dot(self.stability_matrix, safe_exp(log_stability))

        return residual
