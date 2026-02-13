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
from collections.abc import Callable, Iterable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmod.utils import partial_rref, to_hashable
from jaxtyping import Array, Bool, Float, Integer

from atmodeller.constants import GAS_STATE
from atmodeller.containers import ChemicalSpecies, ReservoirSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.thermodata import thermodynamic_data_source
from atmodeller.type_aliases import NpBool, NpFloat, NpInt
from atmodeller.utilities import get_reaction_dictionary

logger: logging.Logger = logging.getLogger(__name__)


class ReactionNetwork(eqx.Module):
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
    number_reactions: int
    """Number of reactions"""
    reaction_matrix: NpFloat
    """Reaction matrix"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        reaction_species, reaction_species_indices = self.species.extract_reaction_species()
        self.reaction_species = reaction_species
        self.reaction_species_indices = reaction_species_indices
        self.number_reactions = max(
            0, reaction_species.number_species - len(reaction_species.unique_elements)
        )

        # Reaction matrix of linearly independent reactions
        transpose_formula_matrix: NpInt = reaction_species.get_formula_matrix().T
        self.reaction_matrix: NpFloat = partial_rref(transpose_formula_matrix)

        logger.debug("reaction_matrix = %s", self.reaction_matrix)
        logger.info("Reaction network = %s", pprint.pformat(self.get_reaction_dictionary()))

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
    def active_reactions(self) -> Bool[Array, " reactions"]:
        """Boolean mask of active reactions in the reaction network"""
        return jnp.ones(self.number_reactions, dtype=bool)

    @property
    def reaction_mask(self) -> Bool[Array, " species"]:
        """Boolean mask of reaction species in the full species collection"""
        return (
            jnp.zeros(self.species.number_species, dtype=bool)
            .at[self.reaction_species_indices]
            .set(True)
        )

    # TODO: Only used to determine output. Relevant for gas only or should be gas only AND no
    # solubility?
    # @property
    # def gas_only(self) -> bool:
    #     """Checks if a gas-only network"""
    #     return len(self.data) == len(self.gas_species_mask)

    def get_log_Kp(self, temperature: Float[Array, "..."]) -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction in terms of partial pressures.

        Args:
            temperature: Temperature in K

        Returns:
            Log of the equilibrium constant of each reaction in terms of partial pressures
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

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.reaction_matrix, self.reaction_species.species_names)

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


class DissolutionNetwork(eqx.Module):
    """Handles all reactions where a reservoir species dissolves into or exchanges with a phase.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    dissolution_species: SpeciesCollection[ReservoirSpecies]
    """Dissolution species collection"""
    dissolution_species_indices: NpInt
    """Indices of dissolution species in the full species collection"""
    reaction_indices_map: NpInt
    """Mapping of dissolution species to corresponding reaction species"""
    dissolution_matrix: NpFloat
    """Dissolution reaction matrix"""
    active_reactions: NpBool
    """Active dissolution reactions"""
    dilute_limit: bool = True
    """Whether to assume dilute limit for all dissolution reactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        dissolution_species, dissolution_species_indices = (
            self.species.extract_dissolution_species()
        )
        self.dissolution_species = dissolution_species
        self.dissolution_species_indices = dissolution_species_indices
        # All dissolution reactions are active by default
        self.active_reactions = np.ones(self.number_reactions, dtype=bool)

        # Construct dissolution reaction matrix
        # For each reservoir species, get the index of the corresponding reaction (gas) species
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
            # TODO: check sign convention for reactants and products
            dissolution_matrix[reaction_index, self.reaction_indices_map[reaction_index]] = -1.0
            dissolution_matrix[
                reaction_index, self.dissolution_species_indices[reaction_index]
            ] = 1.0

        self.dissolution_matrix = dissolution_matrix

        logger.debug("dissolution_matrix = %s", self.dissolution_matrix)
        logger.info("Dissolution network = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def number_reactions(self) -> int:
        """Number of dissolution reactions"""
        return self.dissolution_species.number_species

    @property
    def dissolution_mask(self) -> Bool[Array, " species"]:
        """Boolean mask of dissolution species in the full species collection"""
        return (
            jnp.zeros(self.species.number_species, dtype=bool)
            .at[self.dissolution_species_indices]
            .set(True)
        )

    def get_log_Kp(
        self,
        fugacity: Float[Array, "..."],
        temperature: Float[Array, "..."],
        pressure: Float[Array, ""],
        fO2: Float[Array, ""],
    ) -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K
            pressure: Pressure in bar
            fO2: Oxygen fugacity in bar

        Returns:
            Log of the equilibrium constant of each reaction
        """
        # Return empty array if no reservoir species
        if self.number_reactions == 0:
            return jnp.array([], dtype=jnp.float32)

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
            indices, fugacity, temperature, pressure, fO2
        )
        # jax.debug.print("species_ppmw = {out}", out=species_ppmw)

        # TODO: Check standard state and sign
        log_Kp: Float[Array, " num_reactions"] = jnp.log(species_ppmw) - jnp.log(fugacity)

        return log_Kp

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets dissolution reactions as a dictionary.

        Returns:
            Dissolution reactions as a dictionary
        """
        return get_reaction_dictionary(self.dissolution_matrix, self.species.species_names)


class FullNetwork(eqx.Module):
    """Full reaction network that includes both core chemical reactions and dissolution reactions.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    reaction: ReactionNetwork
    dissolution: DissolutionNetwork
    full_matrix: NpFloat

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        self.reaction = ReactionNetwork(species)
        self.dissolution = DissolutionNetwork(species)

        # Reaction matrix expanded to full species space
        reaction_matrix_padded: NpFloat = np.zeros(
            (self.reaction.number_reactions, self.species.number_species), dtype=float
        )
        # Insert reduced matrix into correct columns
        reaction_matrix_padded[:, self.reaction.reaction_species_indices] = (
            self.reaction.reaction_matrix
        )

        self.full_matrix = np.vstack([reaction_matrix_padded, self.dissolution.dissolution_matrix])

        logger.debug("full_matrix = %s", str(self.full_matrix))
        logger.info("All reactions = %s", pprint.pformat(self.get_reaction_dictionary()))

    @property
    def number_reactions(self) -> int:
        return self.reaction.number_reactions + self.dissolution.number_reactions

    def get_log_Kp(
        self,
        log_activity: Float[Array, " species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, ""],
    ):  # -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction.

        Args:
            log_activity: Log activity of each species
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log of the equilibrium constant of each reaction
        """
        # Assemble log Kps from the reaction and dissolution networks
        log_Kp_reaction: Float[Array, " num_core_reactions"] = self.reaction.get_log_Kp(
            temperature
        )

        # Need to get just the log_activity of reservoir species
        log_activity_dissolution: Float[Array, " num_dissolution_species"] = jnp.take(
            log_activity,
            indices=self.dissolution.dissolution_species_indices,
            unique_indices=True,
            indices_are_sorted=True,
        )
        # jax.debug.print("log_activity_dissolution = {out}", out=log_activity_dissolution)

        activity_dissolution: Float[Array, " num_dissolution_species"] = jnp.exp(
            log_activity_dissolution
        )

        # Could be an integer (but represented as a float) or np.nan
        O2_index: Float[Array, ""] = jnp.array(self.species.O2_index)

        # For type consistency, convert to integer array with nan as 0
        O2_index_: Integer[Array, ""] = jnp.nan_to_num(O2_index, nan=0).astype(jnp.int_)

        # Get fO2, or nan if not present
        fO2: Float[Array, ""] = jnp.where(
            jnp.isnan(O2_index), jnp.nan, jnp.take(activity_dissolution, O2_index_)
        )

        log_Kp_dissolution: Float[Array, " num_dissolution_reactions"] = (
            self.dissolution.get_log_Kp(
                fugacity=jnp.exp(log_activity_dissolution),
                temperature=temperature,
                pressure=pressure,
                fO2=fO2,
            )
        )

        return jnp.concatenate([log_Kp_reaction, log_Kp_dissolution])

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.full_matrix, self.species.species_names)
