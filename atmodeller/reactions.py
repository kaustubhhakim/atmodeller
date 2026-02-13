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
from collections.abc import Callable, Iterable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmod.utils import partial_rref, to_hashable
from jaxtyping import Array, Float, Integer

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
    chemical_species: SpeciesCollection[ChemicalSpecies]
    """Chemical species collection"""
    number_reactions: int
    """Number of reactions"""
    reaction_matrix: NpFloat
    """Reaction matrix"""
    active_reactions: NpBool
    """Active reactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        chemical_species, chemical_indices = self.species.extract_chemical_species()
        self.chemical_species = chemical_species
        self.number_reactions = max(
            0, chemical_species.number_species - len(chemical_species.unique_elements)
        )

        # Reaction matrix of linearly independent reactions
        transpose_formula_matrix: NpInt = chemical_species.get_formula_matrix().T
        core_matrix_small: NpFloat = partial_rref(transpose_formula_matrix)

        # Expand to full species space
        reaction_matrix_full: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )
        # Insert reduced matrix into correct columns
        reaction_matrix_full[:, chemical_indices] = core_matrix_small
        self.reaction_matrix = reaction_matrix_full
        # All core reactions are active by default
        self.active_reactions = np.ones(self.number_reactions, dtype=bool)

        temperature_min, temperature_max = self.get_temperature_range()
        logger.info(
            "Thermodynamic data requires temperatures between %d K and %d K",
            np.ceil(temperature_min),
            np.floor(temperature_max),
        )

    @classmethod
    def available_species(cls) -> tuple[str, ...]:
        return thermodynamic_data_source.available_species()

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
            to_hashable(species_.get_gibbs_over_RT) for species_ in self.chemical_species
        ]

        def apply_gibbs(
            index: Integer[Array, ""], temperature: Float[Array, "..."]
        ) -> Float[Array, "..."]:
            return lax.switch(index, gibbs_funcs, temperature)

        indices: Integer[Array, " species"] = jnp.arange(self.chemical_species.number_species)
        vmap_gibbs: Callable = eqx.filter_vmap(apply_gibbs, in_axes=(0, None))
        gibbs_values: Float[Array, "species 1"] = vmap_gibbs(indices, temperature)
        # jax.debug.print("gibbs_values = {out}", out=gibbs_values)
        reaction_matrix: Float[Array, "reactions species"] = jnp.asarray(self.reaction_matrix)
        log_Kp: Float[Array, "reactions 1"] = -1.0 * reaction_matrix @ gibbs_values

        return jnp.ravel(log_Kp)

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.reaction_matrix, self.species.species_names)

    def get_temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range of the thermodynamic data for the species

        Returns:
            Minimum and maximum temperature that is valid for the species
        """
        temperature_min: list[float] = [
            min(species.thermo.T_min) for species in self.chemical_species
        ]
        temperature_max: list[float] = [
            max(species.thermo.T_max) for species in self.chemical_species
        ]

        return max(temperature_min), min(temperature_max)


class DissolutionReactionNetwork(eqx.Module):
    """Handles all reactions where a reservoir species dissolves into or exchanges with a phase.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    """Species collection"""
    reservoir_species: SpeciesCollection[ReservoirSpecies]
    """Reservoir species collection"""
    number_reactions: int
    """Number of reactions"""
    dissolution_matrix: NpFloat
    """Dissolution reaction matrix"""
    active_reactions: NpBool
    """Active dissolution reactions"""
    dilute_limit: bool = True
    """Whether to assume dilute limit for all dissolution reactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        reservoir_species, reservoir_indices = self.species.extract_reservoir_species()
        self.reservoir_species = reservoir_species
        self.number_reactions = reservoir_species.number_species
        # All dissolution reactions are active by default
        self.active_reactions = np.ones(self.number_reactions, dtype=bool)

        # Construct dissolution reaction matrix
        # For each reservoir species, get the index of the corresponding gas species
        chemical_species, chemical_indices = self.species.extract_chemical_species()
        gas_species_indices: list[int] = []
        for reservoir_species_ in reservoir_species:
            name: str = f"{reservoir_species_.data.hill_formula}_{GAS_STATE}"
            idx: int = chemical_species.species_names.index(name)
            gas_species_indices.append(chemical_indices[idx])

        dissolution_matrix: NpFloat = np.zeros(
            (self.number_reactions, self.species.number_species), dtype=float
        )
        for reaction_index in range(self.number_reactions):
            # TODO: check sign convention for reactants and products
            dissolution_matrix[reaction_index, gas_species_indices[reaction_index]] = -1.0
            dissolution_matrix[reaction_index, reservoir_indices[reaction_index]] = 1.0

        self.dissolution_matrix = dissolution_matrix

    def get_log_Kp(
        self,
        fugacity: Float[Array, "..."],
        temperature: Float[Array, "..."],
        pressure: Float[Array, ""],
        fO2: Float[Array, ""],
    ) -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction in terms of partial pressures.

        Args:
            fugacity: Fugacity in bar
            temperature: Temperature in K
            pressure: Pressure in bar
            fO2: Oxygen fugacity in bar

        Returns:
            Log of the equilibrium constant of each reaction in terms of partial pressures
        """
        # Return empty array if no reservoir species
        if self.number_reactions == 0:
            return jnp.array([], dtype=jnp.float32)

        # Could be an integer (but represented as a float) or np.nan
        # diatomic_oxygen_index: NpFloat = self.species.diatomic_oxygen_index

        # log_activity: Float[Array, " species"] = get_log_activity(parameters, log_number_moles)
        # fugacity: Float[Array, " species"] = safe_exp(log_activity)

        # Need just the fugacity of the gas species associated with the (dissolved) reservoir species.
        # fugacity_reservoir_species: Float[Array, " reservoir_species"] = jnp.take(
        #    fugacity,
        #    indices=species_collection.data.reservoir_species_map,
        #    unique_indices=True,
        #   indices_are_sorted=True,
        # )

        # For type consistency, convert to integer array with nan as 0
        # diatomic_oxygen_index_: Integer[Array, ""] = jnp.nan_to_num(
        #    diatomic_oxygen_index, nan=0
        # ).astype(jnp.int_)

        # Get fO2, or nan if not present
        # diatomic_oxygen_fugacity: Float[Array, ""] = jnp.where(
        #    jnp.isnan(diatomic_oxygen_index), jnp.nan, jnp.take(fugacity, diatomic_oxygen_index_)
        # )

        # NOTE: All solubility formulations must return a JAX array to allow vmap
        solubility_funcs: list[Callable] = [
            to_hashable(species_.solubility.jax_concentration)
            for species_ in self.reservoir_species
        ]

        def apply_solubility(
            index: Integer[Array, ""],
            fugacity_val: Float[Array, ""],
            temp: Float[Array, ""],
            press: Float[Array, ""],
            o2_fug: Float[Array, ""],
        ) -> Float[Array, ""]:
            return lax.switch(index, solubility_funcs, fugacity_val, temp, press, o2_fug)

        indices: Integer[Array, " reservoir_species"] = jnp.arange(
            self.reservoir_species.number_species
        )

        vmap_solubility: Callable = eqx.filter_vmap(
            apply_solubility, in_axes=(0, 0, None, None, None)
        )
        species_ppmw: Float[Array, " reservoir_species"] = vmap_solubility(
            indices, fugacity, temperature, pressure, fO2
        )
        # jax.debug.print("ppmw = {out}", out=ppmw)

        # FIXME: Divide by fugacity (and return log)?

        return species_ppmw

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.dissolution_matrix, self.species.species_names)


class FullReactionNetwork(eqx.Module):
    """Full reaction network that includes both core chemical reactions and dissolution reactions.

    Args:
        species: An iterable of species
    """

    species: SpeciesCollection[SpeciesProtocol]
    core_network: ReactionNetwork
    dissolution_network: DissolutionReactionNetwork
    full_matrix: NpFloat

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        self.core_network = ReactionNetwork(species)
        self.dissolution_network = DissolutionReactionNetwork(species)

        core_matrix = self.core_network.reaction_matrix
        diss_matrix = self.dissolution_network.dissolution_matrix

        self.full_matrix = np.vstack([core_matrix, diss_matrix])

    def get_log_Kp(self, temperature: Float[Array, "..."]):  # -> Float[Array, " reactions"]:
        """Gets log of the equilibrium constant of each reaction in terms of partial pressures.

        Args:
            temperature: Temperature in K

        Returns:
            Log of the equilibrium constant of each reaction in terms of partial pressures
        """

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        return get_reaction_dictionary(self.full_matrix, self.species.species_names)
