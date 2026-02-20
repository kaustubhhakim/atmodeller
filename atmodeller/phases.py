# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase container classes for thermodynamic equilibrium calculations.

This module defines high-level phase abstractions used in the equilibrium solver, including gas
mixtures, silicate melts, solids, and pure unity-activity phases.

Each phase is represented as a `SpeciesCollection` of thermodynamic species (formula + state of
aggregation). Species are constructed from their Hill formulas and assigned an aggregation state
consistent with the JANAF/NASA convention:

    - "g" : gas
    - "l" : liquid
    - "s" : solid

These phase classes provide a structured interface for grouping species by thermodynamic role (gas,
melt, solid, pure phase) while keeping the underlying species-level thermodynamic data separate.

They are used by the equilibrium solver to:
    - distinguish gas and condensed phases,
    - apply appropriate activity conventions,
    - track phase-specific properties (e.g., O2 index in the gas phase),
    - manage multicomponent phase assemblages.
"""

import logging
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Generic, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from jaxmod.utils import to_hashable
from jaxtyping import Array, Float, Integer
from molmass import Formula

from atmodeller import override
from atmodeller.constants import GAS_STATE, LIQUID_STATE, SOLID_STATE
from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.type_aliases import NpFloat, TSpecies_co

logger: logging.Logger = logging.getLogger(__name__)


def _build_species_collection(
    species: str | Iterable[str], factory: Callable[[str], TSpecies_co]
) -> SpeciesCollection[TSpecies_co]:
    """Normalize input and build a species collection using a factory.

    Args:
        species: A single species name or an iterable of names
        factory: A function that takes a Hill formula and returns a species instance

    Returns:
        A SpeciesCollection containing the constructed species
    """
    if isinstance(species, str):
        species = [species]

    species_list: list[TSpecies_co] = []

    for species_ in species:
        hill_formula: str = Formula(species_).formula
        species_list.append(factory(hill_formula))

    return SpeciesCollection(species_list)


class BasePhase(eqx.Module, Generic[TSpecies_co]):
    """Base class for all phases"""

    species: SpeciesCollection[TSpecies_co]
    """Collection of species in the phase"""

    @abstractmethod
    def __init__(self, species: Iterable[TSpecies_co]):
        """Initialize a phase with a collection of species.

        Args:
            species: An iterable of species belonging to the phase
        """

    @classmethod
    def empty(cls) -> Self:
        """Returns an empty phase instance."""
        return cls([])

    # Need to ensure all phases have this method first
    # @abstractmethod
    # def get_log_mole_fractions(
    #     self,
    #     log_number_moles: Float[Array, " num_species"],
    #     log_stability: Float[Array, " num_species"],
    #     log_inert_moles: Float[Array, ""] = jnp.array(-jnp.inf),
    # ) -> Float[Array, " num_species"]:
    #     """Get the log mole fraction of each species.

    #     Args:
    #         log_number_moles: Log number of moles of each species
    #         log_stability: Log stability of each species
    #         log_inert_moles: Log number of moles of the inert, non-reactive component. Defaults to
    #             negative infinity (i.e., no inert component).

    #     Returns:
    #         Log mole fractions of each species
    #     """

    def get_log_effective_moles(
        self,
        log_number_moles: Float[Array, " num_species"],
        log_stability: Float[Array, " num_species"],
    ) -> Float[Array, " num_species"]:
        """Gets the log effective moles of the phase, accounting for stability.

        Args:
            log_number_moles: Log number of moles of each species
            log_stability: Log stability of each species

        Returns:
            Log effective moles of the phase
        """
        log_stability_masked: Float[Array, " num_species"] = jnp.where(
            self.species.active_stability, log_stability, 0.0
        )
        log_effective_moles: Float[Array, " num_species"] = log_number_moles + log_stability_masked

        return log_effective_moles

    # TODO: May not be required here
    # def get_log_molar_mass(self, log_number_moles: Float[Array, " species"]) -> Float[Array, ""]:
    #     """Get the log molar mass of the phase.

    #     Args:
    #         log_number_moles: Log number of moles of each species in the phase

    #     Returns:
    #         Log molar mass of the phase
    #     """
    #     log_molar_mass: Float[Array, ""] = self.get_log_mass(log_number_moles) - logsumexp(
    #         log_number_moles
    #     )

    #     return log_molar_mass

    def __len__(self) -> int:
        return len(self.species)


class GasPhase(BasePhase[ChemicalSpecies]):
    """Multicomponent gas mixture

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

    O2_index: NpFloat
    """Index of O2 or np.nan if not present"""
    vmap_log_activity: Callable
    """Vectorized log activity functions"""

    @override
    def __init__(self, species: Iterable[ChemicalSpecies]):
        self.species = SpeciesCollection(species)
        self.O2_index = self.get_O2_index()

        log_activity_funcs: list[Callable] = [
            to_hashable(species_.activity.log_activity) for species_ in species
        ]

        def apply_log_activity(
            index: Integer[Array, ""], temperature: Float[Array, ""], pressure: Float[Array, ""]
        ) -> Float[Array, ""]:
            return lax.switch(index, log_activity_funcs, temperature, pressure)

        self.vmap_log_activity = eqx.filter_vmap(apply_log_activity, in_axes=(0, None, None))

        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self.species)}"
        )

    @classmethod
    def create(cls, species: str | Iterable[str]) -> Self:
        """Creates an instance.

        Args:
            species: A single gas species name or iterable of names

        Returns
            An instance
        """
        species_collection: SpeciesCollection[ChemicalSpecies] = _build_species_collection(
            species, lambda hill: ChemicalSpecies.create_gas(hill, state=GAS_STATE)
        )

        return cls(species_collection)

    def get_log_activity(
        self,
        log_number_moles: Float[Array, " species"],
        temperature: Float[Array, ""],
        pressure: Float[Array, ""],
    ) -> Float[Array, " species"]:
        """Get the log activity of each species.

        This is an ideal mixture of (potentially) non-ideal gases.

        Args:
            log_number_moles: Log number of moles of each species
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log activity of each species
        """
        # Log activity of pure species
        log_activity: Float[Array, " num_species"] = self.vmap_log_activity(
            jnp.arange(self.species.number_species), temperature, pressure
        )

        # Ideal mixing
        log_activity = log_activity + self.get_log_mole_fractions(log_number_moles)

        return log_activity

    def get_log_mass(
        self,
        log_number_moles: Float[Array, " num_species"],
        log_stability: Float[Array, " num_species"],
    ) -> Float[Array, ""]:
        """Get the log mass.

        Args:
            log_number_moles: Log number of moles of each species
            log_stability: Log stability of each species

        Returns:
            Log mass of the phase
        """
        log_effective_moles: Float[Array, " num_species"] = self.get_log_effective_moles(
            log_number_moles, log_stability
        )

        return logsumexp(log_effective_moles, b=self.species.molar_masses)

    @override
    def get_log_mole_fractions(
        self,
        log_number_moles: Float[Array, " num_species"],
        log_inert_moles: Float[Array, ""] = jnp.array(-jnp.inf),
    ) -> Float[Array, " num_species"]:
        log_total_moles: Float[Array, ""] = logsumexp(
            jnp.append(log_number_moles, log_inert_moles)
        )

        return log_number_moles - log_total_moles

    def get_O2_index(self) -> NpFloat:
        """Gets the species index corresponding to diatomic oxygen.

        Note:
            This returns a float array for type consistency.

        Returns:
            Index of diatomic oxygen, or np.nan if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.species):
            if species_.data.hill_formula == "O2":
                # logger.debug("Found O2 at index = %d", nn)
                return np.array(nn, dtype=float)

        return np.array(np.nan, dtype=float)


class MeltPhase(BasePhase[SpeciesProtocol]):
    """Multicomponent silicate melt with optionally dissolved volatiles

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

    @override
    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self.species)}"
        )

    @classmethod
    def create(cls, species: str | Iterable[str]) -> Self:
        """Creates an instance.

        Args:
            species: A single melt species name or iterable of names

        Returns
            An instance
        """
        species_collection: SpeciesCollection[SpeciesProtocol] = _build_species_collection(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=LIQUID_STATE)
        )

        return cls(species_collection)

    def get_log_mass_fraction(
        self,
        log_number_moles: Float[Array, " num_species"],
        log_inert_mass: Float[Array, ""] = jnp.asarray(-jnp.inf),
        dilute_limit: bool = True,
    ) -> Float[Array, " num_species"]:
        """Gets the log mass fraction of the species.

        Args:
            log_number_moles: Log number of moles
            log_inert_mass: Log mass of the inert, non-reactive component (e.g., silicate).
                Defaults to negative infinity (i.e., no inert component).
            dilute_limit: Whether to assume the dilute limit for dissolution reactions. Defaults to
                ``True``.

        Returns:
            Log mass fraction
        """
        log_mass: Float[Array, " num_species"] = log_number_moles + jnp.log(
            self.species.molar_masses
        )
        # jax.debug.print("log_mass = {out}", out=log_mass)

        if dilute_limit:
            total_log_mass: Float[Array, ""] = log_inert_mass
        else:
            # Must account for an inert, non-reactive melt mass, given by the thermodynamic state
            log_mass_plus: Float[Array, " num_species_plus_one"] = jnp.append(
                log_mass, log_inert_mass
            )
            # jax.debug.print("log_mass_plus = {out}", out=log_mass_plus)

            # Log total (sum in linear space)
            total_log_mass = logsumexp(log_mass_plus)
            # jax.debug.print("total_log_mass = {out}", out=total_log_mass)

        # Log mass fraction = log(m_i) − log(total)
        log_mass_fraction: Float[Array, " num_species"] = log_mass - total_log_mass
        # jax.debug.print("log_mass_fraction = {out}", out=log_mass_fraction)

        # Finally, convert to ppmw
        log_mass_ppmw: Float[Array, " num_species"] = log_mass_fraction + jnp.log(1e6)
        # jax.debug.print("log_mass_ppmw = {out}", out=log_mass_ppmw)

        return log_mass_ppmw

    @override
    def get_log_mole_fractions(
        self,
        log_number_moles: Float[Array, " num_species"],
        log_inert_moles: Float[Array, ""] = jnp.array(-jnp.inf),
    ) -> Float[Array, " num_species"]: ...


class SolidPhase(BasePhase[SpeciesProtocol]):
    """Multicomponent silicate solid

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

    @override
    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self.species)}"
        )

    @classmethod
    def create(cls, species: str | Iterable[str]) -> Self:
        """Creates an instance.

        Args:
            species: A single solid species name or iterable of names

        Returns
            An instance
        """
        species_collection: SpeciesCollection[ChemicalSpecies] = _build_species_collection(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=SOLID_STATE)
        )

        return cls(species_collection)

    def get_log_mass_fraction(
        self,
        log_number_moles: Float[Array, " num_species"],
        log_inert_mass: Float[Array, ""] = jnp.array(-jnp.inf),
    ) -> Float[Array, " num_species"]:
        """Gets the log mass fraction of the species.

        Args:
            log_number_moles: Log number of moles
            log_inert_mass: Log mass of the inert, non-reactive component (e.g., silicate).
                Defaults to negative infinity (i.e., no inert component).

        Returns:
            Log mass fraction in the solid
        """
        log_mass: Float[Array, " species"] = log_number_moles + jnp.log(self.species.molar_masses)
        # jax.debug.print("log_mass = {out}", out=log_mass)

        # Must account for an inert, non-reactive solid mass, given by the thermodynamic state
        log_mass_plus: Float[Array, " species_plus_one"] = jnp.append(log_mass, log_inert_mass)
        # jax.debug.print("log_mass_plus = {out}", out=log_mass_plus)

        # Log total (sum in linear space)
        total_log_mass = logsumexp(log_mass_plus)
        # jax.debug.print("total_log_mass = {out}", out=total_log_mass)

        # Log mass fraction = log(m_i) − log(total)
        log_mass_fraction: Float[Array, " species"] = log_mass - total_log_mass
        # jax.debug.print("log_mass_fraction = {out}", out=log_mass_fraction)

        # Finally, convert to ppmw
        log_mass_ppmw: Float[Array, " species"] = log_mass_fraction + jnp.log(1e6)
        # jax.debug.print("log_mass_ppmw = {out}", out=log_mass_ppmw)

        return log_mass_ppmw


class PurePhase(BasePhase[ChemicalSpecies]):
    """Pure, unity-activity phases

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

    @override
    def __init__(self, species: Iterable[ChemicalSpecies]):
        self.species = SpeciesCollection(species)

        if self.species.number_species != 1:
            raise ValueError(
                f"PurePhase must contain exactly one species, got {len(self.species)}"
            )

        logger.info(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self.species)}"
        )

    @classmethod
    def create(cls, species: str, state: str = SOLID_STATE) -> Self:
        """Creates an instance.

        Args:
            species: Species
            state: State of aggregation. Defaults to :const:`~atmodeller.constants.SOLID_STATE`.

        Returns
            An instance
        """
        species_collection: SpeciesCollection[ChemicalSpecies] = _build_species_collection(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=state)
        )

        return cls(species_collection)

    # Although this could be a method, it is more efficient to not have to vmap over such a simple
    # function that just returns zeros.
    # def get_log_activity(self) -> Float[Array, " species"]:
    #     """Gets the log activity of a pure phase.

    #     The activity of a pure phase is unity by definition.

    #     Returns:
    #         Log activity of a pure phase (zero)
    #     """
    #     return jnp.zeros(self.species.number_species)
