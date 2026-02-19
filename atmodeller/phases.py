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
import numpy as np
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float
from molmass import Formula

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

    def get_log_mass(self, log_number_moles: Float[Array, " species"]) -> Float[Array, ""]:
        """Get the log mass of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mass of the phase
        """
        return logsumexp(log_number_moles, b=self.species.molar_masses)

    def get_log_molar_mass(self, log_number_moles: Float[Array, " species"]) -> Float[Array, ""]:
        """Get the log molar mass of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log molar mass of the phase
        """
        log_molar_mass: Float[Array, ""] = self.get_log_mass(log_number_moles) - logsumexp(
            log_number_moles
        )

        return log_molar_mass

    def get_log_mole_fraction(
        self, log_number_moles: Float[Array, " species"]
    ) -> Float[Array, " species"]:
        """Get the log mole fraction of each species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mole fractions of each species in the phase
        """
        log_total_moles: Float[Array, ""] = logsumexp(log_number_moles)

        return log_number_moles - log_total_moles

    def __len__(self) -> int:
        return len(self.species)


class GasPhase(BasePhase[ChemicalSpecies]):
    """Multicomponent gas mixture

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

    O2_index: NpFloat
    """Index of O2 or np.nan if not present"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        self.species = SpeciesCollection(species)
        self.O2_index = self.get_O2_index()
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


class SolidPhase(BasePhase[SpeciesProtocol]):
    """Multicomponent silicate solid

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

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


class PurePhase(BasePhase[ChemicalSpecies]):
    """Pure, unity-activity phases

    Explicit __init__ is needed to specialize the generic BasePhase constructor, ensuring static
    type checkers (e.g., Pyright) infer types correctly.
    """

    def __init__(self, species: Iterable[ChemicalSpecies]):
        self.species = SpeciesCollection(species)
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
