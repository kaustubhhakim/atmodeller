# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase system containing multiple phases and phase indexing for slicing species arrays."""

import logging
from collections.abc import Iterable
from typing import Optional

import equinox as eqx
import numpy as np
from jaxmod.type_aliases import NpArray, NpBool

from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.interfaces import SpeciesProtocol
from atmodeller.phases import GasPhase, MeltPhase, PurePhase, SolidPhase

logger: logging.Logger = logging.getLogger(__name__)


class PhaseIndex(eqx.Module):
    """Stores start and stop indices of a phase in the full species collection.

    Args:
        start: Starting index of the phase in the full species collection
        stop: Stopping index of the phase in the full species collection
    """

    start: int
    stop: int

    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop

    @property
    def slice(self) -> slice:
        """Slice object for indexing arrays."""
        return slice(self.start, self.stop)

    def mask(self, n_total: int) -> NpArray:
        """Boolean mask for this phase

        Args:
            n_total: Total number of species in the full species collection

        Returns:
            Boolean mask for this phase
        """
        mask: NpBool = np.zeros(n_total, dtype=bool)
        mask[self.start : self.stop] = True
        return mask

    def __len__(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"PhaseIndex(start={self.start}, stop={self.stop})"


class PhaseSystem(eqx.Module):
    """A phase system containing multiple phases.

    This class represents a complete set of coexisting phases (gas, melt, solid, condensates).

    Args:
        gas: Gas phase
        melt: Melt phase. Defaults to an empty melt phase if not provided.
        solid: Solid phase. Defaults to an empty solid phase if not provided.
        condensates: Condensate phases. Defaults to an empty tuple if not provided.
    """

    gas: GasPhase
    """Gas"""
    melt: MeltPhase
    """Melt"""
    solid: SolidPhase
    """Solid"""
    condensates: tuple[PurePhase, ...]
    """Condensates"""
    species: SpeciesCollection[SpeciesProtocol]
    """All species"""
    _phase_indices: dict[str, PhaseIndex]
    """Phase indices for slicing species arrays"""

    def __init__(
        self,
        gas: GasPhase,
        *,
        melt: Optional[MeltPhase] = None,
        solid: Optional[SolidPhase] = None,
        condensates: Optional[Iterable[PurePhase]] = None,
    ):
        self.gas = gas
        self.melt = MeltPhase.empty() if melt is None else melt
        self.solid = SolidPhase.empty() if solid is None else solid
        if condensates is None:
            self.condensates = ()
        else:
            self.condensates = tuple(condensates)

        # The order of phases is significant! "gas" -> "melt" -> "solid" -> "condensates" must be
        # preserved because reaction matrices, phase slices, and activity concatenation rely on
        # this ordering.
        phase_order: tuple[str, ...] = ("gas", "melt", "solid", "condensates")

        # Flatten all species. Index 0 because pure phases can only have one species.
        condensate_species: tuple[ChemicalSpecies, ...] = tuple(
            condensate.species[0] for condensate in self.condensates
        )
        all_species: tuple[SpeciesProtocol, ...] = (
            self.gas.species.species
            + self.melt.species.species
            + self.solid.species.species
            + condensate_species
        )
        self.species = SpeciesCollection(all_species)

        # Phase indexing
        start: int = 0
        self._phase_indices = {}

        for phase_name, phase_collection in zip(
            phase_order, [self.gas, self.melt, self.solid, self.condensates]
        ):
            n: int = len(phase_collection)
            self._phase_indices[phase_name] = PhaseIndex(start, start + n)
            start += n

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

    def phase_slice(self, phase_name: str) -> slice:
        """Slice object for a given phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Slice object for the phase
        """
        return self._phase_indices[phase_name].slice

    def phase_mask(self, phase_name: str) -> NpBool:
        """Boolean mask for a given phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Boolean mask for the phase
        """
        return self._phase_indices[phase_name].mask(len(self.species))
