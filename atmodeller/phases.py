# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Phase container classes for thermodynamic equilibrium calculations.

This module defines four concrete phase types used in the equilibrium solver:

- :class:`atmodeller.phases.GasPhase`: ideal mixture of (potentially) non-ideal gases.
  Tracks the O2 index for redox calculations.
- :class:`atmodeller.phases.MeltPhase`: multicomponent silicate melt, optionally with
  dissolved volatiles and/or condensed species treated as additional to the solvent.
  Phase-level properties (mass, moles, molar mass, and derived fractions) can be computed
  relative to just the solvent mass, while per-species properties such as activity are
  always computed from raw species amounts.
- :class:`atmodeller.phases.SolidPhase`: multicomponent silicate solid.
- :class:`atmodeller.phases.PurePhase`: single-species, unity-activity phase (e.g., a
  pure mineral or ice).

All phases are JAX-compatible :class:`equinox.Module` subclasses. Each phase wraps a
:class:`atmodeller.containers.SpeciesCollection` of thermodynamic species (Hill formula
+ aggregation state). Species are constructed from their Hill formulas and assigned an
aggregation state consistent with the JANAF/NASA convention:

    - ``"g"`` : gas
    - ``"l"`` : liquid
    - ``"s"`` : solid

Quantities are accumulated in log-space throughout for numerical stability. Many methods
accept an optional ``log_background_*`` argument representing a background component
(e.g., the silicate melt mass) that contributes to phase total but is not tracked as an
explicit species in the solver.
"""

import logging
from collections.abc import Callable, Iterable
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmod.constants import GAS_CONSTANT_BAR
from jaxmod.type_aliases import NpFloat
from jaxmod.utils import to_hashable
from jaxtyping import Array, ArrayLike, Float, Integer

from atmodeller import override
from atmodeller.constants import GAS_STATE, LIQUID_STATE, SOLID_STATE
from atmodeller.containers import ChemicalSpecies, SpeciesCollection
from atmodeller.interfaces import RedoxBufferProtocol, SpeciesProtocol
from atmodeller.phase_base import BasePhase, PhaseOutput, build_species_collection
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer

logger: logging.Logger = logging.getLogger(__name__)


class GasPhaseOutput(PhaseOutput["GasPhase"]):
    """Output helper class for GasPhase-specific properties."""

    @property
    def log10dIW_1_bar(self) -> Float[Array, "#n_batch 1"]:
        """Log10 of the oxygen fugacity relative to the IW buffer at 1 bar."""
        O2_index: ArrayLike = self.phase.O2_index

        def no_oxygen() -> Float[Array, "#n_batch 1"]:
            return jnp.full((self.log_number_moles.shape[0], 1), jnp.nan)

        def with_oxygen() -> Float[Array, "#n_batch 1"]:
            log10_fugacity = jnp.log10(self.species_activity[..., O2_index.astype(int)])
            buffer: RedoxBufferProtocol = IronWustiteBuffer()
            buffer_at_one_bar = buffer.log10_fugacity(jnp.squeeze(self.temperature), 1.0)
            log10_shift_at_one_bar = log10_fugacity - buffer_at_one_bar
            return jnp.expand_dims(log10_shift_at_one_bar, axis=-1)

        return lax.cond(jnp.isnan(O2_index), no_oxygen, with_oxygen)

    @property
    def log10dIW_P(self) -> Float[Array, "#n_batch 1"]:
        """Log10 of the oxygen fugacity relative to the IW buffer at the pressure of interest."""
        O2_index: ArrayLike = self.phase.O2_index

        def no_oxygen() -> Float[Array, "#n_batch 1"]:
            return jnp.full((self.log_number_moles.shape[0], 1), jnp.nan)

        def with_oxygen() -> Float[Array, "#n_batch 1"]:
            log10_fugacity = jnp.log10(self.species_activity[..., O2_index.astype(int)])
            buffer: RedoxBufferProtocol = IronWustiteBuffer()
            buffer_at_P = buffer.log10_fugacity(
                jnp.squeeze(self.temperature), jnp.squeeze(self.pressure)
            )
            log10_shift_at_P = log10_fugacity - buffer_at_P
            return jnp.expand_dims(log10_shift_at_P, axis=-1)

        return lax.cond(jnp.isnan(O2_index), no_oxygen, with_oxygen)

    @property
    def species_partial_pressure(self) -> Float[Array, "#n_batch n_species"]:
        """Partial pressure of each species in bar"""
        return self.pressure * self.species_mole_fraction

    @property
    def volume(self) -> Float[Array, "#n_batch 1"]:
        r"""Volume of the gas phase in m\ :sup:`3`"""
        return self.phase_number_moles * GAS_CONSTANT_BAR * self.temperature / self.pressure

    @override
    def asdict(self) -> dict[str, Any]:
        """Dictionary representation of the output for downstream processing or analysis.

        Returns:
            A dictionary containing phase-level and species-level properties
        """
        out = super().asdict()

        out["phase"]["volume"] = self.volume
        out["phase"]["log10dIW_1_bar"] = self.log10dIW_1_bar
        out["phase"]["log10dIW_P"] = self.log10dIW_P
        out["species"]["partial_pressure"] = self.species_partial_pressure

        return out

    @override
    def asdict_split(self) -> dict[str, Any]:
        """Dictionary representation of the output with species-level arrays split into individual
        entries for each species.

        Returns:
            A dictionary containing phase-level and species-level properties
        """
        out = super().asdict_split()

        out["phase"]["volume"] = self.volume
        out["phase"]["log10dIW_1_bar"] = self.log10dIW_1_bar
        out["phase"]["log10dIW_P"] = self.log10dIW_P

        self._split_by_species(self.species_partial_pressure, out["species"], "partial_pressure")

        return out


class GasPhase(BasePhase[ChemicalSpecies]):
    """Multicomponent gas mixture.

    Models gas species as an ideal mixture of (potentially) non-ideal pure gases, where each
    species contributes a fugacity-corrected activity based on its equation of state.
    """

    O2_index: NpFloat
    """Index of O2 or ``np.nan`` if not present"""
    vmap_log_activity: Callable
    """Vectorized log activity function"""

    @override
    def __init__(self, species: Iterable[ChemicalSpecies]):
        self.species = SpeciesCollection(species)
        self.O2_index = self.get_O2_index()

        log_activity_funcs: list[Callable] = [
            to_hashable(species_.activity.log_activity) for species_ in species
        ]

        def apply_log_activity(
            index: Integer[Array, ""],
            temperature: Float[Array, "..."],
            pressure: Float[Array, "..."],
        ) -> Float[Array, "..."]:
            return lax.switch(index, log_activity_funcs, temperature, pressure)

        self.vmap_log_activity = eqx.filter_vmap(
            apply_log_activity, in_axes=(0, None, None), out_axes=-1
        )

        logger.info(
            "Creating %s: %s",
            self.__class__.__name__,
            tuple(str(species) for species in self.species),
        )

    @classmethod
    def create(cls, species: str | Iterable[str]) -> Self:
        """Creates an instance.

        Args:
            species: A single gas species name or iterable of names

        Returns:
            An instance
        """
        species_collection: SpeciesCollection[ChemicalSpecies] = build_species_collection(
            species, lambda hill: ChemicalSpecies.create_gas(hill, state=GAS_STATE)
        )

        return cls(species_collection)

    def get_O2_index(self) -> NpFloat:
        """Gets the species index corresponding to diatomic oxygen.

        Note:
            This returns a float array for type consistency.

        Returns:
            Index of diatomic oxygen, or ``np.nan`` if diatomic oxygen is not in the species
        """
        for nn, species_ in enumerate(self.species):
            if species_.data.hill_formula == "O2":
                # logger.debug("Found O2 at index = %d", nn)
                return np.array(nn, dtype=float)

        return np.array(np.nan, dtype=float)

    @override
    def output(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_stability: Float[Array, "... n_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, "..."],
        log_background_molar_mass: Float[Array, "..."] = jnp.asarray(0.0),
        log_background_mass: Float[Array, "..."] = jnp.asarray(-jnp.inf),
    ) -> GasPhaseOutput:
        r"""Constructs a jittable output helper object for phase-level and species-level properties.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_stability: Log stability of each species in the phase
            temperature: Temperature in K
            pressure: Pressure in bar
            log_background_molar_mass: Log molar mass of the background component in
                kg mol\ :sup:`-1`. Defaults to ``0.0`` (dummy value of 1 kg mol\ :sup:`-1`);
                only meaningful when ``log_background_mass`` is finite.
            log_background_mass: Log mass of the background component in kg. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            An output helper object for accessing and manipulating output quantities in a
            structured, jittable form.
        """
        output: GasPhaseOutput = GasPhaseOutput(
            self,
            log_number_moles,
            log_stability,
            temperature,
            pressure,
            log_background_molar_mass,
            log_background_mass,
        )

        return output

    # @override
    # def output(
    #     self,
    #     log_number_moles: Float[Array, "... n_species"],
    #     log_stability: Float[Array, "... n_species"],
    #     temperature: Float[Array, "..."],
    #     pressure: Float[Array, "..."],
    #     log_background_molar_mass: Float[Array, "..."] = jnp.asarray(0.0),
    #     log_background_mass: Float[Array, ""] = jnp.asarray(-jnp.inf),
    # ) -> dict[str, Any]:
    #     r"""Outputs phase-level and species-level properties in a human-readable format.

    #     Single conversion boundary: JAX -> NumPy -> dict

    #     Args:
    #         log_number_moles: Log number of moles of each species in the phase
    #         log_stability: Log stability of each species in the phase
    #         temperature: Temperature in K
    #         pressure: Pressure in bar
    #         log_background_molar_mass: Log molar mass of the background component in
    #             kg mol\ :sup:`-1`. Defaults to ``0.0`` (i.e., a dummy value of 1 kg mol\ :sup:`-1`)
    #             ; only meaningful when ``log_background_mass`` is finite.
    #         log_background_mass: Log mass of the background component in kg. Defaults to negative
    #             infinity (i.e., no background component).

    #     Returns:
    #         A dictionary containing phase-level and species-level properties
    #     """
    #     out: dict[str, Any] = super().output(
    #         log_number_moles,
    #         log_stability,
    #         temperature,
    #         pressure,
    #         log_background_molar_mass,
    #         log_background_mass,
    #     )

    # log_background_moles: Float[Array, "..."] = log_background_mass - log_background_molar_mass

    # out["phase"]["pressure_bar"] = jnp.squeeze(jnp.asarray(pressure))
    # out["species"]["partial_pressure_bar"] = dict(
    #    zip(
    #        self.species_names,
    #        jnp.asarray(pressure)
    #        * jnp.exp(self.get_log_mole_fraction(log_number_moles, log_background_moles)).T,
    #    )
    # )
    # number_moles: NpFloat = np.squeeze(
    #    jnp.exp(self.get_log_phase_moles(log_number_moles, log_background_moles))
    # )
    # out["phase"]["volume_m3"] = number_moles * GAS_CONSTANT_BAR * temperature / pressure

    # TODO: Need JAX compatible switch
    # # fO2 calculation: if O2 is present, compute fO2 relative to the iron-wustite (IW) buffer
    # if not jnp.isnan(self.O2_index):
    #     log10_fugacity = jnp.log10(out["species"]["activity"]["O2_g"])
    #     buffer: RedoxBufferProtocol = IronWustiteBuffer()
    #     # Shift at 1 bar
    #     buffer_at_one_bar = jnp.asarray(buffer.log10_fugacity(temperature, 1.0))
    #     log10_shift_at_one_bar = log10_fugacity - buffer_at_one_bar
    #     # logger.debug("log10_shift_at_1bar = %s", log10_shift_at_one_bar)
    #     out["phase"]["log10dIW_1_bar"] = log10_shift_at_one_bar
    #     # Shift at actual pressure
    #     buffer_at_P = jnp.asarray(buffer.log10_fugacity(temperature, pressure))
    #     log10_shift_at_P = log10_fugacity - buffer_at_P
    #     # logger.debug("log10_shift_at_P = %s", log10_shift_at_P)
    #     out["phase"]["log10dIW_P"] = log10_shift_at_P

    # return out


class MeltPhase(BasePhase[SpeciesProtocol]):
    """Multicomponent silicate melt with optionally dissolved volatiles.

    The melt phase can optionally treat dissolved and/or condensed species as additional to the
    solvent (the bulk melt mass that volatiles dissolve into). When enabled, these species are
    included on top of the solvent when computing phase-level totals (mass, moles, molar mass,
    and derived fractions). This is useful in the dilute limit, where their contribution to the
    total phase mass is negligible, or when the background component already accounts for them.

    Note that this only affects phase-level aggregations — per-species thermodynamic properties
    (e.g., activity) are always computed from the raw species amounts and are unaffected. Total
    mass and mole conservation is therefore always maintained by the solver.
    """

    vmap_log_activity: Callable
    """Vectorized log activity function"""

    @override
    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        log_activity_funcs: list[Callable] = [
            to_hashable(species_.activity.log_activity) for species_ in species
        ]

        def apply_log_activity(
            index: Integer[Array, ""],
            temperature: Float[Array, "..."],
            pressure: Float[Array, "..."],
        ) -> Float[Array, ""]:
            return lax.switch(index, log_activity_funcs, temperature, pressure)

        self.vmap_log_activity = eqx.filter_vmap(
            apply_log_activity, in_axes=(0, None, None), out_axes=-1
        )

        logger.info(
            "Creating %s: %s",
            self.__class__.__name__,
            tuple(str(species) for species in self.species),
        )

    @classmethod
    def create(cls, species: str | Iterable[str], include_in_phase_mass: bool = True) -> Self:
        """Creates an instance.

        Args:
            species: A single melt species name or iterable of names
            include_in_phase_mass: Whether to include species in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            An instance
        """
        species_collection: SpeciesCollection[SpeciesProtocol] = build_species_collection(
            species,
            lambda hill: ChemicalSpecies.create_condensed(
                hill, state=LIQUID_STATE, include_in_phase_mass=include_in_phase_mass
            ),
        )

        return cls(species_collection)


class SolidPhase(BasePhase[SpeciesProtocol]):
    """Multicomponent silicate solid."""

    vmap_log_activity: Callable
    """Vectorized log activity function"""

    @override
    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.species = SpeciesCollection(species)
        log_activity_funcs: list[Callable] = [
            to_hashable(species_.activity.log_activity) for species_ in species
        ]

        def apply_log_activity(
            index: Integer[Array, ""],
            temperature: Float[Array, "..."],
            pressure: Float[Array, "..."],
        ) -> Float[Array, ""]:
            return lax.switch(index, log_activity_funcs, temperature, pressure)

        self.vmap_log_activity = eqx.filter_vmap(
            apply_log_activity, in_axes=(0, None, None), out_axes=-1
        )

        logger.info(
            "Creating %s: %s",
            self.__class__.__name__,
            tuple(str(species) for species in self.species),
        )

    @classmethod
    def create(cls, species: str | Iterable[str], include_in_phase_mass: bool = True) -> Self:
        """Creates an instance.

        Args:
            species: A single solid species name or iterable of names
            include_in_phase_mass: Whether to include species in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            An instance
        """
        species_collection: SpeciesCollection[ChemicalSpecies] = build_species_collection(
            species,
            lambda hill: ChemicalSpecies.create_condensed(
                hill, state=SOLID_STATE, include_in_phase_mass=include_in_phase_mass
            ),
        )

        return cls(species_collection)


class PurePhase(BasePhase[ChemicalSpecies]):
    """Single-species, unity-activity phase (e.g., a pure mineral, ice, or liquid).

    The activity of the species is fixed at unity by definition, so only one species is permitted.
    """

    @override
    def __init__(self, species: Iterable[ChemicalSpecies]):
        # TODO: Accept a non-iterable since only a single species is consistent with a pure phase?
        self.species = SpeciesCollection(species)

        if self.species.number_species != 1:
            raise ValueError(
                f"PurePhase must contain exactly one species, got {len(self.species)}"
            )

        logger.info(
            "Creating %s: %s",
            self.__class__.__name__,
            tuple(str(species) for species in self.species),
        )

    @property
    def name(self) -> str:
        """Name of the pure phase, given by the single species it contains."""
        return self.species.species_names[0]

    @classmethod
    def create(cls, species: str, state: str = SOLID_STATE) -> Self:
        """Creates an instance.

        Args:
            species: A single species Hill formula
            state: State of aggregation. Defaults to
                :const:`atmodeller.constants.SOLID_STATE`.

        Returns:
            An instance
        """
        species_collection: SpeciesCollection[ChemicalSpecies] = build_species_collection(
            species, lambda hill: ChemicalSpecies.create_condensed(hill, state=state)
        )

        return cls(species_collection)

    @override
    def vmap_log_activity(
        self,
        species_indices: Integer[Array, " n_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, "..."],
    ) -> Float[Array, "... n_species"]:
        del species_indices
        del temperature
        del pressure

        return jnp.zeros(self.species.number_species)
