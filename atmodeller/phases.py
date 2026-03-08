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
from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Generic, Self, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp
from jaxmod.constants import GAS_CONSTANT_BAR
from jaxmod.type_aliases import NpFloat, NpInt
from jaxmod.utils import safe_exp, to_hashable
from jaxtyping import Array, Float, Integer
from molmass import Formula
from scipy.special import logsumexp as sp_logsumexp

from atmodeller import override
from atmodeller.constants import GAS_STATE, LIQUID_STATE, SOLID_STATE
from atmodeller.containers import (
    ChemicalSpecies,
    SpeciesCollection,
    TSpecies_co,
    get_formula_matrix,
)
from atmodeller.interfaces import RedoxBufferProtocol, SpeciesProtocol
from atmodeller.thermodata._redox_buffers import IronWustiteBuffer

logger: logging.Logger = logging.getLogger(__name__)


def _build_species_collection(
    species: str | Iterable[str], factory: Callable[[str], TSpecies_co]
) -> SpeciesCollection[TSpecies_co]:
    """Normalizes input and builds a species collection using a factory.

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
    """Base class for all phases."""

    species: SpeciesCollection[TSpecies_co]
    """Collection of species in the phase"""

    @abstractmethod
    def __init__(self, species: Iterable[TSpecies_co]):
        """Initializes a phase with a collection of species.

        Args:
            species: An iterable of species belonging to the phase
        """

    # Abstract, but commented out because vmap_log_activity may be an instance attribute assigned
    # in __init_, which does not satisfy the ABC mechanism.
    # @abstractmethod
    def vmap_log_activity(
        self,
        species_indices: Integer[Array, " n_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, "..."],
    ) -> Float[Array, "... n_species"]:
        """Applies the log activity function for each species in the phase.

        Args:
            species_indices: Integer array of shape (n_species,) containing the indices of the
                species in the phase. This is passed to the log activity function to identify
                which species' activity to compute.
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log activity of each species in the phase
        """
        del species_indices
        del temperature
        del pressure

        raise NotImplementedError("vmap_log_activity must be implemented by subclasses")

    @classmethod
    def empty(cls) -> Self:
        """Returns an empty phase instance."""
        return cls([])

    @property
    def species_names(self) -> tuple[str, ...]:
        """List of species names in the phase."""
        return self.species.species_names

    def get_log_activity(
        self,
        log_number_moles: Float[Array, "... n_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, "..."],
        log_background_moles: Float[Array, ""] = jnp.array(-jnp.inf),
    ) -> Float[Array, "... n_species"]:
        """Gets the log activity of each species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            temperature: Temperature in K
            pressure: Pressure in bar
            log_background_moles: Log moles of the background component. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            Log activity of each species in the phase
        """
        if self.species.number_species == 0:
            return jnp.zeros_like(log_number_moles)

        # Log activity coefficient of pure species
        log_activity: Float[Array, "... n_species"] = self.vmap_log_activity(
            jnp.arange(self.species.number_species), temperature, pressure
        )

        log_mole_fraction: Float[Array, "... n_species"] = self.get_log_mole_fraction(
            log_number_moles, log_background_moles
        )

        log_activity: Float[Array, "... n_species"] = log_activity + log_mole_fraction

        return log_activity

    def get_log_activity_with_stability(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_stability: Float[Array, "... n_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, "..."],
        log_background_moles: Float[Array, ""] = jnp.array(-jnp.inf),
    ) -> Float[Array, "... n_species"]:
        """Gets the log activity of each species in the phase, accounting for stability.

        Unstable species are assigned a log activity of negative infinity.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_stability: Log stability of each species in the phase
            temperature: Temperature in K
            pressure: Pressure in bar
            log_background_moles: Log moles of the background component. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            Log activity of each species in the phase, with unstable species set to negative
                infinity
        """
        log_activity: Float[Array, "... n_species"] = self.get_log_activity(
            log_number_moles, temperature, pressure, log_background_moles
        )
        log_stability_masked: Float[Array, "... n_species"] = jnp.where(
            self.species.active_stability, log_stability, -jnp.inf
        )
        log_activity_with_stability: Float[Array, "... n_species"] = log_activity - safe_exp(
            log_stability_masked
        )

        return log_activity_with_stability

    def apply_phase_mass_mask(
        self, log_array: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Zeros out (in log-space) species that do not contribute to phase-level aggregations.

        Species with ``include_in_phase_mass=False`` are replaced with ``-inf``; all others are
        passed through unchanged.

        Args:
            log_array: Log-space values whose entries for non-contributing species are to be masked

        Returns:
            The input array with non-contributing species set to ``-inf``.
        """
        log_mask: Float[Array, " n_species"] = jnp.where(
            self.species.phase_mass_mask, 0.0, -jnp.inf
        )

        return log_array + log_mask

    def get_log_mass(
        self, log_number_moles: Float[Array, "... n_species"]
    ) -> Float[Array, "... n_species"]:
        """Gets the log mass of each species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase

        Returns:
            Log mass of each species in the phase in kg
        """
        log_mass: Float[Array, "... n_species"] = log_number_moles + jnp.log(
            self.species.molar_masses
        )

        return log_mass

    def get_log_phase_mass(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_background_mass: Float[Array, "..."] = jnp.asarray(-jnp.inf),
    ) -> Float[Array, "... 1"]:
        """Gets the log mass of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_background_mass: Log mass of the background component in kg. Defaults to negative
                infinity (i.e., no background mass).

        Returns:
            Log mass of the phase in kg
        """
        log_mass: Float[Array, "... n_species"] = self.get_log_mass(log_number_moles)
        log_mass = self.apply_phase_mass_mask(log_mass)

        # jnp.append without axis flattens its inputs, collapsing any batch dimensions.
        # Instead, broadcast the background scalar to the batch shape and concatenate
        # along the species axis so the result remains (... n_species+1).
        background: Float[Array, "... 1"] = jnp.broadcast_to(
            log_background_mass, log_mass.shape[:-1]
        )[..., jnp.newaxis]
        log_mass_with_background: Float[Array, "... n_species_plus_1"] = jnp.concatenate(
            [log_mass, background], axis=-1
        )

        return logsumexp(log_mass_with_background, axis=-1, keepdims=True)

    def get_log_mass_fraction(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_background_mass: Float[Array, "..."] = jnp.asarray(-jnp.inf),
    ) -> Float[Array, "... n_species"]:
        """Gets the log mass fraction of the species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_background_mass: Log mass of the background component in kg. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            Log mass fraction of each species in the phase
        """
        log_mass: Float[Array, "... n_species"] = self.get_log_mass(log_number_moles)
        log_phase_mass: Float[Array, "... 1"] = self.get_log_phase_mass(
            log_number_moles, log_background_mass
        )
        log_mass_fraction: Float[Array, "... n_species"] = log_mass - log_phase_mass
        # jax.debug.print("log_mass_fraction = {out}", out=log_mass_fraction)

        return log_mass_fraction

    def get_log_phase_moles(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_background_moles: Float[Array, "..."] = jnp.asarray(-jnp.inf),
    ) -> Float[Array, "... 1"]:
        """Gets the log moles of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_background_moles: Log moles of the background component. Defaults to negative
                infinity (i.e., no background moles).

        Returns:
            Log moles of the phase in mol
        """
        log_number_moles = self.apply_phase_mass_mask(log_number_moles)

        # jnp.append without axis flattens its inputs, collapsing any batch dimensions.
        # Instead, broadcast the background scalar to the batch shape and concatenate
        # along the species axis so the result remains (... n_species+1).
        background: Float[Array, "... 1"] = jnp.broadcast_to(
            log_background_moles, log_number_moles.shape[:-1]
        )[..., jnp.newaxis]
        log_moles_with_background: Float[Array, "... n_species_plus_1"] = jnp.concatenate(
            [log_number_moles, background], axis=-1
        )

        return logsumexp(log_moles_with_background, axis=-1, keepdims=True)

    def get_log_mole_fraction(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_background_moles: Float[Array, "..."] = jnp.array(-jnp.inf),
    ) -> Float[Array, "... n_species"]:
        """Gets the log mole fraction of the species in the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_background_moles: Log moles of the background component in mol. Defaults to
                negative infinity (i.e., no background component).

        Returns:
            Log mole fraction of each species in the phase
        """
        log_phase_moles: Float[Array, "... 1"] = self.get_log_phase_moles(
            log_number_moles, log_background_moles
        )
        log_mole_fraction: Float[Array, "... n_species"] = log_number_moles - log_phase_moles

        return log_mole_fraction

    def get_log_phase_molar_mass(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_background_molar_mass: Float[Array, "..."] = jnp.asarray(0.0),
        log_background_mass: Float[Array, ""] = jnp.asarray(-jnp.inf),
    ) -> Float[Array, "... 1"]:
        r"""Gets the log molar mass of the phase.

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_background_molar_mass: Log molar mass of the background component in
                kg mol\ :sup:`-1`. Defaults to ``0.0`` (i.e., a dummy value of
                1 kg mol\ :sup:`-1`); only meaningful when ``log_background_mass`` is finite.
            log_background_mass: Log mass of the background component in kg. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            Log molar mass of the phase in kg mol\ :sup:`-1`
        """
        log_phase_mass: Float[Array, "... 1"] = self.get_log_phase_mass(
            log_number_moles, log_background_mass
        )
        log_background_moles: Float[Array, "..."] = log_background_mass - log_background_molar_mass
        log_number_total: Float[Array, "... 1"] = self.get_log_phase_moles(
            log_number_moles, log_background_moles
        )

        return log_phase_mass - log_number_total

    def output(
        self,
        log_number_moles: Float[Array, "... n_species"],
        log_stability: Float[Array, "... n_species"],
        temperature: Float[Array, "..."],
        pressure: Float[Array, "..."],
        log_background_molar_mass: Float[Array, "..."] = jnp.asarray(0.0),
        log_background_mass: Float[Array, ""] = jnp.asarray(-jnp.inf),
    ) -> dict[str, Any]:
        r"""Outputs phase-level and species-level properties in a human-readable format.

        Single conversion boundary: JAX -> NumPy -> dict

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_stability: Log stability of each species in the phase.
            temperature: Temperature in K.
            pressure: Pressure in bar.
            log_background_molar_mass: Log molar mass of the background component in
                kg mol\ :sup:`-1`. Defaults to ``0.0`` (i.e., a dummy value of 1 kg mol\ :sup:`-1`)
                ; only meaningful when ``log_background_mass`` is finite.
            log_background_mass: Log mass of the background component in kg. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            A dictionary containing phase-level and species-level properties
        """
        log_background_moles: Float[Array, "..."] = log_background_mass - log_background_molar_mass

        # Elements
        formula_matrix_: NpInt = get_formula_matrix(self.species)
        log_stoich_matrix: NpFloat = np.where(
            formula_matrix_ > 0, np.log(formula_matrix_), -np.inf
        )
        log_terms: NpFloat = np.asarray(log_number_moles[..., None, :] + log_stoich_matrix)
        log_elements: NpFloat = cast(NpFloat, sp_logsumexp(log_terms, axis=-1))

        log_phase_mass: Float[Array, "... 1"] = self.get_log_phase_mass(
            log_number_moles, log_background_mass
        )

        # Sum of all species mass (no background, no mask), for comparison with the phase total.
        # For a true dilute system, this provides a check that the dissolved content is in the
        # dilute approximation. Otherwise, depending on the assumptions and objectives of the user,
        # it provides a metric of how much the tracked species contribute to the total phase mass.
        log_species_mass_sum: Float[Array, "... 1"] = logsumexp(
            self.get_log_mass(log_number_moles), axis=-1, keepdims=True
        )

        out: dict[str, Any] = {
            "phase": {
                "number_moles": np.squeeze(
                    np.exp(self.get_log_phase_moles(log_number_moles, log_background_moles))
                ),
                "mass_kg": np.squeeze(np.exp(log_phase_mass)),
                "molar_mass_kg_per_mol": np.squeeze(
                    np.exp(
                        self.get_log_phase_molar_mass(
                            log_number_moles, log_background_molar_mass, log_background_mass
                        )
                    )
                ),
                "background_molar_mass_kg_per_mol": np.exp(log_background_molar_mass),
                "background_mass_kg": np.exp(log_background_mass),
                "species_to_phase_mass_ratio": np.squeeze(
                    np.exp(log_species_mass_sum - log_phase_mass)
                ),
            },
            "elements": {
                "number_moles": dict(zip(self.species.unique_elements, np.exp(log_elements).T)),
                "mass_kg": dict(
                    zip(
                        self.species.unique_elements,
                        np.exp(log_elements + np.log(self.species.element_molar_masses)).T,
                    )
                ),
            },
            "species": {
                "number_moles": dict(zip(self.species_names, np.exp(log_number_moles).T)),
                "mole_fraction": dict(
                    zip(
                        self.species_names,
                        np.exp(
                            self.get_log_mole_fraction(log_number_moles, log_background_moles)
                        ).T,
                    )
                ),
                "mass_kg": dict(
                    zip(self.species_names, np.exp(self.get_log_mass(log_number_moles)).T)
                ),
                "mass_fraction": dict(
                    zip(
                        self.species_names,
                        np.exp(
                            self.get_log_mass_fraction(log_number_moles, log_background_mass)
                        ).T,
                    )
                ),
                "activity": dict(
                    zip(
                        self.species_names,
                        np.exp(
                            self.get_log_activity_with_stability(
                                log_number_moles,
                                log_stability,
                                temperature,
                                pressure,
                                log_background_moles,
                            )
                        ).T,
                    )
                ),
                "include_in_phase_mass": dict(
                    zip(self.species_names, self.species.phase_mass_mask.astype(bool))
                ),
            },
        }

        return out

    def __len__(self) -> int:
        return len(self.species)


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
        species_collection: SpeciesCollection[ChemicalSpecies] = _build_species_collection(
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
        log_background_mass: Float[Array, ""] = jnp.asarray(-jnp.inf),
    ) -> dict[str, Any]:
        r"""Outputs phase-level and species-level properties in a human-readable format.

        Single conversion boundary: JAX -> NumPy -> dict

        Args:
            log_number_moles: Log number of moles of each species in the phase
            log_stability: Log stability of each species in the phase
            temperature: Temperature in K
            pressure: Pressure in bar
            log_background_molar_mass: Log molar mass of the background component in
                kg mol\ :sup:`-1`. Defaults to ``0.0`` (i.e., a dummy value of 1 kg mol\ :sup:`-1`)
                ; only meaningful when ``log_background_mass`` is finite.
            log_background_mass: Log mass of the background component in kg. Defaults to negative
                infinity (i.e., no background component).

        Returns:
            A dictionary containing phase-level and species-level properties
        """
        out: dict[str, Any] = super().output(
            log_number_moles,
            log_stability,
            temperature,
            pressure,
            log_background_molar_mass,
            log_background_mass,
        )

        log_background_moles: Float[Array, "..."] = log_background_mass - log_background_molar_mass

        out["phase"]["pressure_bar"] = np.squeeze(np.asarray(pressure))
        out["species"]["partial_pressure_bar"] = dict(
            zip(
                self.species_names,
                np.asarray(pressure)
                * np.exp(self.get_log_mole_fraction(log_number_moles, log_background_moles)).T,
            )
        )
        number_moles: NpFloat = np.squeeze(
            np.exp(self.get_log_phase_moles(log_number_moles, log_background_moles))
        )
        out["phase"]["volume_m3"] = number_moles * GAS_CONSTANT_BAR * temperature / pressure

        # fO2 calculation: if O2 is present, compute fO2 relative to the iron-wustite (IW) buffer
        if not jnp.isnan(self.O2_index):
            log10_fugacity = np.log10(out["species"]["activity"]["O2_g"])
            buffer: RedoxBufferProtocol = IronWustiteBuffer()
            # Shift at 1 bar
            buffer_at_one_bar = np.asarray(buffer.log10_fugacity(temperature, 1.0))
            log10_shift_at_one_bar = log10_fugacity - buffer_at_one_bar
            # logger.debug("log10_shift_at_1bar = %s", log10_shift_at_one_bar)
            out["phase"]["log10dIW_1_bar"] = log10_shift_at_one_bar
            # Shift at actual pressure
            buffer_at_P = np.asarray(buffer.log10_fugacity(temperature, pressure))
            log10_shift_at_P = log10_fugacity - buffer_at_P
            # logger.debug("log10_shift_at_P = %s", log10_shift_at_P)
            out["phase"]["log10dIW_P"] = log10_shift_at_P

        return out


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
        species_collection: SpeciesCollection[SpeciesProtocol] = _build_species_collection(
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
        species_collection: SpeciesCollection[ChemicalSpecies] = _build_species_collection(
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
        species_collection: SpeciesCollection[ChemicalSpecies] = _build_species_collection(
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
