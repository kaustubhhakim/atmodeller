# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Base phase container classes for thermodynamic equilibrium calculations"""

from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Generic, Self, TypeVar, cast

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxmod.utils import safe_exp
from jaxtyping import Array, Bool, Float, Integer
from molmass import Formula

from atmodeller.containers import SpeciesCollection, get_formula_matrix
from atmodeller.interfaces import TSpecies_co


def build_species_collection(
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
        log_mass_with_background: Float[Array, "... n_species_plus_one"] = jnp.concatenate(
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
        log_moles_with_background: Float[Array, "... n_species_plus_one"] = jnp.concatenate(
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
        log_background_mass: Float[Array, "..."] = jnp.asarray(-jnp.inf),
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
        log_background_mass: Float[Array, "..."] = jnp.asarray(-jnp.inf),
    ) -> "PhaseOutput[Self]":
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
        output: PhaseOutput[Self] = PhaseOutput(
            self,
            log_number_moles,
            log_stability,
            temperature,
            pressure,
            log_background_molar_mass,
            log_background_mass,
        )

        return output

    def __len__(self) -> int:
        return len(self.species)


TPhase_co = TypeVar("TPhase_co", bound=BasePhase, covariant=True)


class PhaseOutput(eqx.Module, Generic[TPhase_co]):
    r"""Output helper class for phase-level and species-level results.

    Provides convenient accessors and computed properties for phase output quantities (e.g., mass,
    mole fractions, activities, element totals) in a structured, jittable form. Designed to operate
    within JAX-jitted workflows, enabling efficient manipulation, transformation, and extraction of
    output data for downstream analysis or further computation. Arrays maintain batch dimensions to
    allow flexible use in both single-calculation and batched contexts.

    Args:
        phase: The phase instance associated with this output
        log_number_moles: Log number of moles for each species
        log_stability: Log stability for each species
        temperature: Temperature in K
        pressure: Pressure in bar
        log_background_molar_mass: Log molar mass of the background component in
            kg mol\ :sup:`-1`
        log_background_mass: Log mass of the background component in kg
    """

    phase: TPhase_co
    log_number_moles: Float[Array, "#n_batch n_species"]
    log_stability: Float[Array, "#n_batch n_species"]
    temperature: Float[Array, "#n_batch 1"]
    pressure: Float[Array, "#n_batch 1"]
    log_background_molar_mass: Float[Array, "#n_batch 1"]
    log_background_mass: Float[Array, "#n_batch 1"]

    @property
    def include_in_mass_phase(self) -> Bool[Array, " n_species"]:
        """Boolean mask indicating which species to include in phase-level mass and derived
        aggregations."""
        return jnp.asarray(self.phase.species.phase_mass_mask.astype(bool))

    @property
    def formula_matrix(self) -> Integer[Array, "n_elements n_species"]:
        return jnp.asarray(get_formula_matrix(self.phase.species))

    @property
    def log_stoich_matrix(self) -> Float[Array, "n_element n_species"]:
        formula_matrix: Integer[Array, "n_elements n_species"] = self.formula_matrix
        return jnp.where(formula_matrix > 0, jnp.log(formula_matrix), -jnp.inf)

    # Background component outputs
    @property
    def log_background_number_moles(self) -> Float[Array, "... 1"]:
        return self.log_background_mass - self.log_background_molar_mass

    @property
    def background_mass(self) -> Float[Array, "... 1"]:
        return jnp.exp(self.log_background_mass)

    @property
    def background_molar_mass(self) -> Float[Array, "... 1"]:
        return jnp.exp(self.log_background_molar_mass)

    @property
    def background_number_moles(self) -> Float[Array, "... 1"]:
        return jnp.exp(self.log_background_number_moles)

    # Element outputs
    @property
    def log_element_number_moles(self) -> Float[Array, "#n_batch n_elements"]:
        log_terms: Array = self.log_number_moles[..., None, :] + self.log_stoich_matrix
        return cast(
            Float[Array, "#batch n_elements"], logsumexp(log_terms, axis=-1, keepdims=False)
        )

    @property
    def element_number_moles(self) -> Float[Array, "#n_batch n_elements"]:
        return jnp.exp(self.log_element_number_moles)

    @property
    def element_mass(self) -> Float[Array, "#n_batch n_elements"]:
        return jnp.exp(
            self.log_element_number_moles + jnp.log(self.phase.species.element_molar_masses)
        )

    # Phase outputs
    @property
    def phase_number_moles(self) -> Float[Array, "#n_batch 1"]:
        return jnp.exp(
            self.phase.get_log_phase_moles(
                self.log_number_moles, jnp.squeeze(self.log_background_number_moles)
            )
        )

    @property
    def log_phase_mass(self) -> Float[Array, "#n_batch 1"]:
        return self.phase.get_log_phase_mass(
            self.log_number_moles, jnp.squeeze(self.log_background_mass)
        )

    @property
    def phase_mass(self) -> Float[Array, "#n_batch 1"]:
        return jnp.exp(self.log_phase_mass)

    @property
    def phase_molar_mass(self) -> Float[Array, "#n_batch 1"]:
        return jnp.exp(
            self.phase.get_log_phase_molar_mass(
                self.log_number_moles,
                jnp.squeeze(self.log_background_molar_mass),
                jnp.squeeze(self.log_background_mass),
            )
        )

    @property
    def species_to_phase_mass_ratio(self) -> Float[Array, "#n_batch 1"]:
        """Mass of tracked species divided by total phase mass.

        Sum of all species mass (no background, no mask), for comparison with the phase total.
        For a true dilute system, this provides a check that the dissolved content is in the dilute
        approximation. Otherwise, depending on the assumptions and objectives of the user, it
        provides a metric of how much the tracked species contribute to the total phase mass.
        """
        log_species_mass_sum: Float[Array, "#n_batch 1"] = logsumexp(
            self.phase.get_log_mass(self.log_number_moles), axis=-1, keepdims=True
        )
        log_phase_mass: Float[Array, "#n_batch 1"] = self.phase.get_log_phase_mass(
            self.log_number_moles, jnp.squeeze(self.log_background_mass)
        )
        log_species_to_phase_mass_ratio: Float[Array, "#batch 1"] = (
            log_species_mass_sum - log_phase_mass
        )
        return jnp.exp(log_species_to_phase_mass_ratio)

    # Species outputs
    @property
    def species_activity(self) -> Float[Array, "#n_batch n_species"]:
        log_activity: Float[Array, "#n_batch n_species"] = (
            self.phase.get_log_activity_with_stability(
                self.log_number_moles,
                self.log_stability,
                jnp.squeeze(self.temperature),
                jnp.squeeze(self.pressure),
                jnp.squeeze(self.log_background_number_moles),
            )
        )
        return jnp.exp(log_activity)

    @property
    def species_mass_fraction(self) -> Float[Array, "#n_batch n_species"]:
        log_mass_fraction: Float[Array, "#n_batch n_species"] = self.phase.get_log_mass_fraction(
            self.log_number_moles, jnp.squeeze(self.log_background_mass)
        )
        return jnp.exp(log_mass_fraction)

    @property
    def species_mass(self) -> Float[Array, "#n_batch n_species"]:
        log_mass: Float[Array, "#n_batch n_species"] = self.phase.get_log_mass(
            self.log_number_moles
        )
        return jnp.exp(log_mass)

    @property
    def species_mole_fraction(self) -> Float[Array, "#n_batch n_species"]:
        log_mole_fraction: Float[Array, "#n_batch n_species"] = self.phase.get_log_mole_fraction(
            self.log_number_moles, jnp.squeeze(self.log_background_number_moles)
        )
        return jnp.exp(log_mole_fraction)

    @property
    def species_number_moles(self) -> Float[Array, "#n_batch n_species"]:
        return jnp.exp(self.log_number_moles)

    def asdict(self) -> dict[str, Any]:
        """Dictionary representation of the output for downstream processing or analysis.

        Returns:
            A dictionary containing phase-level and species-level properties
        """
        out: dict[str, Any] = {}
        out["phase"] = {
            "background_mass": self.background_mass,
            "background_number_moles": self.background_number_moles,
            "background_molar_mass": self.background_molar_mass,
            "mass": self.phase_mass,
            "number_moles": self.phase_number_moles,
            "molar_mass": self.phase_molar_mass,
            "species_to_phase_mass_ratio": self.species_to_phase_mass_ratio,
        }
        out["elements"] = {
            "names": self.phase.species.unique_elements,
            "mass": self.element_mass,
            "number_moles": self.element_number_moles,
        }
        out["species"] = {
            "names": self.phase.species_names,
            "activity": self.species_activity,
            "mass": self.species_mass,
            "mass_fraction": self.species_mass_fraction,
            "number_moles": self.species_number_moles,
            "mole_fraction": self.species_mole_fraction,
            "include_in_phase_mass": self.include_in_mass_phase,
        }

        return out
