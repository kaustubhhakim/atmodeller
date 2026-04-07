# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Parameter containers for thermodynamic calculations.

This module defines immutable, JAX-friendly parameter objects used by the solver:

- :class:`ActivityConstraintSet` stores per-species activity/fugacity constraints.
- :class:`MassConstraintSet` stores elemental abundance constraints in moles.
- :class:`Parameters` bundles state, constraints, and solver settings into one object.

Factory methods validate and normalize user inputs, while ``update`` methods return new instances
with leaf shapes kept stable for efficient JAX transformations, also within jitted workflows.
"""

from collections.abc import Callable, Mapping
from typing import Literal, Optional, Self, cast

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import lax
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import CompositionItem, Formula

from atmodeller.containers import FixedActivityConstraint, SolverParameters, SpeciesCollection
from atmodeller.interfaces import ActivityConstraintProtocol, SpeciesProtocol
from atmodeller.jax_utils import FloatArray, as_j64, get_batch_size, to_hashable
from atmodeller.reactions import ReactionSystem
from atmodeller.state import BaseThermodynamicState


class ActivityConstraintSet(eqx.Module):
    """A set of activity/fugacity constraints

    These are applied as constraints on the species' activity.

    Use :meth:`create` to construct a new instance and :meth:`update` to return an updated
    instance with modified activity/fugacity constraints.

    Args:
        constraints_dict: Dictionary mapping species names to activity constraint
        species: Species collection
    """

    constraints_dict: dict[str, ActivityConstraintProtocol]
    """Activity constraints dictionary mapping species name to activity constraint"""
    species: SpeciesCollection
    """Species collection"""

    @property
    def ordered_constraints(self) -> tuple[ActivityConstraintProtocol, ...]:
        """Activity constraints in the canonical species order.

        This explicit ordering is required for stable internal JAX operations. Relying on
        dictionary iteration for semantic species alignment is fragile across transformed code
        paths, so constraints are always materialized in ``species.species_names`` order.
        """
        return tuple(
            self.constraints_dict[species_name] for species_name in self.species.species_names
        )

    @classmethod
    def create(
        cls,
        species: SpeciesCollection,
        activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
    ) -> Self:
        """Creates an instance.

        Args:
            species: Species collection
            activity_constraints: Mapping of a species name and an activity constraint. Defaults to
                ``None``.

        Returns:
            An instance
        """
        activity_constraints_: Mapping[str, ActivityConstraintProtocol] = (
            activity_constraints if activity_constraints is not None else {}
        )

        constraints_dict: dict[str, ActivityConstraintProtocol] = {}

        for species_name in species.species_names:
            if species_name in activity_constraints_:
                constraints_dict[species_name] = activity_constraints_[species_name]
            else:
                # No imposed activity/fugacity since NaNs are returned by default.
                constraints_dict[species_name] = FixedActivityConstraint()

        # jax.debug.print("constraints_dict = {out}", out=constraints_dict)

        return cls(constraints_dict, species)

    def active(self) -> Bool[Array, "... species"]:
        """Active activity constraints

        Returns:
            Mask indicating whether activity constraints are active or not
        """
        active_constraints: Bool[Array, "... species"] = jnp.stack(
            [constraint.active() for constraint in self.ordered_constraints], axis=-1
        )

        return active_constraints

    def log_activity(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> Float[Array, "... species"]:
        """Log activity

        Args:
            temperature: Temperature (K)
            pressure: Pressure (bar)

        Returns:
            Log activity (dimensionless) or log fugacity referenced to 1 bar for gaseous species
        """
        activity_funcs: list[Callable] = [
            to_hashable(constraint.log_activity) for constraint in self.ordered_constraints
        ]
        # jax.debug.print("activity_funcs = {out}", out=activity_funcs)

        # Temperature must be a float array to ensure branches have have identical types
        temperature = as_j64(temperature)

        def apply_activity(index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike) -> Array:
            # jax.debug.print("index = {out}", out=index)
            return lax.switch(index, activity_funcs, temperature, pressure)

        indices: Integer[Array, " species"] = jnp.arange(len(self.ordered_constraints))
        vmap_activity: Callable = eqx.filter_vmap(
            apply_activity, in_axes=(0, None, None), out_axes=-1
        )
        log_activity: Float[Array, "... species"] = vmap_activity(indices, temperature, pressure)
        # jax.debug.print("log_activity = {out}", out=log_activity)

        return log_activity

    def update(self, new_constraints: Mapping[str, ActivityConstraintProtocol]) -> Self:
        """Updates the activity/fugacity constraints with new values from a dictionary

        Args:
            new_constraints: Dictionary with new constraint values for some or all species. The
                keys should be species names and the values should be the new constraint values.
                Original constraints that are not included in the ``new_constraints`` dictionary
                will be retained.

        Returns:
            An instance with updated constraints
        """
        constraints_dict: dict[str, ActivityConstraintProtocol] = dict(self.constraints_dict)

        for species_name, new_value in new_constraints.items():
            original_value: ActivityConstraintProtocol = constraints_dict[species_name]

            original_dynamic, _ = eqx.partition(original_value, eqx.is_array)
            new_dynamic, new_static = eqx.partition(new_value, eqx.is_array)

            # Keep leaf signatures stable to avoid unnecessary retracing under JAX transformations.
            new_dynamic_stable = jtu.tree_map(
                lambda new_leaf, original_leaf: jnp.broadcast_to(
                    as_j64(new_leaf), original_leaf.shape
                ),
                new_dynamic,
                original_dynamic,
            )
            constraints_dict[species_name] = eqx.combine(new_dynamic_stable, new_static)

        activity_constraint_set_updated: ActivityConstraintSet = eqx.tree_at(
            lambda c: c.constraints_dict, self, constraints_dict
        )

        return cast(Self, activity_constraint_set_updated)


class MassConstraintSet(eqx.Module):
    """A set of mass constraints

    Use :meth:`create` to construct a new instance and :meth:`update` to return an updated
    instance with modified abundance constraints.

    Args:
        abundance_dict: Dictionary mapping element names to abundance (in moles) arrays. All
            elements in the species collection must be included as keys in the dictionary and in
            the same order as the unique elements in the species collection. Elements for which
            there are no active constraints should be included with abundance values of NaN.
        species: Species collection
    """

    abundance_dict: dict[str, FloatArray]
    """Abundance dictionary mapping element name to abundance array"""
    species: SpeciesCollection
    """Species collection"""

    @classmethod
    def create(
        cls,
        species: SpeciesCollection,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        units: Literal["mass", "moles"] = "mass",
    ) -> Self:
        """Creates an instance.

        Args:
            species: Species collection
            mass_constraints: Mapping of element name and mass constraint in ``units``. Defaults to
                ``None`` to create an empty set of mass constraints.
            units: Units of ``mass_constraints``. Defaults to ``mass``.

        Returns:
            An instance
        """
        mass_constraints_: Mapping[str, ArrayLike] = (
            mass_constraints if mass_constraints is not None else {}
        )

        abundance_dict: dict[str, Array] = {}

        for element in species.unique_elements:
            element_sum: ArrayLike = 0
            # This accommodates mass constraints given as mass or moles of species as well as
            # elements.
            for species_, value_ in mass_constraints_.items():
                # Does the species formula contain the element? If not, skip to the next species.
                try:
                    element_composition: CompositionItem = Formula(species_).composition()[element]
                except KeyError:
                    continue
                # Always convert to moles for storage
                if units == "mass":
                    # value_ is in mass units, convert to moles
                    # To get moles: (mass of element in species) / (molar mass of element)
                    # But here, value_ is the mass of the species, so:
                    # moles of element = (mass of species * element_composition.fraction) /
                    # element molar mass
                    element_index: int = species.get_element_index(element)
                    element_molar_mass: float = species.element_molar_masses[element_index]
                    scale: float = element_composition.fraction / element_molar_mass
                elif units == "moles":
                    # element_composition.count is the atom count
                    # value_ is in moles of species, so moles of element = count * value_
                    scale = element_composition.count
                element_sum += scale * value_

            # All elements must be included as keys in the abundance dictionary, even if they
            # are not present in any constraints. In the latter case, the abundance is set to
            # NaN to indicate that the constraint is inactive.
            if jnp.any(element_sum != 0):
                abundance_dict[element] = as_j64(element_sum)
            else:
                abundance_dict[element] = as_j64(jnp.nan)

        return cls(abundance_dict, species)

    def abundance(self) -> Float[Array, " n_elements"]:
        """Abundance array constructed from the abundance dictionary

        .. warning::
            This method should only be called inside a vmapped context so the abundance arrays are
            correctly broadcast and the output array is always 1-D.

        Returns:
            Abundance array constructed from the abundance dictionary
        """
        arrays: list[Array] = [
            self.abundance_dict[element] for element in self.species.unique_elements
        ]
        abundance_array: Float[Array, "... n_elements"] = jnp.stack(arrays, axis=-1)
        # jax.debug.print("abundance_array = {out}", out=abundance_array)

        return abundance_array

    def abundance_mol(self, batch_size: int = 1) -> Float[Array, "#n_batch n_elements"]:
        """Abundance by moles for all elements with broadcasting to a specified batch size.

        Args:
            batch_size: Batch size to broadcast the abundance arrays to. Defaults to ``1``.

        Returns:
            Abundance by moles for all elements
        """
        arrays: list[Array] = []

        for element in self.species.unique_elements:
            arr: Array = self.abundance_dict[element]
            arr = jnp.broadcast_to(arr, (batch_size,) + arr.shape[1:])
            arrays.append(arr)

        abundance_array: Float[Array, "... n_elements"] = jnp.stack(arrays, axis=-1)
        # jax.debug.print("abundance_array = {out}", out=abundance_array)

        return abundance_array

    def abundance_mass(self, batch_size: int = 1) -> Float[Array, "#n_batch n_elements"]:
        """Abundance by mass for all elements with broadcasting to a specified batch size.

        Args:
            batch_size: Batch size to broadcast the abundance arrays to. Defaults to ``1``.

        Returns:
            Abundance by mass for all elements
        """
        return self.abundance_mol(batch_size) * self.species.element_molar_masses

    def update(self, new_abundances: Mapping[str, ArrayLike]) -> Self:
        """Updates the abundance with new values from a dictionary

        Note:
            Previously active constraints can be turned off by setting the abundance to ``NaN``
            in the ``new_abundances`` dictionary and previously inactive constraints can be turned
            on by setting the abundance to a non-``NaN`` value in the ``new_abundances``
            dictionary.

        Args:
            new_abundances: Dictionary with new abundance values for some or all elements. The keys
                should be element names and the values should be the new abundance values in moles.
                Original abundances that are not included in the ``new_abundance`` dictionary will
                be retained.

        Returns:
            An instance with updated abundances
        """
        abundance_dict: dict[str, Array] = dict(self.abundance_dict)

        for element, new_value in new_abundances.items():
            original_value: Array = abundance_dict[element]
            # Keep leaf signatures stable to avoid unnecessary retracing under JAX transforms.
            value_array: FloatArray = as_j64(new_value)
            abundance_dict[element] = jnp.broadcast_to(value_array, original_value.shape)

        mass_constraint_set_updated: MassConstraintSet = eqx.tree_at(
            lambda c: c.abundance_dict, self, abundance_dict
        )

        return cast(Self, mass_constraint_set_updated)

    def active(self) -> Bool[Array, "... elements"]:
        """Active mass constraints

        Returns:
            Mask indicating whether elemental mass constraints are active or not
        """
        return ~jnp.isnan(self.abundance())


class Parameters(eqx.Module):
    """Parameters

    Use :meth:`create` to construct a new instance and :meth:`update` to return an updated instance
    with modified activity/fugacity and mass constraints.

    Args:
        state: Thermodynamic state
        activity_constraints: Activity constraints
        mass_constraints: Mass constraints
        solver_parameters: Solver parameters
        batch_size: Batch size. Defaults to ``1``.
    """

    state: BaseThermodynamicState
    """Thermodynamic state"""
    activity_constraints: ActivityConstraintSet
    """Activity constraints"""
    mass_constraints: MassConstraintSet
    """Mass constraints"""
    solver_parameters: SolverParameters
    """Solver parameters"""
    batch_size: int = 1
    """Batch size"""

    @classmethod
    def create(
        cls,
        state: BaseThermodynamicState,
        activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance from a pre-built reaction system.

        Args:
            state: Thermodynamic state
            activity_constraints: Mapping of a species name and an activity constraint. Defaults to
                a new instance of :class:`ActivityConstraintSet`.
            mass_constraints: Mapping of element name and mass constraint in kg. Defaults to
                a new instance of :class:`MassConstraintSet`.
            solver_parameters: Solver parameters. Defaults to a new instance of
                :class:`atmodeller.containers.SolverParameters`.

        Returns:
            An instance
        """
        activity_constraints_: ActivityConstraintSet = ActivityConstraintSet.create(
            state.reaction_system.phase_system.species, activity_constraints
        )
        mass_constraints_: MassConstraintSet = MassConstraintSet.create(
            state.reaction_system.phase_system.species, mass_constraints
        )
        batch_size: int = get_batch_size((state, activity_constraints_, mass_constraints_))
        # jax.debug.print("batch_size (parameters) = {out}", out=batch_size)

        solver_parameters_: SolverParameters = (
            SolverParameters() if solver_parameters is None else solver_parameters
        )

        return cls(state, activity_constraints_, mass_constraints_, solver_parameters_, batch_size)

    @property
    def reaction_system(self) -> ReactionSystem:
        """Reaction system representing the thermodynamic state of the planetary body"""
        return self.state.reaction_system

    @property
    def species(self) -> SpeciesCollection[SpeciesProtocol]:
        """Species in the system"""
        return self.reaction_system.phase_system.species

    @property
    def species_names(self) -> tuple[str, ...]:
        """Species names in the system"""
        return self.species.species_names

    @property
    def element_names(self) -> tuple[str, ...]:
        """Unique elements in the system"""
        return self.reaction_system.species.unique_elements

    def update_constraints(
        self,
        *,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        activity_constraints: Optional[Mapping[str, ActivityConstraintProtocol]] = None,
    ) -> Self:
        """Updates the mass and activity/fugacity constraints of the parameters.

        New values are assumed to be broadcastable to the shapes of the existing fields. Keeping
        leaf shapes stable helps avoid unnecessary JAX recompilation, including in jitted
        workflows.

        Args:
            mass_constraints: New mass constraints. Defaults to ``None``.
            activity_constraints: New activity/fugacity constraints. Defaults to ``None``.

        Returns:
            Updated parameters
        """
        parameters_updated: Parameters = self

        if mass_constraints is not None:
            mass_constraints_updated: MassConstraintSet = self.mass_constraints.update(
                mass_constraints
            )
            parameters_updated = eqx.tree_at(
                lambda p: p.mass_constraints, parameters_updated, mass_constraints_updated
            )

        if activity_constraints is not None:
            activity_constraints_updated: ActivityConstraintSet = self.activity_constraints.update(
                activity_constraints
            )
            parameters_updated = eqx.tree_at(
                lambda p: p.activity_constraints, parameters_updated, activity_constraints_updated
            )

        return cast(Self, parameters_updated)

    def update_state(self, *args, **kwargs) -> Self:
        """Updates the thermodynamic state of the parameters.

        New values are assumed to be broadcastable to the shapes of the existing fields. Keeping
        leaf shapes stable helps avoid unnecessary JAX recompilation, including in jitted
        workflows.

        Args:
            *args: Positional arguments to pass to the ``update`` method of the thermodynamic state
            **kwargs: Keyword arguments to pass to the ``update`` method of the thermodynamic state

        Returns:
            Updated parameters
        """
        state_updated: BaseThermodynamicState = self.state.update(*args, **kwargs)
        parameters_updated = eqx.tree_at(lambda p: p.state, self, state_updated)

        return cast(Self, parameters_updated)
