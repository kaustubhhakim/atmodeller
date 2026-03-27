# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Containers"""

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Generic, Literal, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmod.solvers import RootFindParameters
from jaxmod.type_aliases import NpFloat, NpInt
from jaxmod.units import unit_conversion
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import CompositionItem, Formula

from atmodeller.constants import (
    DISSOLVED_STATE,
    GAS_STATE,
    LOG_NUMBER_MOLES_LOWER,
    LOG_NUMBER_MOLES_UPPER,
    LOG_STABILITY_LOWER,
    LOG_STABILITY_UPPER,
    SOLID_STATE,
    TAU,
)
from atmodeller.eos.core import IdealGas
from atmodeller.interfaces import (
    ActivityProtocol,
    ChemicalSpeciesData,
    FugacityConstraintProtocol,
    SolubilityProtocol,
    SpeciesProtocol,
    TSpecies_co,
)
from atmodeller.jaxhelper import as_j64, to_hashable
from atmodeller.solubility.core import NoSolubility
from atmodeller.thermodata import ActivityCoefficient, thermodynamic_data_source
from atmodeller.thermodata.core import (
    ThermodynamicCoefficients,
    thermodynamic_coefficients_dictionary,
)

logger: logging.Logger = logging.getLogger(__name__)


class ChemicalSpecies(eqx.Module):
    """Chemical species that participate in reactions

    Args:
        data: Chemical species data
        activity: Activity
        solve_for_stability: Solve for stability
        number_solution: Number of solution quantities
        thermo: Thermodynamic coefficients
        include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
            fraction aggregations.
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solve_for_stability: bool
    number_solution: int
    thermo: ThermodynamicCoefficients
    include_in_phase_mass: bool

    @classmethod
    def create(
        cls,
        formula: str,
        state: str,
        activity: ActivityProtocol,
        solve_for_stability: bool,
        number_solution: int,
        include_in_phase_mass: bool,
    ) -> "ChemicalSpecies":
        """Creates an instance.

        Args:
            formula: Formula
            state: State of aggregation, as typically defined by JANAF
            activity: Activity
            solve_for_stability: Solve for stability
            number_solution: Number of solution quantities
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations.

        Returns:
            An instance
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, state)

        try:
            thermo: ThermodynamicCoefficients = thermodynamic_coefficients_dictionary[
                species_data.name
            ]
        except KeyError:
            raise KeyError(
                f"{species_data.name} not available. "
                f"Available species are {thermodynamic_data_source.available_species()}"
            )

        return cls(
            species_data,
            activity,
            solve_for_stability,
            number_solution,
            thermo,
            include_in_phase_mass,
        )

    @classmethod
    def create_condensed(
        cls,
        formula: str,
        *,
        state: str = SOLID_STATE,
        activity: ActivityProtocol = ActivityCoefficient(),
        solve_for_stability: bool = True,
        include_in_phase_mass: bool = True,
    ) -> "ChemicalSpecies":
        """Creates a condensate with some default values.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :const:`~atmodeller.constants.SOLID_STATE`.
            activity: Activity. Defaults to unity activity.
            solve_for_stability: Solve for stability. Defaults to ``True``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            A condensed species
        """
        # Either both the number of moles and stability are solved for, or alternatively stability
        # can be enforced in which case the number of moles is irrelevant and there is nothing to
        # solve for.
        number_solution: int = 2 if solve_for_stability else 0

        return cls.create(
            formula, state, activity, solve_for_stability, number_solution, include_in_phase_mass
        )

    @classmethod
    def create_gas(
        cls,
        formula: str,
        *,
        state: str = GAS_STATE,
        activity: ActivityProtocol = IdealGas(),
        solve_for_stability: bool = False,
        include_in_phase_mass: bool = True,
    ) -> "ChemicalSpecies":
        """Creates a gas species with some default values.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :const:`~atmodeller.constants.GAS_STATE`.
            activity: Activity. Defaults to an ideal gas.
            solve_for_stability: Solve for stability. Defaults to ``False``.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            A gas species
        """
        # The number of moles is always solved for, and stability can be if desired, although
        # this is not typically done for gas species because they are usually stable over the range
        # of conditions considered and truncating the abundance can severely distort the results,
        # notably for O2.
        number_solution: int = 2 if solve_for_stability else 1

        return cls.create(
            formula, state, activity, solve_for_stability, number_solution, include_in_phase_mass
        )

    @property
    def name(self) -> str:
        return self.data.name

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Array:
        """Gets Gibbs energy over RT

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy over RT
        """
        return self.thermo.get_gibbs_over_RT(temperature)

    def __str__(self) -> str:
        return f"{self.data.name}: {self.activity.__class__.__name__}"


class ReservoirSpecies(eqx.Module):
    """Reservoir species

    A species that is not part of the reaction network but can exchange with it. For example, this
    can represent a volatile species dissolved in a melt that can exchange with the gas phase but
    is not explicitly included in the reaction network.

    Args:
        data: Chemical species data
        activity: Activity
        solubility: Solubility
        number_solution: Number of solution quantities
        include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
            fraction aggregations.
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solubility: SolubilityProtocol
    number_solution: int
    include_in_phase_mass: bool

    @classmethod
    def create_dissolved(
        cls,
        formula: str,
        *,
        activity: ActivityProtocol = ActivityCoefficient(),
        solubility: Optional[SolubilityProtocol] = None,
        include_in_phase_mass: bool = True,
    ) -> "ReservoirSpecies":
        """Creates a dissolved species with some default values.

        Args:
            formula: Formula
            activity: Activity. Defaults to unity activity.
            solubility: Solubility. Defaults to no solubility.
            include_in_phase_mass: Whether the species is included in phase-level mass, mole, and
                fraction aggregations. Defaults to ``True``.

        Returns:
            A dissolved species
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, state=DISSOLVED_STATE)

        if solubility is None:
            solubility = NoSolubility()
            number_solution: int = 0
        else:
            number_solution = 1

        return cls(species_data, activity, solubility, number_solution, include_in_phase_mass)

    @property
    def solve_for_stability(self) -> bool:
        """Always ``False`` — reservoir species have no stability variable."""
        return False

    def __str__(self) -> str:
        return f"{self.data.name}: {self.solubility.__class__.__name__}"


class SpeciesCollection(eqx.Module, Generic[TSpecies_co]):
    """Container of species and metadata"""

    species: tuple[TSpecies_co, ...]
    """Species in the collection"""
    species_names: tuple[str, ...]
    """Unique names of all species"""
    number_solution: int
    """Number of solution quantities, which cannot depend on traced quantities"""
    unique_elements_map: dict[str, int]
    """Mapping of unique element name to index in the unique elements array"""

    def __init__(self, species: Iterable[TSpecies_co]):
        self.species = tuple(species)
        self.species_names = tuple(species_.data.name for species_ in self)

        # Ensure number_solution is static
        self.number_solution = sum(species.number_solution for species in self)

        # Unique elements in species in alphabetical order
        elements: list[str] = []
        for species_ in self.species:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        self.unique_elements_map = {
            element: index for index, element in enumerate(sorted(unique_elements))
        }

        logger.debug(
            "Creating %s: %s", self.__class__.__name__, tuple(str(species) for species in self)
        )

    @property
    def active_stability(self) -> Bool[Array, "..."]:
        """Active stability mask"""
        return jnp.array([species.solve_for_stability for species in self], dtype=bool)

    @property
    def element_molar_masses(self) -> NpFloat:
        """Element molar masses for the unique elements in the species"""
        element_molar_masses: list[float] = []

        for element_ in self.unique_elements:
            mformula: Formula = Formula(element_)
            molar_mass: float = mformula.mass * unit_conversion.g_to_kg
            element_molar_masses.append(molar_mass)

        return np.array(element_molar_masses, dtype=float)

    @property
    def molar_masses(self) -> Float[Array, " species"]:
        """Molar masses for all species in the collection"""
        return jnp.array([species_.data.molar_mass for species_ in self], dtype=float)

    @property
    def number_elements(self) -> int:
        """Number of unique elements in the species"""
        return len(self.unique_elements)

    @property
    def number_species(self) -> int:
        """Number of species"""
        return len(self)

    @property
    def phase_mass_mask(self) -> Bool[Array, "..."]:
        """Mask for species included in phase-level mass, mole, and fraction aggregations"""
        return jnp.array([species.include_in_phase_mass for species in self], dtype=bool)

    @property
    def reaction_species(self) -> "SpeciesCollection[ChemicalSpecies]":
        """Reaction species collection"""
        return SpeciesCollection(
            [species for species in self if isinstance(species, ChemicalSpecies)]
        )

    @property
    def reaction_species_mask(self) -> Bool[Array, "..."]:
        """Mask for reaction species in the collection"""
        return jnp.array([isinstance(species_, ChemicalSpecies) for species_ in self], dtype=bool)

    @property
    def reservoir_species(self) -> "SpeciesCollection[ReservoirSpecies]":
        """Reservoir species collection"""
        return SpeciesCollection(
            [species for species in self if isinstance(species, ReservoirSpecies)]
        )

    @property
    def reservoir_species_mask(self) -> Bool[Array, "..."]:
        """Mask for reservoir species in the collection"""
        return jnp.array([isinstance(species_, ReservoirSpecies) for species_ in self], dtype=bool)

    @property
    def unique_elements(self) -> tuple[str, ...]:
        """Unique elements in species in alphabetical order"""
        return tuple(self.unique_elements_map.keys())

    def get_element_index(self, element: str) -> int:
        """Get the index of an element in the unique elements map"""
        # TODO: Returning a non-existent element with an index of -1 is a bit hacky
        return self.unique_elements_map.get(element, -1)

    def __getitem__(self, index: int) -> TSpecies_co:
        return self.species[index]

    def __iter__(self) -> Iterator[TSpecies_co]:
        return iter(self.species)

    def __len__(self) -> int:
        return len(self.species)

    def __str__(self) -> str:
        return str(tuple(str(species) for species in self.species))


def get_formula_matrix(species: SpeciesCollection[SpeciesProtocol]) -> NpInt:
    """Gets the formula matrix.

    Elements are given in rows and species in columns following the convention in :cite:t:`LKS17`.

    Args:
        species: Species collection

    Returns:
        Formula matrix
    """
    formula_matrix: NpInt = np.zeros(
        (len(species.unique_elements), species.number_species), dtype=int
    )

    for element_index, element in enumerate(species.unique_elements):
        for species_index, species_ in enumerate(species):
            count: int = 0
            try:
                count = species_.data.composition[element][0]
            except KeyError:
                count = 0
            formula_matrix[element_index, species_index] = count

    # logger.debug("formula_matrix = %s", formula_matrix)

    return formula_matrix


class FixedFugacityConstraint(eqx.Module):
    """A fixed fugacity constraint

    This must adhere to FugacityConstraintProtocol

    Args:
        fugacity: Fugacity in bar. Defaults to ``np.nan``.
    """

    fugacity: Array = eqx.field(converter=as_j64, default=np.nan)
    """Fugacity"""

    def active(self) -> Bool[Array, "..."]:
        """Active fugacity constraint

        Returns:
            ``True`` if the fugacity constraint is active, otherwise ``False``
        """
        return ~jnp.isnan(self.fugacity)

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Float[Array, "..."]:
        """Log fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        broadcast_shape: tuple[int, ...] = jnp.broadcast_shapes(
            jnp.shape(temperature), jnp.shape(pressure)
        )
        # jax.debug.print("broadcast_shape = {out}", out=broadcast_shape)

        return jnp.broadcast_to(jnp.log(self.fugacity), broadcast_shape)


class FugacityConstraintSet(eqx.Module):
    """A set of fugacity constraints

    These are applied as constraints on the gas activity.

    Args:
        constraints: Fugacity constraints
        species: Species collection
    """

    constraints: tuple[FugacityConstraintProtocol, ...]
    """Fugacity constraints"""
    species: SpeciesCollection
    """Species collection"""

    @classmethod
    def create(
        cls,
        species: SpeciesCollection,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
    ) -> "FugacityConstraintSet":
        """Creates an instance.

        Args:
            species: Species collection
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                ``None``.

        Returns:
            An instance
        """
        fugacity_constraints_: Mapping[str, FugacityConstraintProtocol] = (
            fugacity_constraints if fugacity_constraints is not None else {}
        )

        constraints: list[FugacityConstraintProtocol] = []

        for species_name in species.species_names:
            if species_name in fugacity_constraints_:
                constraints.append(fugacity_constraints_[species_name])
            else:
                # This is applied to all species, which is OK because it returns nans, meaning no
                # imposed activity/fugacity.
                constraints.append(FixedFugacityConstraint())

        return cls(tuple(constraints), species)

    def active(self) -> Bool[Array, "... species"]:
        """Active fugacity constraints

        Returns:
            Mask indicating whether fugacity constraints are active or not
        """
        # TODO: can remove broadcasting now using vmap?
        mask_list: list[Array] = [constraint.active() for constraint in self.constraints]
        broadcast_shape: tuple[int, ...] = jnp.broadcast_shapes(*[jnp.shape(m) for m in mask_list])

        active_constraints: Bool[Array, "... species"] = jnp.stack(
            [jnp.broadcast_to(m, broadcast_shape) for m in mask_list], axis=-1
        )
        # jax.debug.print("active fugacity constraints = {out}", out=active_constraints)

        return active_constraints

    def log_fugacity(
        self, temperature: ArrayLike, pressure: ArrayLike
    ) -> Float[Array, "... species"]:
        """Log fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        fugacity_funcs: list[Callable] = [
            to_hashable(constraint.log_fugacity) for constraint in self.constraints
        ]
        # jax.debug.print("fugacity_funcs = {out}", out=fugacity_funcs)

        # Temperature must be a float array to ensure branches have have identical types
        temperature = as_j64(temperature)

        def apply_fugacity(index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike) -> Array:
            # jax.debug.print("index = {out}", out=index)
            return lax.switch(index, fugacity_funcs, temperature, pressure)

        indices: Integer[Array, " species"] = jnp.arange(len(self.constraints))
        vmap_fugacity: Callable = eqx.filter_vmap(
            apply_fugacity, in_axes=(0, None, None), out_axes=-1
        )
        log_fugacity: Float[Array, "... species"] = vmap_fugacity(indices, temperature, pressure)
        # jax.debug.print("log_fugacity = {out}", out=log_fugacity)

        return log_fugacity


class MassConstraintSet(eqx.Module):
    """A set of mass constraints

    Args:
        abundance_dict: Dictionary mapping element names to abundance (in moles) arrays. Note that
            all elements in the species collection must be included as keys in the dictionary and
            in the same order as the unique elements in the species collection. Elements for which
            there are no active constraints should be included with abundance values of NaN.
        species: Species collection
    """

    abundance_dict: dict[str, Array]
    """Abundance dictionary mapping element name to abundance array"""
    species: SpeciesCollection
    """Species collection"""

    @classmethod
    def create(
        cls,
        species: SpeciesCollection,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        units: Literal["mass", "moles"] = "mass",
    ) -> "MassConstraintSet":
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

        # Populate mass constraints. This accommodates mass constraints given as mass or moles of
        # species as well as elements.
        abundance_dict: dict[str, Array] = {}

        for element in species.unique_elements:
            element_sum: ArrayLike = 0
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

    # TODO: reinstate this later
    # def update_abundance(self, new_abundances: Mapping[str, ArrayLike]) -> "MassConstraintSet":
    #     """Updates the abundance with new values from a dictionary

    #     Args:
    #         new_abundances: Dictionary with new abundance values for some or all elements. The keys
    #             should be element names and the values should be the new abundance values in the
    #             same units as the original abundance. Original abundances that are not included in
    #             the ``new_abundance`` dictionary will be retained.

    #     Returns:
    #         A new MassConstraintSet with the updated abundance
    #     """
    #     abundance_updated: Array = self.abundance

    #     for element, new_value in new_abundances.items():
    #         element_index: int = self.species.get_element_index(element)
    #         # TODO: decide if squeezing is necessary or if the input should be required to have the
    #         # same shape as abundance
    #         abundance_updated = abundance_updated.at[..., element_index].set(
    #             jnp.squeeze(new_value)
    #         )

    #     mass_constaint_set_update: MassConstraintSet = eqx.tree_at(
    #         lambda c: c.abundance, self, abundance_updated
    #     )

    #     return mass_constaint_set_update

    def active(self) -> Bool[Array, "... elements"]:
        """Active mass constraints

        Returns:
            Mask indicating whether elemental mass constraints are active or not
        """
        return ~jnp.isnan(self.abundance())


class SolverParameters(RootFindParameters):
    """Solver parameters

    Args:
        solver: Solver. Defaults to :class:`optimistix.Newton`.
        atol: Absolute tolerance. Defaults to ``1.0e-6``.
        rtol: Relative tolerance. Defaults to ``1.0e-6``.
        linear_solver: Linear solver. Defaults to ``AutoLinearSolver(well_posed=False)``.
        norm: Norm. Defaults to :func:`optimistix.max_norm`.
        throw: How to report any failures. Defaults to ``False``.
        max_steps: The maximum number of steps the solver can take. Defaults to ``256``.
        jac: Whether to use forward- or reverse-mode autodifferentiation to compute the Jacobian.
            Can be either ``fwd`` or ``bwd``. Defaults to ``fwd``.
        max_starts: Maximum number of starts. Defaults to ``10``.
        retry_perturbation: Perturbation for retry. Defaults to ``30``.
        tau: Tau factor for species stability. Defaults to :const:`~atmodeller.constants.TAU`.
    """

    max_starts: int = 10
    """Maximum number of starts"""
    retry_perturbation: float = 20.0
    """Perturbation for retry, in this case for the log number of moles of a species"""
    tau: Array = eqx.field(converter=as_j64, default=TAU)
    """Tau factor for species stability"""

    def get_options(self, number_species: int) -> dict[str, Any]:
        """Gets the solver options.

        Args:
            number_species: Number of species

        Returns:
            Solver options
        """
        options: dict[str, Any] = {
            "lower": self._get_lower_bound(number_species),
            "upper": self._get_upper_bound(number_species),
            "jac": self.jac,
        }

        return options

    def _get_lower_bound(self, number_species: int) -> Float[Array, " dim"]:
        """Gets the lower bound for truncating the solution during the solve.

        Args:
            number_species: Number of species

        Returns:
            Lower bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(
            number_species, LOG_NUMBER_MOLES_LOWER, LOG_STABILITY_LOWER
        )

    def _get_upper_bound(self, number_species: int) -> Float[Array, " dim"]:
        """Gets the upper bound for truncating the solution during the solve.

        Args:
            number_species: Number of species

        Returns:
            Upper bound for truncating the solution during the solve
        """
        return self._get_hypercube_bound(
            number_species, LOG_NUMBER_MOLES_UPPER, LOG_STABILITY_UPPER
        )

    def _get_hypercube_bound(
        self, number_species: int, log_number_moles_bound: float, stability_bound: float
    ) -> Float[Array, " dim"]:
        """Gets the bound on the hypercube.

        Args:
            number_species: Number of species
            log_number_moles_bound: Bound on the log number of moles
            stability_bound: Bound on the stability

        Returns:
            Bound on the hypercube that contains the root
        """
        bound: Array = jnp.concatenate(
            (
                log_number_moles_bound * jnp.ones(number_species),
                stability_bound * jnp.ones(number_species),
            )
        )

        return bound
