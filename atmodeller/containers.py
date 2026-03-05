# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Containers"""

import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict
from typing import Any, Generic, Literal, Optional, TypeVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxmod.constants import GRAVITATIONAL_CONSTANT
from jaxmod.solvers import RootFindParameters
from jaxmod.type_aliases import NpArray, NpBool, NpFloat, NpInt
from jaxmod.units import unit_conversion
from jaxmod.utils import as_j64, get_batch_size, to_hashable
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
)
from atmodeller.solubility.core import NoSolubility
from atmodeller.thermodata import ActivityCoefficient, thermodynamic_data_source
from atmodeller.thermodata.core import (
    ThermodynamicCoefficients,
    thermodynamic_coefficients_dictionary,
)

logger: logging.Logger = logging.getLogger(__name__)

TSpecies_co = TypeVar("TSpecies_co", bound=SpeciesProtocol, covariant=True)


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
    molar_masses: NpFloat
    """Molar masses"""
    active_stability: NpBool
    """Active stability mask"""
    reaction_species_mask: NpBool
    """Mask for reaction species in the collection"""
    reservoir_species_mask: NpBool
    """Mask for reservoir species in the collection"""
    phase_mass_mask: NpBool
    """Mask for species included in phase-level mass, mole, and fraction aggregations"""
    number_solution: int
    """Number of solution quantities, which cannot depend on traced quantities"""

    def __init__(self, species: Iterable[TSpecies_co]):
        self.species = tuple(species)
        self.species_names = tuple(species_.data.name for species_ in self)
        self.molar_masses = np.array([species_.data.molar_mass for species_ in self], dtype=float)
        self.active_stability = np.array(
            [species.solve_for_stability for species in self], dtype=bool
        )
        self.reaction_species_mask = np.array(
            [isinstance(species_, ChemicalSpecies) for species_ in self], dtype=bool
        )
        self.reservoir_species_mask = np.array(
            [isinstance(species_, ReservoirSpecies) for species_ in self], dtype=bool
        )
        self.phase_mass_mask = np.array(
            [species.include_in_phase_mass for species in self], dtype=bool
        )

        # Ensure number_solution is static
        self.number_solution = sum(species.number_solution for species in self)

        logger.debug(
            f"Creating {self.__class__.__name__}: {tuple(str(species) for species in self)}"
        )

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
    def number_elements(self) -> int:
        """Number of unique elements in the species"""
        return len(self.unique_elements)

    @property
    def number_species(self) -> int:
        """Number of species"""
        return len(self)

    @property
    def reaction_species(self) -> "SpeciesCollection[ChemicalSpecies]":
        """Reaction species collection"""
        return SpeciesCollection(
            [species for species in self if isinstance(species, ChemicalSpecies)]
        )

    @property
    def reservoir_species(self) -> "SpeciesCollection[ReservoirSpecies]":
        """Reservoir species collection"""
        return SpeciesCollection(
            [species for species in self if isinstance(species, ReservoirSpecies)]
        )

    @property
    def unique_elements(self) -> tuple[str, ...]:
        """Unique elements in species in alphabetical order"""
        elements: list[str] = []
        for species in self:
            elements.extend(species.data.elements)
        unique_elements: list[str] = list(set(elements))

        return tuple(sorted(unique_elements))

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


class ThermodynamicState(eqx.Module):
    """A generic thermodynamic state

    This must adhere to ThermodynamicStateProtocol.

    Note:
        All parameters are stored as JAX arrays (``jnp.ndarray``) rather than Python floats. This
        ensures that JAX sees a consistent type during transformations (e.g., ``jit``, ``grad``,
        ``vmap``), preventing unnecessary recompilation when values change. In JAX, switching
        between a Python float and an array for the same argument will trigger retracing or
        recompilation, so keeping everything as arrays avoids this overhead.

    Args:
        temperature: Temperature in K
        pressure: Pressure in bar
        mass: Mass in kg. Defaults to ``1`` kg.
        melt_fraction: Melt fraction by weight in kg/kg. Defaults to ``1`` kg/kg.
        molar_mass: Molar mass of the silicate in kg/mol. Defaults to 60 g/mol, which is a typical
            value for silicate melts.
    """

    temperature: Array
    """Temperature in K"""
    pressure: Array
    """Pressure in bar"""
    mass: Array
    """Mass in kg"""
    melt_fraction: Array
    """Mass fraction of melt in kg/kg"""
    molar_mass: Array
    """Molar mass of the silicate in kg/mol"""

    def __init__(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass: ArrayLike = 1,
        melt_fraction: ArrayLike = 1,
        molar_mass: ArrayLike = 60e-3,
    ):
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.mass = as_j64(mass)
        self.melt_fraction = as_j64(melt_fraction)
        self.molar_mass = as_j64(molar_mass)

    @property
    def melt_mass(self) -> Array:
        """Mass of the melt in kg"""
        return self.mass * self.melt_fraction

    @property
    def melt_moles(self) -> Array:
        """Moles of the melt"""
        return self.melt_mass / self.molar_mass

    @property
    def solid_mass(self) -> Array:
        """Mass of the solid in kg"""
        return self.mass * (1.0 - self.melt_fraction)

    @property
    def solid_moles(self) -> Array:
        """Moles of the solid"""
        return self.solid_mass / self.molar_mass

    def get_pressure(self, gas_mass: Float[Array, "..."]) -> Float[Array, "..."]:
        """Gets the pressure.

        Args:
            gas_mass: Gas mass in kg. Unused but required by the interface.

        Returns:
            Pressure in bar
        """
        del gas_mass

        return self.pressure

    def asdict(self) -> dict[str, NpArray]:
        """Gets a dictionary of the values as NumPy arrays.

        Returns:
            A dictionary of the values
        """
        base_dict: dict[str, ArrayLike] = asdict(self)
        base_dict["melt_mass"] = self.melt_mass
        base_dict["solid_mass"] = self.solid_mass

        # Convert all values to NumPy arrays
        base_dict_np: dict[str, NpArray] = {k: np.asarray(v) for k, v in base_dict.items()}

        return base_dict_np


class ThinAtmospherePlanet(eqx.Module):
    """A planet with a thin atmosphere.

    This must adhere to ThermodynamicStateProtocol.

    Default values are for a fully molten Earth.

    Note:
        All parameters are stored as JAX arrays (``jnp.ndarray``) rather than Python floats. This
        ensures that JAX sees a consistent type during transformations (e.g., ``jit``, ``grad``,
        ``vmap``), preventing unnecessary recompilation when values change. In JAX, switching
        between a Python float and an array for the same argument will trigger retracing or
        recompilation, so keeping everything as arrays avoids this overhead.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to ``5.972e24`` kg (Earth).
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to ``0.295334691460966`` kg/kg (Earth).
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to ``1.0`` kg/kg.
        surface_radius: Radius of the planetary surface in m. Defaults to ``6371000`` m (Earth).
        temperature: Temperature in K. Defaults to ``2000`` K.
        pressure: Pressure in bar. Defaults to ``np.nan`` to solve for the mechanical pressure
            balance at the surface.
        molar_mass: Molar mass of the silicate in kg/mol. Defaults to 60 g/mol, which is a typical
            value for silicate melts.
    """

    planet_mass: Array
    """Mass of the planet in kg"""
    core_mass_fraction: Array
    """Mass fraction of the core relative to the planetary mass in kg/kg"""
    mantle_melt_fraction: Array
    """Mass fraction of the molten mantle in kg/kg"""
    surface_radius: Array
    """Radius of the surface in m"""
    temperature: Array
    """Temperature in K"""
    pressure: Array
    """Pressure in bar"""
    molar_mass: Array
    """Molar mass of the silicate in kg/mol"""

    def __init__(
        self,
        planet_mass: ArrayLike = 5.972e24,
        core_mass_fraction: ArrayLike = 0.295334691460966,
        mantle_melt_fraction: ArrayLike = 1.0,
        surface_radius: ArrayLike = 6371000,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = np.nan,
        molar_mass: ArrayLike = 60e-3,
    ):
        self.planet_mass = as_j64(planet_mass)
        self.core_mass_fraction = as_j64(core_mass_fraction)
        self.mantle_melt_fraction = as_j64(mantle_melt_fraction)
        self.surface_radius = as_j64(surface_radius)
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.molar_mass = as_j64(molar_mass)

    @property
    def mantle_mass(self) -> Array:
        """Mantle mass in kg"""
        return self.planet_mass * self.mantle_mass_fraction

    @property
    def mantle_moles(self) -> Array:
        """Moles of the mantle"""
        return self.mantle_mass / self.molar_mass

    @property
    def mantle_mass_fraction(self) -> Array:
        """Mantle mass fraction in kg/kg"""
        return 1 - self.core_mass_fraction

    @property
    def mantle_melt_mass(self) -> Array:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_melt_moles(self) -> Array:
        """Moles of the molten mantle"""
        return self.mantle_melt_mass / self.molar_mass

    @property
    def mantle_solid_mass(self) -> Array:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def mantle_solid_moles(self) -> Array:
        """Moles of the solid mantle"""
        return self.mantle_solid_mass / self.molar_mass

    @property
    def surface_area(self) -> Array:
        """Surface area"""
        return 4.0 * jnp.pi * jnp.square(self.surface_radius)

    @property
    def surface_gravity(self) -> Array:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / jnp.square(self.surface_radius)

    # The following properties ensure compliance with ThermodynamicStateProtocol
    @property
    def mass(self) -> Array:
        """Mantle mass in kg (alias for :attr:`mantle_mass`)"""
        return self.mantle_mass

    @property
    def melt_fraction(self) -> Array:
        """Mantle melt fraction in kg/kg (alias for :attr:`mantle_melt_fraction`)"""
        return self.mantle_melt_fraction

    @property
    def melt_mass(self) -> Array:
        """Mass of the molten mantle in kg (alias for :attr:`mantle_melt_mass`)"""
        return self.mantle_melt_mass

    @property
    def melt_moles(self) -> Array:
        """Moles of the molten mantle (alias for :attr:`mantle_melt_moles`)"""
        return self.mantle_melt_moles

    @property
    def solid_mass(self) -> Array:
        """Mass of the solid mantle in kg (alias for :attr:`mantle_solid_mass`)"""
        return self.mantle_solid_mass

    @property
    def solid_moles(self) -> Array:
        """Moles of the solid mantle (alias for :attr:`mantle_solid_moles`)"""
        return self.mantle_solid_moles

    def get_pressure(self, gas_mass: Float[Array, "..."]) -> Float[Array, "..."]:
        """Gets the pressure.

        A pressure is used if specified, otherwise the default behaviour is to compute the
        pressure from the mechanical pressure balance at the planetary surface assuming the thin
        atmosphere approximation. That is, the surface gravity is computed from the mass of the
        planet alone and is assumed to act on all the mass of the atmosphere.

        Args:
            gas_mass: Gas mass in kg

        Returns:
            Pressure in bar
        """
        pressure_specified: Bool[Array, "..."] = ~jnp.isnan(self.pressure)

        mechanical_pressure: Float[Array, "..."] = (
            gas_mass * self.surface_gravity / self.surface_area * unit_conversion.Pa_to_bar
        )
        # jax.debug.print("mechanical_pressure = {out}", out=mechanical_pressure)

        pressure: Float[Array, "..."] = jnp.where(
            pressure_specified, self.pressure, mechanical_pressure
        )
        # jax.debug.print("pressure = {out}", out=pressure)

        return pressure

    def asdict(self) -> dict[str, NpArray]:
        """Gets a dictionary of the values as NumPy arrays.

        Returns:
            A dictionary of the values
        """
        base_dict: dict[str, ArrayLike] = asdict(self)
        base_dict["mantle_mass"] = self.mantle_mass
        base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        # Convert all values to NumPy arrays
        base_dict_np: dict[str, NpArray] = {k: np.asarray(v) for k, v in base_dict.items()}

        return base_dict_np


# The only planet supported so far is one with a thin atmosphere
Planet = ThinAtmospherePlanet


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
        mask_list: list[Array] = [constraint.active() for constraint in self.constraints]
        broadcast_shape: tuple[int, ...] = jnp.broadcast_shapes(*[jnp.shape(m) for m in mask_list])

        active_constraints: Bool[Array, "... species"] = jnp.stack(
            [jnp.broadcast_to(m, broadcast_shape) for m in mask_list], axis=-1
        )
        # jax.debug.print("active fugacity constraints = {out}", out=active_constraints)

        return active_constraints

    def asdict(self, temperature: ArrayLike, pressure: ArrayLike) -> dict[str, Any]:
        """Gets an output dictionary of the evaluated fugacity constraints

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            An output dictionary
        """
        out: dict[str, Any] = {
            "species": {
                "activity": dict(
                    zip(
                        self.species.species_names,
                        [
                            np.exp(constraint.log_fugacity(temperature, pressure))
                            for constraint in self.constraints
                        ],
                    )
                )
            }
        }

        return out

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
        abundance: Abundance
        species: Species collection
        units: Units of the abundance. Defaults to ``mass``.
    """

    abundance: Float[Array, "... elements"] = eqx.field(converter=as_j64)
    """Abundance"""
    species: SpeciesCollection
    """Species collection"""
    units: Literal["mass", "moles"] = "mass"
    """Units of the abundance"""

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
                ``None``.
            units: Units of the abundance. Defaults to ``mass``.

        Returns:
            An instance
        """
        mass_constraints_: Mapping[str, ArrayLike] = (
            mass_constraints if mass_constraints is not None else {}
        )

        # Determine the maximum length of any array in mass_constraints_
        max_len: int = get_batch_size(mass_constraints_)

        # Initialise to all nans — shape (batch, elements), always 2-D
        shape: tuple[int, int] = (max_len, len(species.unique_elements))
        abundance: NpFloat = np.full(shape, np.nan, dtype=float)

        # Populate mass constraints. This accommodates mass constraints given as mass or moles of
        # species as well as elements
        for nn, element in enumerate(species.unique_elements):
            element_sum: ArrayLike = 0
            for species_, value_ in mass_constraints_.items():
                try:
                    element_composition: CompositionItem = Formula(species_).composition()[element]
                except KeyError:
                    continue
                if units == "mass":
                    # mass fraction
                    scale: float = element_composition.fraction
                elif units == "moles":
                    # element count
                    scale = element_composition.count
                element_sum += scale * value_

            if np.any(element_sum != 0):
                # Broadcasts scalar along that column
                abundance[:, nn] = element_sum

        return cls(abundance, species, units)

    def abundance_mol(self) -> Float[Array, "... elements"]:
        """Abundance by moles for all elements

        Returns:
            Abundance by moles for all elements
        """
        if self.units == "mass":
            return self.abundance / self.species.element_molar_masses
        elif self.units == "moles":
            return self.abundance
        else:
            raise ValueError("Units must be 'mass' or 'moles'")

    def abundance_mass(self) -> Float[Array, "... elements"]:
        """Abundance by mass for all elements

        Returns:
            Abundance by mass for all elements
        """
        if self.units == "mass":
            return self.abundance
        elif self.units == "moles":
            return self.abundance * self.species.element_molar_masses
        else:
            raise ValueError("Units must be 'mass' or 'moles'")

    def log_abundance(self) -> Float[Array, "... elements"]:
        """Element abundances in log-space

        The output shape depends on the calling context:

        - **Unbatched** (``abundance`` shape ``(1, elements)``): the leading singleton is squeezed
          away, returning a 1-D array of shape ``(elements,)``.
        - **Batched, not vmapped** (``abundance`` shape ``(batch, elements)``): returns a 2-D array
          of shape ``(batch, elements)``.
        - **Vmapped** (``abundance`` shape ``(elements,)`` inside the vmap): returns a 1-D array of
          shape ``(elements,)``.

        ``atleast_1d`` guards against collapse to a scalar when there is only a single element.

        Returns:
            Log abundance by moles
        """
        log_abundance: Float[Array, "... elements"] = jnp.log(self.abundance_mol())
        log_abundance = jnp.atleast_1d(log_abundance.squeeze())

        return log_abundance

    def asdict(self) -> dict[str, Any]:
        """Gets an output dictionary

        Returns:
            An output dictionary with the abundance by moles and mass for all elements
        """
        elements: tuple[str, ...] = self.species.unique_elements
        out: dict[str, Any] = {
            "elements": {
                "number_moles": dict(zip(elements, np.asarray(self.abundance_mol()).T)),
                "mass_kg": dict(zip(elements, np.asarray(self.abundance_mass()).T)),
            }
        }

        return out

    def active(self) -> Bool[Array, "... elements"]:
        """Active mass constraints

        Returns:
            Mask indicating whether elemental mass constraints are active or not
        """
        return ~jnp.isnan(self.log_abundance())


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
        multistart: Number of multistarts. Defaults to ``10``.
        multistart_perturbation: Perturbation for multistart. Defaults to ``30``.
        tau: Tau factor for species stability. Defaults to :const:`~atmodeller.constants.TAU`.
    """

    multistart: int = 10
    """Number of multistarts"""
    multistart_perturbation: float = 30.0
    """Perturbation for multistart"""
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
