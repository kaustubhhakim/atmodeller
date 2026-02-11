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
from jaxmod.units import unit_conversion
from jaxmod.utils import as_j64, get_batch_size, partial_rref, to_hashable
from jaxtyping import Array, ArrayLike, Bool, Float
from molmass import CompositionItem, Formula

from atmodeller.constants import (
    CONDENSED_STATE,
    DISSOLVED_STATE,
    GAS_STATE,
    LOG_NUMBER_MOLES_LOWER,
    LOG_NUMBER_MOLES_UPPER,
    LOG_STABILITY_LOWER,
    LOG_STABILITY_UPPER,
    TAU,
)
from atmodeller.eos.core import IdealGas
from atmodeller.interfaces import (
    ActivityProtocol,
    ChemicalSpeciesData,
    FugacityConstraintProtocol,
    SolubilityProtocol,
    SpeciesProtocol,
    ThermodynamicStateProtocol,
)
from atmodeller.solubility.core import NoSolubility
from atmodeller.thermodata import CondensateActivity, thermodynamic_data_source
from atmodeller.thermodata.core import (
    ThermodynamicCoefficients,
    thermodynamic_coefficients_dictionary,
)
from atmodeller.type_aliases import NpArray, NpBool, NpFloat, NpInt

logger: logging.Logger = logging.getLogger(__name__)


class ChemicalSpecies(eqx.Module):
    """Chemical species that participate in reactions

    Args:
        data: Chemical species data
        activity: Activity
        solve_for_stability: Solve for stability
        number_solution: Number of solution quantities
        thermo: Thermodynamic coefficients
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solve_for_stability: bool
    number_solution: int
    thermo: ThermodynamicCoefficients

    @classmethod
    def create(
        cls,
        formula: str,
        state: str,
        activity: ActivityProtocol,
        solve_for_stability: bool,
        number_solution: int,
    ) -> "ChemicalSpecies":
        """Creates an instance.

        Args:
            formula: Formula
            state: State of aggregation, as typically defined by JANAF
            activity: Activity
            solve_for_stability. Solve for stability.
            number_solution: Number of solution quantities

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

        return cls(species_data, activity, solve_for_stability, number_solution, thermo)

    @classmethod
    def create_condensed(
        cls,
        formula: str,
        *,
        state: str = CONDENSED_STATE,
        activity: ActivityProtocol = CondensateActivity(),
        solve_for_stability: bool = True,
    ) -> "ChemicalSpecies":
        """Creates a condensate with some default values.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :const:`~atmodeller.constants.CONDENSED_STATE`
            activity: Activity. Defaults to unity activity.
            solve_for_stability. Solve for stability. Defaults to ``True``.

        Returns:
            A condensed species
        """
        # Either both the number of moles and stability are solved for, or alternatively stability
        # can be enforced in which case the number of moles is irrelevant and there is nothing to
        # solve for.
        number_solution: int = 2 if solve_for_stability else 0

        return cls.create(formula, state, activity, solve_for_stability, number_solution)

    @classmethod
    def create_gas(
        cls,
        formula: str,
        *,
        state: str = GAS_STATE,
        activity: ActivityProtocol = IdealGas(),
        solve_for_stability: bool = False,
    ) -> "ChemicalSpecies":
        """Creates a gas species with some default values.

        Args:
            formula: Formula
            state: State of aggregation as defined by JANAF. Defaults to
                :const:`~atmodeller.constants.GAS_STATE`
            activity: Activity. Defaults to an ideal gas.
            solve_for_stability. Solve for stability. Defaults to ``False``.

        Returns:
            A gas species
        """
        # The number of moles is always solved for, and stability can be if desired, although
        # this is not typically done for gas species because they are usually stable over the range
        # of conditions considered and truncating the abundance can severely distort the results,
        # notably for O2.
        number_solution: int = 2 if solve_for_stability else 1

        return cls.create(formula, state, activity, solve_for_stability, number_solution)

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
    """

    data: ChemicalSpeciesData
    activity: ActivityProtocol
    solubility: SolubilityProtocol
    number_solution: int

    @classmethod
    def create_dissolved(
        cls,
        formula: str,
        *,
        state=DISSOLVED_STATE,
        activity: ActivityProtocol = CondensateActivity(),
        solubility: Optional[SolubilityProtocol] = None,
    ) -> "ReservoirSpecies":
        """Creates a dissolved species with some default values.

        Args:
            formula: Formula
            state: State of aggregation. Defaults to :const:`~atmodeller.constants.DISSOLVED_STATE`
            activity: Activity. Defaults to unity activity.
            solubility: Solubility. Defaults to no solubility.

        Returns:
            A dissolved species
        """
        species_data: ChemicalSpeciesData = ChemicalSpeciesData(formula, state)

        if solubility is None:
            solubility = NoSolubility()
            number_solution: int = 0
        else:
            number_solution = 1

        return cls(species_data, activity, solubility, number_solution)


TSpecies = TypeVar("TSpecies", bound=SpeciesProtocol)


class SpeciesCollectionData(eqx.Module, Generic[TSpecies]):
    """Data container for a species collection

    Args:
        species: An iterable of species
    """

    species: tuple[TSpecies, ...]
    """Species in the collection"""
    species_names: tuple[str, ...]
    """Unique names of all species"""
    gas_species_names: tuple[str, ...]
    """Gas species names"""
    condensed_species_names: tuple[str, ...]
    """Condensed species names"""
    gas_species_mask: NpBool
    """Gas species mask"""
    reaction_species: tuple[ChemicalSpecies, ...]
    """Chemical species that participate in reactions"""
    reservoir_species: tuple[ReservoirSpecies, ...]
    """Reservoir species that do not participate in reactions"""
    reaction_species_mask: NpBool
    """Mask for reaction species in the full species list"""
    molar_masses: NpFloat
    """Molar masses"""
    unique_elements: tuple[str, ...]
    """Unique elements in species in alphabetical order"""
    element_molar_masses: NpFloat
    """Molar masses of the ordered elements"""
    diatomic_oxygen_index: NpFloat
    """Index of diatomic oxygen or np.nan if not present"""
    formula_matrix: NpInt
    """Formula matrix"""
    number_solution: int
    """Number of solution quantities that cannot depend on traced quantities"""

    def __init__(self, species: Iterable[TSpecies]):
        self.species = tuple(species)
        self.species_names = tuple([species_.data.name for species_ in self.species])
        self.gas_species_names = tuple(
            [species.data.name for species in self.species if species.data.state == GAS_STATE]
        )
        self.condensed_species_names = tuple(
            [species.data.name for species in self.species if species.data.state != GAS_STATE]
        )
        self.gas_species_mask = np.array(
            [species.data.state == GAS_STATE for species in self.species], dtype=bool
        )

        reaction_species: list[ChemicalSpecies] = []
        reaction_mask_list: list[bool] = []
        reservoir_species: list[ReservoirSpecies] = []
        for species_ in self.species:
            if isinstance(species_, ChemicalSpecies):
                reaction_species.append(species_)
                reaction_mask_list.append(True)
            elif isinstance(species_, ReservoirSpecies):
                reservoir_species.append(species_)
                reaction_mask_list.append(False)
            else:
                raise TypeError(
                    "Species must be either ChemicalSpecies or ReservoirSpecies, "
                    f"got {type(species_)}"
                )
        self.reaction_species = tuple(reaction_species)
        self.reservoir_species = tuple(reservoir_species)
        self.reaction_species_mask = np.array(reaction_mask_list, dtype=bool)

        self.molar_masses = np.array([species_.data.molar_mass for species_ in self.species])

        # Unique elements
        elements: list[str] = []
        for species_ in self.species:
            elements.extend(species_.data.elements)
        unique_elements: list[str] = list(set(elements))
        self.unique_elements = tuple(sorted(unique_elements))

        # Element molar masses
        element_molar_masses: list[float] = []
        for element_ in self.unique_elements:
            mformula: Formula = Formula(element_)
            molar_mass: float = mformula.mass * unit_conversion.g_to_kg
            element_molar_masses.append(molar_mass)
        self.element_molar_masses = np.array(element_molar_masses)

        self.diatomic_oxygen_index = self.get_diatomic_oxygen_index()

        self.formula_matrix = self.get_formula_matrix()

        # Ensure number_solution is static
        self.number_solution = sum([species.number_solution for species in self.species])

    @property
    def number_species(self) -> int:
        """Number of species"""
        return len(self.species)

    @property
    def reservoir_species_mask(self) -> NpBool:
        """Mask for reservoir species in the full species list"""
        return ~self.reaction_species_mask

    def get_diatomic_oxygen_index(self) -> NpFloat:
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

    def get_formula_matrix(self) -> NpInt:
        """Gets the formula matrix.

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            Formula matrix
        """
        formula_matrix: NpInt = np.zeros(
            (len(self.unique_elements), self.number_species), dtype=np.int_
        )

        for element_index, element in enumerate(self.unique_elements):
            for species_index, species_ in enumerate(self):
                count: int = 0
                try:
                    count = species_.data.composition[element][0]
                except KeyError:
                    count = 0
                formula_matrix[element_index, species_index] = count

        # logger.debug("formula_matrix = %s", formula_matrix)

        return formula_matrix

    def __getitem__(self, index: int) -> SpeciesProtocol:
        return self.species[index]

    def __iter__(self) -> Iterator[SpeciesProtocol]:
        return iter(self.species)

    def __len__(self) -> int:
        return len(self.species)

    def __str__(self) -> str:
        return str(tuple(str(species) for species in self.species))


class SpeciesNetwork(eqx.Module):
    """A network of species

    Args:
        species: An iterable of chemical species
    """

    data: SpeciesCollectionData[ChemicalSpecies]
    """Species collection data"""
    active_stability: NpBool
    """Active stability mask"""
    number_reactions: int
    """Number of reactions"""
    reaction_matrix: NpFloat
    """Reaction matrix"""
    active_reactions: NpBool
    """Active reactions"""

    def __init__(self, species: Iterable[ChemicalSpecies]):
        self.data = SpeciesCollectionData(species)

        active_stability: list[bool] = [
            species.solve_for_stability for species in self.data.species
        ]
        self.active_stability = np.array(active_stability)
        self.number_reactions = max(0, self.data.number_species - len(self.data.unique_elements))
        self.reaction_matrix = self.get_reaction_matrix()
        self.active_reactions = np.ones(self.number_reactions, dtype=bool)

    @classmethod
    def available_species(cls) -> tuple[str, ...]:
        return thermodynamic_data_source.available_species()

    # TODO: Only used to determine output. Relevant for gas only or should be gas only AND no
    # solubility?
    # @property
    # def gas_only(self) -> bool:
    #     """Checks if a gas-only network"""
    #     return len(self.data) == len(self.gas_species_mask)

    def get_reaction_dictionary(self) -> dict[int, str]:
        """Gets reactions as a dictionary.

        Returns:
            Reactions as a dictionary
        """
        reaction_matrix: NpFloat = self.get_reaction_matrix()

        reactions: dict[int, str] = {}
        if reaction_matrix.size != 0:
            for reaction_index in range(reaction_matrix.shape[0]):
                reactants: str = ""
                products: str = ""
                for species_index, name in enumerate(self.data.species_names):
                    coeff: float = reaction_matrix[reaction_index, species_index].item()
                    if coeff != 0:
                        if coeff < 0:
                            reactants += f"{abs(coeff)} {name} + "
                        else:
                            products += f"{coeff} {name} + "

                reactants = reactants.rstrip(" + ")
                products = products.rstrip(" + ")
                reaction: str = f"{reactants} = {products}"
                reactions[reaction_index] = reaction

        return reactions

    def get_reaction_matrix(self) -> NpFloat:
        """Gets the reaction matrix.

        Returns:
            A matrix of linearly independent reactions or an empty array if no reactions
        """
        transpose_formula_matrix: NpInt = self.data.get_formula_matrix().T
        reaction_matrix: NpFloat = partial_rref(transpose_formula_matrix)
        # logger.debug("reaction_matrix = %s", reaction_matrix)

        return reaction_matrix

    def get_temperature_range(self) -> tuple[float, float]:
        """Gets the temperature range of the thermodynamic data for the species

        Returns:
            Minimum and maximum temperature that is valid for the species
        """
        temperature_min: list[float] = [min(species.thermo.T_min) for species in self.data.species]
        temperature_max: list[float] = [max(species.thermo.T_max) for species in self.data.species]

        return max(temperature_min), min(temperature_max)


class SpeciesCollection(eqx.Module):
    """A collection of species that can include both reactive and reservoir species"""

    data: SpeciesCollectionData[SpeciesProtocol]
    """Species collection data"""
    species_network: SpeciesNetwork
    """Network of species and their interactions"""

    def __init__(self, species: Iterable[SpeciesProtocol]):
        self.data = SpeciesCollectionData(species)
        self.species_network = SpeciesNetwork(self.data.reaction_species)

    @classmethod
    def create(cls, species: Iterable[str]) -> "SpeciesCollection":
        """Creates an instance

        Args:
            species: A list or tuple of species names with a state suffix

        Returns
            An instance
        """
        species_list: list[SpeciesProtocol] = []

        for species_ in species:
            formula, state = species_.split("_")
            hill_formula = Formula(formula).formula
            if state == GAS_STATE:
                species_to_add: SpeciesProtocol = ChemicalSpecies.create_gas(
                    hill_formula, state=state
                )
            elif state == CONDENSED_STATE:
                species_to_add = ChemicalSpecies.create_condensed(hill_formula, state=state)
            elif state == DISSOLVED_STATE:
                species_to_add = ReservoirSpecies.create_dissolved(hill_formula, state=state)
            else:
                raise ValueError(
                    f"State must be '{GAS_STATE}', '{CONDENSED_STATE}' or '{DISSOLVED_STATE}', "
                    f"got {state}"
                )
            species_list.append(species_to_add)

        return cls(species_list)


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
    """

    temperature: Array
    """Temperature in K"""
    pressure: Array
    """Pressure in bar"""
    mass: Array
    """Mass in kg"""
    melt_fraction: Array
    """Mass fraction of melt in kg/kg"""

    def __init__(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        mass: ArrayLike = 1,
        melt_fraction: ArrayLike = 1,
    ):
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)
        self.mass = as_j64(mass)
        self.melt_fraction = as_j64(melt_fraction)

    @property
    def melt_mass(self) -> Array:
        """Mass of the melt in kg"""
        return self.mass * self.melt_fraction

    @property
    def solid_mass(self) -> Array:
        """Mass of the solid in kg"""
        return self.mass * (1.0 - self.melt_fraction)

    def get_pressure(self, gas_mass: Array) -> Array:
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
            to ``0.3`` kg/kg (Earth).
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to ``1.0`` kg/kg.
        surface_radius: Radius of the planetary surface in m. Defaults to ``6371000`` m (Earth).
        temperature: Temperature in K. Defaults to ``2000`` K.
        pressure: Pressure in bar. Defaults to ``np.nan`` to solve for the mechanical pressure
            balance at the surface.
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

    def __init__(
        self,
        planet_mass: ArrayLike = 5.972e24,
        core_mass_fraction: ArrayLike = 0.295334691460966,
        mantle_melt_fraction: ArrayLike = 1.0,
        surface_radius: ArrayLike = 6371000,
        temperature: ArrayLike = 2000,
        pressure: ArrayLike = np.nan,
    ):
        self.planet_mass = as_j64(planet_mass)
        self.core_mass_fraction = as_j64(core_mass_fraction)
        self.mantle_melt_fraction = as_j64(mantle_melt_fraction)
        self.surface_radius = as_j64(surface_radius)
        self.temperature = as_j64(temperature)
        self.pressure = as_j64(pressure)

    @property
    def mantle_mass(self) -> Array:
        """Mantle mass"""
        return self.planet_mass * self.mantle_mass_fraction

    @property
    def mantle_mass_fraction(self) -> Array:
        """Mantle mass fraction"""
        return 1 - self.core_mass_fraction

    @property
    def mantle_melt_mass(self) -> Array:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_solid_mass(self) -> Array:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

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
        return self.mantle_mass

    @property
    def melt_fraction(self) -> Array:
        return self.mantle_melt_fraction

    @property
    def melt_mass(self) -> Array:
        return self.mantle_melt_mass

    @property
    def solid_mass(self) -> Array:
        return self.mantle_solid_mass

    def get_pressure(self, gas_mass: Array) -> Array:
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

        pressure: Float[Array, ""] = jnp.where(
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
        del temperature
        del pressure

        return jnp.log(self.fugacity)


class FugacityConstraintSet(eqx.Module):
    """A set of fugacity constraints

    These are applied as constraints on the gas activity.

    Args:
        constraints: Fugacity constraints
        species_collection: Species collection
    """

    constraints: tuple[FugacityConstraintProtocol, ...]
    """Fugacity constraints"""
    species_collection: SpeciesCollection
    """Species collection"""

    @classmethod
    def create(
        cls,
        species_collection: SpeciesCollection,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
    ) -> "FugacityConstraintSet":
        """Creates an instance

        Args:
            species_collection: Species collection
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                ``None``.

        Returns:
            An instance
        """
        fugacity_constraints_: Mapping[str, FugacityConstraintProtocol] = (
            fugacity_constraints if fugacity_constraints is not None else {}
        )

        constraints: list[FugacityConstraintProtocol] = []

        for species_name in species_collection.data.species_names:
            if species_name in fugacity_constraints_:
                constraints.append(fugacity_constraints_[species_name])
            else:
                # NOTE: This is applied to all species, which is OK because it returns nans,
                # meaning no imposed activity/fugacity.
                constraints.append(FixedFugacityConstraint())

        return cls(tuple(constraints), species_collection)

    def active(self) -> Bool[Array, "..."]:
        """Active fugacity constraints

        Returns:
            Mask indicating whether fugacity constraints are active or not
        """
        mask_list: list[Array] = [constraint.active() for constraint in self.constraints]

        return jnp.array(mask_list)

    def asdict(self, temperature: ArrayLike, pressure: ArrayLike) -> dict[str, NpArray]:
        """Gets a dictionary of the evaluated fugacity constraints as NumPy Arrays

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            A dictionary of the evaluated fugacity constraints
        """
        log_fugacity_list: list[NpFloat] = []

        for constraint in self.constraints:
            log_fugacity: NpFloat = np.asarray(constraint.log_fugacity(temperature, pressure))
            log_fugacity_list.append(log_fugacity)

        out: dict[str, NpArray] = {
            # Subtle, but np.exp will collapse scalar array to 0-D, violating the type hint.
            f"{key}_fugacity": np.exp(np.atleast_1d(log_fugacity_list[idx]))
            for idx, key in enumerate(self.species_collection.data.species_names)
            if not np.all(np.isnan(log_fugacity_list[idx]))
        }

        return out

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        # NOTE: Must avoid the late-binding closure issue
        fugacity_funcs: list[Callable] = [
            to_hashable(constraint.log_fugacity) for constraint in self.constraints
        ]
        # jax.debug.print("fugacity_funcs = {out}", out=fugacity_funcs)

        # Temperature must be a float array to ensure branches have have identical types
        temperature = as_j64(temperature)

        def apply_fugacity(index: ArrayLike, temperature: ArrayLike, pressure: ArrayLike) -> Array:
            # jax.debug.print("index = {out}", out=index)
            return lax.switch(index, fugacity_funcs, temperature, pressure)

        indices: Array = jnp.arange(len(self.constraints))
        vmap_fugacity: Callable = eqx.filter_vmap(apply_fugacity, in_axes=(0, None, None))
        log_fugacity: Array = vmap_fugacity(indices, temperature, pressure)
        # jax.debug.print("log_fugacity = {out}", out=log_fugacity)

        return log_fugacity


class MassConstraintSet(eqx.Module):
    """A set of mass constraints

    Note:
        ``abundance`` must be stored as a 2-D array so that the vmapping operation only batches
        over the leading dimension if it has a size greater than unity. Then, the methods that
        process ``abundance`` consistently return 1-D arrays, shape (elements,), to avoid
        triggering JAX recompilation.

    Args:
        abundance: Abundance
        species_collection: Species collection
        units: Units of the abundance. Defaults to ``mass``.
    """

    abundance: Float[Array, "..."] = eqx.field(converter=as_j64)
    """Abundance"""
    species_collection: SpeciesCollection
    """Species collection"""
    units: Literal["mass", "moles"] = "mass"
    """Units of the abundance"""
    oxygen_column_index: Optional[int] = None
    """Column index of oxygen in ``abundance``. Defaults to ``None``."""

    @classmethod
    def create(
        cls,
        species_collection: SpeciesCollection,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        units: Literal["mass", "moles"] = "mass",
    ) -> "MassConstraintSet":
        """Creates an instance

        Args:
            species_collection: Species collection
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

        # Initialise to all nans assuming that there are no mass constraints
        abundance: NpFloat = np.full(
            (max_len, len(species_collection.data.unique_elements)), np.nan, dtype=np.float64
        )

        # Populate mass constraints. This accommodates mass constraints given as mass or moles of
        # species as well as elements
        for nn, element in enumerate(species_collection.data.unique_elements):
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

        # jax.debug.print("abundance = {out}", out=abundance)

        return cls(abundance, species_collection, units)

    def abundance_mol(self) -> Float[Array, "..."]:
        """Abundance by moles for all elements

        Returns:
            Abundance by moles for all elements
        """
        if self.units == "mass":
            return self.abundance / self.species_collection.data.element_molar_masses
        elif self.units == "moles":
            return self.abundance
        else:
            raise ValueError("Units must be 'mass' or 'moles'")

    def abundance_mass(self) -> Float[Array, "..."]:
        """Abundance by mass for all elements

        Returns:
            Abundance by mass for all elements
        """
        if self.units == "mass":
            return self.abundance
        elif self.units == "moles":
            return self.abundance * self.species_collection.data.element_molar_masses
        else:
            raise ValueError("Units must be 'mass' or 'moles'")

    def log_abundance(self) -> Float[Array, "..."]:
        """Element abundances in log-space

        ``abundance`` is stored as a 2-D array with shape (batch, elements) so that ``vmap`` only
        maps over the leading dimension when batching is active. When called inside a ``vmap``,
        each mapped instance receives a single row of the abundance matrix, i.e. an array of shape
        (elements,). When called outside ``vmap``, ``abundance`` has shape (1, elements) and must
        be reduced to a consistent 1-D vector.

        If the batch dimension is greater than one and the method is called outside a vmapped
        workflow, the full 2-D log-abundance array is returned unchanged. This preserves the
        natural behaviour for genuinely batched data while still collapsing the leading singleton
        dimension in unbatched use.

        Returns:
            Log abundance by moles for all elements
        """
        log_abundance: Float[Array, "..."] = jnp.log(self.abundance_mol())

        # Ensure stable 1-D output:
        #  - Unbatched case: shape (1, elements) --> (elements,)
        #  - Vmapped case: shape (elements,) --> already correct
        # ``squeeze`` removes the leading singleton, and ``atleast_1d`` guards against accidental
        # collapse when only a single element exists.
        log_abundance = jnp.atleast_1d(log_abundance.squeeze())

        return log_abundance

    def asdict(self) -> dict[str, NpArray]:
        """Gets a dictionary of the values as NumPy arrays

        Returns:
            A dictionary of the values
        """
        abundance_mol: NpArray = np.asarray(self.abundance_mol())
        abundance_mass: NpArray = np.asarray(self.abundance_mass())

        out: dict[str, NpArray] = {}

        for label, arr in [("number", abundance_mol), ("mass", abundance_mass)]:
            for idx, element in enumerate(self.species_collection.data.unique_elements):
                col = arr[:, idx]
                if not np.all(np.isnan(col)):
                    out[f"{element}_{label}"] = col

        return out

    def active(self) -> Bool[Array, "..."]:
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
    tau: Array = eqx.field(converter=as_j64, default=TAU)  # NOTE: Must be an array to trace tau
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


class Parameters(eqx.Module):
    """Parameters

    Args:
        species: Species collection
        state: Thermodynamic state
        fugacity_constraints: Fugacity constraints
        mass_constraints: Mass constraints
        solver_parameters: Solver parameters
        batch_size: Batch size. Defaults to ``1``.
    """

    species_collection: SpeciesCollection
    """Species collection"""
    state: ThermodynamicStateProtocol
    """Thermodynamic state"""
    fugacity_constraints: FugacityConstraintSet
    """Fugacity constraints"""
    mass_constraints: MassConstraintSet
    """Mass constraints"""
    solver_parameters: SolverParameters
    """Solver parameters"""
    batch_size: int = 1
    """Batch size"""

    @property
    def species_network(self) -> SpeciesNetwork:
        """Species network"""
        return self.species_collection.species_network

    @classmethod
    def create(
        cls,
        species_collection: SpeciesCollection,
        state: Optional[ThermodynamicStateProtocol] = None,
        fugacity_constraints: Optional[Mapping[str, FugacityConstraintProtocol]] = None,
        mass_constraints: Optional[Mapping[str, ArrayLike]] = None,
        solver_parameters: Optional[SolverParameters] = None,
    ):
        """Creates an instance

        Args:
            species_collection: Species collection
            state: Thermodynamic state. Defaults to a new instance of ``Planet``.
            fugacity_constraints: Mapping of a species name and a fugacity constraint. Defaults to
                a new instance of ``FugacityConstraints``.
            mass_constraints: Mapping of element name and mass constraint in kg. Defaults to
                a new instance of ``MassConstraints``.
            solver_parameters: Solver parameters. Defaults to a new instance of
                ``SolverParameters``.

        Returns:
            An instance
        """
        state_: ThermodynamicStateProtocol = Planet() if state is None else state
        fugacity_constraints_: FugacityConstraintSet = FugacityConstraintSet.create(
            species_collection, fugacity_constraints
        )
        mass_constraints_: MassConstraintSet = MassConstraintSet.create(
            species_collection, mass_constraints
        )

        # These pytrees only contain arrays intended for vectorisation (no hidden JAX/NumPy arrays
        # that should remain scalar)
        batch_size: int = get_batch_size((state, fugacity_constraints, mass_constraints))
        solver_parameters_: SolverParameters = (
            SolverParameters() if solver_parameters is None else solver_parameters
        )
        # Always broadcast tau so we can apply vmap to the solver once, even if some calculations
        # need to be repeated due to failures.
        tau_broadcasted: Float[Array, " batch"] = jnp.broadcast_to(
            solver_parameters_.tau, (batch_size,)
        )
        get_leaf: Callable = lambda t: t.tau  # noqa: E731
        solver_parameters_ = eqx.tree_at(get_leaf, solver_parameters_, tau_broadcasted)

        return cls(
            species_collection,
            state_,
            fugacity_constraints_,
            mass_constraints_,
            solver_parameters_,
            batch_size,
        )
