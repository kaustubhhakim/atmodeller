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
"""Pytree containers that can be use by JAX

The leaves of pytrees must be JAX-compliant types, which excludes strings. So the preferred
approach is to encode the data as JAX-compatible types and provide properties and methods that can
reconstruct other desired quantities, notably strings and other objects. This ensures that similar 
functionality can remain together whilst accommodating the requirements of JAX-compliant pytrees.
"""
import sys
from collections.abc import Mapping
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from molmass import Formula

from atmodeller import AVOGADRO, GRAVITATIONAL_CONSTANT
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

phase_mapping: dict[str, int] = {"g": 0, "l": 1, "cr": 2}
"""Mapping from the JANAF phase string to an integer code"""
inverse_phase_mapping: dict[int, str] = {value: key for key, value in phase_mapping.items()}
"""Inverse mapping from the integer code to a JANAF phase string"""


class Planet(NamedTuple):
    """Planet properties

    Default values are for a fully molten Earth.

    Args:
        planet_mass: Mass of the planet in kg. Defaults to Earth.
        core_mass_fraction: Mass fraction of the iron core relative to the planetary mass. Defaults
            to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        surface_radius: Radius of the planetary surface in m. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
    """

    planet_mass: ArrayLike = 5.972e24
    """Mass of the planet in kg"""
    core_mass_fraction: ArrayLike = 0.295334691460966
    """Mass fraction of the core relative to the planetary mass in kg/kg"""
    mantle_melt_fraction: ArrayLike = 1.0
    """Mass fraction of the molten mantle in kg/kg"""
    surface_radius: ArrayLike = 6371000.0
    """Radius of the surface in m"""
    surface_temperature: ArrayLike = 2000.0
    """Temperature of the surface in K"""

    @property
    def mantle_mass(self) -> ArrayLike:
        """Mantle mass"""
        return self.planet_mass * (1.0 - self.core_mass_fraction)

    @property
    def mantle_melt_mass(self) -> ArrayLike:
        """Mass of the molten mantle"""
        return self.mantle_mass * self.mantle_melt_fraction

    @property
    def mantle_solid_mass(self) -> ArrayLike:
        """Mass of the solid mantle"""
        return self.mantle_mass * (1.0 - self.mantle_melt_fraction)

    @property
    def surface_area(self) -> ArrayLike:
        """Surface area"""
        return 4.0 * jnp.pi * self.surface_radius**2

    @property
    def surface_gravity(self) -> ArrayLike:
        """Surface gravity"""
        return GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2

    def asdict(self) -> dict:
        """Gets a dictionary of the values"""
        base_dict: dict = self._asdict()
        base_dict["mantle_mass"] = self.mantle_mass
        base_dict["mantle_melt_mass"] = self.mantle_melt_mass
        base_dict["mantle_solid_mass"] = self.mantle_solid_mass
        base_dict["surface_area"] = self.surface_area
        base_dict["surface_gravity"] = self.surface_gravity

        return base_dict


class SpeciesData(NamedTuple):
    """Species data

    Args:
        composition: Composition
        phase_code: Phase code
        molar_mass: Mass
        gibbs_coefficients: Gibbs coefficients
    """

    composition: dict[str, tuple[int, float, float]]
    """Composition"""
    phase_code: int
    """Phase code"""
    molar_mass: float
    """Molar mass"""
    gibbs_coefficients: tuple[float, ...]
    """Gibbs coefficients"""

    @classmethod
    def create(cls, formula: str, phase: str, gibbs_coefficients: tuple[float, ...]) -> Self:
        """Creates an instance

        Args:
            formula: Formula
            phase: Phase
            gibbs_coefficients: Gibbs coefficients

        Returns:
            An instance
        """
        mformula: Formula = Formula(formula)
        composition: dict[str, tuple[int, float, float]] = mformula.composition().asdict()
        molar_mass: float = mformula.mass * unit_conversion.g_to_kg
        phase_code: int = phase_mapping[phase]

        return cls(
            composition,
            phase_code,
            molar_mass,
            gibbs_coefficients,
        )

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    def formula(self) -> Formula:
        """Formula object"""
        formula: str = ""
        for element, values in self.composition.items():
            count: int = values[0]
            formula += element
            if count > 1:
                formula += str(count)

        return Formula(formula)

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return f"{self.hill_formula}_{self.phase}"

    @property
    def hill_formula(self) -> str:
        """Hill formula"""
        return self.formula().formula

    @property
    def phase(self) -> str:
        """JANAF phase"""
        return inverse_phase_mapping[self.phase_code]


class Solution(NamedTuple):
    """Solution

    Args:
        number: Number density of species
        stability: Stability of species
    """

    number_density: Array
    """Number density of species"""
    stability: Array
    """Stability of species"""


class Parameters(NamedTuple):
    """Parameters

    Args:
        formula_matrix; Formula matrix
        reaction_matrix: Reaction matrix
        species: List of species
        planet: Planet
        constraints: Dictionary of mass constraints
        scaling: Scaling for the number density. Defaults to the Avogadro constant, which converts
            molecules/m^3 to moles/m^3

    Attributes:
        formula_matrix: Formula matrix
        reaction_matrix: Reaction matrix
        species: List of species
        planet: Planet
        constraints: Dictionary of mass constraints
        scaling: Scaling for the number density
    """

    formula_matrix: Array
    """Formula matrix"""
    reaction_matrix: Array
    """Reaction matrix"""
    species: list[SpeciesData]
    """List of species"""
    planet: Planet
    """Planet"""
    constraints: Mapping[str, ArrayLike]
    """Mass constraints"""
    scaling: ArrayLike = AVOGADRO
    """Scaling"""


class Constraints(NamedTuple):
    """Log10 number of molecules constraints

    Args:
        species: A list of species
        log10_molecules: Log10 number of molecules constraints, ordered alphabetically by element
    """

    species: list[SpeciesData]
    """List of species"""
    log10_molecules: dict[str, ArrayLike]
    """Log10 number of molecules constraints, ordered alphabetically by element name"""

    @classmethod
    def create(cls, species: list[SpeciesData], mass: dict[str, ArrayLike]):
        """Creates an instance

        Args:
            species: A list of species
            mass: Dictionary of element name and mass constraint in kg in any order
        """
        sorted_mass: dict[str, ArrayLike] = {k: mass[k] for k in sorted(mass)}
        log10_number_of_molecules: dict[str, ArrayLike] = {}
        for element, mass_constraint in sorted_mass.items():
            molar_mass: ArrayLike = Formula(element).mass * unit_conversion.g_to_kg
            log10_number_of_molecules_: Array = (
                jnp.log10(mass_constraint) + jnp.log10(AVOGADRO) - jnp.log10(molar_mass)
            )
            log10_number_of_molecules[element] = log10_number_of_molecules_

        return cls(species, log10_number_of_molecules)

    def array(self) -> Array:
        return jnp.array(list[self.log10_molecules.values()])


# TODO: Switch convention to use dG = S - Href/T as per the comment of Hugh. Then the
# discontinuities associated with the Gibbs energy of formation are no longer a problem meaning
# that a fit can be accurately made across the temperature range of interest. This will notably
# fix problems with sulphur.

# For all fits, "zero" temperature was set to 0.01 to avoid problems with fitting a/T
number_of_coefficients: int = 5
reference_gibbs: tuple[float, ...] = (0,) * 5

CH4_g: SpeciesData = SpeciesData.create(
    "CH4", "g", (-7.471666e-01, -9.118137e00, -3.418606e01, 1.176984e-01, -4.291384e-07)
)
Cl2_g: SpeciesData = SpeciesData.create("Cl2", "g", reference_gibbs)
CO_g: SpeciesData = SpeciesData.create(
    "CO", "g", (-1.413961e-01, -1.125331e00, -1.048469e02, -8.915501e-02, 1.503998e-06)
)
CO2_g: SpeciesData = SpeciesData.create(
    "CO2",
    "g",
    (-1.884621e-02, -2.387862e-01, -3.923660e02, -2.506878e-03, 7.017329e-07),
)
C_cr: SpeciesData = SpeciesData.create("C", "cr", reference_gibbs)
H2_g: SpeciesData = SpeciesData.create(
    "H2",
    "g",
    reference_gibbs,
)
H2O_g: SpeciesData = SpeciesData.create(
    "H2O",
    "g",
    (-3.817134e-01, -4.469468e00, -2.213329e02, 5.975648e-02, 6.535070e-08),
)
H2O_l: SpeciesData = SpeciesData.create(
    "H2O",
    "l",
    (-9.885210e02, -3.519502e00, -2.658921e02, 1.856359e-01, -3.631301e-05),
)
N2_g: SpeciesData = SpeciesData.create("N2", "g", reference_gibbs)
NH3_g: SpeciesData = SpeciesData.create(
    "NH3", "g", (-4.493772e-01, -5.741781e00, -2.041235e01, 1.233233e-01, -9.071982e-07)
)
O2_g: SpeciesData = SpeciesData.create("O2", "g", reference_gibbs)
