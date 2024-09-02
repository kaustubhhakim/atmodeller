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
"""Temporary module for storing new optimised JAX classes

Most likely these will migrate elsewhere in the code base eventually.
"""
# Convenient to use chemical symbol names so pylint: disable=invalid-name

import sys
from typing import NamedTuple

import jax.numpy as jnp
import pandas as pd
from jax import Array, jit
from molmass import Formula

from atmodeller.reaction_network import partial_rref
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


class SpeciesData(NamedTuple):
    """Species data

    To replace ChemicalSpecies. This is a light weight pytree compliant NamedTuple.
    """

    formula: str
    phase: str
    composition: dict[str, int]
    hill_formula: str
    molar_mass: float
    gibbs_coefficients: Array

    @classmethod
    def create(cls, formula: str, phase: str, gibbs_coefficients: Array) -> Self:
        _mformula: Formula = Formula(formula)
        _composition_df: pd.DataFrame = _mformula.composition().dataframe()
        # Linter complaining but seems fine so pylint: disable=unsubscriptable-object
        composition: dict[str, int] = _composition_df["Count"].to_dict()
        hill_formula: str = _mformula.formula
        molar_mass: float = _mformula.mass * unit_conversion.g_to_kg

        return cls(
            formula,
            phase,
            composition,
            hill_formula,
            molar_mass,
            gibbs_coefficients,
        )

    @property
    def atoms(self) -> int:
        """Atoms"""
        return sum(self.composition.values())

    @property
    def elements(self) -> list[str]:
        """Elements"""
        return list(self.composition.keys())

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return f"{self.hill_formula}_{self.phase}"


@jit
def gibbs_energy_of_formation(species_data: SpeciesData, temperature: float) -> Array:
    r"""Gibbs energy of formation

    Args:
        species_data: Species data
        temperature: Temperature in K

    Returns:
        The standard Gibbs free energy of formation in :math:`\mathrm{J}\mathrm{mol}^{-1}`
    """
    gibbs: Array = (
        species_data.gibbs_coefficients[0] / temperature
        + species_data.gibbs_coefficients[1] * jnp.log(temperature)
        + species_data.gibbs_coefficients[2]
        + species_data.gibbs_coefficients[3] * jnp.power(temperature, 1)
        + species_data.gibbs_coefficients[4] * jnp.power(temperature, 2)
    )

    return gibbs * 1000.0  # kilo


# For all below fits, "zero" temperature was set to 0.01 to avoid problems with fitting a/T
number_of_coefficients: int = 5
reference_gibbs: Array = jnp.zeros(number_of_coefficients)

CH4_g: SpeciesData = SpeciesData.create(
    "CH4", "g", jnp.array([-7.471666e-01, -9.118137e00, -3.418606e01, 1.176984e-01, -4.291384e-07])
)
Cl2_g: SpeciesData = SpeciesData.create("Cl2", "g", reference_gibbs)
CO_g: SpeciesData = SpeciesData.create(
    "CO", "g", jnp.array([-1.413961e-01, -1.125331e00, -1.048469e02, -8.915501e-02, 1.503998e-06])
)
CO2_g: SpeciesData = SpeciesData.create(
    "CO2",
    "g",
    jnp.array([-1.884621e-02, -2.387862e-01, -3.923660e02, -2.506878e-03, 7.017329e-07]),
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
    jnp.array([-3.817134e-01, -4.469468e00, -2.213329e02, 5.975648e-02, 6.535070e-08]),
)
H2O_l: SpeciesData = SpeciesData.create(
    "H2O",
    "l",
    jnp.array([-9.885210e02, -3.519502e00, -2.658921e02, 1.856359e-01, -3.631301e-05]),
)
N2_g: SpeciesData = SpeciesData.create("N2", "g", reference_gibbs)
NH3_g: SpeciesData = SpeciesData.create(
    "NH3", "g", jnp.array([-4.493772e-01, -5.741781e00, -2.041235e01, 1.233233e-01, -9.071982e-07])
)
O2_g: SpeciesData = SpeciesData.create("O2", "g", reference_gibbs)

# TODO: Sulphur species cannot be fit with a single function, so either a multi-part fit needs
# implementing or a (jit-compliant) interpolator.
# https://jax.readthedocs.io/en/latest/_autosummary/
#   jax.scipy.interpolate.RegularGridInterpolator.html


class ReactionNetworkJAX:
    """Primary role is to assemble Python objects to generate JAX-compliants arrays for numerical
    solution. In this regard, this should only generate arrays that are required once, and not
    computed dynamically during the solve.
    """

    @staticmethod
    def unique_elements_in_species(species: list[SpeciesData]) -> tuple[str, ...]:
        """Unique elements in a list of species

        Args:
            species: A list of species

        Returns:
            Unique elements in the list of species
        """
        elements: list[str] = []
        for species_ in species:
            elements.extend(species_.elements)
        unique_elements: list[str] = list(set(elements))
        sorted_elements: list[str] = sorted(unique_elements)

        return tuple(sorted_elements)

    def formula_matrix(self, species: list[SpeciesData]) -> Array:
        """Formula matrix

        Elements are given in rows and species in columns following the convention in
        :cite:t:`LKS17`.

        Returns:
            The formula matrix
        """
        unique_elements: tuple[str, ...] = self.unique_elements_in_species(species)

        formula_matrix: Array = jnp.zeros((len(unique_elements), len(species)), dtype=jnp.int_)
        for element_index, element in enumerate(unique_elements):
            for species_index, species_ in enumerate(species):
                try:
                    count: int = species_.composition[element]
                except KeyError:
                    count = 0
                formula_matrix = formula_matrix.at[element_index, species_index].set(count)

        return formula_matrix

    def reaction_matrix(self, species: list[SpeciesData]) -> Array:
        """Reaction matrix

        Returns:
            A matrix of linearly independent reactions or None
        """
        # TODO: Would prefer to always return an array
        # if self._species.number == 1:
        #    logger.debug("Only one species therefore no reactions")
        #    return None

        transpose_formula_matrix: Array = self.formula_matrix(species).T

        return partial_rref(transpose_formula_matrix)


def test():

    # Define a JAX-compliant species list
    species_list: list[SpeciesData] = [H2_g, H2O_g, CO2_g, O2_g, CH4_g, CO_g]

    reaction_network: ReactionNetworkJAX = ReactionNetworkJAX()

    formula_matrix: Array = reaction_network.formula_matrix(species_list)

    print(formula_matrix)

    reaction_matrix = reaction_network.reaction_matrix(species_list)
    print(reaction_matrix)


if __name__ == "__main__":
    test()
