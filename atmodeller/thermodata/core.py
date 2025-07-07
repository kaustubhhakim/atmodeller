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
"""Core classes and functions for thermochemical and critical data"""

import importlib.resources
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from importlib.abc import Traversable
from pathlib import Path
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import pandas as pd
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller import TEMPERATURE_REFERENCE
from atmodeller.constants import GAS_CONSTANT
from atmodeller.utilities import as_j64, unit_conversion

DATA_DIRECTORY: Traversable = importlib.resources.files(f"{__package__}.data")
"""Data directory"""
THERMODYNAMIC_DATA_SOURCE: Path = Path("nasa_glenn_coefficients.txt")
"""Source of the thermodynamic data"""
CRITICAL_DATA_SOURCE: Path = Path("critical_data.txt")
"""Source of the critical data"""


class CondensateActivity(eqx.Module):
    """Activity of a stable condensate

    Args:
        activity: Activity. Defaults to 1.
    """

    activity: Array = eqx.field(converter=as_j64, default=1.0)
    """Activity"""

    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> Float[Array, ""]:
        """Log activity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log activity, which is dimensionless
        """
        del temperature
        del pressure

        return jnp.log(self.activity)


class ThermodynamicCoefficients(eqx.Module):
    """NASA Glenn coefficients for the thermodynamic properties of an individual species

    Coefficients are available at https://ntrs.nasa.gov/citations/20020085330

    Args:
        b1: Enthalpy constant(s) of integration
        b2: Entropy constant(s) of integration
        cp_coeffs: Heat capacity coefficients
        T_min: Minimum temperature(s) in K in the range
        T_max: Maximum temperature(s) in K in the range
    """

    b1: tuple[float, ...]
    """Enthalpy constant(s) of integration"""
    b2: tuple[float, ...]
    """Entropy constant(s) of integration"""
    cp_coeffs: tuple[tuple[float, ...], ...]
    """Heat capacity coefficients"""
    T_min: Float[Array, " N"] = eqx.field(converter=as_j64, static=True)
    """Minimum temperature(s) in K in the range"""
    T_max: Float[Array, " N"] = eqx.field(converter=as_j64, static=True)
    """Maximum temperature(s) in K in the range"""

    def _get_index(self, temperature: ArrayLike) -> Integer[Array, " T"]:
        """Gets the index of the temperature range for the given temperature

        This assumes the temperature is within one of the ranges and will produce unexpected output
        if the temperature is outside the ranges.

        Args:
            temperature: Temperature in K

        Returns:
            Index of the temperature range
        """
        temperature = jnp.atleast_1d(as_j64(temperature))

        # Reshape for broadcasting
        bool_mask: Bool[Array, "N T"] = (self.T_min[:, None] <= temperature[None, :]) & (
            temperature[None, :] <= self.T_max[:, None]
        )
        index: Integer[Array, " T"] = jnp.argmax(bool_mask, axis=0)

        return index

    def _cp_over_R(
        self, cp_coefficients: Float[Array, "T 7"], temperature: ArrayLike
    ) -> Float[Array, ""]:
        """Heat capacity relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`

        Args:
            cp_coefficients: Heat capacity coefficients
            temperature: Temperature in K

        Returns:
            Heat capacity relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        temperature_terms: Float[Array, "T 7"] = jnp.stack(
            [
                jnp.power(temperature, -2),
                jnp.power(temperature, -1),
                jnp.ones_like(temperature),
                temperature,
                jnp.power(temperature, 2),
                jnp.power(temperature, 3),
                jnp.power(temperature, 4),
            ],
            axis=-1,
        )

        heat_capacity: Float[Array, " T"] = jnp.einsum(
            "ti,ti->t", cp_coefficients, temperature_terms
        )

        return heat_capacity

    def _S_over_R(
        self, cp_coefficients: Float[Array, "T 7"], b2: ArrayLike, temperature: ArrayLike
    ) -> Float[Array, ""]:
        """Entropy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`

        Args:
            cp_coefficients: Heat capacity coefficients
            b2: Entropy integration constant
            temperature: Temperature in K

        Returns:
            Entropy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        temperature_terms: Float[Array, "T 7"] = jnp.stack(
            [
                -jnp.power(temperature, -2) / 2,
                -jnp.power(temperature, -1),
                jnp.log(temperature),
                temperature,
                jnp.power(temperature, 2) / 2,
                jnp.power(temperature, 3) / 3,
                jnp.power(temperature, 4) / 4,
            ],
            axis=-1,
        )

        entropy: Float[Array, " T"] = (
            jnp.einsum("ti,ti->t", cp_coefficients, temperature_terms) + b2
        )

        return entropy

    def _H_over_RT(
        self, cp_coefficients: Float[Array, "T 7"], b1: ArrayLike, temperature: ArrayLike
    ) -> Float[Array, ""]:
        r"""Enthalpy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            temperature: Temperature in K

        Returns:
            Enthalpy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
            :math:`\times T`
        """
        temperature = jnp.atleast_1d(as_j64(temperature))
        temperature_terms: Float[Array, "T 7"] = jnp.stack(
            [
                -jnp.power(temperature, -2),
                jnp.log(temperature) / temperature,
                jnp.ones_like(temperature),
                temperature / 2,
                jnp.power(temperature, 2) / 3,
                jnp.power(temperature, 3) / 4,
                jnp.power(temperature, 4) / 5,
            ],
            axis=-1,
        )

        enthalpy: Float[Array, " T"] = (
            jnp.einsum("ti,ti->t", cp_coefficients, temperature_terms) + b1 / temperature
        )

        return enthalpy

    def _G_over_RT(
        self,
        cp_coefficients: Float[Array, "T 7"],
        b1: ArrayLike,
        b2: ArrayLike,
        temperature: ArrayLike,
    ) -> Float[Array, " T"]:
        r"""Gibbs energy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
        :math:`\times T`

        Args:
            cp_coefficients: Heat capacity coefficients as an array
            b1: Enthalpy integration constant
            b2: Entropy integration constant
            temperature: Temperature

        Returns:
            Gibbs energy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
            :math:`\times T`
        """
        enthalpy: Float[Array, " T"] = self._H_over_RT(cp_coefficients, b1, temperature)
        # jax.debug.print("enthalpy = {out}", out=enthalpy)
        entropy: Float[Array, " T"] = self._S_over_R(cp_coefficients, b2, temperature)
        # jax.debug.print("entropy = {out}", out=entropy)
        # No temperature multiplication is correct since the return is Gibbs energy relative to RT
        gibbs: Float[Array, " T"] = enthalpy - entropy

        return gibbs

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets Gibbs energy to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
        :math:`\times T`

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy relative to :data:`GAS_CONSTANT <atmodeller.constants.GAS_CONSTANT>`
            :math:`\times T`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index)
        b1_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        b2_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b2), index)
        # jax.debug.print("b2_for_index = {out}", out=b2_for_index)
        gibbs_for_index: Float[Array, " T"] = self._G_over_RT(
            cp_coeffs_for_index, b1_for_index, b2_for_index, temperature
        )

        return gibbs_for_index

    def cp(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets heat capacity.

        This is :math:`C_p^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Heat capacity in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        # jax.debug.print("cp_coeffs_for_index = {out}", out=cp_coeffs_for_index.shape)
        cp: Float[Array, " T"] = self._cp_over_R(cp_coeffs_for_index, temperature) * GAS_CONSTANT

        return cp

    def enthalpy(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets enthalpy.

        This is :math:`H` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        b1_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b1), index)
        enthalpy: Float[Array, " T"] = (
            self._H_over_RT(cp_coeffs_for_index, b1_for_index, temperature)
            * GAS_CONSTANT
            * temperature
        )

        return enthalpy

    def reference_enthalpy(self) -> Float[Array, ""]:
        r"""Gets reference enthalpy.

        This is :math:`H^{\circ}(T_r)` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Reference enthalpy in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        index: Integer[Array, ""] = self._get_index(TEMPERATURE_REFERENCE)
        # jax.debug.print("index = {out}", out=index)
        cp_coeffs_for_index: Float[Array, "7"] = jnp.take(jnp.array(self.cp_coeffs), index, axis=0)
        b1_for_index: Float[Array, ""] = jnp.take(jnp.array(self.b1), index)
        # jax.debug.print("b1_for_index = {out}", out=b1_for_index)
        reference_enthalpy: Float[Array, ""] = (
            self._H_over_RT(cp_coeffs_for_index, b1_for_index, TEMPERATURE_REFERENCE)
            * GAS_CONSTANT
            * TEMPERATURE_REFERENCE
        )

        return reference_enthalpy

    def enthalpy_function(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets enthalpy function/increment.

        This is :math:`H-H^{\circ}(T_r)` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Enthalpy increment in :math:`\mathrm{J}\ \mathrm{mol}^{-1}`
        """
        enthalpy: Float[Array, " T"] = self.enthalpy(temperature)
        reference_enthalpy: Float[Array, ""] = self.reference_enthalpy()

        return enthalpy - reference_enthalpy

    def entropy(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets entropy

        This is :math:`S^\circ` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Entropy in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        index: Integer[Array, " T"] = self._get_index(temperature)
        cp_coeffs_for_index: Float[Array, "T 7"] = jnp.take(
            jnp.array(self.cp_coeffs), index, axis=0
        )
        b2_for_index: Float[Array, " T"] = jnp.take(jnp.array(self.b2), index)
        entropy: Float[Array, " T"] = (
            self._S_over_R(cp_coeffs_for_index, b2_for_index, temperature) * GAS_CONSTANT
        )

        return entropy

    def gibbs_function(self, temperature: ArrayLike) -> Float[Array, " T"]:
        r"""Gets Gibbs energy function.

        This is :math:`-[G^\circ-H^{\circ}(T_r)]/T` in the JANAF tables.

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy function in :math:`\mathrm{J}\ \mathrm{K}^{-1} \mathrm{mol}^{-1}`
        """
        gibbs: Float[Array, " T"] = (
            self.get_gibbs_over_RT(temperature) * GAS_CONSTANT * temperature
        )
        gibbs_function: Float[Array, " T"] = -(gibbs - self.reference_enthalpy()) / temperature

        return gibbs_function


@dataclass
class ThermodynamicDataSource:
    """Thermodynamic data source for all species"""

    data: pd.DataFrame = field(init=False)
    """Thermodynamic data for all species"""

    def __post_init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(THERMODYNAMIC_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            self.data = pd.read_csv(datapath, sep=" ", comment="#")

    @property
    def formula_column(self) -> str:
        """Name of the column that refers to the hill formula"""
        return "hill_formula"

    @property
    def state_column(self) -> str:
        """Name of the column that refers to the state of aggregation"""
        return "state"

    def available_species(self) -> tuple[str, ...]:
        """Available species

        Returns:
            Available species
        """
        df: pd.DataFrame = cast(
            pd.DataFrame, self.data[[self.formula_column, self.state_column]].drop_duplicates()
        )
        available_species: tuple[str, ...] = tuple(
            f"{getattr(row, self.formula_column)}_{getattr(row, self.state_column)}"
            for row in df.itertuples(index=False)
        )

        return available_species

    def create_dictionary(self) -> dict[str, ThermodynamicCoefficients]:
        """Dictionary of thermodynamic coefficients for all species

        Returns:
            Dictionary of thermodynamic coefficients for all species
        """
        unique_combinations: pd.DataFrame = cast(
            pd.DataFrame, self.data[[self.formula_column, self.state_column]].drop_duplicates()
        )
        coefficient_dict: dict[str, ThermodynamicCoefficients] = {}

        for row in unique_combinations.itertuples(index=False):
            hill_formula: str = str(getattr(row, self.formula_column))
            state: str = str(getattr(row, self.state_column))
            name: str = f"{hill_formula}_{state}"

            # Find all data across all temperature ranges
            df: pd.DataFrame = cast(
                pd.DataFrame,
                self.data[
                    (self.data[self.formula_column] == hill_formula)
                    & (self.data[self.state_column] == state)
                ],
            )

            # Process and store the thermodynamic coefficients
            T_min: npt.NDArray[np.float64] = df["T_min"].to_numpy(dtype=float)
            T_max: npt.NDArray[np.float64] = df["T_max"].to_numpy(dtype=float)
            b1: tuple[float, ...] = tuple(df["b1"].astype(dtype=float))
            b2: tuple[float, ...] = tuple(df["b2"].astype(dtype=float))
            cp_coeffs: tuple[tuple[np.float64, ...], ...] = tuple(
                map(tuple, df[["a1", "a2", "a3", "a4", "a5", "a6", "a7"]].to_numpy(dtype=float))
            )
            coefficient_dict[name] = ThermodynamicCoefficients(b1, b2, cp_coeffs, T_min, T_max)

        return coefficient_dict


class CriticalData(eqx.Module):
    """Critical temperature and pressure of a gas species

    Args:
        temperature: Critical temperature in K
        pressure: Critical pressure in bar
    """

    temperature: float = 1.0
    """Critical temperature in K"""
    pressure: float = 1.0
    """Critical pressure in bar"""


@dataclass
class CriticalDataSource:
    """Critical data source for all species"""

    data: pd.DataFrame = field(init=False)
    """Critical data for all species"""

    def __post_init__(self):
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(CRITICAL_DATA_SOURCE)  # type: ignore
        )
        with data as datapath:
            self.data = pd.read_csv(datapath, sep=" ", comment="#")

    @property
    def name_column(self) -> str:
        """Name of the column that refers to the hill formula and an optional suffix"""
        return "name"

    @property
    def critical_temperature_column(self) -> str:
        """Name of the column that refers to the critical temperature in K"""
        return "Tc"

    @property
    def critical_pressure_column(self) -> str:
        """Name of the column that refers to the critical pressure"""
        return "Pc"

    def create_dictionary(self) -> dict[str, CriticalData]:
        """Dictionary of critical data for all species

        Returns:
            Dictionary of critical data for all species
        """
        critical_dict: dict[str, CriticalData] = {}

        for row in self.data.itertuples(index=False):
            name: str = str(getattr(row, self.name_column))
            critical_dict[name] = CriticalData(
                temperature=float(getattr(row, self.critical_temperature_column)),
                pressure=float(getattr(row, self.critical_pressure_column)),
            )

        return critical_dict


# Although it might be tempting to create objects on-the-fly when required, this might not play
# nice with JAX, which requires purely functional programming (and no side effects). So instead
# we create dictionaries of instantiated data (JAX-compliant Pytrees) that we can use for lookup.
# It should also be net faster to create these data once and then access (potentially many times).
# These are also set to private to avoid sphinx (autodoc) from printing long strings.
thermodynamic_data_source: ThermodynamicDataSource = ThermodynamicDataSource()
"""Thermodynamic data source

:meta private:
"""
thermodynamic_coefficients_dictionary: dict[str, ThermodynamicCoefficients] = (
    thermodynamic_data_source.create_dictionary()
)
"""Thermodynamic coefficients dictionary

:meta private:
"""
critical_data_source: CriticalDataSource = CriticalDataSource()
"""Critical data source

:meta private:
"""
critical_data_dictionary: dict[str, CriticalData] = critical_data_source.create_dictionary()
"""Critical data dictionary

:meta private:
"""


class IndividualSpeciesData(eqx.Module):
    """Individual species data

    Args:
        formula: Formula
        state: State of aggregation as defined by JANAF
    """

    formula: str
    """Formula"""
    state: str
    """State of aggregation"""
    thermo: ThermodynamicCoefficients = eqx.field(init=False)
    """Thermodynamic coefficient and methods"""
    composition: ImmutableMap[str, tuple[int, float, float]] = eqx.field(init=False)
    """Composition"""
    hill_formula: str = eqx.field(init=False)
    """Hill formula"""
    molar_mass: float = eqx.field(init=False)
    """Molar mass"""

    def __post_init__(self):
        # NOTE: Here, it is intentionally avoided to instantiate ThermodynamicCoefficients, even
        # though the arguments would allow it. Previous tests resulted in an infinite compilation
        # loop for JAX. Instead, we get thermodynamic coefficients from a dictionary.
        mformula: Formula = Formula(self.formula)
        self.composition = ImmutableMap(mformula.composition().asdict())
        self.hill_formula = mformula.formula
        self.molar_mass = mformula.mass * unit_conversion.g_to_kg
        try:
            self.thermo = thermodynamic_coefficients_dictionary[self.name]
        except KeyError:
            raise KeyError(
                f"{self.name} not available. "
                f"Available species are {thermodynamic_data_source.available_species()}"
            )

    @property
    def elements(self) -> tuple[str, ...]:
        """Elements"""
        return tuple(self.composition.keys())

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and state of aggregation"""
        return f"{self.hill_formula}_{self.state}"

    def get_gibbs_over_RT(self, temperature: ArrayLike) -> Array:
        """Gets Gibbs energy over RT

        Args:
            temperature: Temperature in K

        Returns:
            Gibbs energy over RT
        """
        return self.thermo.get_gibbs_over_RT(temperature)
