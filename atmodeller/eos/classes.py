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
"""Concrete classes for real gas equations of state"""

import importlib.resources
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import optimistix as optx
import pandas as pd
from jax import Array, jit
from jax.scipy.integrate import trapezoid
from jax.scipy.interpolate import RegularGridInterpolator
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike
from molmass import Formula

from atmodeller import GAS_CONSTANT_BAR
from atmodeller.eos import DATA_DIRECTORY
from atmodeller.eos.core import RealGas
from atmodeller.utilities import (
    ExperimentalCalibrationNew,
    PyTreeNoData,
    unit_conversion,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


@register_pytree_node_class
class IdealGas(PyTreeNoData, RealGas):
    r"""Ideal gas equation of state:

    .. math::

        R T = P V

    where :math:`R` is the gas constant, :math:`T` is temperature, :math:`P` is pressure, and
    :math:`V` is volume.
    """

    # Validity of ideal gas model depends on species and conditions
    _calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew()

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        del temperature
        return jnp.log(pressure)

    @override
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        return self.ideal_volume(temperature, pressure)


@register_pytree_node_class
class BeattieBridgeman(RealGas):
    r"""Beattie-Bridgeman equation :cite:p:`HWZ58{Equation 1}`.

    .. math::

        PV^2 = RT\left(1-\frac{c}{VT^3}\right)\left(V+B_0-\frac{bB_0}{V}\right)
        - A_0\left(1-\frac{a}{V}\right)

    Args:
        A0: A0 empirical constant determined experimentally
        a: a empirical constant determined experimentally
        B0: B0 empirical constant determined experimentally
        b: b empirical constant determined experimentally
        c: c empirical constant determined experimentally
        calibration: Experimental calibration. Defaults to empty.

    Attributes:
        A0: A0 empirical constant determined experimentally
        a: a empirical constant determined experimentally
        B0: B0 empirical constant determined experimentally
        b: b empirical constant determined experimentally
        c: c empirical constant determined experimentally
        calibration: Experimental calibration. Defaults to empty.
    """

    # Convenient to use symbols from the paper so pylint: disable=invalid-name
    def __init__(
        self,
        A0: float,
        a: float,
        B0: float,
        b: float,
        c: float,
        calibration: ExperimentalCalibrationNew = ExperimentalCalibrationNew(),
    ):
        self.A0: float = A0
        self.a: float = a
        self.B0: float = B0
        self.b: float = b
        self.c: float = c
        self._calibration: ExperimentalCalibrationNew = calibration

    # pylint: enable=invalid-name

    @jit
    def _objective_function(self, volume: ArrayLike, kwargs: dict[str, ArrayLike]) -> Array:
        r"""Objective function to solve for the volume :cite:p:`HWZ58{Equation 2}`

        .. math::

            PV^4 - RTV^3 - \left(RTB_0 - \frac{Rc}{T^2}-A_0\right)V^2
            +\left(RTbB_0+\frac{RcB_0}{T^2}-aA_0\right)V - \frac{RcbB_0}{T^2}=0

        Args:
            volume: Volume
            kwargs: Dictionary with other required parameters

        Returns:
            Residual of the objective function
        """
        temperature: ArrayLike = kwargs["temperature"]
        pressure: ArrayLike = kwargs["pressure"]

        coeff0: Array = 1 / jnp.square(temperature) * -GAS_CONSTANT_BAR * self.c * self.b * self.B0
        coeff1: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self.c * self.B0
            + GAS_CONSTANT_BAR * temperature * self.b * self.B0
            - self.a * self.A0
        )
        coeff2: Array = (
            1 / jnp.square(temperature) * GAS_CONSTANT_BAR * self.c
            - GAS_CONSTANT_BAR * temperature * self.B0
            + self.A0
        )
        coeff3: ArrayLike = -GAS_CONSTANT_BAR * temperature

        residual: Array = (
            coeff0
            + coeff1 * volume
            + coeff2 * jnp.power(volume, 2)
            + coeff3 * jnp.power(volume, 3)
            + pressure * jnp.power(volume, 4)
        )

        return residual

    @override
    @jit
    def log_fugacity(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
    ) -> Array:
        """Log fugacity :cite:p:`HWZ58{Equation 11}`.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        volume: ArrayLike = self.volume(temperature, pressure)
        log_fugacity: Array = (
            jnp.log(GAS_CONSTANT_BAR * temperature / volume)
            + (
                self.B0
                - self.c / jnp.power(temperature, 3)
                - self.A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 2
            / volume
            - (
                self.b * self.B0
                + self.c * self.B0 / temperature**3
                - self.a * self.A0 / (GAS_CONSTANT_BAR * temperature)
            )
            * 3
            / (2 * jnp.square(volume))
            + (self.c * self.b * self.B0 / jnp.power(temperature, 3))
            * 4
            / (3 * jnp.power(volume, 3))
        )

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Volume

        :cite:t:`HWZ58` doesn't say which root to take, but one real root is very small and the
        maximum real root gives a volume that agrees with the tabulated compressibility parameter
        for all species.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        # Start with a large initial guess, say some factor of the ideal gas volume, to guide the
        # Newton solver to the largest root, which agrees with the tabulated data in the paper.
        # The choice of 10 below is somewhat arbitrary, but based on the calibration data for the
        # Holley model should be comfortably larger than the actual volume.
        scaling_factor: float = 10
        initial_volume = scaling_factor * self.ideal_volume(temperature, pressure)

        kwargs: dict[str, ArrayLike] = {"temperature": temperature, "pressure": pressure}

        solver = optx.Newton(rtol=1.0e-8, atol=1.0e-8)
        sol = optx.root_find(
            self._objective_function,
            solver,
            initial_volume,
            args=kwargs,
        )
        volume: ArrayLike = sol.value

        return volume

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "A0": self.A0,
            "a": self.a,
            "B0": self.B0,
            "b": self.b,
            "c": self.c,
            "calibration": self.calibration,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class Chabrier(RealGas):
    r"""Chabrier EOS from :cite:t:`CD21`

    This uses rho-T-P tables to lookup density (rho).

    Args:
        filename: Filename of the density-T-P data

    Attributes:
        filename: Filename of the density-T-P data
    """

    CHABRIER_DIRECTORY: Path = Path("chabrier")
    """Directory of the Chabrier data within :obj:`~atmodeller.eos.data`."""

    def __init__(self, filename: Path, species_name: str):
        self.filename: Path = filename
        self.species_name: str = species_name
        # For self-consistency this should be the same as Pref in atmodeller.thermodata.core
        self._standard_state_pressure: float = 1
        self._log10_density_func: RegularGridInterpolator = self._get_spline()
        self._molar_mass_g_mol: float = Formula(species_name).mass

    def _get_spline(self) -> RegularGridInterpolator:
        """Gets spline lookup for density from :cite:t:`CD21` T-P-rhp tables.

        The first 3 columns contain log10 T [K], log10 P [GPa], log10 rho [g/cc].
        """
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(str(self.CHABRIER_DIRECTORY.joinpath(self.filename)))
        )
        with data as datapath:
            columns: pd.Index = pd.read_fwf(
                datapath, widths=(16, 15, 15, 15, 16, 15, 15, 15, 15, 15)
            ).columns
            df: pd.DataFrame = pd.read_fwf(
                datapath,
                widths=(16, 15, 15, 15, 16, 15, 15, 15, 15, 15),
                header=None,
                comment="#",
            )
        df.columns = columns
        pivot_table: pd.DataFrame = df.pivot(
            index="#log T [K]", columns="log P [GPa]", values="log rho [g/cc]"
        )
        # Convenient to use T and P so pylint: disable=invalid-name
        log_T: Array = jnp.array(pivot_table.index.to_numpy())
        log_P: Array = jnp.array(pivot_table.columns.to_numpy())
        # pylint: enable=invalid-name
        log_rho: Array = jnp.array(pivot_table.to_numpy())

        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (log_T, log_P), log_rho, method="linear"
        )

        return interpolator

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log10_density_gcc: Array = self._log10_density_func(
            (jnp.log10(temperature), jnp.log10(unit_conversion.bar_to_GPa * pressure))
        )
        # Convert units: g/cm3 to mol/cm3 to mol/m3 for H2 (1e6 cm3 = 1 m3; 1 mol H2 = 2.016 g H2)
        molar_density: Array = jnp.power(10, log10_density_gcc) / (
            unit_conversion.cm3_to_m3 * self._molar_mass_g_mol
        )
        volume: Array = 1 / molar_density

        return volume

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        # Pressure range to integrate over
        pressures: Array = jnp.logspace(
            jnp.log10(self._standard_state_pressure), jnp.log10(pressure), num=1000
        )
        temperatures: Array = jnp.full_like(pressures, temperature)
        volumes: Array = self.volume(temperatures, pressures)
        volume_integral: Array = trapezoid(volumes, pressures)
        log_fugacity: Array = volume_integral / (GAS_CONSTANT_BAR * temperature)

        return log_fugacity

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {"filename": self.filename, "species_name": self.species_name}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)
