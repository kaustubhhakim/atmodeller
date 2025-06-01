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
"""Real gas EOS from :cite:t:`CD21`"""

import importlib.resources
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jax.scipy.interpolate import RegularGridInterpolator
from jaxtyping import Array, ArrayLike
from molmass import Formula
from xmmutablemap import ImmutableMap

from atmodeller import PRESSURE_REFERENCE
from atmodeller.constants import GAS_CONSTANT_BAR
from atmodeller.eos import DATA_DIRECTORY
from atmodeller.eos._aggregators import CombinedRealGas
from atmodeller.eos.core import RealGas
from atmodeller.utilities import ExperimentalCalibration, unit_conversion

try:
    from typing import override  # type: ignore valid for Python 3.12+
except ImportError:
    from typing_extensions import override  # Python 3.11 and earlier

logger: logging.Logger = logging.getLogger(__name__)


class Chabrier(RealGas):
    r"""Chabrier EOS from :cite:t:`CD21`

    This uses rho-T-P tables to lookup density (rho).

    Args:
        log10_density_func: Spline lookup for density from :cite:t:`CD21` T-P-rho tables
        He_fraction: He fraction
        H2_molar_mass_g_mol: Molar mass of H2
        He_molar_mass_g_mol: Molar mass of He
        integration_steps: Number of integration steps
    """

    CHABRIER_DIRECTORY: ClassVar[Path] = Path("chabrier")
    """Directory of the Chabrier data within :obj:`~atmodeller.eos.data`"""
    He_fraction_map: ClassVar[ImmutableMap[str, float]] = ImmutableMap(
        {
            "TABLE_H_TP_v1": 0.0,
            "TABLE_HE_TP_v1": 1.0,
            "TABLEEOS_2021_TP_Y0275_v1": 0.275,
            "TABLEEOS_2021_TP_Y0292_v1": 0.292,
            "TABLEEOS_2021_TP_Y0297_v1": 0.297,
        }
    )
    """Mole fraction of He in the gas mixture, the other component being H2.
    
    Dictionary keys should correspond to the name of the Chabrier file.
    """
    log10_density_func: Callable
    """Spline lookup for density from :cite:t:`CD21` T-P-rho tables"""
    He_fraction: float
    """He fraction"""
    H2_molar_mass_g_mol: float
    """Molar mass of H2"""
    He_molar_mass_g_mol: float
    """Molar mass of He"""
    integration_steps: int
    """Number of integration steps"""

    @classmethod
    def create(cls, filename: Path, integration_steps: int = 100) -> RealGas:
        """Creates a Chabrier instance

        Args:
            filename: Filename of the density-T-P data
            integration_steps: Number of integration steps. Defaults to 100, which computes the
                fugacity of H2 to within 4% (relative to 1000 steps, for T from 1000 to 5000 K and
                pressure to 10 GPa), and to within 10% (relative to 1000 steps, for T from 1000 to
                5000 K and pressure to 100 GPa). Increasing the integration steps will increase the
                run time but provide better accuracy.

        Returns:
            Instance
        """
        log10_density_func: Callable = cls._get_interpolator(filename)
        He_fraction: float = cls.He_fraction_map[filename.name]
        H2_molar_mass_g_mol: float = Formula("H2").mass
        He_molar_mass_g_mol: float = Formula("He").mass

        return cls(
            log10_density_func,
            He_fraction,
            H2_molar_mass_g_mol,
            He_molar_mass_g_mol,
            integration_steps,
        )

    @classmethod
    def _get_interpolator(cls, filename: Path) -> Callable:
        """Gets spline lookup for density from :cite:t:`CD21` T-P-rho tables.

        The data tables have a slightly different organisation of the header line. But in all cases
        the first three columns contain the required data: log10 T [K], log10 P [GPa], and
        log10 rho [g/cc].

        Args:
            filename: Filename of the density-T-P data

        Returns:
            Interpolator
        """
        # Define column names for the first three columns
        T_name: str = "log T [K]"
        P_name: str = "log P [GPa]"
        rho_name: str = "log rho [g/cc]"
        column_names: list[str] = [T_name, P_name, rho_name]

        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(str(cls.CHABRIER_DIRECTORY.joinpath(filename)))
        )
        with data as datapath:
            df: pd.DataFrame = pd.read_csv(
                datapath,  # type: ignore
                sep=r"\s+",
                comment="#",
                usecols=[0, 1, 2],  # type: ignore
                names=column_names,
                skiprows=2,
            )
        pivot_table: pd.DataFrame = df.pivot(index=T_name, columns=P_name, values=rho_name)
        log_T: Array = jnp.array(pivot_table.index.to_numpy())
        log_P: Array = jnp.array(pivot_table.columns.to_numpy())
        log_rho: Array = jnp.array(pivot_table.to_numpy())

        interpolator: RegularGridInterpolator = RegularGridInterpolator(
            (log_T, log_P), log_rho, method="linear"
        )

        def interpolator_hashable_function_wrapper(x) -> Array:
            """Converts interpolator to a hashable function"""
            return interpolator(x)

        return interpolator_hashable_function_wrapper

    @eqx.filter_jit
    def _convert_to_molar_density(self, log10_density_gcc: ArrayLike) -> Array:
        r"""Converts density to molar density

        Convert units: g/cm3 to mol/cm3 to mol/m3 for H2 (1e6 cm3 = 1 m3; 1 mol H2 = 2.016 g H2)

        Args:
            log10_density_gcc: Log10 density in g/cc

        Returns:
            Molar density in :math:`\mathrm{mol}\mathrm{m}^{-3}`
        """
        molar_density: Array = jnp.power(10, log10_density_gcc) / unit_conversion.cm3_to_m3
        composition_factor: float = (
            self.He_molar_mass_g_mol * self.He_fraction
            + self.H2_molar_mass_g_mol * (1 - self.He_fraction)
        )
        molar_density = molar_density / composition_factor

        return molar_density

    @override
    @eqx.filter_jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Log fugacity

        This performs a numerical integration to compute the fugacity, although an obvious
        speedup to implement is to precompute the fugacity (integral), either by calculating it
        during initialisation or by storing it in a lookup table that is read in.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Log fugacity in bar
        """
        temperature = jnp.asarray(temperature)
        log10_pressure: Array = jnp.log10(pressure)
        temperature, log10_pressure = jnp.broadcast_arrays(temperature, log10_pressure)

        # Pressure range to integrate over
        pressures: Array = jnp.logspace(
            jnp.log10(PRESSURE_REFERENCE), log10_pressure, num=self.integration_steps
        )
        # jax.debug.print("pressures.shape = {out}", out=pressures.shape)
        dP: Array = jnp.diff(pressures, axis=0)
        # jax.debug.print("dP.shape = {out}", out=dP.shape)

        volumes: Array = self.volume(temperature, pressures)
        # jax.debug.print("volumes.shape = {out}", out=volumes.shape)
        avg_volumes: Array = (volumes[:-1] + volumes[1:]) * 0.5
        # jax.debug.print("avg_volumes.shape = {out}", out=avg_volumes.shape)

        # Trapezoid integration
        volume_integral: Array = jnp.sum(avg_volumes * dP, axis=0)
        # jax.debug.print("volume_integral.shape = {out}", out=volume_integral.shape)

        log_fugacity: Array = volume_integral / (GAS_CONSTANT_BAR * temperature)

        return log_fugacity

    @override
    @eqx.filter_jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log10_density_gcc: Array = self.log10_density_func(
            (jnp.log10(temperature), jnp.log10(unit_conversion.bar_to_GPa * pressure))
        )
        # jax.debug.print("log10_density_gcc = {out}", out=log10_density_gcc)
        molar_density: Array = self._convert_to_molar_density(log10_density_gcc)
        volume: Array = jnp.reciprocal(molar_density)
        # jax.debug.print("volume = {out}", out=volume)

        return volume

    @override
    @eqx.filter_jit
    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        log_fugacity: Array = self.log_fugacity(temperature, pressure)
        volume_integral: Array = log_fugacity * GAS_CONSTANT_BAR * temperature

        return volume_integral


calibration_chabrier21: ExperimentalCalibration = ExperimentalCalibration(
    temperature_min=100, temperature_max=1.0e8, pressure_max=1.0e17
)
"""Calibration for :cite:t:`CD21`"""
H2_chabrier21: RealGas = Chabrier.create(Path("TABLE_H_TP_v1"))
"""H2 :cite:p:`CD21`"""
H2_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_chabrier21],
    [calibration_chabrier21],
)
"""H2 bounded :cite:p:`CD21`"""
He_chabrier21: RealGas = Chabrier.create(Path("TABLE_HE_TP_v1"))
"""He :cite:p:`CD21`"""
He_chabrier21_bounded: RealGas = CombinedRealGas.create([He_chabrier21], [calibration_chabrier21])
"""He bounded :cite:p:`CD21`"""
H2_He_Y0275_chabrier21: RealGas = Chabrier.create(Path("TABLEEOS_2021_TP_Y0275_v1"))
"""H2HeY0275 :cite:p:`CD21`"""
H2_He_Y0275_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_He_Y0275_chabrier21], [calibration_chabrier21]
)
"""H2HeY0275 bounded :cite:p:`CD21`"""
H2_He_Y0292_chabrier21: RealGas = Chabrier.create(Path("TABLEEOS_2021_TP_Y0292_v1"))
"""H2HeY0292 :cite:p:`CD21`"""
H2_He_Y0292_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_He_Y0292_chabrier21], [calibration_chabrier21]
)
"""H2HeY0292 bounded :cite:p:`CD21`"""
H2_He_Y0297_chabrier21: RealGas = Chabrier.create(Path("TABLEEOS_2021_TP_Y0297_v1"))
"""H2HeY0297 :cite:p:`CD21`"""
H2_He_Y0297_chabrier21_bounded: RealGas = CombinedRealGas.create(
    [H2_He_Y0297_chabrier21], [calibration_chabrier21]
)
"""H2HeY0297 bounded :cite:p:`CD21`"""


def get_chabrier_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of EOS models

    Returns:
        Dictionary of EOS models
    """
    eos_models: dict[str, RealGas] = {}
    eos_models["H2_chabrier21"] = H2_chabrier21_bounded
    eos_models["H2_He_Y0275_chabrier21"] = H2_He_Y0275_chabrier21_bounded
    eos_models["H2_He_Y0292_chabrier21"] = H2_He_Y0292_chabrier21_bounded
    eos_models["H2_He_Y0297_chabrier21"] = H2_He_Y0297_chabrier21_bounded
    eos_models["He_chabrier21"] = He_chabrier21_bounded

    return eos_models
