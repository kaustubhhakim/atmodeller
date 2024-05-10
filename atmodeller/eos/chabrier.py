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
"""Real gas EOSs from :cite:t:`CD21`. 

Examples:
    Evaluate the fugacity coefficient for H2 at 2000 K and 1000 bar::

        from atmodeller.eos.chabrier import H2_CD21
        model = H2_CD21
        fugacity_coefficient = model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)
"""

from __future__ import annotations

import importlib.resources
import logging
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

from atmodeller import GAS_CONSTANT_BAR
from atmodeller.eos import DATA_DIRECTORY
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

CHABRIER_DIRECTORY: Path = Path("chabrier")


@dataclass(kw_only=True)
class Chabrier(RealGas):
    r"""A real gas EOS from :cite:t:`CD21`

    This uses the rho-T-P tables to lookup density (rho).

    This form of the EOS can also be used for the models in :cite:t:`CD21`.

    Args:
        filename: Filename of the density-T-P data
    """

    filename: Path
    """Filename of the density-T-P data"""
    standard_state_pressure: float = field(init=False, default=1)
    """Standard state pressure with the appropriate units"""
    log10density_func: RectBivariateSpline = field(init=False)
    """Spline to evalute the density"""

    def __post_init__(self):
        self._create_spline()

    def _create_spline(self) -> None:
        """Sets spline lookup for density from :cite:t:`CD21` rho-T-P tables"""
        data: AbstractContextManager[Path] = importlib.resources.as_file(
            DATA_DIRECTORY.joinpath(str(CHABRIER_DIRECTORY.joinpath(self.filename)))
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
        self.log10density_func: RectBivariateSpline = RectBivariateSpline(
            pivot_table.index.to_numpy(), pivot_table.columns.to_numpy(), pivot_table.to_numpy()
        )

    def get_molar_density(self, temperature: float, pressure: float) -> float:
        """Gets molar density

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Molar density in # FIXME units?
        """

        log10molar_density_gcc = self.log10density_func(
            np.log10(temperature), np.log10(UnitConversion.bar_to_GPa(pressure))
        )
        # Convert units from g/cc to mol/m3 for H2
        molar_density: float = 1e6 / 2.016 * np.power(10, log10molar_density_gcc.item())

        return molar_density

    # FIXME: Must implement this method.
    @override
    def volume(self, temperature: float, pressure: float) -> float: ...

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        volume_integral: float = -1 * np.log(pressure / self.standard_state_pressure) + (
            pressure - self.standard_state_pressure
        ) / (self.get_molar_density(temperature, pressure) * GAS_CONSTANT_BAR * temperature)
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral


H2_CD21: RealGas = Chabrier(filename=Path("TABLE_H_TP_v1"))
"""H2 :cite:p:`CD21`"""


def get_chabrier_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred Chabrier and colleagues EOS models for each species.

    The latest and/or most sophisticated EOS model is chosen for each species.

    Returns:
        Dictionary of EOS models for each species
    """
    models: dict[str, RealGas] = {}
    models["H2"] = H2_CD21

    return models
