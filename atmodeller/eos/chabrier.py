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
"""Real gas EOSs from :cite:t:`CD2021`. 

Examples:
    Evaluate the fugacity coefficient for H2 at 2000 K and 1000 bar::

        from atmodeller.eos.chabrier import H2_CD2021
        model = H2_CD2021
        fugacity_coefficient = model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)
"""

from __future__ import annotations

import logging
import sys
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT_BAR
from atmodeller.eos.interfaces import RealGas
from atmodeller.utilities import UnitConversion

import pandas as pd
import scipy.interpolate as si 

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Chabrier(RealGas):
    """A real gas EOS from :cite:t:`CD2021`

    This form of the EOS can also be used for the models in :cite:t:`CD2021`.

    Args:
        critical_temperature: Critical temperature in K. Defaults to unity meaning not a
            corresponding states model.
        critical_pressure: Critical pressure in bar. Defaults to unity meaning not a corresponding
            states model.
    """

    standard_state_pressure: float = field(init=False, default=1)
    """Standard state pressure with the appropriate units"""

    def __init__(self):
        """Initialise spline

        """
        self._create_spline()

    def _create_spline(self) -> None:
        """Obtain density from Chabrier and Debras (2021) rho-T-P tables
        
        """   

        columns = pd.read_fwf('./data/TABLE_H_TP_v1', 
                              widths=(16, 15, 15, 15, 16, 15, 15, 15, 15, 15)).columns
        df = pd.read_fwf('./data/TABLE_H_TP_v1', 
                         widths=(16, 15, 15, 15, 16, 15, 15, 15, 15, 15), header=None, comment='#')
        df.columns = columns
        pivot_table = df.pivot(index="#log T [K]", columns="log P [GPa]", values="log rho [g/cc]")

        self.log10density_func = si.RectBivariateSpline(pivot_table.index.to_numpy(), 
                                                        pivot_table.columns.to_numpy(),  
                                                        pivot_table.to_numpy())


    def get_molar_density(self, temperature: float, pressure: float) -> float:
        """Obtain density from Chabrier and Debras (2021) rho-T-P tables
        
        """   

        log10molar_density_gcc = self.log10density_func(np.log10(temperature),
                                                   np.log10(UnitConversion.bar_to_GPa(pressure)))
        # convert units from g/cc to mol/m3 for H2
        molar_density = 1e6 / 2.016 * np.power(10, log10molar_density_gcc.item())
        
        return molar_density

    @override
    def volume_integral(self, temperature: float, pressure: float) -> float:
        r"""Volume integral :cite:p:`CD2021`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        volume_integral: float = (
            (
                - 1 * np.log(pressure / self.standard_state_pressure)
                + (pressure - self.standard_state_pressure) / 
                (self.get_molar_density(temperature, pressure) * GAS_CONSTANT_BAR * temperature) 
            )
        )
        volume_integral = UnitConversion.m3_bar_to_J(volume_integral)

        return volume_integral