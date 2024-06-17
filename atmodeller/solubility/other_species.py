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
"""Solubility laws for other species"""

# Convenient to use chemical formulas so pylint: disable=C0103

from __future__ import annotations

import logging
import sys

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.solubility.interfaces import Solubility, SolubilityPowerLaw
from atmodeller.utilities import UnitConversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class Cl2_ano_dio_for_thomas(Solubility):
    """Solubility of chlorine in silicate melts :cite:p:`TW21`

    Solubility law from :cite:t:`TW21{Figure 4}` showing relation between dissolved Cl
    concentration and Cl fugacity for CMAS composition (An50Di28Fo22
    (anorthite-diopside-forsterite), Fe-free low-degree mantle melt) at 1400 C and 1.5 GPa.
    Experiments from 0.5-2 GPa and 1200-1500 C
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        cl_wtp: float = 140.52 * np.sqrt(fugacity)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(cl_wtp)

        return ppmw


class Cl2_basalt_thomas(Solubility):
    """Solubility of chlorine in silicate melts :cite:p:`TW21`

    Solubility law from :cite:t:`TW21{Figure 4}` showing relation between dissolved Cl
    concentration and Cl fugacity for Icelandic basalt at 1400 C and 1.5 GPa. Experiments from
    0.5-2 GPa and 1200-1500 C
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        cl_wtp: float = 78.56 * np.sqrt(fugacity)
        ppmw: float = UnitConversion.weight_percent_to_ppmw(cl_wtp)

        return ppmw


class He_basalt(Solubility):
    """Solubility of He in tholeittic basalt melt :cite:p:`JWB86`

    Experiments determined Henry's law solubility constant in tholetiitic basalt melt at 1 bar and
    1250-1600 C. Using Henry's Law solubility constant for He from the abstract, convert from STP
    units to mol/g*bar.
    """

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs
        henry_sol_constant: float = 56e-5  # cm3*STP/g*bar
        # Convert Henry solubility constant to mol/g*bar, 2.24e4 cm^3/mol at STP
        he_conc: float = (henry_sol_constant / 2.24e4) * fugacity
        # Convert He conc from mol/g to g H2/g total and then to ppmw
        ppmw: float = he_conc * 4.0026 * 1e6

        return ppmw


class N2_basalt_bernadou(Solubility):
    """N2 in basaltic silicate melt :cite:p:`BGF21`

    :cite:t:`BGF21{Equation 18}` and using :cite:t:`BGF21{Equations 19-20}` and the values for the
    thermodynamic constants from :cite:t:`BGF21{Table 6}`. Experiments on basaltic samples at fluid
    saturation in C-H-O-N system, pressure range: 0.8-10 kbar, temperature range: 1200-1300 C;
    fO2 range: IW+4.9 to IW-4.7. Using their experimental results and a database for N
    concentrations at fluid saturation from 1 bar to 10 kbar, calibrated their solubility law.
    """

    @override
    def concentration(
        self, fugacity: float, *, temperature: float, pressure: float, O2: float, **kwargs
    ) -> float:
        del kwargs
        k13: float = np.exp(
            -(29344 + 121 * temperature + 4 * pressure) / (GAS_CONSTANT * temperature)
        )
        k14: float = np.exp(
            -(183733 + 172 * temperature - 5 * pressure) / (GAS_CONSTANT * temperature)
        )
        molfrac: float = (k13 * fugacity) + ((O2 ** (-3 / 4)) * k14 * (fugacity**0.5))
        ppmw: float = UnitConversion.fraction_to_ppm(molfrac)

        return ppmw


class N2_basalt_dasgupta(Solubility):
    """N2 in silicate melts :cite:p:`DFP22`

    Using :cite:t:`DFP22{Equation 10}`, composition parameters from :cite:t:`DFP22{Figure 8}`, and
    Iron-wustite buffer (logIW_fugacity) from :cite:t:`OP93,HGD08`.

    Performed experiments on 80:20 synthetic basalt-Si3N4 mixture at 1.5-3.0 GPa and 1300-1600 C
    fO2 from ~IW-3 to IW-4. Combined this high pressure data with lower pressure studies to derive
    their N solubility law.

    Args:
        xsio2: Mole fraction of SiO2
        xal2o3: Mole fraction of Al2O3
        xtio2: Mole fraction of TiO2
    """

    def __init__(self, xsio2: float = 0.56, xal2o3: float = 0.11, xtio2: float = 0.01):
        self.xsio2: float = xsio2
        self.xal2o3: float = xal2o3
        self.xtio2: float = xtio2

    @override
    def concentration(
        self, fugacity: float, *, temperature: float, pressure: float, O2: float, **kwargs
    ) -> float:
        del kwargs
        fugacity_gpa: float = UnitConversion.bar_to_GPa(fugacity)
        pressure_gpa: float = UnitConversion.bar_to_GPa(pressure)
        logiw_fugacity: float = (
            -28776.8 / temperature
            + 14.057
            + 0.055 * (pressure - 1) / temperature
            - 0.8853 * np.log(temperature)
        )
        fo2_shift = np.log10(O2) - logiw_fugacity
        ppmw: float = (fugacity_gpa**0.5) * np.exp(
            (5908.0 * (pressure_gpa**0.5) / temperature) - (1.6 * fo2_shift)
        )
        ppmw += fugacity_gpa * np.exp(
            4.67 + (7.11 * self.xsio2) - (13.06 * self.xal2o3) - (120.67 * self.xtio2)
        )

        return ppmw


class N2_basalt_libourel(Solubility):
    """N2 in basalt (tholeiitic) magmas :cite:p:`LMH03`

    :cite:t:`LMH03{Equation 23}`, includes dependencies on fN2 and fO2. Experiments conducted at 1
    atm and 1425 C (two experiments at 1400 C), fO2 from IW-8.3 to IW+8.7 using mixtures of CO, CO2
    and N2 gases.
    """

    def __init__(self):
        self._power_law: SolubilityPowerLaw = SolubilityPowerLaw(constant=0.0611, exponent=1)

    @override
    def concentration(self, fugacity: float, *, O2: float, **kwargs) -> float:
        del kwargs
        ppmw: float = self._power_law.concentration(fugacity)
        constant: float = (O2**-0.75) * 5.97e-10
        power_law: SolubilityPowerLaw = SolubilityPowerLaw(constant=constant, exponent=0.5)
        ppmw += power_law.concentration(fugacity)

        return ppmw
