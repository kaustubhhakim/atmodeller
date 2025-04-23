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
"""Solubility laws for hydrogen species

For every law there should be a test in the test suite.
"""

import sys

import jax.numpy as jnp
from jax import jit
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller.eos._chabrier import H2_chabrier21_bounded
from atmodeller.eos._zhang_duan import H2_zhang09_bounded
from atmodeller.interfaces import SolubilityProtocol
from atmodeller.solubility.classes import Solubility, SolubilityPowerLaw, SolubilityPowerLawLog10
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

H2_andesite_hirschmann12: SolubilityProtocol = SolubilityPowerLawLog10(1.01058631, 0.60128868)
"""H2 in synthetic andesite :cite:p:`HWA12`

Log-scale linear fit to fH2 vs H2 concentration for andesite in :cite:t:`HWA12{Table 2}`.
Experiments conducted from 0.7-3 GPa at 1400 C.
"""

H2_basalt_hirschmann12: SolubilityProtocol = SolubilityPowerLawLog10(1.10083602, 0.52413928)
"""H2 in synthetic basalt :cite:p:`HWA12`

Log-scale linear fit to fH2 vs. H2 concentration for basalt in :cite:t:`HWA12{Table 2}`.
Experiments conducted from 0.7-3 GPa, 1400 C.
"""

H2_silicic_melts_gaillard03: SolubilityProtocol = SolubilityPowerLaw(0.163, 1.252)
"""Fe-H redox exchange in silicate glasses :cite:p:`GSM03`

Power law fit for fH2 vs. H2 (ppm-wt) from :cite:t:`GSM03{Table 4}` data. Experiments at
pressures from 0.02-70 bar, temperatures from 300-1000C.
"""

H2O_ano_dio_newcombe17: SolubilityProtocol = SolubilityPowerLaw(727, 0.5)
"""H2O in anorthite-diopside-eutectic compositions :cite:p:`NBB17`

Power law from :cite:t:`NBB17{Figure 5(A)}` for anorthite-diopside glass. Experiments conducted
at 1 atm and 1350 C. Melts equilibrated in 1 atm furnace with H2/CO2 gas mixtures that spanned
fO2 from IW-3 to IW+4.8 and pH2/pH2O from 0.003-24.
"""

H2O_basalt_dixon95: SolubilityProtocol = SolubilityPowerLaw(965, 0.5)
"""H2O in MORB liquids :cite:p:`DSH95`

Refitted data to a power law by Paolo Sossi (fitting :cite:t:`DSH95{Figure 4}`, TODO: CHECK).
Experiments conducted at 1200 C, 200-717 bars with pure H2O.
"""

H2O_basalt_mitchell17: SolubilityProtocol = SolubilityPowerLaw(258.946, 0.669)
"""H2O in basaltic melt :cite:p:`MGO17`

Refitted the H2O wt. % vs. fH2O fitted line from :cite:t:`MGO17{Figure 8}` to a power-law.
Experiments conducted at 1200 C and 1000 MPa total pressure. This fit includes data from
their experiments and prior studies on H2O solubility in basaltic melt at 1200 C and P at or
below 600 MPa.
"""

H2O_lunar_glass_newcombe17: SolubilityProtocol = SolubilityPowerLaw(683, 0.5)
"""H2O in lunar basalt :cite:p:`NBB17`

Power law from :cite:t:`NBB17{Figure 5(A)}` for Lunar glass. Experiments conducted at 1 atm and
1350 C. Melts equilibrated in 1-atm furnace with H2/CO2 gas mixtures that spanned fO2 from IW-3
to IW+4.8.
"""

H2O_peridotite_sossi23: SolubilityProtocol = SolubilityPowerLaw(647, 0.5)
"""H2O in peridotite liquids :cite:p:`STB23`

Power law parameters in the abstract for peridotitic glasses. Experiments conducted at 2173 K
and 1 bar and range of fO2 from IW-1.9 to IW+6.0.
"""


@register_pytree_node_class
class _H2_chachan18(Solubility):
    """H2 solubility :cite:p:`CS18`

    Args:
        f_calibration: Calibration fugacity
        T_calibration: Calibration temperature
        X_calibration: Mass fraction at calibration conditions
        T0: Arrhenius temperature factor in K, which expresses the repulsive interaction of the
            molecule with magma. Defaults to 4000 K, which is the middle of the range the authors
            explore (from 3000 K to 5000 K).

    Attributes:
        T0: Arrhenius temperature factor
    """

    def __init__(
        self,
        f_calibration: ArrayLike,
        T_calibration: ArrayLike,
        X_calibration: ArrayLike,
        T0: float = 4000,
    ):
        self.f_calibration: ArrayLike = f_calibration
        self.T_calibration: ArrayLike = T_calibration
        self.X_calibration: ArrayLike = X_calibration
        self.T0: float = T0
        self.A: ArrayLike = jnp.exp(
            (self.T0 / self.T_calibration) + jnp.log(self.X_calibration / self.f_calibration)
        )
        # jax.debug.print("A = ", self.A)

    @override
    @jit
    def concentration(self, fugacity: ArrayLike, *, temperature: ArrayLike, **kwargs) -> ArrayLike:
        del kwargs
        mass_fraction: ArrayLike = self.A * fugacity * jnp.exp(-self.T0 / temperature)
        ppmw: ArrayLike = mass_fraction * unit_conversion.fraction_to_ppm

        return ppmw

    def tree_flatten(self) -> tuple[tuple, dict[str, float]]:
        children: tuple = ()
        aux_data = {
            "f_calibration": self.f_calibration,
            "T_calibration": self.T_calibration,
            "X_calibration": self.X_calibration,
            "T0": self.T0,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


H2_chachan18: SolubilityProtocol = _H2_chachan18(
    f_calibration=1500, T_calibration=1673, X_calibration=0.001
)
"""H2 by combining theory and experiment :cite:p:`CS18`"""

# At 1 GPa in the presence of pure H2, the molecular H2 concentration is 0.19 wt.%"
# Need to convert pressure to H2 fugacity
T_calibration: float = 1673
P_calibration: float = 1 * unit_conversion.GPa_to_bar
f_calibration: ArrayLike = H2_chabrier21_bounded.fugacity(T_calibration, P_calibration)
print("f_calibration = ", f_calibration)
f_calibration2: ArrayLike = H2_zhang09_bounded.fugacity(T_calibration, P_calibration)
print("f_calibration2 = ", f_calibration2)
X_calibration: float = 0.0019

H2_kite19: SolubilityProtocol = _H2_chachan18(
    f_calibration=f_calibration, T_calibration=T_calibration, X_calibration=X_calibration
)
"""H2 by combining theory and experiment :cite:p:`KFS19`."""
