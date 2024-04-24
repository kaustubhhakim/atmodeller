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
"""Real gas EOS from :cite:t:`HP91,HP98,HP11`

You will usually want to use the CORK models, since these are the most complete. The MRK models are
nevertheless useful for comparison and understanding the influence of the virial compensation term 
that is encapsulated within the CORK model.

Examples:
    Evaluate the fugacity coefficient for the H2O CORK model from :cite:t:`HP98` at 2000 K and
    1000 bar::

        from atmodeller.eos.holland import H2O_CORK_HP98
        model = H2O_CORK_HP98
        fugacity_coefficient = model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)

    Get the preferred EOS models for various species from the Holland and Powell models::
    
        from atmodeller.eos.holland import get_holland_eos_models
        models = get_holland_eos_models()
        # List the available species
        models.keys()
        # Get the EOS model for CO
        co_model = models['CO']
        # Determine the fugacity coefficient at 2000 K and 1000 bar
        fugacity_coefficient = co_model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.constants import kilo

from atmodeller.eos.interfaces import (
    CORK,
    MRKCriticalBehaviour,
    MRKExplicitABC,
    MRKImplicitABC,
    RealGas,
    critical_parameters,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MRKCorrespondingStatesHP91(MRKExplicitABC):
    """An MRK simplified model used for corresponding states :cite:p:`HP91`

    Universal constants from :cite:t:`HP91{Table 2}`

    Note the unit conversion to SI and pressure in bar using the values in Table 2:

        * `a` coefficients have been multiplied by 1e-4
        * `b0` has been multiplied by 1e-2
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(5.45963e-9, -8.63920e-10, 0))
    b0: float = field(init=False, default=9.18301e-6)

    @classmethod
    def get_species(cls, species: str) -> RealGas:
        """Gets an MRK corresponding states model for a given species.

        Args:
            species: A species, which must be a key in
                :obj:`atmodeller.eos.interfaces.critical_parameters`

        Returns:
            A corresponding states model for the species
        """
        return cls(
            critical_temperature=critical_parameters[species].temperature,
            critical_pressure=critical_parameters[species].pressure,
        )


@dataclass(kw_only=True)
class CORKCorrespondingStatesHP91(CORK):
    """A Simplified Compensated-Redlich-Kwong (CORK) equation :cite:p:`HP91`.

    Although originally fit to CO2 data, this predicts the volumes and fugacities for several other
    gases which are known to obey approximately the principle of corresponding states. The
    corresponding states parameters are from :cite:t:`HP91{Table 2}`. Note also in this case it
    appears :attr:`P0` is always zero, even though for the full CORK equations it determines
    whether or not the virial contribution is added. It assumes there are no complications of
    critical behaviour in the P-T range considered.

    The unit conversions to SI and pressure in bar mean that every virial coefficient has been
    multiplied by 1e-2 compared to the values in :cite:t:`HP91{Table 2}`.

    Args:
        critical_temperature: Critical temperature in K
        critical_pressure: Critical pressure in bar
        mrk: Fugacity model for computing the MRK contribution
    """

    P0: float = field(init=False, default=0)
    a_virial: tuple[float, float] = field(init=False, default=(6.93054e-9, -8.38293e-10))
    b_virial: tuple[float, float] = field(init=False, default=(-3.30558e-7, 2.30524e-8))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))

    @classmethod
    def get_species(cls, species: str) -> RealGas:
        """Gets a CORK corresponding states model for a given species

        Args:
            species: A species, which must be a key in
                :obj:`atmodeller.eos.interfaces.critical_parameters`

        Returns:
            A corresponding states model for the species
        """
        mrk: RealGas = MRKCorrespondingStatesHP91.get_species(species)

        return cls(
            mrk=mrk,
            critical_temperature=critical_parameters[species].temperature,
            critical_pressure=critical_parameters[species].pressure,
        )


CO2_MRK_simple_HP91: RealGas = MRKCorrespondingStatesHP91.get_species("CO2")
"""CO2 MRK corresponding states :cite:p:`HP91`"""
CH4_MRK_HP91: RealGas = MRKCorrespondingStatesHP91.get_species("CH4")
"""CH4 MRK corresponding states :cite:p:`HP91`"""
H2_MRK_HP91: RealGas = MRKCorrespondingStatesHP91.get_species("H2_Holland")
"""H2 MRK corresponding states :cite:p:`HP91`"""
CO_MRK_HP91: RealGas = MRKCorrespondingStatesHP91.get_species("CO")
"""CO MRK corresponding states :cite:p:`HP91`"""
N2_MRK_HP91: RealGas = MRKCorrespondingStatesHP91.get_species("N2")
"""N2 MRK corresponding states :cite:p:`HP91`"""
S2_MRK_HP11: RealGas = MRKCorrespondingStatesHP91.get_species("S2")
"""S2 MRK corresponding states :cite:p:`HP91`"""
H2S_MRK_HP11: RealGas = MRKCorrespondingStatesHP91.get_species("H2S")
"""H2S MRK corresponding states :cite:p:`HP91`"""

CO2_CORK_simple_HP91: RealGas = CORKCorrespondingStatesHP91.get_species("CO2")
"""CO2 CORK corresponding states :cite:p:`HP91`"""
CH4_CORK_HP91: RealGas = CORKCorrespondingStatesHP91.get_species("CH4")
"""CH4 CORK corresponding states :cite:p:`HP91`"""
H2_CORK_HP91: RealGas = CORKCorrespondingStatesHP91.get_species("H2_Holland")
"""H2 CORK corresponding states :cite:p:`HP91`"""
CO_CORK_HP91: RealGas = CORKCorrespondingStatesHP91.get_species("CO")
"""CO CORK corresponding states :cite:p:`HP91`"""
N2_CORK_HP91: RealGas = CORKCorrespondingStatesHP91.get_species("N2")
"""N2 CORK corresponding states :cite:p:`HP91`"""
S2_CORK_HP11: RealGas = CORKCorrespondingStatesHP91.get_species("S2")
"""S2 CORK corresponding states :cite:p:`HP11`"""
H2S_CORK_HP11: RealGas = CORKCorrespondingStatesHP91.get_species("H2S")
"""H2S CORK corresponding states :cite:p:`HP11`"""

# For any subclass of MRKImplicitABC, note the unit conversion to SI and pressure in bar compared
# to the values that Holland and Powell present in Table 1. These are different by 1000 compared to
# the corresponding states scaling, because in the corresponding states formulation the
# coefficients contain a (kilo) pressure scaling as well.
#   a coefficients have been multiplied by 1e-7
#   b0 coefficient has been multiplied by 1e-5

# The critical temperature for the CORK H2O model
Tc_H2O: float = 695  # K
# The temperature at which a_gas = a; hence the critical point is handled by a single a parameter
Ta_H2O: float = 673  # K
# b parameter value is the same across all phases (i.e. gas, fluid, liquid)
b0_H2O: float = 1.465e-5


@dataclass(kw_only=True)
class _MRKH2OLiquidHP91(MRKImplicitABC):
    """MRK for liquid H2O. Equation 6, Holland and Powell (1991)"""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1113.4e-7, -0.88517e-7, 4.53e-10, -1.3183e-12),
    )
    b0: float = field(init=False, default=b0_H2O)
    Ta: float = field(init=False, default=Ta_H2O)

    def delta_temperature_for_a(self, temperature: float) -> float:
        """Temperature difference for the calculation of the a parameter"""
        return self.Ta - temperature

    def volume(self, *args, **kwargs) -> float:
        """Volume

        Args:
            *args: Positional arguments to pass to self.volume_roots
            **kwargs: Keyword arguments to pass to self.volume_roots

        Returns:
            Volume in m^3/mol
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        return np.min(volume_roots)


@dataclass(kw_only=True)
class _MRKH2OGasHP91(MRKImplicitABC):
    """MRK for gaseous H2O. Equation 6a, Holland and Powell (1991)"""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4e-7,
            5.8487e-7,
            -2.1370e-9,
            6.8133e-12,
        ),
    )
    b0: float = field(init=False, default=b0_H2O)
    Ta: float = field(init=False, default=Ta_H2O)

    def delta_temperature_for_a(self, temperature: float) -> float:
        """Temperature difference for the calculation of the a parameter"""
        return self.Ta - temperature

    def volume(self, *args, **kwargs) -> float:
        """Volume

        Args:
            *args: Positional arguments to pass to self.volume_roots
            **kwargs: Keyword arguments to pass to self.volume_roots

        Returns:
            Volume in m^3/mol
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        return np.max(volume_roots)


@dataclass(kw_only=True)
class _MRKH2OFluidHP91(MRKImplicitABC):
    """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991)"""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4e-7,
            -0.22291e-7,
            -3.8022e-11,
            1.7791e-14,
        ),
    )
    b0: float = field(init=False, default=b0_H2O)
    Ta: float = field(init=False, default=Ta_H2O)
    Tc: float = field(init=False, default=Tc_H2O)

    def delta_temperature_for_a(self, temperature: float) -> float:
        """Temperature difference for the calculation of the a parameter"""
        return temperature - self.Ta

    def volume(self, *args, **kwargs) -> float:
        """Volume

        Args:
            *args: Positional arguments to pass to self.volume_roots
            **kwargs: Keyword arguments to pass to self.volume_roots

        Returns:
            Volume in m^3/mol
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        # It appears that there is only ever a single root, even if Ta < temperature < Tc. Holland
        # and Powell state that a single root exists if temperature > Tc, but this appears to be
        # true if temperature > Ta.
        assert volume_roots.size == 1

        return volume_roots[0]


@dataclass(kw_only=True)
class MRKCO2HP91(MRKImplicitABC):
    """MRK for CO2. Holland and Powell (1991)"""

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(741.2e-7, -0.10891e-7, -3.4203e-11, 0),
    )
    b0: float = field(init=False, default=3.057e-5)
    Ta: float = field(init=False, default=0)

    def delta_temperature_for_a(self, temperature: float) -> float:
        return temperature - self.Ta

    def volume(self, *args, **kwargs) -> float:
        """Volume

        Args:
            *args: Positional arguments to pass to self.volume_roots
            **kwargs: Keyword arguments to pass to self.volume_roots

        Returns:
            Volume in m^3/mol
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        # In some cases there are more than a single root, in which case the maximum value
        # maintains continuity/monotonicity with the single root cases. Furthermore, the max value
        # passed the tests that were previously configured when Newton's method was used instead.
        return np.max(volume_roots)


CO2_MRK_HP91: RealGas = MRKCO2HP91()

# For completeness, the MRK model for CO2 in 1998 is the same as the 1991 paper.
CO2_MRK_HP98: RealGas = MRKCO2HP91()


@dataclass(kw_only=True)
class MRKH2OHP91(MRKCriticalBehaviour):
    """MRK for H2O that spans the range across the critical behaviour"""

    mrk_fluid: MRKImplicitABC = field(init=False, default_factory=_MRKH2OFluidHP91)
    mrk_gas: MRKImplicitABC = field(init=False, default_factory=_MRKH2OGasHP91)
    mrk_liquid: MRKImplicitABC = field(init=False, default_factory=_MRKH2OLiquidHP91)
    Ta: float = field(init=False, default=Ta_H2O)
    Tc: float = field(init=False, default=Tc_H2O)

    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991)

        Args:
            temperature: Temperature in kelvin

        Returns:
            Saturation curve pressure in bar
        """
        Psat: float = (
            -13.627
            + 7.29395e-4 * temperature**2
            - 2.34622e-6 * temperature**3
            + 4.83607e-12 * temperature**5
        )
        return Psat


H2O_MRK_HP91: RealGas = MRKH2OHP91()

# For completeness, the MRK model for H2O in 1998 is the same as the 1991 paper.
H2O_MRK_HP98: RealGas = MRKH2OHP91()

# For the Full CORK models below, the virial coefficients in the Holland and Powell papers need
# converting to SI units and pressure in bar as follows, where k = kilo = 1000:
#    a_virial = a_virial (Holland and Powell) * 10**(-5) / k
#    b_virial = b_virial (Holland and Powell) * 10**(-5) / k**(1/2)
#    c_virial = c_virial (Holland and Powell) * 10**(-5) / k**(1/4)

a_conversion: Callable[[tuple[float, ...]], tuple[float, ...]] = lambda x: tuple(
    map(lambda y: y * 1e-5 / kilo, x)
)
b_conversion: Callable[[tuple[float, ...]], tuple[float, ...]] = lambda x: tuple(
    map(lambda y: y * 1e-5 / kilo**0.5, x)
)
c_conversion: Callable[[tuple[float, ...]], tuple[float, ...]] = lambda x: tuple(
    map(lambda y: y * 1e-5 / kilo**0.25, x)
)

CO2_CORK_HP91: RealGas = CORK(
    P0=5000,
    mrk=CO2_MRK_HP91,
    a_virial=a_conversion((1.33790e-2, -1.01740e-5)),
    b_virial=b_conversion((-2.26924e-1, 7.73793e-5)),
)

CO2_CORK_HP98: RealGas = CORK(
    P0=5000,
    mrk=CO2_MRK_HP98,
    a_virial=a_conversion((5.40776e-3, -1.59046e-6)),
    b_virial=b_conversion((-1.78198e-1, 2.45317e-5)),
)

H2O_CORK_HP91: RealGas = CORK(
    P0=2000,
    mrk=MRKH2OHP91(),
    a_virial=a_conversion((-3.2297554e-3, 2.2215221e-6)),
    b_virial=b_conversion((-3.025650e-2, -5.343144e-6)),
)

H2O_CORK_HP98: RealGas = CORK(
    P0=2000,
    mrk=H2O_MRK_HP98,
    a_virial=a_conversion((1.9853e-3, 0)),
    b_virial=b_conversion((-8.9090e-2, 0)),
    c_virial=c_conversion((8.0331e-2, 0)),
)


def get_holland_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred EOS models to use for each species.

    Returns:
        Dictionary of preferred EOS models for each species
    """
    models: dict[str, RealGas] = {}
    models["CH4"] = CH4_CORK_HP91
    models["CO"] = CO_CORK_HP91
    models["CO2"] = CO2_CORK_HP98
    models["H2"] = H2_CORK_HP91
    models["H2O"] = H2O_CORK_HP98
    models["H2S"] = H2S_CORK_HP11
    models["N2"] = N2_CORK_HP91
    models["S2"] = S2_CORK_HP11

    return models
