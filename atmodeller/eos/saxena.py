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
"""Real gas EOSs from :cite:t:`SF87,SF87a,SF88,SS92`.

Examples:
    Evaluate the fugacity coefficient for the CO2 corresponding states model from :cite:t:`SS92` at
    2000 K and 1000 bar::

        from atmodeller.eos.saxena import CO2_SS92
        model = CO2_SS92
        fugacity_coefficient = model.fugacity_coefficient(temperature=2000, pressure=1000)
        print(fugacity_coefficient)

    Get the preferred EOS models for various species from the Saxena and colleagues models::

        from atmodeller.eos.saxena import get_saxena_eos_models
        models = get_saxena_eos_models()
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
import sys
from abc import abstractmethod
from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from atmodeller import GAS_CONSTANT_BAR
from atmodeller.eos.interfaces import (
    CombinedEOSModel,
    CorrespondingStatesMixin,
    ExperimentalCalibration,
    RealGas,
)
from atmodeller.thermodata._gases import critical_data
from atmodeller.utilities import unit_conversion

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SaxenaABC(CorrespondingStatesMixin, RealGas):
    """A real gas EOS from :cite:t:`SS92`

    This form of the EOS can also be used for the models in :cite:t:`SF87a,SF87,SF88`.

    Args:
        a_coefficients: `a` coefficients. Defaults to empty.
        b_coefficients: `b` coefficients. Defaults to empty.
        c_coefficients: `c` coefficients. Defaults to empty.
        d_coefficients: `d` coefficients. Defaults to empty.
        critical_temperature: Critical temperature in K. Defaults to unity meaning not a
            corresponding states model.
        critical_pressure: Critical pressure in bar. Defaults to unity meaning not a corresponding
            states model.
    """

    a_coefficients: Array = field(default_factory=lambda: jnp.array(()))
    """`a` coefficients"""
    b_coefficients: Array = field(default_factory=lambda: jnp.array(()))
    """`b` coefficients"""
    c_coefficients: Array = field(default_factory=lambda: jnp.array(()))
    """`c` coefficients"""
    d_coefficients: Array = field(default_factory=lambda: jnp.array(()))
    """`d` coefficients"""
    standard_state_pressure: ArrayLike = field(init=False, default=1)
    """Standard state pressure with the appropriate units"""

    @abstractmethod
    def _get_compressibility_coefficient(
        self, scaled_temperature: float, coefficients: Array
    ) -> Array:
        """General form of the coefficients for the compressibility calculation
        :cite:p:`SS92{Equation 1}`

        Args:
            temperature: Scaled temperature
            coefficients: Tuple of the coefficients `a`, `b`, `c`, or `d`.

        Returns
            The relevant coefficient
        """

    def _a(self, scaled_temperature: float) -> Array:
        """`a` parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            a parameter
        """
        a: Array = self._get_compressibility_coefficient(scaled_temperature, self.a_coefficients)

        return a

    def _b(self, scaled_temperature: float) -> Array:
        """`b` parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            b parameter
        """
        b: Array = self._get_compressibility_coefficient(scaled_temperature, self.b_coefficients)

        return b

    def _c(self, scaled_temperature: float) -> Array:
        """`c` parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            c parameter
        """
        c: Array = self._get_compressibility_coefficient(scaled_temperature, self.c_coefficients)

        return c

    def _d(self, scaled_temperature: float) -> Array:
        """`d` parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            d parameter
        """
        d: Array = self._get_compressibility_coefficient(scaled_temperature, self.d_coefficients)

        return d

    @override
    def compressibility_factor(self, temperature: float, pressure: ArrayLike) -> Array:
        """Compressibility parameter :cite:p:`SS92{Equation 2}`

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            The compressibility parameter, which is dimensionless
        """
        Tr: float = self.scaled_temperature(temperature)
        Pr: ArrayLike = self.scaled_pressure(pressure)
        Z: Array = self._a(Tr) + self._b(Tr) * Pr + self._c(Tr) * Pr**2 + self._d(Tr) * Pr**3

        return Z

    @override
    def volume(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume :cite:p:`SS92{Equation 1}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`
        """
        Z: Array = self.compressibility_factor(temperature, pressure)
        volume: Array = Z * self.ideal_volume(temperature, pressure)

        return volume

    @override
    def volume_integral(self, temperature: float, pressure: ArrayLike) -> Array:
        r"""Volume integral :cite:p:`SS92{Equation 11}`

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar

        Returns:
            Volume integral in :math:`\mathrm{J}\mathrm{mol}^{-1}`
        """
        Tr: float = self.scaled_temperature(temperature)
        Pr: ArrayLike = self.scaled_pressure(pressure)
        P0r: ArrayLike = self.scaled_pressure(self.standard_state_pressure)
        volume_integral: Array = (
            (
                self._a(Tr) * jnp.log(Pr / P0r)
                + self._b(Tr) * (Pr - P0r)  # type: ignore
                + (1.0 / 2) * self._c(Tr) * (Pr**2 - P0r**2)
                + (1.0 / 3) * self._d(Tr) * (Pr**3 - P0r**3)
            )
            * GAS_CONSTANT_BAR
            * temperature
        )
        volume_integral = volume_integral * unit_conversion.m3_bar_to_J

        return volume_integral


@dataclass(kw_only=True)
class SaxenaFiveCoefficients(SaxenaABC):
    """Real gas EOS with five coefficients, which is generally used for low pressures"""

    @override
    def _get_compressibility_coefficient(
        self, scaled_temperature: float, coefficients: Array
    ) -> Array:
        """General form of the coefficients for the compressibility calculation
        :cite:p:`SS92{Equation 3b}`

        Args:
            temperature: Scaled temperature
            coefficients: Tuple of the coefficients `a`, `b`, `c`, or `d`.

        Returns
            The relevant coefficient
        """
        coefficient: Array = (
            coefficients[0]
            + coefficients[1] / scaled_temperature
            + coefficients[2] / scaled_temperature ** (3 / 2)
            + coefficients[3] / scaled_temperature**3
            + coefficients[4] / scaled_temperature**4
        )

        return coefficient


@dataclass(kw_only=True)
class SaxenaEightCoefficients(SaxenaABC):
    """Real gas EOS with eight coefficients, which is generally used for high pressures"""

    @override
    def _get_compressibility_coefficient(
        self, scaled_temperature: float, coefficients: Array
    ) -> Array:
        """General form of the coefficients for the compressibility calculation
        :cite:p:`SS92{Equation 3a}`

        Args:
            temperature: Scaled temperature
            coefficients: Tuple of the coefficients `a`, `b`, `c`, or `d`.

        Returns
            The relevant coefficient
        """
        coefficient: Array = (
            coefficients[0]
            + coefficients[1] * scaled_temperature
            + coefficients[2] / scaled_temperature
            + coefficients[3] * scaled_temperature**2
            + coefficients[4] / scaled_temperature**2
            + coefficients[5] * scaled_temperature**3
            + coefficients[6] / scaled_temperature**3
            + coefficients[7] * jnp.log(scaled_temperature)
        )

        return coefficient


_H2_low_pressure_SS92: RealGas = SaxenaFiveCoefficients(
    critical_temperature=critical_data["H2_g"].temperature,
    critical_pressure=critical_data["H2_g"].pressure,
    a_coefficients=jnp.array((1, 0, 0, 0, 0)),
    b_coefficients=jnp.array((0, 0.9827e-1, 0, -0.2709, 0)),
    # Saxena and Fei (1987a), Eq. 23, C final coefficient = 0.1472e-1 (not 0.1427e-1)
    c_coefficients=jnp.array((0, 0, -0.1030e-2, 0, 0.1472e-1)),
    d_coefficients=jnp.array((0, 0, 0, 0, 0)),
    calibration=ExperimentalCalibration(pressure_max=1000),
)
"""H2 low pressure (<1000 bar) :cite:p:`SS92{Table 1b}`

The coefficients are the same as for the corresponding states model in :cite:t:`SS92{Table 1a}`
because they originate from :cite:t:`SF87a{Equation 23}`.

In :cite:t:`SF87a{Equation 23}` the final `c` coefficient is 0.1472e-1, not 0.1427e-1 as given in
:cite:t:`SS92{Table 1b}`. The earlier work is assumed to be correct.
"""
_H2_high_pressure_SS92: RealGas = SaxenaEightCoefficients(
    a_coefficients=jnp.array((2.2615, 0, -6.8712e1, 0, -1.0573e4, 0, 0, -1.6936e-1)),
    b_coefficients=jnp.array((-2.6707e-4, 0, 2.0173e-1, 0, 4.5759, 0, 0, 3.1452e-5)),
    c_coefficients=jnp.array((-2.3376e-9, 0, 3.4091e-7, 0, -1.4188e-3, 0, 0, 3.0117e-10)),
    d_coefficients=jnp.array((-3.2606e-15, 0, 2.4402e-12, 0, -2.4027e-9, 0, 0, 0)),
    calibration=ExperimentalCalibration(pressure_min=1000),
)
"""H2 high pressure (>1000 bar) :cite:p:`SS92{Table 1b}`

This model cannot be a corresponding states model because the data do not appear correct when
plotted, so presumably it requires the actual temperature and pressure (hence
`critical_temperature` and `critical_pressure` are not provided as arguments). Visually, the fit
compares well to :obj:`_H2_high_pressure_SS92_refit`.
"""

_H2_high_pressure_SS92_refit: RealGas = SaxenaEightCoefficients(
    critical_temperature=critical_data["H2_g"].temperature,
    critical_pressure=critical_data["H2_g"].pressure,
    a_coefficients=jnp.array(
        (1.00574428e00, 0, 1.93022092e-03, 0, -3.79261142e-01, 0, 0, -2.44217972e-03)
    ),
    b_coefficients=jnp.array(
        (1.31517888e-03, 0, 7.22328441e-02, 0, 4.84354163e-02, 0, 0, -4.19624507e-04)
    ),
    c_coefficients=jnp.array(
        (2.64454401e-06, 0, -5.18445629e-05, 0, -2.05045979e-04, 0, 0, -3.64843213e-07)
    ),
    d_coefficients=jnp.array((2.28281107e-11, 0, -1.07138603e-08, 0, 3.67720815e-07, 0, 0, 0)),
    calibration=ExperimentalCalibration(pressure_min=1000),
)
"""H2 high pressure (>1000 bar)

This model has been refitted using a least square regression using the experimental volume,
pressure, and temperature data from :cite:t:`P69,RRY83` assuming the same functional form for
pressures above 1 kbar as given by :cite:t:`SS92{Equation 2 and 3a}`, including which coefficients
are set to zero :cite:p:`SS92{Table 1b}`. The refitting is performed using reduced temperature and
pressure.
"""

H2_SS92: RealGas = CombinedEOSModel(
    models=(_H2_low_pressure_SS92, _H2_high_pressure_SS92),
    upper_pressure_bounds=jnp.array((1000,)),
)
"""H2 EOS, which combines the low and high pressure EOS :cite:p:`SS92{Table 1b}`"""

_H2_high_pressure_SF88: RealGas = SaxenaEightCoefficients(
    critical_temperature=critical_data["H2_g"].temperature,
    critical_pressure=critical_data["H2_g"].pressure,
    a_coefficients=jnp.array((1.6688, 0, -2.0759, 0, -9.6173, 0, 0, -0.1694)),
    b_coefficients=jnp.array((-2.0410e-3, 0, 7.9230e-2, 0, 5.4295e-2, 0, 0, 4.0887e-4)),
    c_coefficients=jnp.array((-2.1693e-7, 0, 1.7406e-6, 0, -2.1885e-4, 0, 0, 5.0897e-5)),
    d_coefficients=jnp.array((-7.1635e-12, 0, 1.6197e-10, 0, -4.8181e-9, 0, 0, 0)),
)
"""H2 high pressure :cite:p:`SF88{Table on page 1196}`

This model does not at all agree with :cite:t:`SS92` or data, regardless of whether the temperature
and pressure inputs are the actual or reduced values.

Further investigations are warranted before this model should be used.
"""

SO2_SS92: RealGas = SaxenaEightCoefficients(
    critical_temperature=critical_data["SO2_g"].temperature,
    critical_pressure=critical_data["SO2_g"].pressure,
    a_coefficients=jnp.array(
        (0.92854, 0.43269e-1, -0.24671, 0, 0.24999, 0, -0.53182, -0.16461e-1)
    ),
    b_coefficients=jnp.array(
        (
            0.84866e-3,
            -0.18379e-2,
            0.66787e-1,
            0,
            -0.29427e-1,
            0,
            0.29003e-1,
            0.54808e-2,
        )
    ),
    c_coefficients=jnp.array(
        (
            -0.35456e-3,
            0.23316e-4,
            0.94159e-3,
            0,
            -0.81653e-3,
            0,
            0.23154e-3,
            0.55542e-4,
        )
    ),
    d_coefficients=jnp.array((0, 0, 0, 0, 0, 0, 0, 0)),
    calibration=ExperimentalCalibration(pressure_min=1, pressure_max=10e3),
)
"""SO2 EOS :cite:p:`SS92{Table 1c}`"""

_H2S_low_pressure_SS92: RealGas = SaxenaEightCoefficients(
    critical_temperature=critical_data["H2S_g"].temperature,
    critical_pressure=critical_data["H2S_g"].pressure,
    a_coefficients=jnp.array(
        (0.14721e1, 0.11177e1, 0.39657e1, 0, -0.10028e2, 0, 0.45484e1, -0.38200e1)
    ),
    b_coefficients=jnp.array((0.16066, 0.10887, 0.29014, 0, -0.99593, 0, -0.18627, -0.45515)),
    c_coefficients=jnp.array(
        (-0.28933, -0.70522e-1, 0.39828, 0, -0.50533e-1, 0, 0.11760, 0.33972)
    ),
    d_coefficients=jnp.array((0, 0, 0, 0, 0, 0, 0, 0)),
    calibration=ExperimentalCalibration(pressure_min=1, pressure_max=500),
)
"""H2S low pressure (1-500 bar) :cite:p:`SS92{Table 1d}`"""

_H2S_high_pressure_SS92: RealGas = SaxenaEightCoefficients(
    critical_temperature=critical_data["H2S_g"].temperature,
    critical_pressure=critical_data["H2S_g"].pressure,
    a_coefficients=jnp.array((0.59941, -0.15570e-2, 0.45250e-1, 0, 0.36687, 0, -0.79248, 0.26058)),
    b_coefficients=jnp.array(
        (
            0.22545e-1,
            0.17473e-2,
            0.48253e-1,
            0,
            -0.19890e-1,
            0,
            0.32794e-1,
            -0.10985e-1,
        )
    ),
    c_coefficients=jnp.array(
        (
            0.57375e-3,
            -0.20944e-5,
            -0.11894e-2,
            0,
            0.14661e-2,
            0,
            -0.75605e-3,
            -0.27985e-3,
        )
    ),
    d_coefficients=jnp.array((0, 0, 0, 0, 0, 0, 0, 0)),
    calibration=ExperimentalCalibration(pressure_min=500, pressure_max=10e3),
)
"""H2S high pressure (500-10000 bar) :cite:p:`SS92{Table 1d}`"""

H2S_SS92: RealGas = CombinedEOSModel(
    models=(_H2S_low_pressure_SS92, _H2S_high_pressure_SS92),
    upper_pressure_bounds=jnp.array((500,)),
    calibration=ExperimentalCalibration(pressure_min=1, pressure_max=10e3),
)
"""H2S EOS, which combines the low and high pressure EOS :cite:p:`SS92{Table 1d}`"""


def get_corresponding_states_SS92(
    species: str, calibration: ExperimentalCalibration = ExperimentalCalibration()
) -> RealGas:
    """Corresponding states :cite:p:`SS92{Table 1a}`

    Coefficients for the low and medium pressure regimes are from
    :cite:t:`SF87a{Equation 23 and 21}`, respectively, although some values disagree either in
    value or sign with :cite:t:`SS92{Table 1a}`. Eventually it should be determined which is right,
    but for the time being it is assumed that the earlier work is correct. Coefficients for the
    high pressure regime are from :cite:t:`SF87{Equation 11}`.

    Args:
        species: A species, which must be a key in
            :obj:`.interfaces.critical_data`
        calibration: Calibration temperature and pressure range. Defaults to empty.

    Returns:
        A corresponding states model for the species
    """

    critical_temperature: float = critical_data[species].temperature
    critical_pressure: float = critical_data[species].pressure

    # Table 1a, <1000 bar
    low_pressure: RealGas = SaxenaFiveCoefficients(
        critical_temperature=critical_temperature,
        critical_pressure=critical_pressure,
        a_coefficients=jnp.array((1, 0, 0, 0, 0)),
        b_coefficients=jnp.array((0, 0.9827e-1, 0, -0.2709, 0)),
        # Saxena and Fei (1987) CMP, Eq. 23, C final coefficient = 0.1472e-1 (not 0.1427e-1)
        c_coefficients=jnp.array((0, 0, -0.1030e-2, 0, 0.1472e-1)),
        d_coefficients=jnp.array((0, 0, 0, 0, 0)),
        calibration=ExperimentalCalibration(pressure_max=1000),
    )

    # Table 1a, 1000-5000 bar
    medium_pressure: RealGas = SaxenaEightCoefficients(
        critical_temperature=critical_temperature,
        critical_pressure=critical_pressure,
        a_coefficients=jnp.array((1, 0, 0, 0, -5.917e-1, 0, 0, 0)),
        b_coefficients=jnp.array((0, 0, 9.122e-2, 0, 0, 0, 0, 0)),
        # Saxena and Fei (1987) CMP, Eq. 21, C first coefficient = 1.4164e-4 (not negative)
        c_coefficients=jnp.array((0, 0, 0, 0, 1.4164e-4, 0, 0, -2.8349e-6)),
        d_coefficients=jnp.array((0, 0, 0, 0, 0, 0, 0, 0)),
        calibration=ExperimentalCalibration(pressure_min=1000, pressure_max=5000),
    )

    # Table 1a, >5000 bar
    # High precision coefficients taken from Saxena and Fei (1987), GCA, and agree with Table 1(a)
    high_pressure: RealGas = SaxenaEightCoefficients(
        critical_temperature=critical_temperature,
        critical_pressure=critical_pressure,
        a_coefficients=jnp.array((2.0614, 0, 0, 0, -2.2351, 0, 0, -3.9411e-1)),
        b_coefficients=jnp.array((0, 0, 5.5125e-2, 0, 3.9344e-2, 0, 0, 0)),
        c_coefficients=jnp.array((0, 0, -1.8935e-6, 0, -1.1092e-5, 0, -2.1892e-5, 0)),
        d_coefficients=jnp.array((0, 0, 5.0527e-11, 0, 0, -6.3033e-21, 0, 0)),
        calibration=ExperimentalCalibration(pressure_min=5000),
    )

    combined_model: RealGas = CombinedEOSModel(
        models=(low_pressure, medium_pressure, high_pressure),
        upper_pressure_bounds=jnp.array((1000, 5000)),
        calibration=calibration,
    )

    return combined_model


CH4_SS92: RealGas = get_corresponding_states_SS92("CH4_g")
"""CH4 corresponding states :cite:p:`SS92`"""
CO_SS92: RealGas = get_corresponding_states_SS92("CO_g")
"""CO corresponding states :cite:p:`SS92`"""
CO2_SS92: RealGas = get_corresponding_states_SS92("CO2_g")
"""CO2 corresponding states :cite:p:`SS92`"""
COS_SS92: RealGas = get_corresponding_states_SS92("COS_g")
"""COS corresponding states :cite:p:`SS92`"""
O2_SS92: RealGas = get_corresponding_states_SS92("O2_g")
"""O2 corresponding states :cite:p:`SS92`"""
S2_SS92: RealGas = get_corresponding_states_SS92("S2_g")
"""S2 corresponding states :cite:p:`SS92`"""
Ar_SF87: RealGas = get_corresponding_states_SS92("Ar_g")
"""Ar corresponding states :cite:p:`SF87{Equation 11}`

The low pressure extension given by the corresponding states model of :cite:t:`SS92{Table 1a}` is
also adopted.
"""
H2_SF87: RealGas = get_corresponding_states_SS92("H2_g")
"""H2 corresponding states :cite:p:`SF87{Equation 11}`

The low pressure extension given by the corresponding states model of :cite:t:`SS92{Table 1a}` is
also adopted.
"""
N2_SF87: RealGas = get_corresponding_states_SS92("N2_g")
"""N2 corresponding states :cite:p:`SF87{Equation 11}`

The low pressure extension given by the corresponding states model of :cite:t:`SS92{Table 1a}` is
also adopted.
"""


def get_saxena_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of the preferred Saxena and colleagues EOS models for each species.

    The latest and/or most sophisticated EOS model is chosen for each species. Corresponding
    states models are used when a bespoke fit to just that species is not available.

    Returns:
        Dictionary of EOS models for each species
    """
    models: dict[str, RealGas] = {}
    models["Ar"] = Ar_SF87
    models["CH4"] = CH4_SS92
    models["CO"] = CO_SS92
    models["CO2"] = CO2_SS92
    models["COS"] = COS_SS92
    models["H2"] = H2_SS92
    models["H2S"] = H2S_SS92
    models["N2"] = N2_SF87
    models["O2"] = O2_SS92
    models["S2"] = S2_SS92
    models["SO2"] = SO2_SS92

    return models
