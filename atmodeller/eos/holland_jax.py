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
"""Real gas EOS from :cite:t:`HP91,HP98,HP11`"""

import logging
import sys
from abc import abstractmethod
from typing import Any, Callable

import jax.numpy as jnp
from jax import Array, jit, lax
from jax.tree_util import register_pytree_node_class
from jax.typing import ArrayLike

from atmodeller import GAS_CONSTANT_BAR
from atmodeller.eos.core import (
    RealGas,
    RedlichKwongABC,
    RedlichKwongImplicitDenseFluidABC,
    RedlichKwongImplicitGasABC,
)
from atmodeller.thermodata import CriticalData, select_critical_data
from atmodeller.utilities import PyTreeNoData

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

logger: logging.Logger = logging.getLogger(__name__)

Tc_H2O: float = 695
"""Critical temperature of H2O in K for the MRK/CORK model :cite:p:`HP91`"""
Ta_H2O: float = 673  # K
r"""Temperature at which :math:`a_{\mathrm gas} = a` for H2O by fitting :cite:p:`HP91`"""
b0_H2O: float = 1.465e-5
"""b parameter value which is the same across all H2O phases :cite:p:`HP91`

Compared to :cite:t:`HP91` the value accounts for unit conversion.
"""


@register_pytree_node_class
class MRKCorrespondingStatesHP91(RedlichKwongABC):
    """MRK corresponding states :cite:p:`HP91`

    Universal constants from :cite:t:`HP91{Table 2}`

    Note the unit conversion to SI and pressure in bar using the values in Table 2:

        * `a` coefficients are multiplied by 1e-4
        * `b` is multiplied by 1e-2

    Args:
        critical_data: Critical data
    """

    def __init__(self, critical_data: CriticalData):
        self._critical_data: CriticalData = critical_data
        self._a_coefficients: tuple[float, ...] = (5.45963e-9, -8.63920e-10, 0)
        self._b: float = 9.18301e-6

    @property
    def critical_pressure(self) -> float:
        """Critical pressure"""
        return self._critical_data.pressure

    @property
    def critical_temperature(self) -> float:
        """Critical temperature"""
        return self._critical_data.temperature

    @classmethod
    def get_species(cls, species: str) -> Self:
        """Gets an MRK corresponding states model for a given species.

        Args:
            species: Species

        Returns:
            An MRK corresponding states model for the species
        """
        critical_data: CriticalData = select_critical_data(species)

        return cls(critical_data)

    @override
    def a(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""MRK `a` parameter :cite:p:`HP91{Equation 9}`

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            MRK `a` parameter in
            :math:`(\mathrm{m}^3\mathrm{mol}^{-1})^2\mathrm{K}^{1/2}\mathrm{bar}`
        """
        del pressure

        a: ArrayLike = (
            self._a_coefficients[0] * jnp.power(self.critical_temperature, (5.0 / 2))
            + self._a_coefficients[1]
            * jnp.power(self.critical_temperature, (3.0 / 2))
            * temperature
            + self._a_coefficients[2]
            * jnp.power(self.critical_temperature, (1.0 / 2))
            * jnp.square(temperature)
        )
        a = a / self.critical_pressure

        return a

    @override
    def b(self) -> ArrayLike:
        r"""MRK `b` parameter computed from :attr:`b0` :cite:p:`HP91{Equation 9}`.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            MRK `b` parameter in :math:`\mathrm{m}^3\mathrm{mol}^{-1}`.
        """
        b: ArrayLike = self._b * self.critical_temperature / self.critical_pressure

        return b

    def tree_flatten(self) -> tuple[tuple, tuple]:
        children: tuple = ()
        aux_data = (self._critical_data,)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(*aux_data)


class MRKImplicitHP91ABCMixin:
    """MRK implicit :cite:p:`HP91`

    Universal constants from :cite:t:`HP91{Table 1}`. Note the unit conversion to SI and pressure
    in bar:

        * `a` coefficients are multiplied by 1e-7
        * `b` is multiplied by 1e-5

    These scalings are different by 1e3 compared to the corresponding states scaling because in the
    corresponding states formulation the coeffficients contain a kilo pressure scaling as well.

    Args:
        a_coefficients: `a` coefficients
        b: `b` coefficient
        Ta: Temperature at which the `a` parameter is equal for the dense fluid and gas
        Tc: Critical temperature
    """

    def __init__(
        self, a_coefficients: tuple[float, float, float, float], b: float, Ta: float, Tc: float
    ):
        self._a_coefficients: tuple[float, float, float, float] = a_coefficients
        self._b: float = b
        self._Ta: float = Ta
        self._Tc: float = Tc

    @abstractmethod
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        """Temperature difference for the calculation of the `a` parameter

        Args:
            temperature: Temperature

        Returns:
            Temperature difference
        """
        ...

    def a(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""MRK `a` parameter :cite:p:`HP91{Equation 6}`

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            MRK `a` parameter
        """
        del pressure

        delta_temperature: ArrayLike = self.delta_temperature_for_a(temperature)
        a: ArrayLike = (
            self._a_coefficients[0]
            + self._a_coefficients[1] * delta_temperature
            + self._a_coefficients[2] * jnp.square(delta_temperature)
            + self._a_coefficients[3] * jnp.power(delta_temperature, 3)
        )

        return a

    def b(self) -> ArrayLike:
        return self._b

    def tree_flatten(self) -> tuple[tuple, dict[str, Any]]:
        children: tuple = ()
        aux_data = {
            "a_coefficients": self._a_coefficients,
            "b": self._b,
            "Ta": self._Ta,
            "Tc": self._Tc,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children) -> Self:
        del children
        return cls(**aux_data)


@register_pytree_node_class
class MRKImplicitGasHP91(MRKImplicitHP91ABCMixin, RedlichKwongImplicitGasABC):
    """MRK for gaseous phase :cite:p:`HP91{Equation 6a}`"""

    @override
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        return self._Ta - temperature


H2OMrkGasHolland91: RealGas = MRKImplicitGasHP91(
    (1113.4e-7, 5.8487e-7, -2.1370e-9, 6.8133e-12), b0_H2O, Ta_H2O, Tc_H2O
)
"""H2O MRK for gas phase :cite:p:`HP91`"""


@register_pytree_node_class
class MRKImplicitLiquidHP91(MRKImplicitHP91ABCMixin, RedlichKwongImplicitDenseFluidABC):
    """MRK for liquid phase :cite:p`HP91{Equation 6}`"""

    @override
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        return self._Ta - temperature


H2OMrkLiquidHolland91: RealGas = MRKImplicitLiquidHP91(
    (1113.4e-7, -0.88517e-7, 4.53e-10, -1.3183e-12), b0_H2O, Ta_H2O, Tc_H2O
)
"""H2O MRK for liquid phase :cite:p`HP91`"""


@register_pytree_node_class
class MRKImplicitFluidHP91(MRKImplicitHP91ABCMixin, RedlichKwongImplicitDenseFluidABC):
    """MRK for supercritical fluid :cite:p:`HP91{Equation 6}`"""

    @override
    def delta_temperature_for_a(self, temperature: ArrayLike) -> ArrayLike:
        return temperature - self._Ta

    @override
    def initial_volume(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike:
        r"""Initial guess volume to ensure convergence to the correct root

        See :cite:t:`HP91{Appendix}`. It appears that there is only ever a single root, even if
        Ta < temperature < Tc. Holland and Powell state that a single root exists if
        temperature > Tc, but this appears to be true if temperature > Ta. Nevertheless, the
        initial guess is changed accordingly.

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Initial volume
        """

        def low_temperature_case() -> ArrayLike:
            return self.b() / 2

        def high_temperature_case() -> ArrayLike:
            return GAS_CONSTANT_BAR * temperature / pressure + self.b()

        initial_volume: Array = jnp.where(
            temperature < jnp.array(self._Tc),
            low_temperature_case(),
            high_temperature_case(),
        )

        return initial_volume


H2OMrkFluidHolland91: RealGas = MRKImplicitFluidHP91(
    (
        1113.4e-7,
        -0.22291e-7,
        -3.8022e-11,
        1.7791e-14,
    ),
    b0_H2O,
    Ta_H2O,
    Tc_H2O,
)
"""H2O MRK for fluid phase :cite:p`HP91`"""

CO2_critical_data: CriticalData = select_critical_data("CO2_g")
"""Alternative values from :cite:t:`HP91` are 304.2 K and 73.8 bar"""
CO2MrkHolland91: RealGas = MRKImplicitFluidHP91(
    (741.2e-7, -0.10891e-7, -3.4203e-11, 0), 3.057e-5, 0, CO2_critical_data.temperature
)
"""CO2 MRK :cite:p:`HP91{Above Equation 7}`

Critical behaviour is not considered for CO2 by :cite:t:`HP91`, but for consistency with the 
formulation for H2O, the CO2 critical temperature is set.
"""

# TODO: Stitch together the H2O gas and H2O fluid for a single EOS


@register_pytree_node_class
class H2OMrkHP91(PyTreeNoData, RealGas):
    """A MRK model for H2O that accommodates critical behaviour

    Args:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation
        Tc: Critical temperature

    Attributes:
        mrk_fluid: The MRK for the supercritical fluid
        mrk_gas: The MRK for the subcritical gas
        mrk_liquid: The MRK for the subcritical liquid
        Ta: Temperature at which a_gas = a in the MRK formulation
        Tc: Critical temperature
    """

    mrk_fluid: MRKImplicitFluidHP91 = H2OMrkFluidHolland91
    mrk_gas: MRKImplicitGasHP91 = H2OMrkGasHolland91
    mrk_liquid: MRKImplicitLiquidHP91 = H2OMrkLiquidHolland91
    Ta: float = Ta_H2O
    Tc: float = Tc_H2O

    @jit
    def Psat(self, temperature: ArrayLike) -> Array:
        """Saturation curve

        Args:
            temperature: Temperature

        Returns:
            Saturation curve pressure
        """
        Psat: Array = (
            -13.627
            + 7.29395e-4 * jnp.square(temperature)
            - 2.34622e-6 * jnp.power(temperature, 3)
            + 4.83607e-12 * jnp.power(temperature, 5)
        )

        return Psat

    @jit
    def _select_condition(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Selects the condition

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Integer denoting the condition, i.e. the region of phase space
        """
        Psat: Array = self.Psat(temperature)
        temperature_array: Array = jnp.asarray(temperature)
        pressure_array: Array = jnp.asarray(pressure)

        # Supercritical (saturation pressure irrelevant)
        cond0: Array = temperature_array >= self.Tc
        # jax.debug.print("cond0 = {cond}", cond=cond0)
        # Below the saturation pressure and below Ta
        cond1: Array = jnp.logical_and(temperature_array <= self.Ta, pressure_array <= Psat)
        # jax.debug.print("cond1 = {cond}", cond=cond1)
        # Below the saturation pressure and below Tc
        cond2: Array = jnp.logical_and(temperature_array < self.Tc, pressure_array <= Psat)
        # Ensure cond2 is exclusive of cond1
        cond2 = jnp.logical_and(cond2, ~cond1)
        # jax.debug.print("cond2 = {cond}", cond=cond2)
        # Above the saturation pressure and below Ta
        cond3: Array = jnp.logical_and(temperature_array <= self.Ta, pressure_array > Psat)
        # jax.debug.print("cond3 = {cond}", cond=cond3)
        # Above the saturation pressure and below Tc
        cond4: Array = jnp.logical_and(temperature_array < self.Tc, pressure_array > Psat)
        # Ensure cond4 is exclusive of cond3
        cond4 = jnp.logical_and(cond4, ~cond3)
        # jax.debug.print("cond4 = {cond}", cond=cond4)

        # All conditions are mutually exclusive
        condition: Array = jnp.select([cond0, cond1, cond2, cond3, cond4], [0, 1, 2, 3, 4])
        # jax.debug.print("condition = {condition}", condition=condition)

        return condition

    def volume_integral(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Volume integral :cite:p:`HP91{Appendix A}`

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            volume integral
        """
        condition: Array = self._select_condition(temperature, pressure)
        Psat: Array = self.Psat(temperature)

        def volume_integral0() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        def volume_integral1() -> Array:
            return self.mrk_gas.volume_integral(temperature, pressure)

        def volume_integral2() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        def volume_integral3() -> Array:
            value: Array = self.mrk_gas.volume_integral(temperature, Psat)
            value = value - self.mrk_liquid.volume_integral(temperature, Psat)
            value = value + self.mrk_liquid.volume_integral(temperature, pressure)

            return value

        def volume_integral4() -> Array:
            return self.mrk_fluid.volume_integral(temperature, pressure)

        volume_integral_funcs: list[Callable] = [
            volume_integral0,
            volume_integral1,
            volume_integral2,
            volume_integral3,
            volume_integral4,
        ]

        volume_integral: Array = lax.switch(condition, volume_integral_funcs)
        # jax.debug.print("volume_integral = {out}", out=volume_integral)

        return volume_integral

    @override
    @jit
    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        r"""Log fugacity :cite:p:`HP91{Equation 8}`

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Log fugacity
        """
        # 1e-5 to convert volume integral back to appropriate units
        log_fugacity: Array = (
            1e-5 * self.volume_integral(temperature, pressure) / (GAS_CONSTANT_BAR * temperature)
        )

        return log_fugacity

    @override
    @jit
    def volume(self, temperature: ArrayLike, pressure: ArrayLike) -> Array:
        """Volume

        Args:
            temperature: Temperature
            pressure: Pressure

        Returns:
            Volume
        """
        condition: Array = self._select_condition(temperature, pressure)

        def volume0() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        def volume1() -> Array:
            return self.mrk_gas.volume(temperature, pressure)

        def volume2() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        def volume3() -> Array:
            return self.mrk_liquid.volume(temperature, pressure)

        def volume4() -> Array:
            return self.mrk_fluid.volume(temperature, pressure)

        volume_funcs: list[Callable] = [volume0, volume1, volume2, volume3, volume4]

        volume: Array = lax.switch(condition, volume_funcs)
        # jax.debug.print("volume = {out}", out=volume)

        return volume


H2OMrkHolland91: RealGas = H2OMrkHP91()
"""H2O MRK that includes critical behaviour"""
