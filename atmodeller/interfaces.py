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
"""Interfaces"""

from typing import Protocol, runtime_checkable

from jaxtyping import Array, ArrayLike, Bool


@runtime_checkable
class ActivityProtocol(Protocol):
    def log_activity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class FugacityConstraintProtocol(Protocol):
    def active(self) -> Bool[Array, "..."]: ...

    def log_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> ArrayLike: ...


@runtime_checkable
class RedoxBufferProtocol(FugacityConstraintProtocol, Protocol):
    evaluation_pressure: ArrayLike | None

    @property
    def log10_shift(self) -> Array: ...

    def log10_fugacity_buffer(self, temperature: ArrayLike, pressure: ArrayLike) -> Array: ...

    def log10_fugacity(self, temperature: ArrayLike, pressure: ArrayLike) -> Array: ...


@runtime_checkable
class SolubilityProtocol(Protocol):
    """Solubility protocol

    :meth:`~SolubilityProtocol.jax_concentration` is defined in order to allow arguments to be
    passed by position to lax.switch.
    """

    def concentration(
        self,
        fugacity: ArrayLike,
        *,
        temperature: ArrayLike | None = None,
        pressure: ArrayLike | None = None,
        fO2: ArrayLike | None = None,
    ) -> ArrayLike: ...

    def jax_concentration(
        self, fugacity: ArrayLike, temperature: ArrayLike, pressure: ArrayLike, fO2: ArrayLike
    ) -> Array: ...
