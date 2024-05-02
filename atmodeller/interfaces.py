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

# Protocol so pylint: disable=C0115

from __future__ import annotations

from typing import Protocol, runtime_checkable

from atmodeller.core import _ChemicalSpecies


@runtime_checkable
class ConstraintProtocol(Protocol):

    @property
    def constraint(self) -> str: ...

    @property
    def name(self) -> str: ...

    def get_value(self, *args, **kwargs) -> float: ...

    def get_log10_value(self, *args, **kwargs) -> float: ...


class SpeciesConstraintProtocol(ConstraintProtocol, Protocol):

    @property
    def species(self) -> _ChemicalSpecies: ...


class ElementConstraintProtocol(ConstraintProtocol, Protocol):

    @property
    def element(self) -> str: ...


class MassConstraintProtocol(ConstraintProtocol, Protocol):
    def mass(self, temperature: float, pressure: float) -> float: ...


class ReactionNetworkConstraintProtocol(SpeciesConstraintProtocol, Protocol):
    def fugacity(self, temperature: float, pressure: float) -> float: ...


class TotalPressureConstraintProtocol(ConstraintProtocol, Protocol):
    def total_pressure(self, temperature: float, pressure: float) -> float: ...
