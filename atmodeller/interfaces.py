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

from __future__ import annotations

from typing import Protocol


# Protocol so pylint: disable=C0115
class ConstraintProtocol(Protocol):

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def species(self) -> str:
        raise NotImplementedError

    @property
    def full_name(self) -> str:
        raise NotImplementedError

    def get_log10_value(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def get_value(self, *args, **kwargs) -> float:
        raise NotImplementedError
