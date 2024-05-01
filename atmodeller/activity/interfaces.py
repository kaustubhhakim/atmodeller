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
"""Interfaces for activity models"""

import logging
import sys
from abc import ABC, abstractmethod
from typing import Protocol

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class ActivityProtocol(Protocol):
    def activity(self, temperature: float, pressure: float, **kwargs) -> float: ...


class Activity(ABC, ActivityProtocol):
    """An activity model"""

    @abstractmethod
    def activity(self, temperature: float, pressure: float, **kwargs) -> float:
        """Activity

        Args:
            temperature: Temperature in K
            pressure: Pressure in bar
            **kwargs: Keyword arguments relating to compositional parameters

        Returns:
            A constant activity
        """


class ConstantActivity(Activity):
    """A constant activity

    Args:
        activity: The constant activity. Defaults to unity for a pure component.
    """

    def __init__(self, activity: float = 1):
        self._activity: float = activity

    @override
    def activity(self, *args, **kwargs) -> float:
        del args
        del kwargs

        return self._activity
