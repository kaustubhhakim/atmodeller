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
"""Interfaces for solubility laws"""

# Use symbols from the relevant papers for consistency so pylint: disable=C0103

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from typing import Protocol

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)


class SolubilityProtocol(Protocol):
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float,
        pressure: float,
        **kwargs,
    ) -> float: ...


class Solubility(ABC):
    """A solubility law"""

    @abstractmethod
    def concentration(
        self,
        fugacity: float,
        *,
        temperature: float | None = None,
        pressure: float | None = None,
        **kwargs,
    ) -> float:
        """Dissolved volatile concentration in the melt in ppmw

        Args:
            fugacity: Fugacity of the species in bar
            temperature: Temperature in K. Defaults to None.
            pressure: Total pressure in bar. Defaults to None.
            **kwargs: Arbitrary keyword arguments. Keyword arguments that are fugacities must
                adhere to Hill notation: O2, H2, H2O, O2S, etc.

        Returns:
            Dissolved volatile concentration in the melt in ppmw
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SolubilityPowerLaw(Solubility):
    """A solubility power law

    Args:
        constant: Constant
        exponent: Exponent
    """

    def __init__(self, constant: float, exponent: float):
        self.constant: float = constant
        self.exponent: float = exponent

    @override
    def concentration(self, fugacity: float, **kwargs) -> float:
        del kwargs

        return self.constant * fugacity**self.exponent

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"constant={self.constant!r}, "
            f"exponent={self.exponent!r})"
        )


class NoSolubility(Solubility):
    """No solubility"""

    @override
    def concentration(self, *args, **kwargs) -> float:
        del args
        del kwargs

        return 0.0
