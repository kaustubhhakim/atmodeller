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

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Generic, Iterator, TypeVar

from molmass import Composition, Formula

from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDataset,
)
from atmodeller.thermodata.janaf import ThermodynamicDatasetJANAF
from atmodeller.utilities import unit_conversion

logger: logging.Logger = logging.getLogger(__name__)


class ChemicalSpecies:
    """A chemical species and its properties

    Args:
        formula: Chemical formula (e.g., CO2, C, CH4, etc.)
        phase: cr, g, and l for (crystalline) solid, gas, and liquid, respectively
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF.
        thermodata_name: Name of the component in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
    """

    def __init__(
        self,
        formula: str,
        phase: str,
        *,
        thermodata_dataset: ThermodynamicDataset = ThermodynamicDatasetJANAF(),
        thermodata_name: str | None = None,
        thermodata_filename: str | None = None,
    ):
        self._formula: Formula = Formula(formula)
        self._phase: str = phase
        self._thermodynamic_dataset: ThermodynamicDataset = thermodata_dataset
        self._thermodata_name: str | None = thermodata_name
        self._thermodata_filename: str | None = thermodata_filename
        thermodata: ThermodynamicDataForSpeciesProtocol | None = (
            thermodata_dataset.get_species_data(
                self, name=thermodata_name, filename=thermodata_filename
            )
        )
        assert thermodata is not None
        self._thermodata: ThermodynamicDataForSpeciesProtocol = thermodata
        logger.info(
            "Creating %s for %s using thermodynamic data in %s",
            self.__class__.__name__,
            self.hill_formula,
            self.thermodata.data_source,
        )

    @property
    def atoms(self) -> int:
        """Number of atoms"""
        return self._formula.atoms

    def composition(self, isotopic: bool = False) -> Composition:
        """Composition of the species

        Args:
            isotopic: list isotopes separately as opposed to part of an element.

        Returns:
            Composition
        """
        return self._formula.composition(isotopic)

    @property
    def elements(self) -> list[str]:
        """Elements in species"""
        return list(self.composition().keys())

    @property
    def hill_formula(self) -> str:
        """Hill formula"""
        return self._formula.formula

    @property
    def molar_mass(self) -> float:
        r"""Molar mass in :math:\mathrm{kg}\mathrm{mol}^{-1}"""
        return self._formula.mass * unit_conversion.g_to_kg

    @property
    def name(self) -> str:
        """Unique name by combining Hill notation and phase"""
        return f"{self.hill_formula}_{self.phase}"

    @property
    def phase(self) -> str:
        """Phase"""
        return self._phase

    @property
    def thermodata(self) -> ThermodynamicDataForSpeciesProtocol:
        """Thermodynamic data for the species"""
        return self._thermodata

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._formula!s})"

    def __str__(self) -> str:
        return self.name


class CondensedSpecies(ChemicalSpecies):
    """A condensed species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        phase: Phase
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
    """


TypeChemicalSpecies = TypeVar("TypeChemicalSpecies", bound=ChemicalSpecies)
TypeChemicalSpecies_co = TypeVar("TypeChemicalSpecies_co", bound=ChemicalSpecies, covariant=True)
T = TypeVar("T")
U = TypeVar("U")


class ImmutableList(Sequence[T], Generic[T]):
    """An immutable list

    Args:
        iterable: Iterable
    """

    def __init__(self, iterable: Iterable[T]):
        self.data: tuple[T, ...] = tuple(iterable)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int | slice) -> T | ImmutableList[T]:
        if isinstance(index, slice):
            return ImmutableList(self.data[index])
        return self.data[index]

    def __repr__(self) -> str:
        return f"ImmutableList({self.data})"


class ImmutableDict(Mapping[T, U], Generic[T, U]):
    """An immutable dictionary

    Args:
        data: Mapping
    """

    def __init__(self, data: Mapping[T, U] | None = None):
        if data is None:
            self.data: dict[T, U] = {}
        else:
            self.data = dict(data)

    def __getitem__(self, key) -> U:
        return self.data[key]

    def __iter__(self) -> Iterator[T]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"
