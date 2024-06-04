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

import logging
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from molmass import Composition, Formula

from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDataset,
)
from atmodeller.thermodata.janaf import ThermodynamicDatasetJANAF
from atmodeller.utilities import UnitConversion

if TYPE_CHECKING:
    from atmodeller.constraints import SystemConstraints
    from atmodeller.output import Output

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
        return UnitConversion.g_to_kg(self._formula.mass)

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


class CondensedSpecies(ChemicalSpecies):
    """A condensed species

    Args:
        formula: Chemical formula (e.g., C, SiO2, etc.)
        phase: Phase
        thermodata_dataset: The thermodynamic dataset. Defaults to JANAF
        thermodata_name: Name in the thermodynamic dataset. Defaults to None.
        thermodata_filename: Filename in the thermodynamic dataset. Defaults to None.
    """


@runtime_checkable
class ConstraintProtocol(Protocol):

    @property
    def constraint(self) -> str: ...

    @property
    def name(self) -> str: ...

    def get_value(self, temperature: float, pressure: float) -> float: ...

    def get_log10_value(self, temperature: float, pressure: float) -> float: ...


class SpeciesConstraintProtocol(ConstraintProtocol, Protocol):

    @property
    def species(self) -> ChemicalSpecies: ...


class ElementConstraintProtocol(ConstraintProtocol, Protocol):

    @property
    def element(self) -> str: ...


class MassConstraintProtocol(ConstraintProtocol, Protocol):
    def mass(self, temperature: float, pressure: float) -> float: ...


class ReactionNetworkConstraintProtocol(SpeciesConstraintProtocol, Protocol):
    def fugacity(self, temperature: float, pressure: float) -> float: ...


class TotalPressureConstraintProtocol(ConstraintProtocol, Protocol):
    def total_pressure(self, temperature: float, pressure: float) -> float: ...


class InitialSolutionProtocol(Protocol):
    def get_log10_value(
        self,
        constraints: SystemConstraints,
        *,
        temperature: float,
        pressure: float,
        degree_of_condensation_number: int,
        number_of_condensed_species: int,
        perturb: bool = False,
        perturb_log10: float = 2,
    ) -> npt.NDArray[np.float_]: ...

    def update(self, output: Output) -> None: ...


# Type hint indicating covariance using type comments
TypeChemicalSpecies_co = TypeVar("TypeChemicalSpecies_co", bound=ChemicalSpecies, covariant=True)
