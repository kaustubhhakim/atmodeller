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
"""Thermodynamic data from JANAF :cite:p:`Cha98`"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from molmass import Composition
from scipy.constants import kilo
from thermochem import janaf

from atmodeller import NOBLE_GASES
from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesABC,
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDataset,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

if TYPE_CHECKING:
    from atmodeller.interfaces import ChemicalSpecies

logger: logging.Logger = logging.getLogger(__name__)


def is_homonuclear_diatomic(species: ChemicalSpecies) -> bool:
    """True if species is homonuclear diatomic, otherwise False."""
    composition: Composition = species.composition()
    if len(list(composition.keys())) == 1 and list(composition.values())[0].count == 2:
        return True
    else:
        return False


def is_noble(species: ChemicalSpecies) -> bool:
    """True if a species is a noble gas, otherwise False"""
    if species.hill_formula in NOBLE_GASES:
        return True
    else:
        return False


class ThermodynamicDatasetJANAF(ThermodynamicDataset):
    """The JANAF thermodynamic dataset :cite:p:`Cha98`.

    The modified Hill indexing system for chemical compounds is used to order the tables.

    Attributes:
        data: Thermodynamic data used for calculations
        cache: Whether to cache the JANAF database. Setting this to False will download the JANAF
            databse every time it is used. Set to True.
    """

    _DATA_SOURCE: str = "JANAF"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15
    _STANDARD_STATE_PRESSURE: float = 1

    def __init__(self):
        self.data: janaf.Janafdb = janaf.Janafdb()
        self.cache: bool = True

    @override
    def get_species_data(
        self,
        species: ChemicalSpecies,
        *,
        name: str | None = None,
        filename: str | None = None,
        **kwargs,
    ) -> ThermodynamicDataForSpeciesProtocol | None:
        """Gets the thermodynamic data for a species.

        Args:
            species: A chemical species
            name: Select records that match the chemical/mineral name. Defaults to None.
            filename: Select only records that match the filename on the website, which is very
                unique. Defaults to None.
            kwargs: Catches unused keyword arguments.

        Returns:
            Thermodynamic data for the species or None if not available
        """
        del kwargs

        def get_phase_data(phases: list[str]) -> janaf.JanafPhase | None:
            """Gets the phase data from a list of phases in order of priority.

            Args:
                phases: Phases to search for in the JANAF database in priority order.

            Returns:
                Phase data if it exists in JANAF or None if not available
            """
            try:
                logger.debug(
                    "Searching for %s (name = %s, phase = %s) in %s",
                    species.hill_formula,
                    name,
                    phases[0],
                    self.data_source,
                )
                phase_data = self.data.getphasedata(
                    formula=species.hill_formula, name=name, phase=phases[0], cache=self.cache
                )
            except ValueError:
                # Cannot find the phase, so keep iterating through the list of options.
                phase_data = get_phase_data(phases[1:])
            except IndexError:
                # Reached the end of all options therefore no phase data was found.
                phase_data = None

            return phase_data

        # First, check exclusively for a filename match if a filename has been specified.
        if filename is not None:
            logger.debug(
                "Searching for %s (filename = %s) in %s",
                species.hill_formula,
                filename,
                self.data_source,
            )
            phase_data: janaf.JanafPhase | None = self.data.getphasedata(filename=filename)

        # Otherwise, find the phase data based on the phase (solid, liquid, gas).
        elif species.phase == "g":
            if is_homonuclear_diatomic(species) or is_noble(species):
                phase_data = get_phase_data(["ref", "g"])
            else:
                phase_data = get_phase_data(["g"])

        elif species.phase == "cr":
            # ref is included for C (graphite)
            phase_data = get_phase_data(["cr", "ref"])

        elif species.phase == "l":
            # l,g is included for water at 1, 10, and 100 bar
            phase_data = get_phase_data(["l", "l,g"])

        else:
            msg: str = f"{self.__class__.__name__} does not support {species.__class__.__name__} "
            msg += " because it has no phase information"
            raise ValueError(msg)

        if phase_data is None:
            logger.warning("Thermodynamic data not found")
            return None
        else:
            logger.debug("Thermodynamic data found = %s", phase_data)
            return self.ThermodynamicDataForSpecies(species, self.data_source, phase_data)

    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesABC):
        """Thermodynamic data for a species"""

        @override
        def __init__(self, species: ChemicalSpecies, data_source: str, data: janaf.JanafPhase):
            super().__init__(species, data_source, data)

        @override
        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            r"""Gets the standard Gibbs free energy of formation.

            Note that thermochem v0.8.2 returns J not kJ. But the main branch, which atmodeller
            uses, now returns kJ. This gives rise to the kilo conversion. See
            https://github.com/adelq/thermochem/pull/25

            Args:
                temperature: Temperature in K
                pressure: Pressure in bar

            Returns:
                The standard Gibbs free energy of formation in :math:`\mathrm{J}\mathrm{mol}^{-1}`
            """

            del pressure
            gibbs: float = self.data.DeltaG(temperature) * kilo

            # logger.debug(
            #     "Species = %s, standard Gibbs energy of formation = %f",
            #     self.species.hill_formula,
            #     gibbs,
            # )

            return gibbs
