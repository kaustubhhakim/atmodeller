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
"""Thermodynamic data from JANAF"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from scipy.constants import kilo
from thermochem import janaf

from atmodeller.thermodata.interfaces import (
    ThermodynamicDataForSpeciesABC,
    ThermodynamicDataForSpeciesProtocol,
    ThermodynamicDatasetABC,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

logger: logging.Logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from atmodeller.core import ChemicalComponent


class ThermodynamicDatasetJANAF(ThermodynamicDatasetABC):
    """JANAF thermodynamic dataset"""

    _DATA_SOURCE: str = "JANAF"
    _ENTHALPY_REFERENCE_TEMPERATURE: float = 298.15  # K
    _STANDARD_STATE_PRESSURE: float = 1  # bar

    @override
    def get_species_data(
        self,
        species: ChemicalComponent,
        *,
        name: str | None = None,
        filename: str | None = None,
        **kwargs,
    ) -> ThermodynamicDataForSpeciesProtocol | None:
        del kwargs

        db: janaf.Janafdb = janaf.Janafdb()

        # Defined by JANAF convention
        janaf_formula: str = species.hill_formula  # modified_hill_formula()

        def get_phase_data(phases: list[str]) -> janaf.JanafPhase | None:
            """Gets the phase data for a list of phases in order of priority.

            Args:
                phases: Phases to search for in the JANAF database.

            Returns:
                Phase data if it exists in JANAF, otherwise None
            """
            if filename is not None:
                phase_data: janaf.JanafPhase | None = db.getphasedata(filename=filename)
            else:
                try:
                    phase_data = db.getphasedata(formula=janaf_formula, name=name, phase=phases[0])
                except ValueError:
                    # Cannot find the phase, so keep iterating through the list of options
                    phase_data = get_phase_data(phases[1:])
                except IndexError:
                    # Reached the end of the phases to try meaning no phase data was found
                    phase_data = None

            return phase_data

        if species.phase == "g":
            if species.is_homonuclear_diatomic or species.is_noble:
                phase_data = get_phase_data(["ref", "g"])
            else:
                phase_data = get_phase_data(["g"])

        elif species.phase == "cr":
            phase_data = get_phase_data(["cr", "ref"])  # ref included for C (graphite)

        elif species.phase == "l":
            phase_data = get_phase_data(["l", "l,g"])  # l,g included for Water at 1, 10, 100 bar

        else:
            logger.error("Thermodynamic data is unknown for %s", species.__class__.__name__)
            msg: str = f"{self.__class__.__name__} does not support {species.__class__.__name__} "
            msg += " because it has no phase information"
            raise ValueError(msg)

        if phase_data is None:
            logger.warning(
                "Thermodynamic data for %s (%s) not found in %s",
                species.formula,
                janaf_formula,
                self.data_source,
            )

            return None
        else:
            logger.info(
                "Thermodynamic data for %s (%s) found in %s",
                species.formula,
                janaf_formula,
                self.data_source,
            )
            logger.info("Phase data = %s", phase_data)

            return self.ThermodynamicDataForSpecies(species, self.data_source, phase_data)

    class ThermodynamicDataForSpecies(ThermodynamicDataForSpeciesABC):
        """JANAF thermodynamic data for a species"""

        @override
        def __init__(self, species: ChemicalComponent, data_source: str, data: janaf.JanafPhase):
            super().__init__(species, data_source, data)

        @override
        def get_formation_gibbs(self, *, temperature: float, pressure: float) -> float:
            del pressure
            # thermochem v0.8.2 returns J not kJ. Main branch now returns kJ hence kilo conversion.
            # https://github.com/adelq/thermochem/pull/25
            gibbs: float = self.data.DeltaG(temperature) * kilo

            return gibbs
