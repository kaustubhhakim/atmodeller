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
"""Collections of solubility laws for compositions

Keys (species) of the dictionaries must use the hill notation.
"""

from __future__ import annotations

import logging

from atmodeller.solubility.carbon_species import (
    CH4_basalt_ardia,
    CO2_basalt_dixon,
    CO_basalt_yoshioka,
    CO_rhyolite_yoshioka,
)
from atmodeller.solubility.hydrogen_species import (
    H2_andesite_hirschmann,
    H2_basalt_hirschmann,
    H2O_ano_dio_newcombe,
    H2O_basalt_dixon,
    H2O_peridotite_sossi,
)
from atmodeller.solubility.interfaces import SolubilityProtocol
from atmodeller.solubility.other_species import (
    Cl2_basalt_thomas,
    He_basalt,
    N2_basalt_libourel,
)
from atmodeller.solubility.sulfur_species import (
    S2_andesite_boulliung,
    S2_basalt_boulliung,
    S_mercury_magma_namur,
)

logger: logging.Logger = logging.getLogger(__name__)

andesite_solubilities: dict[str, SolubilityProtocol] = {
    "H2": H2_andesite_hirschmann(),
    "S2": S2_andesite_boulliung(),
}
anorthdiop_solubilities: dict[str, SolubilityProtocol] = {"H2O": H2O_ano_dio_newcombe()}
basalt_solubilities: dict[str, SolubilityProtocol] = {
    "H2O": H2O_basalt_dixon(),
    "CO2": CO2_basalt_dixon(),
    "H2": H2_basalt_hirschmann(),
    "N2": N2_basalt_libourel(),
    "S2": S2_basalt_boulliung(),
    "CO": CO_basalt_yoshioka(),
    "He": He_basalt(),
    "Cl2": Cl2_basalt_thomas(),
    "CH4": CH4_basalt_ardia(),
}
rhyolite_solubilities: dict[str, SolubilityProtocol] = {
    "CO": CO_rhyolite_yoshioka(),
}
peridotite_solubilities: dict[str, SolubilityProtocol] = {"H2O": H2O_peridotite_sossi()}
reduced_magma_solubilities: dict[str, SolubilityProtocol] = {"H2S": S_mercury_magma_namur()}

# Dictionary of all the composition solubilities. All of the dictionaries with solubility laws for
# a given composition (above) should be included in this dictionary.
composition_solubilities: dict[str, dict[str, SolubilityProtocol]] = {
    "basalt": basalt_solubilities,
    "andesite": andesite_solubilities,
    "peridotite": peridotite_solubilities,
    "anorthite_diopside_euctectic": anorthdiop_solubilities,
    "reduced_magma": reduced_magma_solubilities,
    "rhyolite": rhyolite_solubilities,
}
