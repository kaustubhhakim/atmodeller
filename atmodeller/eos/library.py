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
"""Real gas EOS library

Usage:
    from atmodeller.eos.library import get_eos_models
    eos_models = get_eos_models()
    CH4_beattie = eos_models["CH4_beattie_holley58"]
    # Evaluate fugacity at 10 bar and 800 K
    fugacity = CH4_beattie.fugacity(800, 10)
    print(fugacity)
"""

from atmodeller.eos._chabrier import get_chabrier_eos_models
from atmodeller.eos._holland_powell import get_holland_eos_models
from atmodeller.eos._holley import get_holley_eos_models
from atmodeller.eos._saxena import get_saxena_eos_models
from atmodeller.eos._vanderwaals import get_vanderwaals_eos_models
from atmodeller.eos._zhang_duan import get_zhang_eos_models
from atmodeller.eos.core import RealGas


def get_eos_models() -> dict[str, RealGas]:
    """Gets a dictionary of EOS models

    Returns:
        Dictionary of EOS models
    """
    eos_models = get_chabrier_eos_models()
    eos_models |= get_holley_eos_models()
    eos_models |= get_holland_eos_models()
    eos_models |= get_saxena_eos_models()
    eos_models |= get_vanderwaals_eos_models()
    eos_models |= get_zhang_eos_models()

    return eos_models
