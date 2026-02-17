# SPDX-FileCopyrightText: 2024 Dan J. Bower <dbower@eaps.ethz.ch>
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Real gas EOS library

.. code-block::
   :caption: Usage

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
from atmodeller.eos._reid_connolly import get_reid_connolly_eos_models
from atmodeller.eos._saxena import get_saxena_eos_models
from atmodeller.eos._vanderwaals import get_vanderwaals_eos_models
from atmodeller.eos._wang import get_wang_eos_models
from atmodeller.eos._zhang_duan import get_zhang_eos_models
from atmodeller.eos.core import RealGasBase


def get_eos_models() -> dict[str, RealGasBase]:
    """Gets a dictionary of EOS models

    Returns:
        Dictionary of EOS models
    """
    eos_models = get_chabrier_eos_models()
    eos_models |= get_holley_eos_models()
    eos_models |= get_holland_eos_models()
    eos_models |= get_reid_connolly_eos_models()
    eos_models |= get_saxena_eos_models()
    eos_models |= get_vanderwaals_eos_models()
    eos_models |= get_wang_eos_models()
    eos_models |= get_zhang_eos_models()

    return eos_models
