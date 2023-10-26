"""Fugacity models from Shi and Saxena (1992) and Saxena and Fei (1988).

See the LICENSE file for licensing information.

This module contains concrete classes for the fugacity models presented in Shi and Saxena (1992)
and Saxena and Fei (1988).

Concrete classes:
    H2LowPressureSS92: Low pressure model for H2 from Shi and Saxena (1992).
    H2HighPressureSS92: High pressure model for H2 from Shi and Saxena (1992).
    H2SS92: Full model for H2 from Shi and Saxena (1992).
    H2HighPressureSF88: High pressure model for H2 from Saxena and Fei (1988).
    SO2SS92: Model for SO2 from Shi and Saxena (1992).
    H2SLowPressureSS92: Low pressure model for H2S from Shi and Saxena (1992).
    H2SHighPressureSS92: High pressure model for H2S from Shi and Saxena (1992).
    H2SSS92: Model for H2S from Shi and Saxena (1992).
    O2SS92: Corresponding states for O2 from Shi and Saxena (1992).
    CO2SS92: Corresponding states for CO2 from Shi and Saxena (1992).
    COSS92: Corresponding states for CO from Shi and Saxena (1992).
    CH4SS92: Corresponding states for CH4 from Shi and Saxena (1992).
    S2SS92: Corresponding states for S2 from Shi and Saxena (1992).
    COSSS92: Correponding states for COS from Shi and Saxena (1992).

Examples:
    Get the fugacity coefficient for the CO2 corresponding states model from Shi and Saxena (1992)
    Note that the input pressure should always be in bar:

    ```python
    >>> from atmodeller.eos.saxena import CO2SS92
    >>> model = CO2SS92()
    >>> fugacity_coefficient = model.get_value(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.09669352805837
    ```

    Get the preferred fugacity models for various species from the Saxena models. Note that the 
    input pressure should always be in bar:
    
    ```python
    >>> from atmodeller.eos.saxena import get_saxena_fugacity_models
    >>> models = get_saxena_fugacity_models()
    >>> # list the available species
    >>> models.keys()
    >>> # Get the fugacity model for CO
    >>> co_model = models['CO']
    >>> # Determine the fugacity coefficient at 2000 K and 1000 bar
    >>> fugacity_coefficient = co_model.get_value(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.164203382026238
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Type

from atmodeller.eos.interfaces import FugacityModelABC
from atmodeller.eos.saxena_base import (
    SaxenaABC,
    SaxenaCombined,
    SaxenaEightCoefficients,
    SaxenaFiveCoefficients,
)

logger: logging.Logger = logging.getLogger(__name__)


# Shi and Saxena (1992), Table 2.
@dataclass(frozen=True)
class critical:
    Tc: float
    Pc: float


table2: dict[str, critical] = {
    "H2O": critical(647.25, 221.1925),
    "CO2": critical(304.15, 73.8659),
    "CH4": critical(191.05, 46.4069),
    "CO": critical(133.15, 34.9571),
    "O2": critical(154.75, 50.7638),
    "H2": critical(33.25, 12.9696),
    "S2": critical(208.15, 72.954),
    "SO2": critical(430.95, 78.7295),
    "COS": critical(377.55, 65.8612),
    "H2S": critical(373.55, 90.0779),
}


@dataclass(kw_only=True)
class H2LowPressureSS92(SaxenaFiveCoefficients):
    """Low pressure model for H2 from Shi and Saxena (1992).

    Table 1(b), <1000 bar.

    See base class.
    """

    Tc: float = table2["H2"].Tc
    Pc: float = table2["H2"].Pc
    a_coefficients: tuple[float, ...] = field(init=False, default=(1, 0, 0, 0, 0, 0))
    b_coefficients: tuple[float, ...] = field(init=False, default=(0, 0.9827e-1, 0, -0.2709, 0))
    c_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, -0.1030e-2, 0, 0.1427e-1))
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class H2HighPressureSS92(SaxenaEightCoefficients):
    """High pressure model for H2 from Shi and Saxena (1992).

    Table 1(b), >1 kbar.

    See base class.
    """

    Tc: float = table2["H2"].Tc
    Pc: float = table2["H2"].Pc

    a_coefficients: tuple[float, ...] = field(
        init=False, default=(2.2615, 0, -6.8712e1, 0, -1.0573e-4, 0, 0, -1.6936e-1)
    )
    b_coefficients: tuple[float, ...] = field(
        init=False, default=(-2.6707e-4, 0, 2.0173e-1, 0, 4.5759, 0, 0, 3.1452e-5)
    )
    c_coefficients: tuple[float, ...] = field(
        init=False, default=(-2.3376e-9, 0, 3.4091e-7, 0, -1.4188e-3, 0, 0, 3.0117e-10)
    )
    d_coefficients: tuple[float, ...] = field(
        init=False, default=(-3.2606e-15, 0, 2.4402e-12, 0, -2.4027e-9, 0, 0, 0)
    )


@dataclass(kw_only=True)
class H2HighPressureSS92_Refit(SaxenaEightCoefficients):
    """High pressure model for H2 from Shi and Saxena (1992), Refitted using V, P, T Data from
    Presnall 1969 and Ross & Ree 1983, assuming same functional form as Shi & Saxena, including which
    coefficients they put at zero

    Table 1(b), >1 kbar.

    See base class.
    """

    Tc: float = table2["H2"].Tc
    Pc: float = table2["H2"].Pc

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1.00574428e00, 0, 1.93022092e-03, 0, -3.79261142e-01, 0, 0, -2.44217972e-03),
    )

    b_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1.31517888e-03, 0, 7.22328441e-02, 0, 4.84354163e-02, 0, 0, -4.19624507e-04),
    )

    c_coefficients: tuple[float, ...] = field(
        init=False,
        default=(2.64454401e-06, 0, -5.18445629e-05, 0, -2.05045979e-04, 0, 0, -3.64843213e-07),
    )

    d_coefficients: tuple[float, ...] = field(
        init=False, default=(2.28281107e-11, 0, -1.07138603e-08, 0, 3.67720815e-07, 0, 0, 0)
    )


@dataclass(kw_only=True)
class H2SS92(SaxenaCombined):
    """H2 fugacity model from Shi and Saxena (1992).

    Combines the low pressure and high pressure models into a single model. See Table 1(b).

    See base class.
    """

    Tc: float = field(init=False, default=table2["H2"].Tc)
    Pc: float = field(init=False, default=table2["H2"].Pc)
    classes: tuple[Type[SaxenaABC], ...] = field(
        init=False,
        default=(
            H2LowPressureSS92,
            H2HighPressureSS92_Refit,
        ),
    )
    upper_pressure_bounds: tuple[float, ...] = (1000,)


@dataclass(kw_only=True)
class H2HighPressureSF88(SaxenaEightCoefficients):
    """High pressure model for H2 from Saxena and Fei (1988).

    Table on p1196.

    See base class.
    """

    Tc: float = field(init=False, default=table2["H2"].Tc)
    Pc: float = field(init=False, default=table2["H2"].Pc)
    a_coefficients: tuple[float, ...] = field(
        init=False, default=(1.6688, 0, -2.0759, 0, -9.6173, 0, 0, -0.1694)
    )
    b_coefficients: tuple[float, ...] = field(
        init=False, default=(-2.0410e-3, 0, 7.9230e-2, 0, 5.4295e-2, 0, 0, 4.0887e-4)
    )
    c_coefficients: tuple[float, ...] = field(
        init=False, default=(-2.1693e-7, 0, 1.7406e-6, 0, -2.1885e-4, 0, 0, 5.0897e-5)
    )
    d_coefficients: tuple[float, ...] = field(
        init=False, default=(-7.1635e-12, 0, 1.6197e-10, 0, -4.8181e-9, 0, 0, 0)
    )


@dataclass(kw_only=True)
class H2HighPressureSF88_Refit(SaxenaEightCoefficients):
    """High pressure model for H2 from Saxena and Fei (1988), Refitted with Data from Presnall 1969 and
    Ross and Ree 1983. Using same functional form as Saxena & Fei, including which coefficient is zero

    See base class.
    """

    Tc: float = field(init=False, default=table2["H2"].Tc)
    Pc: float = field(init=False, default=table2["H2"].Pc)

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1.00574429e00, 0, 1.93017653e-03, 0, -3.79261119e-01, 0, 0, -2.44218196e-03),
    )
    b_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1.31517894e-03, 0, 7.22328436e-02, 0, 4.84354184e-02, 0, 0, -4.19624518e-04),
    )
    c_coefficients: tuple[float, ...] = field(
        init=False,
        default=(2.64454394e-06, 0, -5.18445624e-05, 0, -2.05045980e-04, 0, 0, -3.64843202e-07),
    )
    d_coefficients: tuple[float, ...] = field(
        init=False, default=(2.28281092e-11, 0, -1.07138600e-08, 0, 3.67720812e-07, 0, 0, 0)
    )


@dataclass(kw_only=True)
class SO2SS92(SaxenaEightCoefficients):
    """Fugacity model for SO2 from Shi and Saxena (1992).

    Table 1(c).

    See base class.
    """

    Tc: float = table2["SO2"].Tc
    Pc: float = table2["SO2"].Pc
    a_coefficients: tuple[float, ...] = field(
        init=False, default=(0.92854, 0.43269e-1, -0.24671, 0, 0.24999, 0, -0.53182, -0.16461e-1)
    )
    b_coefficients: tuple[float, ...] = field(
        init=False,
        default=(0.84866e-3, -0.18379e-2, 0.66787e-1, 0, -0.29427e-1, 0, 0.29003e-1, 0.54808e-2),
    )
    c_coefficients: tuple[float, ...] = field(
        init=False,
        default=(-0.35456e-3, 0.23316e-4, 0.94159e-3, 0, -0.81653e-3, 0, 0.23154e-3, 0.55542e-4),
    )
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class H2SLowPressureSS92(SaxenaEightCoefficients):
    """Fugacity model for H2S from Shi and Saxena (1992).

    Table 1(d), 1-500 bar.

    See base class.
    """

    Tc: float = table2["H2S"].Tc
    Pc: float = table2["H2S"].Pc
    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(0.14721e1, 0.11177e1, 0.39657e1, 0, -0.10028e2, 0, 0.45484e1, -0.38200e1),
    )
    b_coefficients: tuple[float, ...] = field(
        init=False,
        default=(0.16066, 0.10887, 0.29014, 0, -0.99593, 0, -0.18627, -0.45515),
    )
    c_coefficients: tuple[float, ...] = field(
        init=False,
        default=(-0.28933, -0.70522e-1, 0.39828, 0, -0.50533e-1, 0, 0.11760, 0.33972),
    )
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class H2SHighPressureSS92(SaxenaEightCoefficients):
    """Fugacity model for H2S from Shi and Saxena (1992).

    Table 1(d), 500-10000 bar.

    See base class.
    """

    Tc: float = table2["H2S"].Tc
    Pc: float = table2["H2S"].Pc
    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(0.59941, -0.15570e-2, 0.45250e-1, 0, 0.36687, 0, -0.79248, 0.26058),
    )
    b_coefficients: tuple[float, ...] = field(
        init=False,
        default=(0.22545e-1, 0.17473e-2, 0.48253e-1, 0, -0.19890e-1, 0, 0.32794e-1, -0.10985e-1),
    )
    c_coefficients: tuple[float, ...] = field(
        init=False,
        default=(0.57375e-3, -0.20944e-5, -0.11894e-2, 0, 0.14661e-2, 0, -0.75605e-3, -0.27985e-3),
    )
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class H2SSS92(SaxenaCombined):
    """H2S fugacity model from Shi and Saxena (1992).

    Combines the low pressure and high pressure models into a single model. See Table 1(d).

    See base class.
    """

    Tc: float = field(init=False, default=table2["H2S"].Tc)
    Pc: float = field(init=False, default=table2["H2S"].Pc)
    classes: tuple[Type[SaxenaABC], ...] = field(
        init=False,
        default=(
            H2SLowPressureSS92,
            H2SHighPressureSS92,
        ),
    )
    upper_pressure_bounds: tuple[float, ...] = (500,)


@dataclass(kw_only=True)
class CorrespondingStatesLowPressureSS92(SaxenaFiveCoefficients):
    """Low pressure model for corresponding fluid species from Shi and Saxena (1992).

    Table 1(a), <1000 bar.

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(1, 0, 0, 0, 0, 0))
    b_coefficients: tuple[float, ...] = field(init=False, default=(0, 0.9827e-1, 0, -0.2709, 0))
    c_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, -0.1030e-2, 0, 0.1427e-1))
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class CorrespondingStatesMediumPressureSS92(SaxenaEightCoefficients):
    """Medium pressure model for corresponding fluid species from Shi and Saxena (1992).

    Table 1(a), 1000-5000 bar.

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(1, 0, 0, 0, -5.917e-1, 0, 0, 0))
    b_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 9.122e-2, 0, 0, 0, 0, 0))
    c_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, 0, 0, -1.416e-4, 0, 0, -2.835e-6)
    )
    d_coefficients: tuple[float, ...] = field(init=False, default=(0, 0, 0, 0, 0, 0, 0, 0))


@dataclass(kw_only=True)
class CorrespondingStatesHighPressureSS92(SaxenaEightCoefficients):
    """High pressure model for corresponding fluid species from Shi and Saxena (1992).

    Table 1(a), >5000 bar.

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False, default=(2.0614, 0, 0, 0, -2.235, 0, 0, -3.941e-1)
    )
    b_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, 5.513e-2, 0, 3.934e-2, 0, 0, 0)
    )
    c_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, -1.894e-6, 0, -1.109e-5, 0, -2.189e-5, 0)
    )
    d_coefficients: tuple[float, ...] = field(
        init=False, default=(0, 0, 5.053e-11, 0, 0, -6.303e-21, 0, 0)
    )


@dataclass(kw_only=True)
class CorrespondingStates(SaxenaCombined):
    """Corresponding states for O2, CO2, CO, CH4, S2, and COS, from Shi and Saxena (1992).

    Table 1(a).

    See base class.
    """

    classes: tuple[Type[SaxenaABC], ...] = field(
        init=False,
        default=(
            CorrespondingStatesLowPressureSS92,
            CorrespondingStatesMediumPressureSS92,
            CorrespondingStatesHighPressureSS92,
        ),
    )
    upper_pressure_bounds: tuple[float, ...] = (
        1000,
        5000,
    )


@dataclass(kw_only=True)
class O2SS92(CorrespondingStates):
    """Corresponding states for O2 from Shi and Saxena (1992)."""

    Tc: float = field(init=False, default=table2["O2"].Tc)
    Pc: float = field(init=False, default=table2["O2"].Pc)


@dataclass(kw_only=True)
class CO2SS92(CorrespondingStates):
    """Corresponding states for CO2 from Shi and Saxena (1992)."""

    Tc: float = field(init=False, default=table2["CO2"].Tc)
    Pc: float = field(init=False, default=table2["CO2"].Pc)


@dataclass(kw_only=True)
class COSS92(CorrespondingStates):
    """Corresponding states for CO from Shi and Saxena (1992)."""

    Tc: float = field(init=False, default=table2["CO"].Tc)
    Pc: float = field(init=False, default=table2["CO"].Pc)


@dataclass(kw_only=True)
class CH4SS92(CorrespondingStates):
    """Corresponding states for CH4 from Shi and Saxena (1992)."""

    Tc: float = field(init=False, default=table2["CH4"].Tc)
    Pc: float = field(init=False, default=table2["CH4"].Pc)


@dataclass(kw_only=True)
class S2SS92(CorrespondingStates):
    """Corresponding states for S2 from Shi and Saxena (1992)."""

    Tc: float = field(init=False, default=table2["S2"].Tc)
    Pc: float = field(init=False, default=table2["S2"].Pc)


@dataclass(kw_only=True)
class COSSS92(CorrespondingStates):
    """Corresponding states for COS from Shi and Saxena (1992)."""

    Tc: float = field(init=False, default=table2["COS"].Tc)
    Pc: float = field(init=False, default=table2["COS"].Pc)


def get_saxena_fugacity_models() -> dict[str, FugacityModelABC]:
    """Gets a dictionary of the preferred fugacity models to use for each species.

    Returns:
        Dictionary of preferred fugacity models for each species.
    """
    models: dict[str, FugacityModelABC] = {}
    models["CH4"] = CH4SS92()
    models["CO"] = COSS92()
    models["CO2"] = CO2SS92()
    models["COS"] = COSSS92()
    models["H2"] = H2SS92()
    models["H2S"] = H2SSS92()
    models["O2"] = O2SS92()
    models["S2"] = S2SS92()
    models["SO2"] = SO2SS92()

    return models
