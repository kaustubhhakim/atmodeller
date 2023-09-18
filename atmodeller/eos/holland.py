"""Fugacity models from Holland and Powell (1991, 1998, 2011).

See the LICENSE file for licensing information.

This module contains concrete classes for the fugacity models presented in Holland and Powell 
(1991, 1998, 2011). You will usually want to use the CORK models, since these are the most
complete. The MRK models are nevertheless useful for comparison and understanding the influence
of the virial compensation term that is encapsulated within the CORK model.

Concrete classes:
  
    CORKCO2HP91: Full CORK for CO2 in Holland and Powell (1991).
    CORKCO2HP98: Full CORK for CO2 in Holland and Powell (1998).
    CORKSimpleCO2HP91: Simple CORK model for CO2 in Holland and Powell (1991).
    CORKH2OHP91: Full CORK for H2O in Holland and Powell (1991).
    CORKH2OHP98: Full CORK for H2O in Holland and Powell (1998).
    CORKCH4HP91: CORK corresponding states for CH4 in Holland and Powell (1991).
    CORKH2HP91: CORK corresponding states for H2 in Holland and Powell (1991).
    CORKCOHP91: CORK corresponding states for CO in Holland and Powell (1991).
    CORKS2HP11: CORK corresponding states for S2 in Holland and Powell (2011).
    CORKH2SHP11: CORK corresponding states for H2S in Holland and Powell (2011).
    MRKH2OLiquidHP91: MRK for liquid H2O (only) in Holland and Powell (1991).
    MRKH2OGasHP91: MRK for gaseous H2O (only) in Holland and Powell (1991).
    MRKH2OFluidHP91: MRK for fluid H2O (only) in Holland and Powell (1991).
    MRKH2OHP91: MRK for H2O with critical behaviour in Holland and Powell (1991).
    MRKH2OHP98: MRK for H2O with critical behaviour in Holland and Powell (1998).
    MRKCO2HP91: Full MRK for CO2 in Holland and Powell (1991).
    MRKCO2HP98: Full MRK for CO2 in Holland and Powell (1998).
    MRKSimpleCO2HP91: Simple MRK for CO2 in Holland and Powell (1991).
    MRKCH4HP91: MRK corresponding states for CH4 in Holland and Powell (1991).
    MRKH2HP91: MRK corresponding states for H2 in Holland and Powell (1991).
    MRKCOHP91: MRK corresponding states for CO in Holland and Powell (1991).
    MRKS2HP11: MRK corresponding states for S2 in Holland and Powell (2011).
    MRKH2SHP11: MRK corresponding states for H2S in Holland and Powell (2011).

Examples:
    Get the fugacity coefficient for the H2O CORK model from Holland and Powell (1998). Note that
    the input pressure should always be in bar:

    ```python
    >>> from atmodeller.eos.holland import CORKH2OHP98
    >>> model = CORKH2OHP98()
    >>> fugacity_coefficient = model.get_value(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.048278616058322
    ```

    Get the preferred fugacity models for various species from the Holland and Powell models. Note
    that the input pressure should always be in bar:
    
    ```python
    >>> from atmodeller.eos.holland import get_holland_fugacity_models
    >>> models = get_holland_and_powell_fugacity_models()
    >>> # list the available species
    >>> models.keys()
    >>> # Get the fugacity model for CO
    >>> co_model = models['CO']
    >>> # Determine the fugacity coefficient at 2000 K and 1000 bar
    >>> fugacity_coefficient = co_model.get_value(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.2664435476696503
    ```
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from scipy.constants import kilo

from atmodeller.eos.holland_base import (
    CORKABC,
    FugacityModelABC,
    MRKCriticalBehaviour,
    MRKExplicitABC,
    MRKImplicitABC,
)

logger: logging.Logger = logging.getLogger(__name__)

# The critical temperature for the CORK H2O model.
Tc_H2O: float = 695  # K
# The temperature at which a_gas = a; hence the critical point is handled by a single a parameter.
Ta_H2O: float = 673  # K
# b parameter value is the same across all phases (i.e. gas, fluid, liquid).
b0_H2O: float = 1.465


@dataclass(kw_only=True)
class Unitskbar:
    """Mixin to use kbar for the pressure units, which is required for Holland and Powell data."""

    scaling: float = field(init=False, default=kilo)


@dataclass(kw_only=True)
class MRKH2OLiquidHP91(Unitskbar, MRKImplicitABC):
    """MRK for liquid H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1113.4, -0.88517, 4.53e-3, -1.3183e-5),
    )
    b0: float = field(init=False, default=b0_H2O)
    Ta: float = field(init=False, default=Ta_H2O)

    def a(self, temperature: float) -> float:
        """MRK a parameter for liquid H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter for liquid H2O.
        """
        assert temperature <= self.Ta

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (self.Ta - temperature)
            + self.a_coefficients[2] * (self.Ta - temperature) ** 2
            + self.a_coefficients[3] * (self.Ta - temperature) ** 3
        )

        return a

    def initial_solution_volume(self, *args, **kwargs) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        For the liquid phase a suitably low value must be chosen. See appendix in Holland and
        Powell (1991).

        Args:
            *args: Unused positional arguments
            **kwargs: Unused keyword arguments.

        Returns:
            Initial solution volume.
        """
        del args
        del kwargs
        initial_volume = self.b / 2

        return initial_volume


@dataclass(kw_only=True)
class MRKH2OGasHP91(Unitskbar, MRKImplicitABC):
    """MRK for gaseous H2O. Equation 6a, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4,
            5.8487,
            -2.1370e-2,
            6.8133e-5,
        ),
    )
    b0: float = field(init=False, default=b0_H2O)
    Ta: float = field(init=False, default=Ta_H2O)

    def a(self, temperature: float) -> float:
        """MRK a parameter for gaseous H2O. Equation 6a, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter for gaseous H2O.
        """
        assert temperature <= self.Ta

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (self.Ta - temperature)
            + self.a_coefficients[2] * (self.Ta - temperature) ** 2
            + self.a_coefficients[3] * (self.Ta - temperature) ** 3
        )

        return a

    def initial_solution_volume(self, temperature: float, pressure: float) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        See appendix in Holland and Powell (1991).

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            Initial solution volume.
        """
        initial_volume: float = self.GAS_CONSTANT * temperature / pressure + 10 * self.b

        return initial_volume


@dataclass(kw_only=True)
class MRKH2OFluidHP91(Unitskbar, MRKImplicitABC):
    """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4,
            -0.22291,
            -3.8022e-4,
            1.7791e-7,
        ),
    )
    b0: float = field(init=False, default=b0_H2O)
    Ta: float = field(init=False, default=Ta_H2O)
    Tc: float = field(init=False, default=Tc_H2O)

    def a(self, temperature: float) -> float:
        """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter supercritical H2O.
        """
        assert temperature >= self.Ta

        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * (temperature - self.Ta)
            + self.a_coefficients[2] * (temperature - self.Ta) ** 2
            + self.a_coefficients[3] * (temperature - self.Ta) ** 3
        )

        return a

    def initial_solution_volume(self, temperature: float, pressure: float) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        See appendix in Holland and Powell (1991).

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            Initial solution volume.
        """
        if temperature >= self.Tc:
            initial_volume: float = self.GAS_CONSTANT * temperature / pressure + self.b
        else:
            initial_volume = self.b / 2

        return initial_volume


@dataclass(kw_only=True)
class MRKCO2HP91(Unitskbar, MRKImplicitABC):
    """MRK for CO2. Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(741.2, -0.10891, -3.903e-4))
    b0: float = field(init=False, default=3.057)

    def a(self, temperature: float) -> float:
        """MRK a parameter. Holland and Powell (1991), p270.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            MRK a parameter.
        """
        a: float = (
            self.a_coefficients[0]
            + self.a_coefficients[1] * temperature
            + self.a_coefficients[2] * temperature**2
        )
        return a

    def initial_solution_volume(self, temperature: float, pressure: float) -> float:
        """Initial guess volume for the solution to ensure convergence to the correct root.

        Args:
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            Initial solution volume.
        """
        initial_volume: float = self.GAS_CONSTANT * temperature / pressure + self.b

        return initial_volume


# For completeness, the MRK model for CO2 in 1998 is the same as the 1991 paper.
MRKCO2HP98 = MRKCO2HP91


@dataclass(kw_only=True)
class MRKH2OHP91(Unitskbar, MRKCriticalBehaviour):
    """MRK for H2O that spans the range across the critical behaviour.

    See base class.
    """

    mrk_fluid: MRKImplicitABC = field(init=False, default_factory=MRKH2OFluidHP91)
    mrk_gas: MRKImplicitABC = field(init=False, default_factory=MRKH2OGasHP91)
    mrk_liquid: MRKImplicitABC = field(init=False, default_factory=MRKH2OLiquidHP91)
    Ta: float = field(init=False, default=Ta_H2O)
    Tc: float = field(init=False, default=Tc_H2O)

    def Psat(self, temperature: float) -> float:
        """Saturation curve. Equation 5, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Saturation curve pressure in kbar.
        """
        Psat: float = (
            -13.627e-3
            + 7.29395e-7 * temperature**2
            - 2.34622e-9 * temperature**3
            + 4.83607e-15 * temperature**5
        )
        return Psat


# For completeness, the MRK model for H2O in 1998 is the same as the 1991 paper.
MRKH2OHP98 = MRKH2OHP91


@dataclass(kw_only=True)
class CORKCO2HP91(Unitskbar, CORKABC):
    """Full CORK equation for CO2 from Holland and Powell (1991).

    See base class.
    """

    P0: float = field(init=False, default=5.0)
    mrk: FugacityModelABC = field(init=False, default_factory=MRKCO2HP91)
    a_virial: tuple[float, float] = field(init=False, default=(1.33790e-2, -1.01740e-5))
    b_virial: tuple[float, float] = field(init=False, default=(-2.26924e-1, 7.73793e-5))


@dataclass(kw_only=True)
class CORKCO2HP98(Unitskbar, CORKABC):
    """Full CORK equation for CO2 from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.

    See base class.
    """

    P0: float = field(init=False, default=5.0)
    mrk: FugacityModelABC = field(init=False, default_factory=MRKCO2HP98)
    a_virial: tuple[float, float] = field(init=False, default=(5.40776e-3, -1.59046e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-1.78198e-1, 2.45317e-5))


@dataclass(kw_only=True)
class CORKH2OHP91(Unitskbar, CORKABC):
    """Full CORK equation for H2O from Holland and Powell (1991).

    See base class.
    """

    P0: float = field(init=False, default=2.0)
    mrk: FugacityModelABC = field(init=False, default_factory=MRKH2OHP91)
    a_virial: tuple[float, float] = field(init=False, default=(-3.2297554e-3, 2.2215221e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-3.025650e-2, -5.343144e-6))


@dataclass(kw_only=True)
class CORKH2OHP98(Unitskbar, CORKABC):
    """Full CORK equation for H2O from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.

    See base class.
    """

    P0: float = field(init=False, default=2.0)
    mrk: FugacityModelABC = field(init=False, default_factory=MRKH2OHP98)
    a_virial: tuple[float, float] = field(init=False, default=(1.9853e-3, 0))
    b_virial: tuple[float, float] = field(init=False, default=(-8.9090e-2, 0))
    c_virial: tuple[float, float] = field(init=False, default=(8.0331e-2, 0))


@dataclass(kw_only=True)
class MRKCorrespondingStatesHP91(Unitskbar, MRKExplicitABC):
    """A MRK simplified model used for corresponding states from Holland and Powell (1991).

    Universal constants from Table 2, Holland and Powell (1991). See base class.

    Args:
        Tc: Critical temperature in kelvin for corresponding states.
        Pc: Critical pressure for corresponding states.

    Attributes:
        Tc: Critical temperature in kelvin for corresponding states.
        Pc: Critical pressure for corresponding states.
    """

    Tc: float
    Pc: float
    a_coefficients: tuple[float, float] = field(init=False, default=(5.45963e-5, -8.63920e-6))
    b0: float = field(init=False, default=9.18301e-4)

    def a(self, temperature: float) -> float:
        """Parameter a in Equation 9, Holland and Powell (1991).

        Args:
            temperature: Temperature in kelvin.

        Returns:
            Parameter a in kJ^2 kbar^(-1) K^(1/2) mol^(-2).
        """
        a: float = (
            self.a_coefficients[0] * self.Tc ** (5.0 / 2.0) / self.Pc
            + self.a_coefficients[1] * self.Tc ** (3.0 / 2.0) / self.Pc * temperature
        )
        return a

    @property
    def b(self) -> float:
        """Parameter b in Equation 9, Holland and Powell (1991).

        Returns:
            Parameter b in kJ kbar^(-1) mol^(-1).
        """
        b: float = self.b0 * self.Tc / self.Pc
        return b


@dataclass(kw_only=True)
class CORKCorrespondingStatesHP91(Unitskbar, CORKABC):
    """A Simplified Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    Although originally fit to CO2 data, this predicts the volumes and fugacities for several other
    gases which are known to obey approximately the principle of corresponding states. The
    corresponding states parameters are from Table 2 in Holland and Powell (1991). Note also in
    this case it appears P0 is always zero, even though for the full CORK equations it determines
    whether or not the virial contribution is added. It assumes there are no complications of
    critical behaviour in the P-T range considered.

    Args:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        mrk: Fugacity model for computing the MRK contribution.
        scaling: See base class. The scaling scales the virial compensation term.

    Attributes:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume. Set to 0.
        a_virial: Constants for the virial contribution (d0 and d1 in Table 2).
        b_virial: Constants for the virial contribution (c0 and c1 in Table 2).
        c_virial: Constants for the virial contribution (unused).
        virial: Virial contribution object.
        scaling: See base class.
        GAS_CONSTANT: See base class.
    """

    P0: float = field(init=False, default=0)
    a_virial: tuple[float, float] = field(init=False, default=(6.93054e-7, -8.38293e-8))
    b_virial: tuple[float, float] = field(init=False, default=(-3.30558e-5, 2.30524e-6))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))


# Critical parameters for CO2.
Tc_CO2: float = 304.2
Pc_CO2: float = 0.0738


@dataclass(kw_only=True)
class MRKSimpleCO2HP91(MRKCorrespondingStatesHP91):
    """MRK for CO. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=Tc_CO2)
    Pc: float = field(init=False, default=Pc_CO2)


@dataclass(kw_only=True)
class CORKSimpleCO2HP91(CORKCorrespondingStatesHP91):
    """CORK for CO2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    mrk: FugacityModelABC = field(init=False, default_factory=MRKSimpleCO2HP91)
    Tc: float = field(init=False, default=Tc_CO2)
    Pc: float = field(init=False, default=Pc_CO2)


# Critical parameters for CH4.
Tc_CH4: float = 190.6
Pc_CH4: float = 0.0460


@dataclass(kw_only=True)
class MRKCH4HP91(MRKCorrespondingStatesHP91):
    """MRK for CH4. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=Tc_CH4)
    Pc: float = field(init=False, default=Pc_CH4)


@dataclass(kw_only=True)
class CORKCH4HP91(CORKCorrespondingStatesHP91):
    """CORK for CH4. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    mrk: FugacityModelABC = field(init=False, default_factory=MRKCH4HP91)
    Tc: float = field(init=False, default=Tc_CH4)  # K
    Pc: float = field(init=False, default=Pc_CH4)


# Critical parameters for H2.
Tc_H2: float = 41.2
Pc_H2: float = 0.0211


@dataclass(kw_only=True)
class MRKH2HP91(MRKCorrespondingStatesHP91):
    """MRK constants for H2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=Tc_H2)  # K
    Pc: float = field(init=False, default=Pc_H2)


@dataclass(kw_only=True)
class CORKH2HP91(CORKCorrespondingStatesHP91):
    """CORK for H2. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    mrk: FugacityModelABC = field(init=False, default_factory=MRKH2HP91)
    Tc: float = field(init=False, default=Tc_H2)
    Pc: float = field(init=False, default=Pc_H2)


# Critical parameters for CO.
Tc_CO: float = 132.9
Pc_CO: float = 0.0350


@dataclass(kw_only=True)
class MRKCOHP91(MRKCorrespondingStatesHP91):
    """MRK for CO. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    Tc: float = field(init=False, default=Tc_CO)
    Pc: float = field(init=False, default=Pc_CO)


@dataclass(kw_only=True)
class CORKCOHP91(CORKCorrespondingStatesHP91):
    """CORK for CO. See base class.

    See Table below Figure 8 in Holland and Powell (1991).
    """

    mrk: FugacityModelABC = field(init=False, default_factory=MRKCOHP91)
    Tc: float = field(init=False, default=Tc_CO)
    Pc: float = field(init=False, default=Pc_CO)


# Critical parameters for S2.
Tc_S2: float = 208.15
Pc_S2: float = 0.072954


@dataclass(kw_only=True)
class MRKS2HP11(MRKCorrespondingStatesHP91):
    """MRK for S2. See base class.

    Holland and Powell (2011) state that the critical constants for S2 are taken from:

        Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
        McGraw-Hill, New York.

    In the fifth edition of this book S2 is not given (only S is), so instead the critical
    constants for S2 are taken from:

        Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
        Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.
        http://www.minsocam.org/ammin/AM77/AM77_1038.pdf
    """

    Tc: float = field(init=False, default=Tc_S2)
    Pc: float = field(init=False, default=Pc_S2)


@dataclass(kw_only=True)
class CORKS2HP11(CORKCorrespondingStatesHP91):
    """CORK for S2. See base class.

    Holland and Powell (2011) state that the critical constants for S2 are taken from:

        Reid, R.C., Prausnitz, J.M. & Sherwood, T.K., 1977. The Properties of Gases and Liquids.
        McGraw-Hill, New York.

    In the fifth edition of this book S2 is not given (only S is), so instead the critical
    constants for S2 are taken from:

        Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
        Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.
        http://www.minsocam.org/ammin/AM77/AM77_1038.pdf
    """

    mrk: FugacityModelABC = field(init=False, default_factory=MRKS2HP11)
    Tc: float = field(init=False, default=Tc_S2)
    Pc: float = field(init=False, default=Pc_S2)


# Critical parameters for S2.
Tc_H2S: float = 373.4
Pc_H2S: float = 0.08963


@dataclass(kw_only=True)
class MRKH2SHP11(MRKCorrespondingStatesHP91):
    """MRK for H2S. See base class.

    Appendix A.19 in:

        Poling, Prausnitz, and O'Connell, 2001. The Properties of Gases and Liquids, 5th edition.
        McGraw-Hill, New York. DOI: 10.1036/0070116822.
    """

    Tc: float = field(init=False, default=Tc_H2S)
    Pc: float = field(init=False, default=Pc_H2S)


@dataclass(kw_only=True)
class CORKH2SHP11(CORKCorrespondingStatesHP91):
    """CORK for H2S. See base class.

    Appendix A.19 in:

        Poling, Prausnitz, and O'Connell, 2001. The Properties of Gases and Liquids, 5th edition.
        McGraw-Hill, New York. DOI: 10.1036/0070116822.
    """

    mrk: FugacityModelABC = field(init=False, default_factory=MRKH2SHP11)
    Tc: float = field(init=False, default=Tc_H2S)
    Pc: float = field(init=False, default=Pc_H2S)


def get_holland_fugacity_models() -> dict[str, FugacityModelABC]:
    """Gets a dictionary of the preferred fugacity models to use for each species.

    Returns:
        Dictionary of preferred fugacity models for each species.
    """
    models: dict[str, FugacityModelABC] = {}
    models["CH4"] = CORKCH4HP91()
    models["CO"] = CORKCOHP91()
    models["CO2"] = CORKCO2HP98()
    models["H2"] = CORKH2HP91()
    models["H2O"] = CORKH2OHP98()
    models["H2S"] = CORKH2SHP11()
    models["S2"] = CORKS2HP11()

    return models
