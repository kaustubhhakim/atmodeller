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

from atmodeller import GAS_CONSTANT
from atmodeller.eos.interfaces import (
    CORKABC,
    MRKCriticalBehaviour,
    MRKExplicitABC,
    MRKImplicitABC,
    RealGasABC,
    critical_data_dictionary,
)

logger: logging.Logger = logging.getLogger(__name__)


# region MRK Corresponding States


@dataclass(kw_only=True)
class MRKCorrespondingStatesHP91(MRKExplicitABC):
    """A MRK simplified model used for corresponding states from Holland and Powell (1991).

    Universal constants from Table 2, Holland and Powell (1991). Note the unit conversion to SI.

    a coefficients have been multiplied by 1e6 to convert kJ^2 to J^2 in the numerator. The
        pressure units effectively cancel because the ratio a/b is calculated.
    b coefficients have been multiplied by 1e3 to convert kJ to J in the numerator. The pressure
        units also cancel because b is multiplied by a pressure.

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(5.45963e1, -8.63920e0, 0))
    b0: float = field(init=False, default=9.18301e-1)

    @classmethod
    def get_species(cls, species: str) -> RealGasABC:
        """Instantiates a MRK corresponding states model for a given species.

        Args:
            species: A species which is a key in the critical_data_dictionary

        Returns:
            A corresponding states model for the species
        """
        return cls(
            critical_temperature=critical_data_dictionary[species].temperature,
            critical_pressure=critical_data_dictionary[species].pressure,
        )


@dataclass(kw_only=True)
class CORKCorrespondingStatesHP91(CORKABC):
    """A Simplified Compensated-Redlich-Kwong (CORK) equation from Holland and Powell (1991).

    Although originally fit to CO2 data, this predicts the volumes and fugacities for several other
    gases which are known to obey approximately the principle of corresponding states. The
    corresponding states parameters are from Table 2 in Holland and Powell (1991). Note also in
    this case it appears P0 is always zero, even though for the full CORK equations it determines
    whether or not the virial contribution is added. It assumes there are no complications of
    critical behaviour in the P-T range considered.

    The unit conversions to SI mean that every coefficient has been multiplied by 1e3 to convert
    kJ to J in the numerator. The pressure units cancel in the calculation.

    Args:
        critical_temperature: Critical temperature in kelvin.
        critical_pressure: Critical pressure.
        mrk: Fugacity model for computing the MRK contribution.

    Attributes:
        critical_temperature: Critical temperature in kelvin.
        critical_pressure: Critical pressure.
        P0: Pressure at which the MRK equation begins to overestimate the molar volume. Set to 0.
        a_virial: Constants for the virial contribution (d0 and d1 in Table 2).
        b_virial: Constants for the virial contribution (c0 and c1 in Table 2).
        c_virial: Constants for the virial contribution (unused).
        virial: Virial contribution object.
    """

    P0: float = field(init=False, default=0)
    a_virial: tuple[float, float] = field(init=False, default=(6.93054e-4, -8.38293e-5))
    b_virial: tuple[float, float] = field(init=False, default=(-3.30558e-2, 2.30524e-3))
    c_virial: tuple[float, float] = field(init=False, default=(0, 0))

    @classmethod
    def get_species(cls, species: str) -> RealGasABC:
        """Instantiates a CORK corresponding states model for a given species.

        Args:
            species: A species which is a key in the critical_data_dictionary

        Returns:
            A corresponding states model for the species
        """
        mrk: RealGasABC = MRKCorrespondingStatesHP91.get_species(species)

        return cls(
            mrk=mrk,
            critical_temperature=critical_data_dictionary[species].temperature,
            critical_pressure=critical_data_dictionary[species].pressure,
        )


# MRK concrete classes
MRKSimpleCO2HP91: RealGasABC = MRKCorrespondingStatesHP91.get_species("CO2")
MRKCH4HP91: RealGasABC = MRKCorrespondingStatesHP91.get_species("CH4")
MRKH2HP91: RealGasABC = MRKCorrespondingStatesHP91.get_species("H2_Holland")
MRKCOHP91: RealGasABC = MRKCorrespondingStatesHP91.get_species("CO")
MRKS2HP11: RealGasABC = MRKCorrespondingStatesHP91.get_species("S2")
MRKH2SHP11: RealGasABC = MRKCorrespondingStatesHP91.get_species("H2S")

# CORK concrete classes
CORKSimpleCO2HP91: RealGasABC = CORKCorrespondingStatesHP91.get_species("CO2")
CORKCH4HP91: RealGasABC = CORKCorrespondingStatesHP91.get_species("CH4")
CORKH2HP91: RealGasABC = CORKCorrespondingStatesHP91.get_species("H2_Holland")
CORKCOHP91: RealGasABC = CORKCorrespondingStatesHP91.get_species("CO")
CORKS2HP11: RealGasABC = CORKCorrespondingStatesHP91.get_species("S2")
CORKH2SHP11: RealGasABC = CORKCorrespondingStatesHP91.get_species("H2S")

# endregion

# For any subclass of MRKImplicitABC, note the unit conversion to SI compared to the values that
# Holland and Powell present
# a coefficients have been multiplied by 1e3.
# b coefficients remain the same

# The critical temperature for the CORK H2O model.
Tc_H2O: float = 695  # K
# The temperature at which a_gas = a; hence the critical point is handled by a single a parameter.
Ta_H2O: float = 673  # K
# b parameter value is the same across all phases (i.e. gas, fluid, liquid).
b0_H2O: float = 1.465


@dataclass(kw_only=True)
class MRKH2OLiquidHP91(MRKImplicitABC):
    """MRK for liquid H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(1113.4e3, -0.88517e3, 4.53, -1.3183e-2),
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
class MRKH2OGasHP91(MRKImplicitABC):
    """MRK for gaseous H2O. Equation 6a, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4e3,
            5.8487e3,
            -2.1370e1,
            6.8133e-2,
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
        initial_volume: float = GAS_CONSTANT * temperature / pressure + 10 * self.b

        return initial_volume


@dataclass(kw_only=True)
class MRKH2OFluidHP91(MRKImplicitABC):
    """MRK a parameter for supercritical H2O. Equation 6, Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(
        init=False,
        default=(
            1113.4e3,
            -0.22291e3,
            -3.8022e-1,
            1.7791e-4,
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
            initial_volume: float = GAS_CONSTANT * temperature / pressure + self.b
        else:
            initial_volume = self.b / 2

        return initial_volume


@dataclass(kw_only=True)
class MRKCO2HP91(MRKImplicitABC):
    """MRK for CO2. Holland and Powell (1991).

    See base class.
    """

    a_coefficients: tuple[float, ...] = field(init=False, default=(741.2e3, -0.10891e3, -3.903e-1))
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
        initial_volume: float = GAS_CONSTANT * temperature / pressure + self.b

        return initial_volume


# For completeness, the MRK model for CO2 in 1998 is the same as the 1991 paper.
MRKCO2HP98 = MRKCO2HP91


@dataclass(kw_only=True)
class MRKH2OHP91(MRKCriticalBehaviour):
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
            -13.627
            + 7.29395e-4 * temperature**2
            - 2.34622e-6 * temperature**3
            + 4.83607e-12 * temperature**5
        )
        return Psat


# For completeness, the MRK model for H2O in 1998 is the same as the 1991 paper.
MRKH2OHP98 = MRKH2OHP91


@dataclass(kw_only=True)
class CORKCO2HP91(CORKABC):
    """Full CORK equation for CO2 from Holland and Powell (1991).

    See base class.
    """

    P0: float = field(init=False, default=5.0e3)
    mrk: RealGasABC = field(init=False, default_factory=MRKCO2HP91)
    a_virial: tuple[float, float] = field(init=False, default=(1.33790e-2, -1.01740e-5))
    b_virial: tuple[float, float] = field(init=False, default=(-2.26924e-1, 7.73793e-5))


@dataclass(kw_only=True)
class CORKCO2HP98(CORKABC):
    """Full CORK equation for CO2 from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.

    See base class.
    """

    P0: float = field(init=False, default=5.0e3)
    mrk: RealGasABC = field(init=False, default_factory=MRKCO2HP98)
    a_virial: tuple[float, float] = field(init=False, default=(5.40776e-3, -1.59046e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-1.78198e-1, 2.45317e-5))


@dataclass(kw_only=True)
class CORKH2OHP91(CORKABC):
    """Full CORK equation for H2O from Holland and Powell (1991).

    See base class.
    """

    P0: float = field(init=False, default=2.0e3)
    mrk: RealGasABC = field(init=False, default_factory=MRKH2OHP91)
    a_virial: tuple[float, float] = field(init=False, default=(-3.2297554e-3, 2.2215221e-6))
    b_virial: tuple[float, float] = field(init=False, default=(-3.025650e-2, -5.343144e-6))


@dataclass(kw_only=True)
class CORKH2OHP98(CORKABC):
    """Full CORK equation for H2O from Holland and Powell (1998).

    Holland and Powell (1998) updated the virial-like terms compared to their 1991 paper.

    See base class.
    """

    P0: float = field(init=False, default=2.0e3)
    mrk: RealGasABC = field(init=False, default_factory=MRKH2OHP98)
    a_virial: tuple[float, float] = field(init=False, default=(1.9853e-3, 0))
    b_virial: tuple[float, float] = field(init=False, default=(-8.9090e-2, 0))
    c_virial: tuple[float, float] = field(init=False, default=(8.0331e-2, 0))


def get_holland_fugacity_models() -> dict[str, RealGasABC]:
    """Gets a dictionary of the preferred fugacity models to use for each species.

    Returns:
        Dictionary of preferred fugacity models for each species.
    """
    models: dict[str, RealGasABC] = {}
    models["CH4"] = CORKCH4HP91
    models["CO"] = CORKCOHP91
    models["CO2"] = CORKCO2HP98()
    models["H2"] = CORKH2HP91
    models["H2O"] = CORKH2OHP98()
    models["H2S"] = CORKH2SHP11
    models["S2"] = CORKS2HP11

    return models
