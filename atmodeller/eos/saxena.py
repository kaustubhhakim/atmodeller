"""Real gas EOSs from Shi and Saxena (1992), Saxena and Fei (1988), and Saxena and Fei (1987a,b)

See the LICENSE file for licensing information.

Functions:
    get_saxena_eos_models: Gets the preferred EOS models to use for each species

Real gas EOSs (class instances) in this module that can be imported:
    Ar_SF87: Corresponding states for Ar from Saxena and Fei (1987)
    CH4_SS92: Corresponding states for CH4 from Shi and Saxena (1992)
    COS_S92: Corresponding states for CO from Shi and Saxena (1992)
    CO2_SS92: Corresponding states for CO2 from Shi and Saxena (1992)
    COS_SS92: Correponding states for COS from Shi and Saxena (1992)
    H2S_F87: Corresponding states for H2 from Saxena and Fei (1987)
    H2_SS92: H2 from Shi and Saxena with refitted high pressure data (1992)
    H2S_SS92: H2S from Shi and Saxena (1992)
    N2_SF87: Corresponding states for N2 from Saxena and Fei (1987)
    O2_SS92: Corresponding states for O2 from Shi and Saxena (1992)
    S2_SS92: Corresponding states for S2 from Shi and Saxena (1992)
    SO2_SS92: SO2 from Shi and Saxena (1992)

Examples:
    Get the fugacity coefficient for the CO2 corresponding states model from Shi and Saxena (1992)
    Note that the input pressure should always be in bar:

    ```python
    >>> from atmodeller.eos.saxena import CO2_SS92
    >>> fugacity_coefficient = CO2SS92.fugacity_coefficient(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.0973520647000174
    ```

    Get the preferred EOS models for various species from the Saxena models. Note that the input 
    pressure should always be in bar:
    
    ```python
    >>> from atmodeller.eos.saxena import get_saxena_eos_models
    >>> models = get_saxena_eos_models()
    >>> # list the available species
    >>> models.keys()
    >>> # Get the EOS model for CO
    >>> co_model = models['CO']
    >>> # Determine the fugacity coefficient at 2000 K and 1000 bar
    >>> fugacity_coefficient = co_model.fugacity_coefficient(temperature=2000, pressure=1000)
    >>> print(fugacity_coefficient)
    1.1648016694290084
    ```
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np

from atmodeller import GAS_CONSTANT
from atmodeller.eos.interfaces import (
    CombinedEOSModel,
    RealGasABC,
    critical_data_dictionary,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SaxenaABC(RealGasABC):
    """Shi and Saxena (1992) fugacity model

    The model presented in Shi and Saxena (1992) is a general form that can be adapted to the
    previous work of Saxena and Fei (1988) and Saxena and Fei (1987).

    Shi and Saxena (1992), Thermodynamic modeling of the C-H-O-S fluid system, American
    Mineralogist, Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid
    phases.

    http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

    Args:
        critical_temperature: Critical temperature in kelvin. Defaults to unity (not used)
        critical_pressure: Critical pressure in bar. Defaults to unity (not used)
        a_coefficients: a coefficients (see paper). Defaults to empty
        b_coefficients: b coefficients (see paper). Defaults to empty
        c_coefficients: c coefficients (see paper). Defaults to empty
        d_coefficients: d coefficients (see paper). Defaults to empty

    Attributes:
        critical_temperature: Critical temperature in kelvin
        critical_pressure: Critical pressure in bar
        a_coefficients: a coefficients
        b_coefficients: b coefficients
        c_coefficients: c coefficients
        d_coefficients: d coefficients
        standard_state_pressure: Scaled standard state pressure with the appropriate units
    """

    a_coefficients: tuple[float, ...] = field(default_factory=tuple)
    b_coefficients: tuple[float, ...] = field(default_factory=tuple)
    c_coefficients: tuple[float, ...] = field(default_factory=tuple)
    d_coefficients: tuple[float, ...] = field(default_factory=tuple)

    @abstractmethod
    def _get_compressibility_coefficient(
        self, scaled_temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation

        Shi and Saxena (1992), Equation 1

        Args:
            temperature: Scaled temperature
            coefficients: Tuple of the coefficients a, b, c, or d

        Returns
            The relevant coefficient
        """
        ...

    def _a(self, scaled_temperature: float) -> float:
        """a parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            a parameter
        """
        a: float = self._get_compressibility_coefficient(scaled_temperature, self.a_coefficients)

        return a

    def _b(self, scaled_temperature: float) -> float:
        """b parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            b parameter
        """
        b: float = self._get_compressibility_coefficient(scaled_temperature, self.b_coefficients)

        return b

    def _c(self, scaled_temperature: float) -> float:
        """c parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            c parameter
        """
        c: float = self._get_compressibility_coefficient(scaled_temperature, self.c_coefficients)

        return c

    def _d(self, scaled_temperature: float) -> float:
        """d parameter

        Args:
            scaled_temperature: Scaled temperature

        Returns:
            d parameter
        """
        d: float = self._get_compressibility_coefficient(scaled_temperature, self.d_coefficients)

        return d

    def compressibility_parameter(self, temperature: float, pressure: float) -> float:
        """Compressibility parameter at temperature and pressure

        This overrides the base class because the compressibility factor is used to determine the
        volume, whereas in the base class the volume is used to determine the compressibility
        factor.

        Shi and Saxena (1992), Equation 2

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            The compressibility parameter, Z
        """
        Tr: float = self.scaled_temperature(temperature)
        Pr: float = self.scaled_pressure(pressure)
        Z: float = self._a(Tr) + self._b(Tr) * Pr + self._c(Tr) * Pr**2 + self._d(Tr) * Pr**3

        return Z

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume

        Shi and Saxena (1992), Equation 1.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume
        """
        Z: float = self.compressibility_parameter(temperature, pressure)
        volume: float = Z * self.ideal_volume(temperature, pressure)

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (VdP)

        Shi and Saxena (1992), Equation 11.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral
        """
        Tr: float = self.scaled_temperature(temperature)
        Pr: float = self.scaled_pressure(pressure)
        P0r: float = self.scaled_pressure(self.standard_state_pressure)
        volume_integral: float = (
            (
                self._a(Tr) * np.log(Pr / P0r)
                + self._b(Tr) * (Pr - P0r)
                + (1.0 / 2) * self._c(Tr) * (Pr**2 - P0r**2)
                + (1.0 / 3) * self._d(Tr) * (Pr**3 - P0r**3)
            )
            * GAS_CONSTANT
            * temperature
        )

        return volume_integral


@dataclass(kw_only=True)
class SaxenaFiveCoefficients(SaxenaABC):
    """Fugacity model with five coefficients, which is generally used for low pressures"""

    def _get_compressibility_coefficient(
        self, scaled_temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation

        Shi and Saxena (1992), Equation 3b

        Args:
            scaled_temperature: Temperature
            coefficients: Tuple of the coefficients a, b, c, or d

        Returns
            The relevant coefficient
        """
        coefficient: float = (
            coefficients[0]
            + coefficients[1] / scaled_temperature
            + coefficients[2] / scaled_temperature ** (3 / 2)
            + coefficients[3] / scaled_temperature**3
            + coefficients[4] / scaled_temperature**4
        )

        return coefficient


@dataclass(kw_only=True)
class SaxenaEightCoefficients(SaxenaABC):
    """Fugacity model with eight coefficients, which is generally used for high pressures"""

    def _get_compressibility_coefficient(
        self, scaled_temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation

        Shi and Saxena (1992), Equation 3a

        Args:
            scaled_temperature: Temperature
            coefficients: Tuple of the coefficients a, b, c, or d

        Returns
            The relevant coefficient
        """
        coefficient: float = (
            coefficients[0]
            + coefficients[1] * scaled_temperature
            + coefficients[2] / scaled_temperature
            + coefficients[3] * scaled_temperature**2
            + coefficients[4] / scaled_temperature**2
            + coefficients[5] * scaled_temperature**3
            + coefficients[6] / scaled_temperature**3
            + coefficients[7] * np.log(scaled_temperature)
        )

        return coefficient


# Low pressure model for H2 from Shi and Saxena (1992)
# The coefficients are the same as for the corresponding states model in Table 1(a) because they
# originate from Saxena and Fei (1987a). Table 1(b), <1000 bar
_H2_low_pressure_SS92: RealGasABC = SaxenaFiveCoefficients(
    critical_temperature=critical_data_dictionary["H2"].temperature,
    critical_pressure=critical_data_dictionary["H2"].pressure,
    a_coefficients=(1, 0, 0, 0, 0, 0),
    b_coefficients=(0, 0.9827e-1, 0, -0.2709, 0),
    # Saxena and Fei (1987a), Eq. 23, C final coefficient = 0.1472e-1 (not 0.1427e-1)
    c_coefficients=(0, 0, -0.1030e-2, 0, 0.1472e-1),
    d_coefficients=(0, 0, 0, 0, 0),
)

# High pressure model for H2 from Shi and Saxena (1992)
# This model is currently not used because it has been superseded by the refitted model
# Coefficients require the actual temperature and pressure. Table 1(b), >1 kbar
_H2_high_pressure_SS92: RealGasABC = SaxenaEightCoefficients(
    a_coefficients=(2.2615, 0, -6.8712e1, 0, -1.0573e4, 0, 0, -1.6936e-1),
    b_coefficients=(-2.6707e-4, 0, 2.0173e-1, 0, 4.5759, 0, 0, 3.1452e-5),
    c_coefficients=(-2.3376e-9, 0, 3.4091e-7, 0, -1.4188e-3, 0, 0, 3.0117e-10),
    d_coefficients=(-3.2606e-15, 0, 2.4402e-12, 0, -2.4027e-9, 0, 0, 0),
)

# High pressure model for H2 from Shi and Saxena (1992) refitted using the experimental volume,
# pressure, and temperature data from Presnall (1969) and Ross, Ree and Young (1983).
# For this fit, we assume the same functional form as Shi and Saxena for pressures above 1 kbar
# (Equation 2 and Equation 3a), including which coefficients are set to zero
# (as in Table 1b at pressures > 1 kbar) and a least squares regression. The refitting is performed
# using reduced temperature and pressure.
_H2_high_pressure_SS92_refit: RealGasABC = SaxenaEightCoefficients(
    critical_temperature=critical_data_dictionary["H2"].temperature,
    critical_pressure=critical_data_dictionary["H2"].pressure,
    a_coefficients=(1.00574428e00, 0, 1.93022092e-03, 0, -3.79261142e-01, 0, 0, -2.44217972e-03),
    b_coefficients=(1.31517888e-03, 0, 7.22328441e-02, 0, 4.84354163e-02, 0, 0, -4.19624507e-04),
    c_coefficients=(2.64454401e-06, 0, -5.18445629e-05, 0, -2.05045979e-04, 0, 0, -3.64843213e-07),
    d_coefficients=(2.28281107e-11, 0, -1.07138603e-08, 0, 3.67720815e-07, 0, 0, 0),
)

# H2 fugacity model from Shi and Saxena (1992)
# Combines the low pressure and high pressure models into a single model. Table 1(b)
models: tuple[RealGasABC, ...] = (_H2_low_pressure_SS92, _H2_high_pressure_SS92_refit)
upper_pressure_bounds: tuple[float, ...] = (1000,)
H2_SS92: RealGasABC = CombinedEOSModel(models=models, upper_pressure_bounds=upper_pressure_bounds)

# High pressure model for H2 from Saxena and Fei (1988). Table on p1196
# This model does not at all agree with Shi and Saxena or data, regardless of whether the
# temperature and pressure are the actual values or reduced values. Since this model cannot be
# trusted it is commented out.
# H2_high_pressure_SF88: RealGasABC = SaxenaEightCoefficients(
#     critical_temperature=critical_data_dictionary["H2"].temperature,
#     critical_pressure=critical_data_dictionary["H2"].pressure,
#     a_coefficients=(1.6688, 0, -2.0759, 0, -9.6173, 0, 0, -0.1694),
#     b_coefficients=(-2.0410e-3, 0, 7.9230e-2, 0, 5.4295e-2, 0, 0, 4.0887e-4),
#     c_coefficients=(-2.1693e-7, 0, 1.7406e-6, 0, -2.1885e-4, 0, 0, 5.0897e-5),
#     d_coefficients=(-7.1635e-12, 0, 1.6197e-10, 0, -4.8181e-9, 0, 0, 0),
# )

# Fugacity model for SO2 from Shi and Saxena (1992). Table 1(c)
SO2_SS92: RealGasABC = SaxenaEightCoefficients(
    critical_temperature=critical_data_dictionary["SO2"].temperature,
    critical_pressure=critical_data_dictionary["SO2"].pressure,
    a_coefficients=(0.92854, 0.43269e-1, -0.24671, 0, 0.24999, 0, -0.53182, -0.16461e-1),
    b_coefficients=(
        0.84866e-3,
        -0.18379e-2,
        0.66787e-1,
        0,
        -0.29427e-1,
        0,
        0.29003e-1,
        0.54808e-2,
    ),
    c_coefficients=(
        -0.35456e-3,
        0.23316e-4,
        0.94159e-3,
        0,
        -0.81653e-3,
        0,
        0.23154e-3,
        0.55542e-4,
    ),
    d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
)

# Fugacity model for H2S from Shi and Saxena (1992)
# Table 1(d), 1-500 bar
_H2S_low_pressure_SS92: RealGasABC = SaxenaEightCoefficients(
    critical_temperature=critical_data_dictionary["H2S"].temperature,
    critical_pressure=critical_data_dictionary["H2S"].pressure,
    a_coefficients=(0.14721e1, 0.11177e1, 0.39657e1, 0, -0.10028e2, 0, 0.45484e1, -0.38200e1),
    b_coefficients=(0.16066, 0.10887, 0.29014, 0, -0.99593, 0, -0.18627, -0.45515),
    c_coefficients=(-0.28933, -0.70522e-1, 0.39828, 0, -0.50533e-1, 0, 0.11760, 0.33972),
    d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
)

# Fugacity model for H2S from Shi and Saxena (1992).
# Table 1(d), 500-10000 bar
_H2S_high_pressure_SS92: RealGasABC = SaxenaEightCoefficients(
    critical_temperature=critical_data_dictionary["H2S"].temperature,
    critical_pressure=critical_data_dictionary["H2S"].pressure,
    a_coefficients=(0.59941, -0.15570e-2, 0.45250e-1, 0, 0.36687, 0, -0.79248, 0.26058),
    b_coefficients=(
        0.22545e-1,
        0.17473e-2,
        0.48253e-1,
        0,
        -0.19890e-1,
        0,
        0.32794e-1,
        -0.10985e-1,
    ),
    c_coefficients=(
        0.57375e-3,
        -0.20944e-5,
        -0.11894e-2,
        0,
        0.14661e-2,
        0,
        -0.75605e-3,
        -0.27985e-3,
    ),
    d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
)

# H2S fugacity model from Shi and Saxena (1992).
# Combines the low pressure and high pressure models into a single model. See Table 1(d)
models: tuple[RealGasABC, ...] = (_H2S_low_pressure_SS92, _H2S_high_pressure_SS92)
upper_pressure_bounds: tuple[float, ...] = (500,)
H2S_SS92: RealGasABC = CombinedEOSModel(models=models, upper_pressure_bounds=upper_pressure_bounds)


def get_corresponding_states_SS92(species: str) -> RealGasABC:
    """Corresponding states from Shi and Saxena (1992)

    The coefficients for the low and medium pressure regime are actually lifted from Saxena and
    Fei (1987a), although some values disagree in either value or sign with Table 1(a) in the 1992
    paper. We need to decide which one is right, probably by computing the RMSE and taking the
    best-fitting combination. I'd assume the 1987 study is correct and the error is copying the
    data into the 1992 paper?  To be investigated.

    Table 1(a)

    Args:
        species: Species name. Must corresponding to an entry (key) in critical_data_dictionary

    Returns:
        Corresponding states fugacity model
    """

    critical_temperature: float = critical_data_dictionary[species].temperature
    critical_pressure: float = critical_data_dictionary[species].pressure

    # Table 1(a), <1000 bar
    low_pressure: RealGasABC = SaxenaFiveCoefficients(
        critical_temperature=critical_temperature,
        critical_pressure=critical_pressure,
        a_coefficients=(1, 0, 0, 0, 0, 0),
        b_coefficients=(0, 0.9827e-1, 0, -0.2709, 0),
        # Saxena and Fei (1987a), Eq. 23, C final coefficient = 0.1472e-1 (not 0.1427e-1)
        c_coefficients=(0, 0, -0.1030e-2, 0, 0.1472e-1),
        d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
    )

    # Table 1(a), 1000-5000 bar
    medium_pressure: RealGasABC = SaxenaEightCoefficients(
        critical_temperature=critical_temperature,
        critical_pressure=critical_pressure,
        a_coefficients=(1, 0, 0, 0, -5.917e-1, 0, 0, 0),
        b_coefficients=(0, 0, 9.122e-2, 0, 0, 0, 0, 0),
        # Saxena and Fei (1987a), Eq. 21, C first coefficient = 1.4164e-4 (not negative)
        c_coefficients=(0, 0, 0, 0, 1.4164e-4, 0, 0, -2.8349e-6),
        d_coefficients=(0, 0, 0, 0, 0, 0, 0, 0),
    )

    # Table 1(a), >5000 bar
    # Higher precision coefficients taken from Saxena and Fei (1987b), but agrees with Table 1(a)
    # in the 1992 paper.
    high_pressure: RealGasABC = SaxenaEightCoefficients(
        critical_temperature=critical_temperature,
        critical_pressure=critical_pressure,
        a_coefficients=(2.0614, 0, 0, 0, -2.2351, 0, 0, -3.9411e-1),
        b_coefficients=(0, 0, 5.5125e-2, 0, 3.9344e-2, 0, 0, 0),
        c_coefficients=(0, 0, -1.8935e-6, 0, -1.1092e-5, 0, -2.1892e-5, 0),
        d_coefficients=(0, 0, 5.0527e-11, 0, 0, -6.3033e-21, 0, 0),
    )

    models: tuple[RealGasABC, ...] = (low_pressure, medium_pressure, high_pressure)
    upper_pressure_bounds: tuple[float, ...] = (1000, 5000)

    combined_model: RealGasABC = CombinedEOSModel(
        models=models, upper_pressure_bounds=upper_pressure_bounds
    )

    return combined_model


# Corresponding states fugacity models from Shi and Saxena (1992)
CH4_SS92: RealGasABC = get_corresponding_states_SS92("CH4")
CO_SS92: RealGasABC = get_corresponding_states_SS92("CO")
CO2_SS92: RealGasABC = get_corresponding_states_SS92("CO2")
COS_SS92: RealGasABC = get_corresponding_states_SS92("COS")
O2_SS92: RealGasABC = get_corresponding_states_SS92("O2")
S2_SS92: RealGasABC = get_corresponding_states_SS92("S2")

# N2, H2, Ar are presented in Saxena and Fei for the high pressure fit only, but here we adopt the
# same low pressure extension as in Shi and Saxena (1992), Table 1(a).
Ar_SF87: RealGasABC = get_corresponding_states_SS92("Ar")
H2_SF87: RealGasABC = get_corresponding_states_SS92("H2")
N2_SF87: RealGasABC = get_corresponding_states_SS92("N2")


def get_saxena_eos_models() -> dict[str, RealGasABC]:
    """Gets a dictionary of the preferred EOS models to use for each species.

    The keys are the species and the values are class instances.

    Returns:
        Dictionary of preferred EOS models for each species
    """
    models: dict[str, RealGasABC] = {}
    models["Ar"] = Ar_SF87
    models["CH4"] = CH4_SS92
    models["CO"] = CO_SS92
    models["CO2"] = CO2_SS92
    models["COS"] = COS_SS92
    models["H2"] = H2_SS92
    models["H2S"] = H2S_SS92
    models["N2"] = N2_SF87
    models["O2"] = O2_SS92
    models["S2"] = S2_SS92
    models["SO2"] = SO2_SS92

    return models
