"""Real gas EOSs from Holley et al. (1958)

See the LICENSE file for licensing information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from atmodeller import ATMOSPHERE, GAS_CONSTANT
from atmodeller.eos.interfaces import RealGasABC
from atmodeller.utilities import UnitConversion, debug_decorator

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class BeattieBridgeman(RealGasABC):
    """Beattie-Bridgeman model

    Args must have the same units as Holley et al. (1958) because they are converted within the
    class to SI and pressure in bar.

    Holley et al (1958) units are atmospheres and litres per mole.

    Args:
        A0: A0 constant
        a: a constant
        B0: B0 constant
        b: b constant
        c: c constant
    """

    A0: float
    a: float
    B0: float
    b: float
    c: float

    def __post_init__(self):
        """Scale the coefficients to SI and pressure in bar

        Based on dimensional analysis of Equation 1, we can see that the constants must have the
        following units:

            c-> V*T**3
            B0 -> V
            b -> V
            A0 -> P*V**2
            a -> V
        """
        print(self)
        self.c = UnitConversion.litre_to_m3(self.c)  # Temperature already in K
        self.B0 = UnitConversion.litre_to_m3(self.B0)
        self.b = UnitConversion.litre_to_m3(self.b)
        self.A0 /= ATMOSPHERE
        self.A0 *= UnitConversion.litre_to_m3() ** 2
        self.a = UnitConversion.litre_to_m3(self.a)
        print(self)

    @debug_decorator(logger)
    def volume_roots(self, temperature: float, pressure: float) -> np.ndarray:
        """Real and (potentially) physically meaningful volume solutions

        Equation 2

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume solutions of the MRK equation in m^3/mol
        """
        coefficients: list[float] = []
        coefficients.append(-GAS_CONSTANT * self.c * self.b * self.B0 / temperature**2)
        coefficients.append(
            GAS_CONSTANT * temperature * self.b * self.B0
            + GAS_CONSTANT * self.c * self.B0 / temperature**2
            - self.a * self.A0
        )
        coefficients.append(
            -GAS_CONSTANT * temperature * self.B0
            + GAS_CONSTANT * self.c / temperature**2
            + self.A0
        )
        coefficients.append(-GAS_CONSTANT * temperature)
        coefficients.append(pressure)

        polynomial: Polynomial = Polynomial(np.array(coefficients), symbol="V")
        logger.debug("Beattie-Bridgeman equation = %s", polynomial)
        volume_roots: np.ndarray = polynomial.roots()
        logger.debug("volume_roots = %s", volume_roots)
        # Numerical solution could result in a small imaginery component, even though the real
        # root is purely real.
        real_roots: np.ndarray = np.real(volume_roots[np.isclose(volume_roots.imag, 0)])
        # Physically meaningful volumes must be positive.
        positive_roots: np.ndarray = real_roots[real_roots > 0]
        # In general, several roots could be returned, and subclasses will need to determine which
        # is the correct volume to use.
        logger.debug("V = %s", positive_roots)

        return positive_roots

    @debug_decorator(logger)
    def volume(self, *args, **kwargs) -> float:
        """Volume

        Args:
            *args: Positional arguments to pass to self.volume_roots
            **kwargs: Keyword arguments to pass to self.volume_roots

        Returns:
            Volume in m^3/mol
        """
        volume_roots: np.ndarray = self.volume_roots(*args, **kwargs)

        # FIXME: Which root to return?  May require trial and error and comparison to the data.
        return np.max(volume_roots)

    @debug_decorator(logger)
    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral. Equation 8. Holley et al. (1958)

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure in bar

        Returns:
            Volume integral in J mol^(-1)
        """
        vol: float = self.volume(temperature, pressure)  # Volume evaluated at T and P conditions
        # This is equation 11. Update docstring.
        # volume_integral: float = (
        #     GAS_CONSTANT
        #     * temperature
        #     * (
        #         np.log(GAS_CONSTANT * temperature / vol)
        #         + (self.B0 - self.c / temperature**3 - self.A0 / (GAS_CONSTANT * temperature))
        #         * 2
        #         / vol
        #         - (
        #             self.b * self.B0
        #             + self.c * self.B0 / temperature**3
        #             - self.a * self.A0 / (GAS_CONSTANT * temperature)
        #         )
        #         * 3
        #         / (2 * vol**2)
        #         + (self.c * self.b * self.B0 / temperature**3) * 4 / (3 * vol**3)
        #     )
        # )
        volume_integral: float = (
            -GAS_CONSTANT * temperature * np.log(vol)
            + (
                GAS_CONSTANT * temperature * self.B0
                - GAS_CONSTANT * self.c / temperature**2
                - self.A0
            )
            * 2
            / vol
            - (
                GAS_CONSTANT * temperature * self.b * self.B0
                + GAS_CONSTANT * self.c * self.B0 / temperature**2
                - self.a * self.A0
            )
            * 3
            / (2 * vol**2)
            + (GAS_CONSTANT * self.c * self.b * self.B0 / temperature**2) * 4 / (3 * vol**3)
        )

        return volume_integral


H2_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(
    A0=0.1975, a=-0.00506, B0=0.02096, b=-0.04359, c=0.0504e4
)
N2_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(
    A0=1.3445, a=0.02617, B0=0.05046, b=-0.00691, c=4.2e4
)
O2_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(
    A0=1.4911, a=0.02562, B0=0.04624, b=0.004208, c=4.8e4
)
CO2_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(
    A0=5.0065, a=0.07132, B0=0.10476, b=0.07235, c=66e4
)
NH3_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(
    A0=2.3930, a=0.17031, B0=0.03415, b=0.19112, c=476.87e4
)
CH4_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(
    A0=2.2769, a=0.01855, B0=0.05587, b=-0.01587, c=12.83e4
)
He_BEATTIE_HWZ58: RealGasABC = BeattieBridgeman(A0=0.0216, a=0.05984, B0=0.01400, b=0, c=0.004e4)


def get_holley_eos_models() -> dict[str, RealGasABC]:
    """Gets a dictionary of the preferred EOS models to use for each species.

    Returns:
        Dictionary of prefered EOS models for each species
    """
    models: dict[str, RealGasABC] = {}
    models["CH4"] = CH4_BEATTIE_HWZ58
    models["CO2"] = CO2_BEATTIE_HWZ58
    models["H2"] = H2_BEATTIE_HWZ58
    models["He"] = He_BEATTIE_HWZ58
    models["N2"] = N2_BEATTIE_HWZ58
    models["NH3"] = NH3_BEATTIE_HWZ58
    models["O2"] = O2_BEATTIE_HWZ58

    return models
