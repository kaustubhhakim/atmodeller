"""Functions originally written by Meng to compute pure gas fugacities from Holland and Powell.

This module can be deleted once testing is complete.
"""

import logging

import numpy as np
from scipy.constants import kilo
from scipy.optimize import fsolve

from atmodeller import GAS_CONSTANT, debug_logger
from atmodeller.eos import CorkFullCO2

logger: logging.Logger = logging.getLogger(__name__)

GAS_CONSTANT_KJ: float = GAS_CONSTANT / kilo


# calculate pure gas fugacity coefficient and Modified Redlich-Kwang volume
def Calc_lambda(P, T, a, b, V_init):
    """
    Because HP98 dataset uses pressure in kbar, for easier implementation,
    pressure in this function is in kbar, temperature in K
    """

    EOS = (
        lambda V: P * V**3
        - GAS_CONSTANT_KJ * T * V**2
        - (b * GAS_CONSTANT_KJ * T + b**2 * P - a / np.sqrt(T)) * V
        - a * b / np.sqrt(T)
    )
    Jacob = (
        lambda V: 3 * P * V**2
        - 2 * GAS_CONSTANT_KJ * T * V
        - (b * GAS_CONSTANT_KJ * T + b**2 * P - a / np.sqrt(T))
    )
    V_mrk = fsolve(EOS, V_init, fprime=Jacob, xtol=1e-6)

    # compressibility factor
    Z = P * V_mrk / (GAS_CONSTANT_KJ * T)
    B = b * P / (GAS_CONSTANT_KJ * T)
    A = a / (b * GAS_CONSTANT_KJ * T**1.5)
    lnlambda = Z - 1.0 - np.log(Z - B) - A * np.log(1.0 + B / Z)
    print("Z = %f" % Z)
    print("A = %f" % A)
    print("B = %f" % B)
    print("lnlambda = %f" % lnlambda)

    return lnlambda, V_mrk


# calculate pure gas fugacity using the CORK equation from HP98
# return value is RTlnf in Joules, rather than f
def Calc_V_f(P, T, name):
    R = GAS_CONSTANT * 1e-3  # Note unit in kJ requires P in kbar
    if (name == "H2O") | (name == "CO2"):
        if name == "H2O":
            a0 = 1113.4
            a1 = -0.88517
            a2 = 4.53e-3
            a3 = -1.3183e-5
            a4 = -0.22291
            a5 = -3.8022e-4
            a6 = 1.7791e-7
            a7 = 5.8487
            a8 = -2.1370e-2
            a9 = 6.8133e-5
            b = 1.465
            c = 1.9853e-3
            d = -8.9090e-2
            e = 8.0331e-2
            Tc = 673.0
            P0 = 2.0
            Psat = -13.627e-3 + 7.29395e-7 * T**2 - 2.34622e-9 * T**3 + 4.83607e-15 * T**5

            if T < Tc:
                a = a0 + a1 * (Tc - T) + a2 * (Tc - T) ** 2 + a3 * (Tc - T) ** 3
                a_gas = a0 + a7 * (Tc - T) + a8 * (Tc - T) ** 2 + a9 * (Tc - T) ** 3

            else:
                a = a0 + a4 * (T - Tc) + a5 * (T - Tc) ** 2 + a6 * (T - Tc) ** 3

        else:
            a0 = 741.2
            a1 = -0.10891
            a2 = -3.903e-4
            Tc = 304.2
            P0 = 5.0
            a = a0 + a1 * T + a2 * T**2
            b = 3.057
            c = 5.40776e-3 - 1.59046e-6 * T
            d = -1.78198e-1 + 2.45317e-5 * T
            e = 0
            Psat = 0

        if T >= Tc:
            V_init = R * T / P + b
            lnlambda_mrk, V_mrk = Calc_lambda(P, T, a, b, V_init)

        else:
            if P <= Psat:
                V_init = R * T / P + 10.0 * b
                lnlambda_mrk, V_mrk = Calc_lambda(P, T, a_gas, b, V_init)  # type: ignore
                # print(lnlambda_mrk, V_mrk)

            else:
                # print("Psat = %f" % Psat)
                V_init = R * T / P + 10.0 * b
                lnlambda1, V_mrk = Calc_lambda(Psat, T, a_gas, b, V_init)  # type: ignore
                # print(lnlambda1, V_mrk)

                V_init = b / 2.0
                lnlambda2, V_mrk = Calc_lambda(Psat, T, a, b, V_init)
                # print(lnlambda2, V_mrk)

                V_init = R * T / P + b
                lnlambda3, V_mrk = Calc_lambda(P, T, a, b, V_init)
                # print(lnlambda3, V_mrk)

                lnlambda_mrk = lnlambda1 - lnlambda2 + lnlambda3

        # virial contribution to MRK (modified Redlich-Kwang)
        if P >= P0:
            lnlambda_vir = (
                1.0
                / (R * T)
                * (
                    c / 2.0 * (P - P0) ** 2
                    + 2.0 / 3.0 * d * (P - P0) ** (3.0 / 2.0)
                    + 4.0 / 5.0 * e * (P - P0) ** (5.0 / 4.0)
                )
            )
            V_vir = c * (P - P0) + d * (P - P0) ** 0.5 + e * (P - P0) ** 0.25

        else:
            lnlambda_vir = 0.0
            V_vir = 0.0

        lnlambda = lnlambda_mrk + lnlambda_vir
        V = V_mrk + V_vir  # type: ignore
        RTlnf = R * T * lnlambda + R * T * np.log(P / 1e-3)
        RTlnf = 1e3 * RTlnf  # convert to J

        return V, RTlnf

    else:
        if name == "CO":
            Tc = 132.9
            Pc = 0.0350

        elif name == "CH4":
            Tc = 190.6
            Pc = 0.0460

        elif name == "H2":
            Tc = 41.2
            Pc = 0.0211

        a0 = 5.45963e-5
        a1 = -8.63920e-6
        b0 = 9.18301e-4
        c0 = -3.30558e-5
        c1 = 2.30524e-6
        d0 = 6.93054e-7
        d1 = -8.38293e-8

        a = a0 * Tc ** (5.0 / 2.0) / Pc + a1 * Tc ** (3.0 / 2.0) / Pc * T  # type: ignore
        b = b0 * Tc / Pc  # type: ignore
        c = c0 * Tc / Pc ** (3.0 / 2.0) + c1 / Pc ** (3.0 / 2.0) * T  # type: ignore
        d = d0 * Tc / Pc**2 + d1 / Pc**2 * T  # type: ignore

        V = (
            R * T / P
            + b
            - a * R * np.sqrt(T) / (R * T + b * P) / (R * T + 2.0 * b * P)
            + c * np.sqrt(P)
            + d * P
        )
        RTlnf = (
            R * T * np.log(1000 * P)
            + b * P
            + a / b / np.sqrt(T) * (np.log(R * T + b * P) - np.log(R * T + 2.0 * b * P))
            + 2 / 3 * c * P * np.sqrt(P)
            + d / 2 * P**2
        )

        return V, 1e3 * RTlnf


def main():
    """For testing."""

    # 1 bar = 10^5 Pa
    # 1 kbar = 10^8 Pa
    # 10 kbar = 1 GPa

    # Comparison with Kite's H2 fugacity coefficient is not great. But around >30kbar the fugacity
    # coefficient for H2 maxes out and then decreases again.

    debug_logger()

    pressure: float = 20  # kbar
    temperature: float = 2000  # K

    print("\nMengs functions\n")
    print("temperature = %s, pressure = %s\n" % (temperature, pressure))
    # print("Corresponding states:\n")
    # V, RTlnf = Calc_V_f(pressure, temperature, "CO")
    # print("CO: V = %f, RTlnf = %f" % (V, RTlnf))
    # V, RTlnf = Calc_V_f(pressure, temperature, "CH4")
    # print("CH4: V = %f, RTlnf = %f" % (V, RTlnf))
    # V, RTlnf = Calc_V_f(pressure, temperature, "H2")
    # print("H2: V = %f, RTlnf = %f" % (V, RTlnf))
    # print("\n")
    print("Full CORK:\n")
    V, RTlnf = Calc_V_f(pressure, temperature, "CO2")
    print("CO2: V = %f, RTlnf = %f" % (V, RTlnf))
    fugacity: float = np.exp(RTlnf / (GAS_CONSTANT * temperature))
    fugacity_coeff: float = fugacity / 1000 / pressure
    print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    # V, RTlnf = Calc_V_f(pressure, temperature, "H2O")
    # print("H2O: V = %f, RTlnf = %f" % (V, RTlnf))
    # fugacity: float = np.exp(RTlnf / (GAS_CONSTANT * temperature))
    # fugacity_coeff: float = fugacity / 1000 / pressure
    # print("H2O: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))
    # print("\n")
    print("Dan Full CORK:\n")
    cork = CorkFullCO2()
    fugacity_coeff = cork.get_value(temperature=temperature, pressure=pressure * kilo)
    fugacity = cork.fugacity(temperature=temperature, pressure=pressure)
    volume = cork.volume(temperature=temperature, pressure=pressure)
    volume_integral = cork.volume_integral(temperature=temperature, pressure=pressure)
    print("CO2: V = %f, RTlnf = %f" % (volume, volume_integral))
    print("CO2: fugacity = %f, fugacity_coefficient = %f" % (fugacity, fugacity_coeff))


if __name__ == "__main__":
    main()
