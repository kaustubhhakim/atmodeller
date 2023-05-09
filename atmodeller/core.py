"""Core classes and functions."""

import csv
import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import fsolve

from atmodeller.reaction import IvtanthermoCH4, JanafC, JanafH, MolarMasses
from atmodeller.solubility import BasaltDixonCO2, LibourelN2, PeridotiteH2O

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalParameters:
    """Global parameters."""

    gravitational_acceleration: float = 9.81  # m/s^2
    # pylint: disable=invalid-name
    is_CH4: bool = False  # Include CH4.
    mantle_mass: float = 4.208261222595111e24  # kg
    mantle_melt_fraction: float = 1.0  # Fraction of mantle that is molten.
    # Below is moles of H2 (or H2O) in one present-day Earth ocean.
    ocean_moles: float = 7.68894973907177e22
    planetary_radius: float = 6371000.0  # m
    # TODO: Make input parameter.
    temperature: float = 2000.0  # K
    molar_masses: MolarMasses = field(default_factory=MolarMasses)


def get_partial_pressures(
    solution_pressures_in: Iterable[float],
    fo2_shift: float,
    global_parameters: GlobalParameters,
) -> dict[str, float]:
    """Get a dictionary of all the species partial pressures.

    Args:
        solution_pressures_in: The solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_parameters: The global parameters.

    Returns:
        A dictionary of all the species and their partial pressures.
    """
    # We only need to know p_h2O, p_co2, and p_n2, since other (reduced) species can be directly
    # determined from equilibrium chemistry.
    p_h2o, p_co2, p_n2 = solution_pressures_in

    # Populate the solution species immediately from the input.
    p_d: dict[str, float] = {}
    p_d["H2O"] = p_h2o
    p_d["CO2"] = p_co2
    p_d["N2"] = p_n2

    # p_h2 from equilibrium chemistry.
    gamma: float = JanafH().modified_equilibrium_constant(
        temperature=global_parameters.temperature, fo2_shift=fo2_shift
    )
    p_d["H2"] = gamma * p_h2o

    # p_co from equilibrium chemistry.
    gamma = JanafC().modified_equilibrium_constant(
        temperature=global_parameters.temperature, fo2_shift=fo2_shift
    )
    p_d["CO"] = gamma * p_co2

    if global_parameters.is_CH4 is True:
        gamma = IvtanthermoCH4().modified_equilibrium_constant(
            temperature=global_parameters.temperature, fo2_shift=fo2_shift
        )
        p_d["CH4"] = gamma * p_co2 * p_d["H2"] ** 2.0
    else:
        p_d["CH4"] = 0

    return p_d


def get_total_pressure(
    solution_pressures_in: Iterable[float],
    fo2_shift: float,
    global_parameters: GlobalParameters,
) -> float:
    """Sum partial pressures of each species to get the total pressure.

    Args:
        solution_pressures_in: The solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global parameters.

    Returns:
        The total pressure.
    """
    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_parameters
    )
    pressure_total: float = sum(p_d.values())

    return pressure_total


def atmosphere_mass(
    solution_pressures_in: Iterable[float],
    fo2_shift: float,
    global_parameters: GlobalParameters,
) -> dict[str, float]:
    """Atmospheric mass of species and totals for H, C, and N.

    Args:
        solution_pressures_in: The solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_parameters: The global parameters.

    Returns:
        A dictionary of the masses of H, C, and N.
    """
    masses: MolarMasses = global_parameters.molar_masses
    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_parameters
    )
    mu_atm: float = atmosphere_mean_molar_mass(
        solution_pressures_in, fo2_shift, global_parameters
    )
    mass_atm_d: dict[str, float] = {}
    for key, value in p_d.items():
        # 1.0E5 because pressures are in bar.
        mass_atm_d[key] = value * 1.0e5 / global_parameters.gravitational_acceleration
        mass_atm_d[key] *= 4.0 * np.pi * global_parameters.planetary_radius**2.0
        mass_atm_d[key] *= getattr(masses, key) / mu_atm

    # Total mass of H.
    mass_atm_d["H"] = mass_atm_d["H2"] / masses.H2
    mass_atm_d["H"] += mass_atm_d["H2O"] / masses.H2O
    # Factor 2 below to account for stoichiometry.
    mass_atm_d["H"] += mass_atm_d["CH4"] * 2 / masses.CH4
    # Convert moles of H2 to mass of H.
    mass_atm_d["H"] *= masses.H2

    # Total mass of C.
    mass_atm_d["C"] = mass_atm_d["CO"] / masses.CO
    mass_atm_d["C"] += mass_atm_d["CO2"] / masses.CO2
    mass_atm_d["C"] += mass_atm_d["CH4"] / masses.CH4
    # Convert moles of C to mass of C.
    mass_atm_d["C"] *= masses.C

    # Total mass of N.
    mass_atm_d["N"] = mass_atm_d["N2"]

    return mass_atm_d


def atmosphere_mean_molar_mass(
    solution_pressures_in: Iterable[float],
    fo2_shift: float,
    global_parameters: GlobalParameters,
) -> float:
    """Mean molar mass of the atmosphere.

    Args:
        solution_pressures_in: The solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_parameters: The global parameters.

    Returns:
        Mean molar mass of the atmosphere.
    """

    masses: MolarMasses = global_parameters.molar_masses

    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_parameters
    )
    pressure_total: float = get_total_pressure(
        solution_pressures_in, fo2_shift, global_parameters
    )

    mu_atm: float = 0
    for key, value in p_d.items():
        mu_atm += getattr(masses, key) * value
    mu_atm /= pressure_total

    return mu_atm


def dissolved_mass(
    solution_pressures_in: Iterable[float],
    fo2_shift: float,
    global_parameters: GlobalParameters,
) -> dict[str, float]:
    """Volatile masses in the molten mantle.

    Args:
        solution_pressures_in: The solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_parameters: The global parameters.

    Returns:
        A dictionary of the species and their mass that is dissolved in the melt (molten mantle).
    """

    mass_int_d: dict[str, float] = {}

    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_parameters
    )

    prefactor: float = (
        1e-6 * global_parameters.mantle_mass * global_parameters.mantle_melt_fraction
    )
    masses: MolarMasses = global_parameters.molar_masses

    # H2O
    sol_h2o = PeridotiteH2O()  # Gets the default solubility model.
    ppmw_h2o = sol_h2o(p_d["H2O"], global_parameters.temperature)
    mass_int_d["H2O"] = prefactor * ppmw_h2o

    # CO2
    sol_co2 = BasaltDixonCO2()  # Gets the default solubility model.
    ppmw_co2 = sol_co2(p_d["CO2"], global_parameters.temperature)
    mass_int_d["CO2"] = prefactor * ppmw_co2

    # N2
    sol_n2 = LibourelN2()  # Gets the default solubility model.
    ppmw_n2 = sol_n2(p_d["N2"], global_parameters.temperature)
    mass_int_d["N2"] = prefactor * ppmw_n2

    # now get totals of H, C, N
    mass_int_d["H"] = mass_int_d["H2O"] * (masses.H2 / masses.H2O)
    mass_int_d["C"] = mass_int_d["CO2"] * (masses.C / masses.CO2)
    mass_int_d["N"] = mass_int_d["N2"]

    return mass_int_d


def mass_residual_objective_func(
    solution_pressures_in: Iterable[float],
    fo2_shift: float,
    global_parameters: GlobalParameters,
    mass_target_d: dict[str, float],
) -> list[float]:
    """Computes the residual of the volatile mass balance for H, C, and N.

    Args:
        solution_pressures_in: The solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_parameters: The global parameters.
        mass_target_d: A dictionary of the target masses for H, C, and N.

    Returns:
        A list of the mass residuals for H, C, and N.
    """
    mass_atm_d: dict[str, float] = atmosphere_mass(
        solution_pressures_in, fo2_shift, global_parameters
    )
    mass_int_d: dict[str, float] = dissolved_mass(
        solution_pressures_in, fo2_shift, global_parameters
    )

    # Compute residuals.
    all_residuals: list[float] = []
    for volatile in ["H", "C", "N"]:
        # Absolute residual.
        residual: float = (
            mass_atm_d[volatile] + mass_int_d[volatile] - mass_target_d[volatile]
        )
        # If target is not zero, compute relative residual.
        if mass_target_d[volatile]:
            residual /= mass_target_d[volatile]
        all_residuals.append(residual)

    return all_residuals


def get_initial_pressures(target_d) -> tuple[float, float, float]:
    """Initial guesses of partial pressures for H2O, CO2, and N2.

    Args:
        target_d: The target masses for H, C, and N.

    Returns:
        A tuple of the pressures in bar for H2O, CO2, and N2.
    """

    # All units are bar.
    # These are just a guess, mostly from the simple observation that H2O is less soluble than CO2.
    # If the target mass is zero, then the pressure must also be exactly zero.
    if target_d["H"] == 0:
        ph2o: float = 0
    else:
        ph2o = np.random.random_sample()
    if target_d["C"] == 0:
        pco2: float = 0
    else:
        pco2 = 10 * np.random.random_sample()
    if target_d["N"] == 0:
        pn2: float = 0
    else:
        pn2 = 10 * np.random.random_sample()

    return ph2o, pco2, pn2


def equilibrium_atmosphere(
    n_ocean_moles: float,
    ch_ratio: float,
    fo2_shift: float,
    nitrogen_ppmw: float,
    global_parameters: GlobalParameters,
) -> dict[str, float]:
    """Calculates the equilibrium chemistry of the atmosphere with mass balance.

    Args:
        n_ocean_moles: Number of Earth oceans.
        ch_ratio: C/H ratio by mass.
        fo2_shift: fO2 shift relative to the chosen buffer.
        nitrogen_ppmw: Target mass of nitrogen.
        global_parameters: The global parameters.

    Returns:
        A dictionary of the solution and input parameters.
    """

    masses: MolarMasses = global_parameters.molar_masses

    h_kg: float = n_ocean_moles * global_parameters.ocean_moles * masses.H2
    c_kg: float = ch_ratio * h_kg
    n_kg: float = nitrogen_ppmw * 1.0e-6 * global_parameters.mantle_mass
    target_d: dict[str, float] = {"H": h_kg, "C": c_kg, "N": n_kg}

    count: int = 0
    ier: int = 0
    initial_pressures: tuple[float, float, float] = (
        0,
        0,
        0,
    )  # Initialise only for the linter/typing.
    sol: np.ndarray = np.array([0, 0, 0])  # Initialise only for the linter/typing.
    # Below could in theory result in an infinite loop, if randomising the initial condition never
    # finds the physical solution, but in practice this doesn't seem to happen.
    while ier != 1:
        initial_pressures = get_initial_pressures(target_d)
        sol, _, ier, _ = fsolve(
            mass_residual_objective_func,
            initial_pressures,
            args=(fo2_shift, global_parameters, target_d),
            full_output=True,
        )
        count += 1
        # Sometimes, a solution exists with negative pressures, which is clearly non-physical.
        # Assert we must have positive pressures and restart the solve if needs be.
        if any(sol < 0):
            # If any negative pressures, report ier!=1 which means a solution has not been found.
            ier = 0

    logger.info("Randomised initial conditions = %d", count)

    p_d: dict[str, float] = get_partial_pressures(sol, fo2_shift, global_parameters)
    all_residuals: list[float] = mass_residual_objective_func(
        sol, fo2_shift, global_parameters, target_d
    )

    p_d["N_ocean_moles"] = n_ocean_moles
    p_d["CH_ratio"] = ch_ratio
    p_d["fo2_shift"] = fo2_shift
    p_d["pH2O_initial"] = initial_pressures[0]
    p_d["pCO2_initial"] = initial_pressures[1]
    p_d["pN2_initial"] = initial_pressures[2]
    p_d["H_mass_residual"] = all_residuals[0]
    p_d["C_mass_residual"] = all_residuals[1]
    p_d["N_mass_residual"] = all_residuals[2]

    return p_d


def equilibrium_atmosphere_monte_carlo(nitrogen_ppmw: float):
    """Monte Carlo simulation to produce realisations of atmospheric conditions.

    Args:
        nitrogen_ppmw: Nitrogen mass concentration in ppmw.
    """
    number: int = (
        100  # Number of realisations. Could be made an input parameter instead.
    )
    # Other parameters are normally distributed between bounds as given below.
    n_ocean_moles_a: np.ndarray = np.random.uniform(1, 10, number)
    ch_ratio_a: np.ndarray = np.random.uniform(0.1, 1, number)
    fo2_shift_a: np.ndarray = np.random.uniform(-4, 4, number)
    global_parameters: GlobalParameters = GlobalParameters()

    out: list[dict[str, float]] = []

    for realisation in range(number):
        logger.info("Realisation number = %d", realisation)
        n_ocean_moles: float = n_ocean_moles_a[realisation]
        ch_ratio: float = ch_ratio_a[realisation]
        fo2_shift: float = fo2_shift_a[realisation]
        p_d: dict[str, float] = equilibrium_atmosphere(
            n_ocean_moles, ch_ratio, fo2_shift, nitrogen_ppmw, global_parameters
        )
        out.append(p_d)

    filename: str = "equilibrium_atmosphere_monte_carlo.csv"
    logger.info("Writing output to: %s", filename)
    fieldnames: list[str] = list(out[0].keys())
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer: csv.DictWriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
