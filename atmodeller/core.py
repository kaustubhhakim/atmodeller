"""Core classes and functions."""

import csv
import logging
from typing import Any

import numpy as np
from scipy.optimize import fsolve

logger: logging.Logger = logging.getLogger(__name__)


def get_global_parameters() -> dict:
    """Global parameters in SI units."""
    global_d: dict[str, Any] = {}
    global_d["mantle_mass"] = 4.208261222595111e24  # kg
    global_d["mantle_melt_fraction"] = 1.0  # Fraction of mantle that is molten.
    # Below is moles of H2 (or H2O) in one present-day Earth ocean.
    global_d["ocean_moles"] = 7.68894973907177e22
    global_d["little_g"] = 9.81  # m/s^2
    global_d["planetary_radius"] = 6371000.0  # m
    global_d["is_CH4"] = False  # Include CH4.
    global_d["temperature"] = 2000.0  # K
    global_d["molar_mass_d"] = get_molar_masses()
    return global_d


def get_molar_masses() -> dict:
    """Molar masses of atoms and molecules in kg/mol."""
    mass_d: dict[str, Any] = {}
    # Atoms are all given in g/mol, then converted before return at end of this function.
    mass_d["H"] = 1.0079
    mass_d["O"] = 15.9994
    mass_d["C"] = 12.0107
    mass_d["N"] = 14.0067
    # Molecules.
    mass_d["H2"] = mass_d["H"] * 2
    mass_d["H2O"] = mass_d["H2"] + mass_d["O"]
    mass_d["CO"] = mass_d["C"] + mass_d["O"]
    mass_d["CO2"] = mass_d["C"] + 2 * mass_d["O"]
    mass_d["CH4"] = mass_d["C"] + 4 * mass_d["H"]
    mass_d["N2"] = 2 * mass_d["N"]
    # Convert all to kg/mol.
    mass_d = {k: v * 1e-3 for k, v in mass_d.items()}
    return mass_d


class _OxygenFugacity:
    """log10 oxygen fugacity as a function of temperature.

    Args:
        model: Iron-wustite (IW) model to use. Either 'oneill' or 'fischer'.
    """

    def __init__(self, model: str = "oneill"):
        self._callmodel = getattr(self, model)

    def __call__(self, temperature: float, fo2_shift: float = 0) -> float:
        """log10 of fo2."""
        return self._callmodel(temperature) + fo2_shift

    def fischer(self, temperature: float) -> float:
        """Fischer et al. (2011) IW."""
        return 6.94059 - 28.1808 * 1e3 / temperature

    def oneill(self, temperature: float) -> float:
        """O'Neill and Eggin (2002) IW."""
        return (
            2
            * (
                -244118
                + 115.559 * temperature
                - 8.474 * temperature * np.log(temperature)
            )
            / (np.log(10) * 8.31441 * temperature)
        )


class ModifiedKeq:
    """Modified equilibrium constant, i.e. includes fo2.

    Args:
        keq_model: Equilibrium model to use. Options are give below __call__.
        fo2_model: fo2 model to use. See class _OxygenFugacity for options.
    """

    def __init__(self, keq_model: str, fo2_model: str = "oneill"):
        self.fo2: _OxygenFugacity = _OxygenFugacity(fo2_model)
        self._callmodel = getattr(self, keq_model)

    def __call__(self, temperature: float, fo2_shift: float = 0) -> float:
        fo2: float = self.fo2(temperature, fo2_shift)
        keq, fo2_stoich = self._callmodel(temperature)
        geq: float = 10 ** (keq - fo2_stoich * fo2)
        return geq

    # For the methods below, the second argument returns the stoichiometry of O2.

    def schaefer_ch4(self, temperature: float) -> tuple[float, float]:
        """Schaefer log10Keq for CO2 + 2H2 = CH4 + fo2."""
        return (-16276 / temperature - 5.4738, 1)

    def schaefer_c(self, temperature: float) -> tuple[float, float]:
        """Schaefer log10Keq for CO2 = CO + 0.5 fo2."""
        return (-14787 / temperature + 4.5472, 0.5)

    def schaefer_h(self, temperature: float) -> tuple[float, float]:
        """Schaefer log10Keq for H2O = H2 + 0.5 fo2."""
        return (-12794 / temperature + 2.7768, 0.5)

    def janaf_c(self, temperature: float) -> tuple[float, float]:
        """JANAF log10Keq, 1500 < K < 3000 for CO2 = CO + 0.5 fo2."""
        return (-14467.511400133637 / temperature + 4.348135473316284, 0.5)

    def janaf_h(self, temperature: float) -> tuple[float, float]:
        """JANAF log10Keq, 1500 < K < 3000 for H2O = H2 + 0.5 fo2."""
        return (-13152.477779978302 / temperature + 3.038586383273608, 0.5)


class Solubility:
    """Solubility base class. All pressures are in bar.

    Args:
        composition: Melt composition.
    """

    def __init__(self, composition: str):
        self._callmodel = getattr(self, composition)

    def power_law(self, pressure: float, constant: float, exponent: float) -> float:
        """Solubility power law."""
        return constant * pressure**exponent

    def __call__(self, pressure: float, *args) -> float:
        """Dissolved concentration in ppmw in the melt."""
        return self._callmodel(pressure, *args)


class SolubilityH2O(Solubility):
    """H2O solubility models.

    Args:
        composition: Melt composition.
    """

    def __init__(self, composition: str = "peridotite"):
        super().__init__(composition)

    def anorthite_diopside(self, pressure: float) -> float:
        """Newcombe et al. (2017)."""
        return self.power_law(pressure, 727, 0.5)

    def peridotite(self, pressure: float) -> float:
        """Sossi et al. (2022)."""
        return self.power_law(pressure, 534, 0.5)

    def basalt_dixon(self, pressure: float) -> float:
        """Dixon et al. (1995) refit by Paolo Sossi."""
        return self.power_law(pressure, 965, 0.5)

    def basalt_wilson(self, pressure: float) -> float:
        """Hamilton (1964) and Wilson and Head (1981)."""
        return self.power_law(pressure, 215, 0.7)

    def lunar_glass(self, pressure: float) -> float:
        """Newcombe et al. (2017)."""
        return self.power_law(pressure, 683, 0.5)


class SolubilityCO2(Solubility):
    """CO2 solubility models.

    Args:
        composition: Melt composition.
    """

    def __init__(self, composition: str = "basalt_dixon"):
        super().__init__(composition)

    def basalt_dixon(self, pressure: float, temperature: float) -> float:
        """Dixon et al. (1995)."""
        ppmw: float = (
            (3.8e-7) * pressure * np.exp(-23 * (pressure - 1) / (83.15 * temperature))
        )
        ppmw = 1.0e4 * (4400 * ppmw) / (36.6 - 44 * ppmw)
        return ppmw


class SolubilityN2(Solubility):
    """N2 solubility models.

    Args:
        composition: Melt composition.
    """

    def __init__(self, composition: str = "libourel"):
        super().__init__(composition)

    def libourel(self, pressure: float) -> float:
        """Libourel et al. (2003)."""
        ppmw: float = self.power_law(pressure, 0.0611, 1.0)
        return ppmw


def get_partial_pressures(
    solution_pressures_in: tuple[float, float, float],
    fo2_shift: float,
    global_d: dict[str, Any],
) -> dict[str, float]:
    """Get a dictionary of all the species partial pressures.

    Args:
        solution_pressures_in: A tuple of the solution pressures (e.g. H2O, CO2, and N2)
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global dictionary of parameters.

    Returns:
        A dictionary of all the species and their partial pressures.
    """
    # We only need to know p_h2O, p_co2, and p_n2, since reduced species can be directly determined
    # from equilibrium chemistry.
    p_h2o, p_co2, p_n2 = solution_pressures_in

    # Populate the oxidised species immediately from the input.
    p_d: dict[str, float] = {}
    p_d["H2O"] = p_h2o
    p_d["CO2"] = p_co2
    p_d["N2"] = p_n2

    # p_h2 from equilibrium chemistry.
    keq: ModifiedKeq = ModifiedKeq("janaf_h")
    gamma: float = keq(global_d["temperature"], fo2_shift)
    p_d["H2"] = gamma * p_h2o

    # p_co from equilibrium chemistry.
    keq = ModifiedKeq("janaf_c")
    gamma = keq(global_d["temperature"], fo2_shift)
    p_d["CO"] = gamma * p_co2

    if global_d["is_CH4"] is True:
        keq = ModifiedKeq("schaefer_ch4")
        gamma = keq(global_d["temperature"], fo2_shift)
        p_d["CH4"] = gamma * p_co2 * p_d["H2"] ** 2.0
    else:
        p_d["CH4"] = 0

    return p_d


def get_total_pressure(
    solution_pressures_in: tuple[float, float, float],
    fo2_shift: float,
    global_d: dict[str, Any],
) -> float:
    """Sum partial pressures of each species to get the total pressure.

    Args:
        solution_pressures_in: A tuple of the solution pressures (e.g. H2O, CO2, and N2)
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global dictionary of parameters.

    Returns:
        The total pressure.
    """
    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_d
    )
    pressure_total: float = sum(p_d.values())

    return pressure_total


def atmosphere_mass(
    solution_pressures_in: tuple[float, float, float],
    fo2_shift: float,
    global_d: dict[str, Any],
) -> dict[str, float]:
    """Atmospheric mass of species and totals for H, C, and N.

    Args:
        solution_pressures_in: A tuple of the solution pressures (e.g. H2O, CO2, and N2)
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global dictionary of parameters.

    Returns:
        A dictionary of the masses of H, C, and N.
    """
    mass_d: dict[str, float] = global_d["molar_mass_d"]
    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_d
    )
    mu_atm: float = atmosphere_mean_molar_mass(
        solution_pressures_in, fo2_shift, global_d
    )
    mass_atm_d: dict[str, float] = {}
    for key, value in p_d.items():
        # 1.0E5 because pressures are in bar.
        mass_atm_d[key] = value * 1.0e5 / global_d["little_g"]
        mass_atm_d[key] *= 4.0 * np.pi * global_d["planetary_radius"] ** 2.0
        mass_atm_d[key] *= mass_d[key] / mu_atm

    # Total mass of H.
    mass_atm_d["H"] = mass_atm_d["H2"] / mass_d["H2"]
    mass_atm_d["H"] += mass_atm_d["H2O"] / mass_d["H2O"]
    # Factor 2 below to account for stoichiometry.
    mass_atm_d["H"] += mass_atm_d["CH4"] * 2 / mass_d["CH4"]
    # Convert moles of H2 to mass of H.
    mass_atm_d["H"] *= mass_d["H2"]

    # Total mass of C.
    mass_atm_d["C"] = mass_atm_d["CO"] / mass_d["CO"]
    mass_atm_d["C"] += mass_atm_d["CO2"] / mass_d["CO2"]
    mass_atm_d["C"] += mass_atm_d["CH4"] / mass_d["CH4"]
    # Convert moles of C to mass of C.
    mass_atm_d["C"] *= mass_d["C"]

    # Total mass of N.
    mass_atm_d["N"] = mass_atm_d["N2"]

    return mass_atm_d


def atmosphere_mean_molar_mass(
    solution_pressures_in: tuple[float, float, float],
    fo2_shift: float,
    global_d: dict[str, Any],
) -> float:
    """Mean molar mass of the atmosphere.

    Args:
        solution_pressures_in: A tuple of the solution pressures (e.g. H2O, CO2, and N2)
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global dictionary of parameters.

    Returns:
        Mean molar mass of the atmosphere.
    """

    mass_d: dict[str, float] = global_d["molar_mass_d"]

    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_d
    )
    pressure_total: float = get_total_pressure(
        solution_pressures_in, fo2_shift, global_d
    )

    mu_atm: float = 0
    for key, value in p_d.items():
        mu_atm += mass_d[key] * value
    mu_atm /= pressure_total

    return mu_atm


def dissolved_mass(
    solution_pressures_in: tuple[float, float, float],
    fo2_shift: float,
    global_d: dict[str, Any],
):
    """Volatile masses in the molten mantle.

    Args:
        solution_pressures_in: A tuple of the solution pressures (e.g. H2O, CO2, and N2)
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global dictionary of parameters.

    Returns:
        A dictionary of the species and their mass that is dissolved in the melt (molten mantle).
    """

    mass_int_d: dict[str, float] = {}

    p_d: dict[str, float] = get_partial_pressures(
        solution_pressures_in, fo2_shift, global_d
    )

    prefactor: float = 1e-6 * global_d["mantle_mass"] * global_d["mantle_melt_fraction"]
    mass_d: dict[str, float] = global_d["molar_mass_d"]

    # H2O
    sol_h2o = SolubilityH2O()  # Gets the default solubility model.
    ppmw_h2o = sol_h2o(p_d["H2O"])
    mass_int_d["H2O"] = prefactor * ppmw_h2o

    # CO2
    sol_co2 = SolubilityCO2()  # Gets the default solubility model.
    ppmw_co2 = sol_co2(p_d["CO2"], global_d["temperature"])
    mass_int_d["CO2"] = prefactor * ppmw_co2

    # N2
    sol_n2 = SolubilityN2()  # Gets the default solubility model.
    ppmw_n2 = sol_n2(p_d["N2"])
    mass_int_d["N2"] = prefactor * ppmw_n2

    # now get totals of H, C, N
    mass_int_d["H"] = mass_int_d["H2O"] * (mass_d["H2"] / mass_d["H2O"])
    mass_int_d["C"] = mass_int_d["CO2"] * (mass_d["C"] / mass_d["CO2"])
    mass_int_d["N"] = mass_int_d["N2"]

    return mass_int_d


def mass_residual_objective_func(
    solution_pressures_in: tuple[float, float, float],
    fo2_shift: float,
    global_d: dict[str, Any],
    mass_target_d: dict[str, float],
) -> list[float]:
    """Computes the residual of the volatile mass balance for H, C, and N.

    Args:
        solution_pressures_in: A tuple of the solution pressures (e.g. H2O, CO2, and N2).
        fo2_shift: Shift relative to the fO2 buffer.
        global_d: The global dictionary of parameters.
        mass_target_d: A dictionary of the target masses for H, C, and N.

    Returns:
        A list of the mass residuals for H, C, and N.
    """
    mass_atm_d = atmosphere_mass(solution_pressures_in, fo2_shift, global_d)
    mass_int_d = dissolved_mass(solution_pressures_in, fo2_shift, global_d)

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


def get_initial_pressures(target_d):
    """Get initial guesses of partial pressures"""

    # all bar
    pH2O = 1 * np.random.random_sample()  # H2O less soluble than CO2
    pCO2 = 10 * np.random.random_sample()  # just a guess
    pN2 = 10 * np.random.random_sample()

    if target_d["H"] == 0:
        pH2O = 0
    if target_d["C"] == 0:
        pCO2 = 0
    if target_d["N"] == 0:
        pN2 = 0

    return pH2O, pCO2, pN2


def equilibrium_atmosphere(N_ocean_moles, CH_ratio, fo2_shift, global_d, Nitrogen):
    """Calculate equilibrium chemistry of the atmosphere"""

    mass_d = global_d["molar_mass_d"]

    H_kg = N_ocean_moles * global_d["ocean_moles"] * mass_d["H2"]
    C_kg = CH_ratio * H_kg
    N_kg = Nitrogen * 1.0e-6 * global_d["mantle_mass"]
    target_d = {"H": H_kg, "C": C_kg, "N": N_kg}

    count = 0
    ier = 0
    # could in principle result in an infinite loop, if randomising
    # the ic never finds the physical solution (but in practice,
    # this doesn't seem to happen)
    while ier != 1:
        x0 = get_initial_pressures(target_d)
        sol, info, ier, msg = fsolve(
            mass_residual_objective_func,
            x0,
            args=(fo2_shift, global_d, target_d),
            full_output=True,
        )
        count += 1
        # sometimes, a solution exists with negative pressures, which
        # is clearly non-physical.  Here, assert we must have positive
        # pressures.
        if any(sol < 0):
            # if any negative pressures, report ier!=1
            ier = 0

    logger.info(f"Randomised initial conditions= {count}")

    p_d = get_partial_pressures(sol, fo2_shift, global_d)
    # get residuals for output
    res_l = mass_residual_objective_func(sol, fo2_shift, global_d, target_d)

    # for convenience, add inputs to same dict
    p_d["N_ocean_moles"] = N_ocean_moles
    p_d["CH_ratio"] = CH_ratio
    p_d["fo2_shift"] = fo2_shift
    # for debugging/checking, add success initial condition
    # that resulted in a converged solution with positive pressures
    p_d["pH2O_0"] = x0[0]
    p_d["pCO2_0"] = x0[1]
    p_d["pN2_0"] = x0[2]
    # also for debugging/checking, report residuals
    p_d["res_H"] = res_l[0]
    p_d["res_C"] = res_l[1]
    p_d["res_N"] = res_l[2]

    return p_d


def equilibrium_atmosphere_MC(Nitrogen):
    """Monte Carlo"""

    NN = 10
    N_ocean_moles_l = np.random.uniform(1, 10, NN)
    CH_ratio_l = np.random.uniform(0.1, 1, NN)
    fo2_shift_l = np.random.uniform(-4, 4, NN)
    global_d = get_global_parameters()

    out_l = []

    for ii in range(NN):
        logging.info(f"Simulation number= {ii}")
        N_ocean_moles = N_ocean_moles_l[ii]
        CH_ratio = CH_ratio_l[ii]
        fo2_shift = fo2_shift_l[ii]
        p_d = equilibrium_atmosphere(
            N_ocean_moles, CH_ratio, fo2_shift, global_d, Nitrogen
        )
        out_l.append(p_d)

    filename = "equilibrium_atmosphere_MC.csv"
    write_output(filename, out_l)


def write_output(filename, out_l):
    """Write output (list of dictionaries) to a CSV"""

    fieldnames = list(out_l[0].keys())
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_l)
