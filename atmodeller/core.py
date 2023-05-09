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

# Module constants.
GRAVITATIONAL_CONSTANT: float = 6.6743e-11  # SI units.
OCEAN_MOLES: float = (
    7.68894973907177e22  # Moles of H2 (or H2O) in one present-day Earth ocean.
)


@dataclass(kw_only=True)
class InteriorAtmosphereSystem:
    """An interior-atmosphere system.

    Args:
        mantle_mass: Mass of the planetary mantle. Defaults to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to all molten.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass. Defaults to
            Earth.
        planetary_radius: Radius of the planet. Defaults to Earth.
    """

    mantle_mass: float = 4.208261222595111e24  # kg, Earth's mantle mass
    mantle_melt_fraction: float = 1.0  # Completely molten
    core_mass_fraction: float = 0.295334691460966  # Earth's core mass fraction
    planetary_radius: float = 6371000.0  # m, Earth's radius
    _fo2_shift: float = field(init=False)  # fo2 shift in log0 units.
    _surface_temperature: float = field(init=False)  # K
    # pylint: disable=invalid-name
    _is_CH4: bool = field(init=False)
    molar_masses: MolarMasses = field(init=False, default_factory=MolarMasses)
    planetary_mass: float = field(init=False)
    surface_gravity: float = field(init=False)
    _solution: Iterable[float] = field(init=False)  # To store the solution.
    # Species pressures in the atmosphere.
    _pressures: dict[str, float] = field(init=False, default_factory=dict)
    # Species mass in the atmosphere and the interior.
    atmospheric_mass: dict[str, float] = field(init=False, default_factory=dict)
    interior_mass: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.planetary_mass = self.mantle_mass / (1 - self.core_mass_fraction)
        self.surface_gravity = (
            GRAVITATIONAL_CONSTANT * self.planetary_mass / self.planetary_radius**2
        )
        logger.info("Mantle mass (kg) = %s", self.mantle_mass)
        logger.info("Mantle melt fraction = %s", self.mantle_melt_fraction)
        logger.info("Core mass fraction = %s", self.core_mass_fraction)
        logger.info("Planetary radius (m) = %s", self.planetary_radius)
        logger.info("Planetary mass (kg) = %s", self.planetary_mass)
        logger.info("Surface gravity (m/s^2) = %s", self.surface_gravity)

    @property
    def pressures(self) -> dict[str, float]:
        """Returns pressures of all species."""
        return self._pressures

    def _set_partial_pressures(self):
        """Sets the partial pressures of all species."""
        # We only need to know p_h2O, p_co2, and p_n2, since other (reduced) species can be
        # directly determined from equilibrium chemistry.
        p_h2o, p_co2, p_n2 = self._solution

        self._pressures["H2O"] = p_h2o
        self._pressures["CO2"] = p_co2
        self._pressures["N2"] = p_n2

        # Get from equilibrium chemistry.
        h2_h2o_ratio: float = JanafH().modified_equilibrium_constant(
            temperature=self._surface_temperature, fo2_shift=self._fo2_shift
        )
        co_co2_ratio = JanafC().modified_equilibrium_constant(
            temperature=self._surface_temperature, fo2_shift=self._fo2_shift
        )
        self._pressures["H2"] = h2_h2o_ratio * p_h2o
        self._pressures["CO"] = co_co2_ratio * p_co2

        if self._is_CH4 is True:
            gamma = IvtanthermoCH4().modified_equilibrium_constant(
                temperature=self._surface_temperature, fo2_shift=self._fo2_shift
            )
            self._pressures["CH4"] = gamma * p_co2 * self._pressures["H2"] ** 2.0
        else:
            self._pressures["CH4"] = 0

    @property
    def _atmospheric_total_pressure(self) -> float:
        """Total atmospheric pressure."""
        return sum(self.pressures.values())

    @property
    def _atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere."""
        mu_atm: float = 0
        for species, partial_pressure in self.pressures.items():
            mu_atm += getattr(self.molar_masses, species) * partial_pressure
        mu_atm /= self._atmospheric_total_pressure
        return mu_atm

    def _set_species_mass_in_atmosphere(self):
        """Sets atmospheric mass of species and totals for H, C, and N."""
        masses: MolarMasses = self.molar_masses
        mass_atm: dict[str, float] = self.atmospheric_mass
        for species, partial_pressure in self.pressures.items():
            # 1.0e5 because pressures are in bar.
            mass_atm[species] = partial_pressure * 1.0e5 / self.surface_gravity
            mass_atm[species] *= 4.0 * np.pi * self.planetary_radius**2.0
            mass_atm[species] *= (
                getattr(masses, species) / self._atmospheric_mean_molar_mass
            )

        # Total mass of H.
        mass_atm["H"] = mass_atm["H2"] / masses.H2
        mass_atm["H"] += mass_atm["H2O"] / masses.H2O
        # Factor 2 below to account for stoichiometry.
        mass_atm["H"] += mass_atm["CH4"] * 2 / masses.CH4
        # Convert moles of H2 to mass of H.
        mass_atm["H"] *= masses.H2

        # Total mass of C.
        mass_atm["C"] = mass_atm["CO"] / masses.CO
        mass_atm["C"] += mass_atm["CO2"] / masses.CO2
        mass_atm["C"] += mass_atm["CH4"] / masses.CH4
        # Convert moles of C to mass of C.
        mass_atm["C"] *= masses.C

        # Total mass of N.
        mass_atm["N"] = mass_atm["N2"]

    def _set_species_mass_in_interior(self):
        """Sets interior mass of species and totals for H, C, and N."""

        masses: MolarMasses = self.molar_masses
        mass_int: dict[str, float] = self.interior_mass
        prefactor: float = 1e-6 * self.mantle_mass * self.mantle_melt_fraction

        # H2O
        sol_h2o = PeridotiteH2O()  # Gets the default solubility model.
        ppmw_h2o = sol_h2o(self.pressures["H2O"], self._surface_temperature)
        mass_int["H2O"] = prefactor * ppmw_h2o

        # CO2
        sol_co2 = BasaltDixonCO2()  # Gets the default solubility model.
        ppmw_co2 = sol_co2(self.pressures["CO2"], self._surface_temperature)
        mass_int["CO2"] = prefactor * ppmw_co2

        # N2
        sol_n2 = LibourelN2()  # Gets the default solubility model.
        ppmw_n2 = sol_n2(self.pressures["N2"], self._surface_temperature)
        mass_int["N2"] = prefactor * ppmw_n2

        # now get totals of H, C, N
        mass_int["H"] = mass_int["H2O"] * (masses.H2 / masses.H2O)
        mass_int["C"] = mass_int["CO2"] * (masses.C / masses.CO2)
        mass_int["N"] = mass_int["N2"]

    def _mass_residual_objective_func(
        self, solution: Iterable[float], mass_target_d: dict[str, float]
    ) -> list[float]:
        """Computes the residual of the volatile mass balance for H, C, and N.

        Args:
            mass_target_d: A dictionary of the target masses for H, C, and N.

        Returns:
            A list of the mass residuals for H, C, and N.
        """
        self._solution = solution
        self._set_partial_pressures()
        self._set_species_mass_in_atmosphere()
        self._set_species_mass_in_interior()
        # Compute residuals.
        all_residuals: list[float] = []
        for volatile in ["H", "C", "N"]:
            # Absolute residual.
            residual: float = (
                self.atmospheric_mass[volatile]
                + self.interior_mass[volatile]
                - mass_target_d[volatile]
            )
            # If target is not zero, compute relative residual.
            if mass_target_d[volatile]:
                residual /= mass_target_d[volatile]
            all_residuals.append(residual)

        return all_residuals

    def _get_initial_pressures(self, target_d) -> tuple[float, float, float]:
        """Initial guesses of partial pressures for H2O, CO2, and N2.

        Args:
            target_d: The target masses for H, C, and N.

        Returns:
            A tuple of the pressures in bar for H2O, CO2, and N2.
        """
        # All units are bar.
        # These are just a guess, mostly from the simple observation that H2O is less soluble than
        # CO2. If the target mass is zero, then the pressure must also be exactly zero.
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

    def solve(
        self,
        *,
        n_ocean_moles: float,
        ch_ratio: float,
        nitrogen_ppmw: float,
        fo2_shift: float = 0,
        temperature: float = 2000,
        is_CH4: bool = False,
    ) -> dict[str, float]:
        """Calculates the equilibrium chemistry of the atmosphere with mass balance.

        Args:
            n_ocean_moles: Number of Earth oceans.
            ch_ratio: C/H ratio by mass.
            fo2_shift: fO2 shift relative to the iron-wustite buffer.
            nitrogen_ppmw: Mantle concentration of nitrogen.
            fo2_shift: Log10 fo2 shift.
            temperature: Surface temperature.
            is_CH4: Include CH4.

        Returns:
            A dictionary of the solution and input parameters.
        """
        # Store on object so other methods can access these parameters.
        self._fo2_shift = fo2_shift
        self._surface_temperature = temperature
        self._is_CH4 = is_CH4

        masses: MolarMasses = self.molar_masses
        h_kg: float = n_ocean_moles * OCEAN_MOLES * masses.H2
        c_kg: float = ch_ratio * h_kg
        n_kg: float = nitrogen_ppmw * 1.0e-6 * self.mantle_mass
        target_d: dict[str, float] = {"H": h_kg, "C": c_kg, "N": n_kg}

        count: int = 0
        ier: int = 0
        initial_pressures: tuple[float, float, float] = (
            0,
            0,
            0,
        )  # Initialise only for the linter/typing.
        sol: np.ndarray = np.array([0, 0, 0])  # Initialise only for the linter/typing.
        # Below could in theory result in an infinite loop, if randomising the initial condition
        # never finds the physical solution, but in practice this doesn't seem to happen.
        while ier != 1:
            initial_pressures = self._get_initial_pressures(target_d)
            sol, _, ier, _ = fsolve(
                self._mass_residual_objective_func,
                initial_pressures,
                args=(target_d),
                full_output=True,
            )
            count += 1
            # Sometimes, a solution exists with negative pressures, which is clearly non-physical.
            # Assert we must have positive pressures and restart the solve if needs be.
            if any(sol < 0):
                # If any negative pressures, report ier!=1 which means a solution has not been
                # found.
                ier = 0

        logger.info("Number of randomised initial conditions = %d", count)

        all_residuals: list[float] = self._mass_residual_objective_func(sol, target_d)
        output: dict[str, float] = self.pressures.copy()

        output["n_ocean_moles"] = n_ocean_moles
        output["ch_ratio"] = ch_ratio
        output["fo2_shift"] = fo2_shift
        output["pH2O_initial"] = initial_pressures[0]
        output["pCO2_initial"] = initial_pressures[1]
        output["pN2_initial"] = initial_pressures[2]
        output["H_mass_residual"] = all_residuals[0]
        output["C_mass_residual"] = all_residuals[1]
        output["N_mass_residual"] = all_residuals[2]

        return output


# TODO: Convert to use new class
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
