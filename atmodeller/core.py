"""Core classes and functions."""

import logging
import pprint
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Iterable, Optional

import numpy as np
from scipy import linalg
from scipy.optimize import fsolve

from atmodeller.reaction import (
    FormationEquilibriumConstants,
    IronWustiteBufferOneill,
    IvtanthermoCH4,
    JanafC,
    JanafH,
    MolarMasses,
    _OxygenFugacity,
)
from atmodeller.solubility import BasaltDixonCO2, LibourelN2, PeridotiteH2O, Solubility

logger: logging.Logger = logging.getLogger(__name__)

from atmodeller import (
    GAS_CONSTANT,
    GRAVITATIONAL_CONSTANT,
    OCEAN_MOLES,
    TEMPERATURE_JANAF_HIGH,
    TEMPERATURE_JANAF_LOW,
)


@dataclass(kw_only=True)
class InteriorAtmosphereSystemOld:
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
    planet_mass: float = field(init=False)
    surface_gravity: float = field(init=False)
    _solution: Iterable[float] = field(init=False)  # To store the solution.
    # Species pressures in the atmosphere.
    _pressures: dict[str, float] = field(init=False, default_factory=dict)
    # Species mass in the atmosphere and the interior.
    atmospheric_mass: dict[str, float] = field(init=False, default_factory=dict)
    interior_mass: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        logger.info("Creating a new interior-atmosphere system")
        self.planet_mass = self.mantle_mass / (1 - self.core_mass_fraction)
        self.surface_gravity = (
            GRAVITATIONAL_CONSTANT * self.planet_mass / self.planetary_radius**2
        )
        logger.info("    Mantle mass (kg) = %s", self.mantle_mass)
        logger.info("    Mantle melt fraction = %s", self.mantle_melt_fraction)
        logger.info("    Core mass fraction = %s", self.core_mass_fraction)
        logger.info("    Planetary radius (m) = %s", self.planetary_radius)
        logger.info("    Planetary mass (kg) = %s", self.planet_mass)
        logger.info("    Surface gravity (m/s^2) = %s", self.surface_gravity)

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
        logger.info("Solving the mass balance with the following parameters:")
        logger.info("    n_ocean_moles = %s", n_ocean_moles)
        logger.info("    C/H mass ratio = %s", ch_ratio)
        logger.info("    nitrogen ppmw = %s", nitrogen_ppmw)
        logger.info("    log10(fo2) shift = %s", self._fo2_shift)
        logger.info("    surface temperature = %s", self._surface_temperature)
        logger.info("    is_CH4 = %s", self._is_CH4)

        masses: MolarMasses = self.molar_masses
        h_kg: float = n_ocean_moles * OCEAN_MOLES * masses.H2
        c_kg: float = ch_ratio * h_kg
        n_kg: float = nitrogen_ppmw * 1.0e-6 * self.mantle_mass
        target_d: dict[str, float] = {"H": h_kg, "C": c_kg, "N": n_kg}
        logger.info("target_d = %s", target_d)

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

        logger.debug("Number of randomised initial conditions = %d", count)

        all_residuals: list[float] = self._mass_residual_objective_func(sol, target_d)
        output: dict[str, float] = self.pressures.copy()

        logger.info("Solution is:")
        for species, pressure in sorted(self.pressures.items()):
            logger.info("    %s pressure (bar) = %f", species, pressure)

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


@dataclass(kw_only=True)
class PlanetProperties:
    """The properties of a planet.

    Default values are the fully molten Earth.

    Args:
        mantle_mass: Mass of the planetary mantle. Defaults to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to all molten.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass. Defaults to
            Earth.
        surface_radius: Radius of the planetary surface. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.

    Attributes:
        mantle_mass
        mantle_melt_fraction
        core_mass_fraction
        surface_radius
        surface_temperature
        surface_gravity
        planet_mass
    """

    mantle_mass: float = 4.208261222595111e24  # kg, Earth's mantle mass
    mantle_melt_fraction: float = 1.0  # Completely molten
    core_mass_fraction: float = 0.295334691460966  # Earth's core mass fraction
    surface_radius: float = 6371000.0  # m, Earth's radius
    surface_temperature: float = 2000.0  # K
    planet_mass: float = field(init=False)
    surface_gravity: float = field(init=False)

    def __post_init__(self):
        logger.info("Creating a new planet")
        self.planet_mass = self.mantle_mass / (1 - self.core_mass_fraction)
        self.surface_gravity = (
            GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2
        )
        logger.info("Mantle mass (kg) = %s", self.mantle_mass)
        logger.info("Mantle melt fraction = %s", self.mantle_melt_fraction)
        logger.info("Core mass fraction = %s", self.core_mass_fraction)
        logger.info("Planetary radius (m) = %s", self.surface_radius)
        logger.info("Planetary mass (kg) = %s", self.planet_mass)
        logger.info("Surface temperature (K) = %f", self.surface_temperature)
        logger.info("Surface gravity (m/s^2) = %s", self.surface_gravity)


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the molecule or one of its elements."""

    @wraps(func)
    def mass_wrapper(
        self: "Molecule", element: Optional[str] = None, **kwargs
    ) -> float:
        """Wrapper for mass.

        Args:
            element: Returns the mass of this element. Defaults to None to return the molecule
                mass.
            **kwargs: Catches keyword arguments to forward to func.

        Returns:
            Mass of either the molecule or element.
        """
        mass: float = func(self, **kwargs)
        if element is not None:
            mass *= self.element_masses.get(element, 0) / self.molar_mass

        return mass

    return mass_wrapper


@dataclass
class Molecule:
    """Defines a molecule and its properties.

    Args:
        name: Chemical formula of the molecule.
        solubility: Solubility law.
        solid_melt_distribution_coefficient: Distribution coefficient. Defaults to 0.
        planet: Planet properties. Defaults to a fully molten Earth.

    Attributes:
        name: Chemical formula of the molecule.
        solubility: Solubility law.
        solid_melt_distribution_coefficient: Distribution coefficient.
        planet: Planet properties.
        elements: The elements and their counts in the molecule.
        element_masses: The elements and their masses in the molecule.
        formation_constants: The constants for computing the formation equilibrium constant.
        molar_mass: Molar mass of the molecule.
    """

    name: str
    solubility: Solubility
    solid_melt_distribution_coefficient: float = 0
    planet: PlanetProperties = field(default_factory=PlanetProperties)
    elements: dict[str, int] = field(init=False)
    element_masses: dict[str, float] = field(init=False)
    formation_constants: tuple[float, float] = field(init=False)
    molar_mass: float = field(init=False)

    def __post_init__(self):
        masses: MolarMasses = MolarMasses()
        self.elements = self._count_elements()
        self.element_masses = {
            key: value * getattr(masses, key) for key, value in self.elements.items()
        }
        self.molar_mass = getattr(masses, self.name)
        formation_constants: FormationEquilibriumConstants = (
            FormationEquilibriumConstants()
        )
        self.formation_constants = getattr(formation_constants, self.name)

    def _count_elements(self) -> dict[str, int]:
        """Count the number of atoms.

        Returns:
            A dictionary of the elements and their counts.
        """
        element_count: dict[str, int] = {}
        current_element: str = ""
        current_count: str = ""

        for char in self.name:
            if char.isupper():
                if current_element != "":
                    count = int(current_count) if current_count else 1
                    element_count[current_element] = (
                        element_count.get(current_element, 0) + count
                    )
                    current_count = ""
                current_element = char
            elif char.islower():
                current_element += char
            elif char.isdigit():
                current_count += char

        if current_element != "":
            count: int = int(current_count) if current_count else 1
            element_count[current_element] = (
                element_count.get(current_element, 0) + count
            )

        return element_count

    def get_formation_equilibrium_constant(self, *, temperature: float) -> float:
        """Gets the formation equilibrium constant (log Kf) in the JANAF tables.

        Args:
            temperature: Temperature.

        Returns:
            The formation equilibrium constant at the specified temperature.
        """
        return self.formation_constants[0] + self.formation_constants[1] / temperature

    @_mass_decorator
    def mass_in_atmosphere(
        self,
        *,
        partial_pressure_bar: float,
        atmosphere_mean_molar_mass: float,
        element: Optional[str] = None,
    ) -> float:
        """Mass in the atmosphere.

        Args:
            partial_pressure_bar: Partial pressure in bar.
            atmosphere_mean_molar_mass: Mean molar mass of the atmosphere.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Mass of the molecule (element=None) or element (element=element) in the atmosphere.
        """
        del element
        mass: float = partial_pressure_bar * 1e5 / self.planet.surface_gravity
        mass *= 4.0 * np.pi * self.planet.surface_radius**2
        mass *= self.molar_mass / atmosphere_mean_molar_mass

        return mass

    @_mass_decorator
    def mass_in_melt(
        self, *, partial_pressure_bar: float, element: Optional[str] = None
    ) -> float:
        """Mass in the molten interior.

        Args:
            partial_pressure_bar: Partial pressure in bar.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Mass of the molecule (element=None) or element (element=element) in the melt.
        """
        del element
        prefactor: float = (
            1e-6 * self.planet.mantle_mass * self.planet.mantle_melt_fraction
        )
        ppmw_in_melt: float = self.solubility(
            partial_pressure_bar, self.planet.surface_temperature
        )
        mass: float = prefactor * ppmw_in_melt

        return mass

    @_mass_decorator
    def mass_in_solid(
        self, *, partial_pressure_bar: float, element: Optional[str] = None
    ) -> float:
        """Mass in the solid interior.

        Args:
            partial_pressure_bar: Partial pressure in bar.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Mass of the molecule (element=None) or element (element=element) in the solid.
        """
        del element
        prefactor: float = (
            1e-6 * self.planet.mantle_mass * (1 - self.planet.mantle_melt_fraction)
        )
        ppmw_in_melt: float = self.solubility(
            partial_pressure_bar, self.planet.surface_temperature
        )
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass: float = prefactor * ppmw_in_solid

        return mass

    def mass(
        self,
        *,
        partial_pressure_bar: float,
        atmosphere_mean_molar_mass: float,
        element: Optional[str] = None,
    ) -> float:
        """Total mass.

        Args:
            partial_pressure_bar: Partial pressure in bar.
            atmosphere_mean_molar_mass: Mean molar mass of the atmosphere.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the molecule (element=None) or element (element=element).
        """
        mass_in_atmosphere: float = self.mass_in_atmosphere(
            partial_pressure_bar=partial_pressure_bar,
            atmosphere_mean_molar_mass=atmosphere_mean_molar_mass,
            element=element,
        )
        mass_in_melt: float = self.mass_in_melt(
            partial_pressure_bar=partial_pressure_bar, element=element
        )
        mass_in_solid: float = self.mass_in_solid(
            partial_pressure_bar=partial_pressure_bar, element=element
        )
        total_mass: float = mass_in_atmosphere + mass_in_melt + mass_in_solid

        return total_mass


@dataclass(kw_only=True)
class Constraint:
    """A constraint to apply to the system of equations."""

    species: str
    value: float
    field: str


class ReactionNetwork:
    """Determines the necessary (formation) reactions to solve a chemical network.

    Args:
        molecules: A list of molecules.
    """

    def __init__(self, molecules: list[Molecule]):
        self.molecules: list[Molecule] = molecules
        self.molecule_names: list[str] = [molecule.name for molecule in self.molecules]
        logger.info("Molecules = %s", self.molecule_names)
        self.number_molecules: int = len(molecules)
        self.elements, self.number_elements = self.find_elements()
        self.number_reactions: int = self.number_molecules - self.number_elements
        self.molecule_matrix: np.ndarray = self.find_matrix()
        self.reaction_matrix: np.ndarray = self.partial_gaussian_elimination()
        self.oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions))

    def find_elements(self) -> tuple[list, int]:
        """Determines the elements that compose the molecules.

        Returns:
            A tuple: (list of elements, number of elements).
        """
        elements: list[str] = []
        for molecule in self.molecules:
            elements.extend(list(molecule.elements.keys()))
        elements_unique: list[str] = list(set(elements))

        return elements_unique, len(elements_unique)

    def find_matrix(self) -> np.ndarray:
        """Creates a matrix where molecules (rows) are split into their element counts (columns).

        Returns:
            For example, self.molecules = ['CO2', 'H2O'] would return:
                [[0, 1, 2],
                 [2, 0, 1]]
            if the columns represent the elements H, C, and O, respectively.
        """
        matrix: np.ndarray = np.zeros((self.number_molecules, self.number_elements))
        for molecule_index, molecule in enumerate(self.molecules):
            for element_index, element in enumerate(self.elements):
                try:
                    count: int = molecule.elements[element]
                except KeyError:
                    count = 0
                matrix[molecule_index, element_index] = count
        return matrix

    def partial_gaussian_elimination(self) -> np.ndarray:
        """Performs a partial gaussian elimination to determine the required reactions.

        A copy of `self.molecule_matrix` is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of molecules - number of
        elements. These reactions are given in the last r rows of the reduced matrix.

        Returns:
            A matrix of the reaction stoichiometry.
        """
        matrix1: np.ndarray = self.molecule_matrix
        matrix2: np.ndarray = np.eye(self.number_molecules)
        augmented_matrix: np.ndarray = np.hstack((matrix1, matrix2))
        logger.debug("augmented_matrix = \n%s", augmented_matrix)

        # Forward elimination.
        for i in range(self.number_elements):  # Note only over the number of elements.
            # Check if the pivot element is zero.
            if augmented_matrix[i, i] == 0:
                # Swap rows to get a non-zero pivot element.
                nonzero_row: int = np.nonzero(augmented_matrix[i:, i])[0][0] + i
                augmented_matrix[[i, nonzero_row]] = augmented_matrix[[nonzero_row, i]]

            # Perform row operations to eliminate values below the pivot.
            for j in range(i + 1, self.number_molecules):
                ratio: float = augmented_matrix[j, i] / augmented_matrix[i, i]
                augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug(
            "augmented_matrix after forward elimination = \n%s", augmented_matrix
        )

        # Backward substitution.
        for i in range(self.number_elements - 1, -1, -1):
            # Normalize the pivot row.
            augmented_matrix[i] /= augmented_matrix[i, i]
            # Eliminate values above the pivot.
            for j in range(i - 1, -1, -1):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug(
            "augmented_matrix after backward substitution = \n%s", augmented_matrix
        )

        reduced_matrix1: np.ndarray = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix: np.ndarray = augmented_matrix[
            self.number_elements :, matrix1.shape[1] :
        ]
        logger.debug("reduced_matrix1 = \n%s", reduced_matrix1)
        logger.debug("reaction_matrix = \n%s", reaction_matrix)

        return reaction_matrix

    @property
    def reactions(self) -> dict[int, str]:
        """The reactions as a dictionary."""
        reactions: dict[int, str] = {}
        for reaction_index in range(self.number_reactions):
            reactants: str = ""
            products: str = ""
            for molecule_index, molecule in enumerate(self.molecules):
                coeff: float = self.reaction_matrix[reaction_index, molecule_index]
                if coeff != 0:
                    if coeff < 0:
                        reactants += f"{abs(coeff)} {molecule.name} + "
                    else:
                        products += f"{coeff} {molecule.name} + "

            reactants = reactants.rstrip(" + ")  # Removes the extra + at the end.
            products = products.rstrip(" + ")  # Removes the extra + at the end.
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction

        return reactions

    def get_reaction_log10_equilibrium_constant(
        self, *, reaction_index: int, temperature: float
    ) -> float:
        """Gets the log10 of the reaction equilibrium constant.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.

        Returns:
            log10 of the reaction equilibrium constant.
        """
        equilibrium_constant: float = 0
        for molecule_index, molecule in enumerate(self.molecules):
            equilibrium_constant += self.reaction_matrix[
                reaction_index, molecule_index
            ] * molecule.get_formation_equilibrium_constant(temperature=temperature)
        return equilibrium_constant

    def get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.

        Returns:
            The Gibb's free energy of the reaction.
        """
        gibbs: float = -self.get_reaction_log10_equilibrium_constant(
            reaction_index=reaction_index, temperature=temperature
        )
        gibbs *= np.log(10) * GAS_CONSTANT * temperature

        return gibbs

    def get_reaction_equilibrium_constant(
        self, *, reaction_index: int, temperature: float
    ) -> float:
        """Gets the equilibrium constant of a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.

        Returns:
            The equilibrium constant of the reaction.
        """
        equilibrium_constant: float = (
            10
            ** self.get_reaction_log10_equilibrium_constant(
                reaction_index=reaction_index, temperature=temperature
            )
        )
        return equilibrium_constant

    def get_coefficient_matrix_and_rhs(
        self,
        *,
        constraints: list[Constraint],
        temperature: float,
        oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill(),
        fo2_shift: float = 0,
        fo2_constraint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the coefficient matrix and the RHS.

        Args:
            constraints: Constraints for the system of equations.
            temperature: Temperature.
            oxygen_fugacity: Oxygen fugacity model. Defaults to IronWustiteBufferOneill. This is
                only used if `fo2_constraint` is True.
            fo2_shift: log10 fo2 shift from the buffer. Defaults to 0. This is only used if
                `fo2_constraint` is True.
            fo2_constraint: Include fo2 as a pressure constraint. Defaults to False.

        Returns:
            A dictionary of all the molecules and their partial pressures.
        """
        pressure_constraints: list[Constraint] = [
            constraint for constraint in constraints if constraint.field == "pressure"
        ]

        if fo2_constraint:
            logger.info(
                "Adding fO2 as an additional constraint using %s with fO2_shift = %0.2f",
                oxygen_fugacity.__class__.__name__,
                fo2_shift,
            )
            fo2: float = 10 ** oxygen_fugacity(
                temperature=temperature, fo2_shift=fo2_shift
            )
            constraint: Constraint = Constraint(
                species="O2", value=fo2, field="pressure"
            )
            pressure_constraints.append(constraint)

        number_pressure_constraints: int = len(pressure_constraints)
        nrows: int = number_pressure_constraints + self.number_reactions

        if nrows == self.number_molecules:
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.info(msg)
        else:
            num: int = self.number_molecules - nrows
            # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
            msg = "%d additional (not pressure) constraint(s) are necessary " % num
            msg += "to solve the system"
            logger.info(msg)

        # Build coefficient matrix and RHS vector.
        coeff: np.ndarray = np.zeros((nrows, self.number_molecules))
        rhs: np.ndarray = np.zeros(nrows)

        # Reactions.
        coeff[0 : self.number_reactions] = self.reaction_matrix.copy()
        for reaction_index in range(self.number_reactions):
            logger.info(
                "Row %02d: Reaction %d: %s",
                reaction_index,
                reaction_index,
                self.reactions[reaction_index],
            )
            # Gibb's reaction is log10 of the equilibrium constant.
            rhs[reaction_index] = self.get_reaction_log10_equilibrium_constant(
                reaction_index=reaction_index, temperature=temperature
            )

        for index, constraint in enumerate(pressure_constraints):
            row_index: int = self.number_reactions + index
            molecule_index: int = self.molecule_names.index(constraint.species)
            logger.info(
                "Row %02d: Setting %s partial pressure", row_index, constraint.species
            )
            coeff[row_index, molecule_index] = 1
            rhs[row_index] = np.log10(constraint.value)

        logger.debug("Coefficient matrix = \n%s", coeff)
        logger.debug("RHS vector = \n%s", rhs)

        return coeff, rhs

    def solve(self, **kwargs) -> np.ndarray:
        """Solves the reaction network to determine the partial pressures of all species.

        Applies the law of mass action.

        We solve for the log10 of the partial pressures of each species. Operating in log10 space
        has two advantages: 1) The dynamic range of the partial pressures is reduced, for example
        fO2 is typically very small compared to other pressures in the system, and 2) In log10
        space the reaction network can be expressed as a linear system which is trivial to solve.

        One could of course use a different log space (natural log), but log10 is chosen because
        the formation reactions are expressed in terms of log10 as well as the oxygen fugacity.

        Args:
            **kwargs: Keyword arguments to pass through.

        Returns:
            The log10 of the pressures.
        """
        logger.info("Solving the reaction network")

        coeff_matrix, rhs = self.get_coefficient_matrix_and_rhs(**kwargs)

        if len(rhs) != self.number_molecules:
            num: int = self.number_molecules - len(rhs)
            raise ValueError(f"Missing {num} constraint(s) to solve the system")

        log10_pressures: np.ndarray = linalg.solve(coeff_matrix, rhs)
        logger.info("The solution converged.")  # For similarity with fsolve.

        return log10_pressures


@dataclass(kw_only=True)
class InteriorAtmosphereSystem:
    """An interior-atmosphere system."""

    molecules: list[Molecule]
    planet: PlanetProperties = field(default_factory=PlanetProperties)
    molecule_names: list[str] = field(init=False)
    _log10_pressures: np.ndarray = field(init=False)  # Aligned with self.molecules.
    _reaction_network: ReactionNetwork = field(init=False)

    def __post_init__(self):
        logger.info("Creating a new interior-atmosphere system")
        self.molecules.sort(key=self.molecule_sorter)
        self.molecule_names: list[str] = [molecule.name for molecule in self.molecules]
        self._log10_pressures = np.zeros_like(self.molecules, dtype="float64")
        self._reaction_network = ReactionNetwork(molecules=self.molecules)

    def molecule_sorter(self, molecule: Molecule) -> tuple[int, str]:
        """Sorter for the molecules.

        Sorts first by molecule complexity and second by molecule name.

        Arg:
            molecule: Molecule.
        """
        return (sum(molecule.elements.values()), molecule.name)

    @property
    def pressures(self) -> np.ndarray:
        """Pressures."""
        return 10**self.log10_pressures

    @property
    def log10_pressures(self) -> np.ndarray:
        """Log10 pressures."""
        return self._log10_pressures

    @property
    def pressures_dict(self) -> dict[str, float]:
        """Pressures in a dictionary."""
        output: dict[str, float] = {
            molecule: pressure
            for (molecule, pressure) in zip(self.molecule_names, self.pressures)
        }
        return output

    @property
    def atmospheric_total_pressure(self) -> float:
        """Total atmospheric pressure."""
        return sum(self.pressures)

    @property
    def atmospheric_mean_molar_mass(self) -> float:
        """Mean molar mass of the atmosphere."""
        mu_atmosphere: float = 0
        for index, molecule in enumerate(self.molecules):
            mu_atmosphere += molecule.molar_mass * self.pressures[index]
        mu_atmosphere /= self.atmospheric_total_pressure

        return mu_atmosphere

    def solve(
        self,
        constraints: list[Constraint],
        *,
        temperature: float,
        oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill(),
        fo2_shift: float = 0,
        fo2_constraint: bool = False,
        use_fsolve: Optional[bool] = None,
    ) -> None:
        """Solves the system with provided constraints.

        Depending on the user-input, this can operate with only pressure constraints, only
        mass constraints, or a combination of both.

        Args:
            constraints: Constraints for the system of equations.
            Temperature: temperature,
            oxygen_fugacity: Oxygen fugacity model. Defaults to IronWustiteBufferOneill. This is
                only used if `fo2_constraint` is True.
            fo2_shift: log10 fo2 shift from the buffer. Defaults to 0. This is only used if
                `fo2_constraint` is True.
            fo2_constraint: Include fo2 as a pressure constraint. Defaults to False.
            use_fsolve: Use fsolve to solve the system of equations. Defaults to None, which means
                to auto select depending if the system is linear or not (which depends on the
                applied constraints).
        """

        if (temperature <= TEMPERATURE_JANAF_LOW) or (
            temperature >= TEMPERATURE_JANAF_HIGH
        ):
            msg: str = "Temperature must be in the range {TEMPERATURE_JANAF_LOW} K to "
            msg += f"{TEMPERATURE_JANAF_HIGH} K"
            raise ValueError(msg)

        logger.info("Constraints: %s", pprint.pformat(constraints))

        all_pressures: bool = all(
            [constraint.field == "pressure" for constraint in constraints]
        )

        if all_pressures and not use_fsolve:
            logger.info(
                "Pressure constraints only so attempting to solve a linear reaction network"
            )
            self._log10_pressures = self._reaction_network.solve(
                constraints=constraints,
                temperature=temperature,
                oxygen_fugacity=oxygen_fugacity,
                fo2_shift=fo2_shift,
                fo2_constraint=fo2_constraint,
            )
        else:
            if all_pressures and use_fsolve:
                msg = "Pressure constraints only and solving with fsolve"
            else:
                msg: str = (
                    "Mixed pressure and mass constraints so attempting to solve a "
                )
                msg += "non-linear system of equations"
            logger.info(msg)
            self._log10_pressures = self.solve_fsolve(
                constraints=constraints,
                temperature=temperature,
                oxygen_fugacity=oxygen_fugacity,
                fo2_shift=fo2_shift,
                fo2_constraint=fo2_constraint,
            )

        logger.info(pprint.pformat(self.pressures_dict))

    def solve_fsolve(self, **kwargs) -> np.ndarray:
        """Solves the non-linear system of equations.

        Args:
            **kwargs: Keyword argument. See `self.solve`.
        """
        coeff_matrix, rhs = self._reaction_network.get_coefficient_matrix_and_rhs(
            **kwargs
        )

        mass_constraints: list[Constraint] = [
            constraint
            for constraint in kwargs["constraints"]
            if constraint.field == "mass"
        ]
        for constraint in mass_constraints:
            logger.info("Adding constraint from mass balance: %s", constraint.species)

        initial_log10_pressures: np.ndarray = np.zeros_like(
            self.molecules, dtype="float64"
        )
        sol, infodict, _, mesg = fsolve(
            self.objective_func,
            initial_log10_pressures,
            args=(coeff_matrix, rhs, mass_constraints),
            full_output=True,
        )
        logger.info(mesg)
        logger.info("Number of function calls = %d", infodict["nfev"])
        logger.info("Final objective function evaluation = %s", infodict["fvec"])  # type: ignore

        return sol

    def objective_func(
        self,
        log10_pressures: np.ndarray,
        coeff_matrix: np.ndarray,
        rhs: np.ndarray,
        mass_constraints: list[Constraint],
    ) -> np.ndarray:
        """Objective function for the non-linear system."""

        self._log10_pressures = log10_pressures

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = coeff_matrix.dot(self.log10_pressures) - rhs
        logger.debug("residual_reaction = %s", residual_reaction)

        # Compute residual for the mass balance.
        residual_mass: np.ndarray = np.zeros_like(mass_constraints, dtype="float64")
        for constraint_index, constraint in enumerate(mass_constraints):
            for molecule_index, molecule in enumerate(self.molecules):
                residual_mass[constraint_index] += molecule.mass(
                    partial_pressure_bar=self.pressures[molecule_index],
                    atmosphere_mean_molar_mass=self.atmospheric_mean_molar_mass,
                    element=constraint.species,
                )
            residual_mass[constraint_index] -= constraint.value
            # Normalise by target mass to compute a relative residual.
            residual_mass[constraint_index] /= constraint.value
        logger.debug("residual_mass = %s", residual_mass)

        # Combined residual.
        residual: np.ndarray = np.concatenate((residual_reaction, residual_mass))
        logger.debug("residual = %s", residual)

        return residual
