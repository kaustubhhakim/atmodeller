"""Core classes and functions."""

import logging
import pprint
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Optional

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg  # type: ignore
from scipy.optimize import fsolve  # type: ignore

from atmodeller import (
    GAS_CONSTANT,
    GRAVITATIONAL_CONSTANT,
    TEMPERATURE_JANAF_HIGH,
    TEMPERATURE_JANAF_LOW,
)
from atmodeller.thermodynamics import (
    GibbsConstants,
    IronWustiteBufferOneill,
    MolarMasses,
    NoSolubility,
    OxygenFugacity,
    Solubility,
    master_container
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Planet:
    """The properties of a planet.

    Default values are for a reduced (at the Iron-Wustite buffer) and fully molten Earth.

    Args:
        mantle_mass: Mass of the planetary mantle. Defaults to Earth.
        mantle_melt_fraction: Mass fraction of the mantle that is molten. Defaults to 1.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass. Defaults to
            Earth.
        surface_radius: Radius of the planetary surface. Defaults to Earth.
        surface_temperature: Temperature of the planetary surface. Defaults to 2000 K.
        fo2_model: Oxygen fugacity model for the mantle. Defaults to
            IronWustiteBufferOneill,
        fo2_shift: log10 shift of the oxygen fugacity relative to the prescribed model.
        composition: melt composition of the planet. Default is Basalt

    Attributes:
        mantle_mass: Mass of the planetary mantle.
        mantle_melt_fraction: mass fraction of the mantle that is molten.
        core_mass_fraction: Mass fraction of the core relative to the planetary mass.
        surface_radius: Radius of the planetary surface.
        surface_temperature: Temperature of the planetary surface.
        fo2_model: Oxygen fugacity model for the mantle.
        fo2_shift: log10 shift of the oxygen fugacity relative to `oxygen_fugacity`.
        composition: melt composition of the planet. Default is Basalt 
        planet_mass: Mass of the planet.
        surface_gravity: Gravitational acceleration at the planetary surface.
    """

    mantle_mass: float = 4.208261222595111e24  # kg, Earth's mantle mass
    mantle_melt_fraction: float = 1.0  # Completely molten
    core_mass_fraction: float = 0.295334691460966  # Earth's core mass fraction
    surface_radius: float = 6371000.0  # m, Earth's radius
    surface_temperature: float = 2000.0  # K
    fo2_model: OxygenFugacity = field(default_factory=IronWustiteBufferOneill)
    fo2_shift: float = 0
    composition: str = 'Basalt' #TODO: Eventually we want to format this so that it accepts no composition and then uses NoSolubility, allowing the user to then assign solubility laws explicity for each molecule 
    planet_mass: float = field(init=False)
    surface_gravity: float = field(init=False)
    master_container: dict[str, Solubility] = field(init=False)

    def __post_init__(self):
        logger.info("Creating a new planet")
        self.planet_mass = self.mantle_mass / (1 - self.core_mass_fraction)
        self.surface_gravity = GRAVITATIONAL_CONSTANT * self.planet_mass / self.surface_radius**2
        self.master_container = master_container[self.composition.casefold()]
        logger.info("Mantle mass (kg) = %s", self.mantle_mass)
        logger.info("Mantle melt fraction = %s", self.mantle_melt_fraction)
        logger.info("Core mass fraction = %s", self.core_mass_fraction)
        logger.info("Planetary radius (m) = %s", self.surface_radius)
        logger.info("Planetary mass (kg) = %s", self.planet_mass)
        logger.info("Surface temperature (K) = %f", self.surface_temperature)
        logger.info("Surface gravity (m/s^2) = %s", self.surface_gravity)
        logger.info("Oxygen fugacity model (mantle) = %s", self.fo2_model.__class__.__name__)
        logger.info("Oxygen fugacity log10 shift = %f", self.fo2_shift)
        logger.info("Melt Composition = %s", self.composition)


def _mass_decorator(func) -> Callable:
    """A decorator to return the mass of either the molecule or one of its elements."""

    @wraps(func)
    def mass_wrapper(self: "Molecule", element: Optional[str] = None, **kwargs) -> float:
        """Wrapper to return the mass of either the molecule or one of its elements.

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


@dataclass(kw_only=True)
class Molecule:
    """A molecule and its properties.

    Args:
        name: Chemical formula of the molecule.
        solubility: Solubility model. Defaults to no solubility.
        solid_melt_distribution_coefficient: Distribution coefficient. Defaults to 0.

    Attributes:
        name: Chemical formula of the molecule.
        solubility: Solubility model.
        solid_melt_distribution_coefficient: Distribution coefficient between solid and melt.
        elements: The elements and their (stoichiometric) counts in the molecule.
        element_masses: The elements and their total masses in the molecule.
        formation_constants: The constants for computing the formation equilibrium constant.
        molar_mass: Molar mass of the molecule.
    """

    name: str
    solubility: Solubility = field(default_factory=NoSolubility)
    solid_melt_distribution_coefficient: float = 0
    elements: dict[str, int] = field(init=False)
    element_masses: dict[str, float] = field(init=False)
    formation_constants: tuple[float, float] = field(init=False)
    molar_mass: float = field(init=False)

    def __post_init__(self):
        logger.info("Creating a molecule: %s", self.name)
        masses: MolarMasses = MolarMasses()
        self.elements = self._count_elements()
        self.element_masses = {
            key: value * getattr(masses, key) for key, value in self.elements.items()
        }
        self.molar_mass = sum(self.element_masses.values())
        formation_constants: GibbsConstants = GibbsConstants()
        self.formation_constants = getattr(formation_constants, self.name)

    def _count_elements(self) -> dict[str, int]:
        """Counts the number of atoms.

        Returns:
            A dictionary of the elements and their (stoichiometric) counts.
        """
        element_count: dict[str, int] = {}
        current_element: str = ""
        current_count: str = ""

        for char in self.name:
            if char.isupper():
                if current_element != "":
                    count = int(current_count) if current_count else 1
                    element_count[current_element] = element_count.get(current_element, 0) + count
                    current_count = ""
                current_element = char
            elif char.islower():
                current_element += char
            elif char.isdigit():
                current_count += char

        if current_element != "":
            count: int = int(current_count) if current_count else 1
            element_count[current_element] = element_count.get(current_element, 0) + count
        logger.debug("element count = \n%s", element_count)
        return element_count

    def get_gibbs_constant(self, *, temperature: float) -> float:
        """Gets the standard Gibbs free energy of formation (Gf) from our fit to JANAF datatables.

        Args:
            temperature: Temperature.

        Returns:
            The formation equilibrium constant at the specified temperature.
        """
        return (self.formation_constants[0] * temperature) + self.formation_constants[1]

    @_mass_decorator
    def mass_in_atmosphere(
        self,
        *,
        planet: Planet,
        partial_pressure_bar: float,
        atmosphere_mean_molar_mass: float,
        element: Optional[str] = None,
    ) -> float:
        """Mass in the atmosphere.

        Args:
            planet: Planet properties.
            partial_pressure_bar: Partial pressure in bar.
            atmosphere_mean_molar_mass: Mean molar mass of the atmosphere.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Mass of the molecule (element=None) or element (element=element) in the atmosphere.
        """
        del element
        mass: float = partial_pressure_bar * 1e5 / planet.surface_gravity
        mass *= 4.0 * np.pi * planet.surface_radius**2
        mass *= self.molar_mass / atmosphere_mean_molar_mass

        return mass

    @_mass_decorator
    def mass_in_melt(
        self,
        *,
        planet: Planet,
        partial_pressure_bar: float,
        element: Optional[str] = None,
    ) -> float:
        """Mass in the molten interior.

        Args:
            planet: Planet properties.
            partial_pressure_bar: Partial pressure in bar.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Mass of the molecule (element=None) or element (element=element) in the melt.
        """
        del element
        prefactor: float = 1e-6 * planet.mantle_mass * planet.mantle_melt_fraction
        ppmw_in_melt: float = self.solubility(partial_pressure_bar, planet.surface_temperature, planet.fo2_model(temperature=planet.surface_temperature, fo2_shift=planet.fo2_shift))
        mass: float = prefactor * ppmw_in_melt

        return mass

    @_mass_decorator
    def mass_in_solid(
        self,
        *,
        planet: Planet,
        partial_pressure_bar: float,
        element: Optional[str] = None,
    ) -> float:
        """Mass in the solid interior.

        Args:
            planet: Planet properties.
            partial_pressure_bar: Partial pressure in bar.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Mass of the molecule (element=None) or element (element=element) in the solid.
        """
        del element
        prefactor: float = 1e-6 * planet.mantle_mass * (1 - planet.mantle_melt_fraction)
        ppmw_in_melt: float = self.solubility(partial_pressure_bar, planet.surface_temperature, planet.fo2_model(temperature=planet.surface_temperature, fo2_shift=planet.fo2_shift))
        ppmw_in_solid: float = ppmw_in_melt * self.solid_melt_distribution_coefficient
        mass: float = prefactor * ppmw_in_solid

        return mass

    def mass(
        self,
        *,
        planet: Planet,
        partial_pressure_bar: float,
        atmosphere_mean_molar_mass: float,
        element: Optional[str] = None,
    ) -> float:
        """Total mass.

        Args:
            planet: Planet properties.
            partial_pressure_bar: Partial pressure in bar.
            atmosphere_mean_molar_mass: Mean molar mass of the atmosphere.
            element: Returns the mass for an element. Defaults to None to return the molecule mass.
               This argument is used by the @_mass_decorator.

        Returns:
            Total mass of the molecule (element=None) or element (element=element).
        """
        mass_in_atmosphere: float = self.mass_in_atmosphere(
            planet=planet,
            partial_pressure_bar=partial_pressure_bar,
            atmosphere_mean_molar_mass=atmosphere_mean_molar_mass,
            element=element,
        )
        mass_in_melt: float = self.mass_in_melt(
            planet=planet, partial_pressure_bar=partial_pressure_bar, element=element
        )
        mass_in_solid: float = self.mass_in_solid(
            planet=planet, partial_pressure_bar=partial_pressure_bar, element=element
        )
        total_mass: float = mass_in_atmosphere + mass_in_melt + mass_in_solid

        return total_mass


@dataclass(kw_only=True)
class SystemConstraint:
    """A constraint to apply to an interior-atmosphere system.

    Args:
        species: The species to constrain. Usually a molecule for a pressure constraint or an
            element for a mass constraint.
        value: Imposed value in kg for masses and bar for pressures.
        field: Either 'pressure' or 'mass'.

    Attributes:
        See Args.
    """

    species: str
    value: float
    field: str


class ReactionNetwork:
    """Determines the necessary (often formation) reactions to solve a chemical network.

    Args:
        molecules: A list of molecules.

    Attributes:
        molecules: A list of molecules.
        molecule_names: The names of the molecules.
        number_molecules: The number of molecules.
        elements: The elements in the molecule and their counts.
        number_elements: The number of (unique) elements in the molecule.
        number_reactions: The number of reactions.
        molecule_matrix: The stoichiometry matrix of the molecules in terms of elements.
        reaction_matrix: The reaction stoichiometry matrix.
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
        logger.debug("number of unique elements = \n%s", elements_unique)
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
                logger.debug("ratio = \n%s", ratio)
                augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("augmented_matrix after forward elimination = \n%s", augmented_matrix)

        # Backward substitution.
        for i in range(self.number_elements - 1, -1, -1):
            # Normalize the pivot row.
            augmented_matrix[i] /= augmented_matrix[i, i]
            # Eliminate values above the pivot.
            for j in range(i - 1, -1, -1):
                if augmented_matrix[j, i] != 0:
                    ratio = augmented_matrix[j, i] / augmented_matrix[i, i]
                    augmented_matrix[j] -= ratio * augmented_matrix[i]
        logger.debug("augmented_matrix after backward substitution = \n%s", augmented_matrix)

        reduced_matrix1: np.ndarray = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix: np.ndarray = augmented_matrix[self.number_elements :, matrix1.shape[1] :]
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
        From the Gibbs free energy, we can calculate logKf:
        logKf = - G/(ln(10)*R*T)
        *Note: R needs to be in kJ/mol for units consistency with G and logKf

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.

        Returns:
            log10 of the reaction equilibrium constant.
        """
        j_to_kj: float = 0.001
        equilibrium_constant: float = 0
        for molecule_index, molecule in enumerate(self.molecules):
            equilibrium_constant += (
                self.reaction_matrix[reaction_index, molecule_index]
                * -molecule.get_gibbs_constant(temperature=temperature)
                / (np.log(10) * (GAS_CONSTANT * j_to_kj) * temperature)
            )
        return equilibrium_constant

    def get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction from our linear fit to the JANAF
        tables.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.

        Returns:
            The Gibb's free energy of the reaction.
        """
        gibbs_energy: float = 0
        for molecule_index, molecule in enumerate(self.molecules):
            gibbs_energy += self.reaction_matrix[
                reaction_index, molecule_index
            ] * molecule.get_gibbs_constant(temperature=temperature)
        return gibbs_energy

    def get_reaction_equilibrium_constant(
        self, *, reaction_index: int, temperature: float
    ) -> float:
        """Gets the equilibrium constant of a reaction Kf

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.

        Returns:
            The equilibrium constant of the reaction.
        """
        equilibrium_constant: float = 10 ** self.get_reaction_log10_equilibrium_constant(
            reaction_index=reaction_index, temperature=temperature
        )
        return equilibrium_constant

    def get_design_matrix_and_rhs(
        self,
        *,
        constraints: list[SystemConstraint],
        planet: Planet,
        fo2_constraint: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the design matrix and the right hand side (RHS) vector.

        Args:
            constraints: Constraints for the system of equations.
            planet: Planet properties.
            fo2_constraint: Include fo2 as a pressure constraint. Defaults to False.

        Returns:
            A dictionary of all the molecules and their partial pressures.
        """
        pressure_constraints: list[SystemConstraint] = [
            constraint for constraint in constraints if constraint.field == "pressure"
        ]

        if fo2_constraint:
            logger.info(
                "Adding fO2 as an additional constraint using %s with fO2_shift = %0.2f",
                planet.fo2_model.__class__.__name__,
                planet.fo2_shift,
            )
            fo2: float = 10 ** planet.fo2_model(
                temperature=planet.surface_temperature, fo2_shift=planet.fo2_shift
            )
            constraint: SystemConstraint = SystemConstraint(
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

        # Build design matrix and RHS vector.
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
                reaction_index=reaction_index, temperature=planet.surface_temperature
            )

        for index, constraint in enumerate(pressure_constraints):
            row_index: int = self.number_reactions + index
            molecule_index: int = self.molecule_names.index(constraint.species)
            logger.info("Row %02d: Setting %s partial pressure", row_index, constraint.species)
            coeff[row_index, molecule_index] = 1
            rhs[row_index] = np.log10(constraint.value)

        logger.debug("Design matrix = \n%s", coeff)
        logger.debug("RHS vector = \n%s", rhs)

        return coeff, rhs

    def solve(self, **kwargs) -> np.ndarray:
        """Solves the reaction network to determine the partial pressures of all species.

        Applies the law of mass action.

        We solve for the log10 of the partial pressures of each species. Operating in log10 space
        has two advantages: 1) The dynamic range of the partial pressures is reduced, for example
        fO2 is typically very small compared to other pressures in the system, and 2) In log10
        space the reaction network can be expressed as a linear system which can be solved
        directly.

        One could of course use a different log space (e.g., natural log), but log10 is chosen
        because the formation reactions are expressed in terms of log10 as well as the oxygen
        fugacity.

        Args:
            **kwargs: Keyword arguments to pass through.

        Returns:
            The log10 of the pressures.
        """
        logger.info("Solving the reaction network")

        design_matrix, rhs = self.get_design_matrix_and_rhs(**kwargs)

        if len(rhs) != self.number_molecules:
            num: int = self.number_molecules - len(rhs)
            raise ValueError(f"Missing {num} constraint(s) to solve the system")

        try:
            log10_pressures: np.ndarray = linalg.solve(design_matrix, rhs)
        except LinAlgError as exc:
            msg: str = "There is not a single solution to the equation set because you did not "
            msg += "specify a sufficient range of constraints"
            raise RuntimeError(msg) from exc

        logger.info("The solution converged.")  # For similarity with fsolve message.

        return log10_pressures


@dataclass(kw_only=True)
class InteriorAtmosphereSystem:
    """An interior-atmosphere system.

    Args:
        molecules: A list of molecules.

    Attributes:
        molecules: A list of molecules.
        planet: A planet. Defaults to a molten Earth.
        molecule_names: A list of the molecule names.
        number_molecules: The number of molecules.
        pressures: The molecule pressures (bar).
        log10_pressures: Log10 of the molecule pressures.
        atmospheric_total_pressure: Total atmospheric pressure.
        atmospheric_mean_molar_mass: Mean molar mass of the atmosphere.
        pressures_dict: The pressures of the molecules (bar) in a dictionary.
    """

    molecules: list[Molecule]
    planet: Planet = field(default_factory=Planet)
    molecule_names: list[str] = field(init=False)
    number_molecules: int = field(init=False)
    _log10_pressures: np.ndarray = field(init=False)  # Aligned with self.molecules.
    _reaction_network: ReactionNetwork = field(init=False)

    def __post_init__(self):
        logger.info("Creating a new interior-atmosphere system")
        self.molecules.sort(key=self._molecule_sorter)
        self.number_molecules: int = len(self.molecules)
        self.molecule_names: list[str] = [molecule.name for molecule in self.molecules]
        logger.info("Molecules = %s", self.molecule_names)
        self._solubility_check()
        self._log10_pressures = np.zeros_like(self.molecules, dtype="float64")
        self._reaction_network = ReactionNetwork(molecules=self.molecules)
    
    def _solubility_check(self)->None:
        #loop over species that exist in molecule_names to see if a solubility law exists
        for molecule in self.molecules:
            if molecule.name in self.planet.master_container:
                molecule.solubility = self.planet.master_container[molecule.name]
                logger.info("Found Solubility for \n%s", molecule.name)
                logger.info("Solubility Law is \n%s", molecule.solubility)

    def _molecule_sorter(self, molecule: Molecule) -> tuple[int, str]:
        """Sorter for the molecules.

        Sorts first by molecule complexity and second by molecule name.

        Arg:
            molecule: Molecule.

        Returns:
            A tuple to sort first by number of elements and second by molecule name.
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
            molecule: pressure for (molecule, pressure) in zip(self.molecule_names, self.pressures)
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
        constraints: list[SystemConstraint],
        *,
        fo2_constraint: bool = False,
        use_fsolve: Optional[bool] = None,
    ) -> dict[str, float]:
        """Solves the system to determine the partial pressures with provided constraints.

        Depending on the user-input, this can operate with only pressure constraints, only
        mass constraints, or a combination of both.

        Args:
            constraints: Constraints for the system of equations.
            fo2_constraint: Include fo2 as a pressure constraint. Defaults to False.
            use_fsolve: Use fsolve to solve the system of equations. Defaults to None, which means
                to auto select depending if the system is linear or not (which depends on the
                applied constraints).

        Returns:
            The pressures in bar.
        """
        # The formation energy data is only fit between a certain temperature range.
        if (self.planet.surface_temperature < TEMPERATURE_JANAF_LOW) or (
            self.planet.surface_temperature > TEMPERATURE_JANAF_HIGH
        ):
            msg: str = f"Surface temperature must be in the range {TEMPERATURE_JANAF_LOW} K to "
            msg += f"{TEMPERATURE_JANAF_HIGH} K"
            raise ValueError(msg)

        logger.info("Constraints: %s", pprint.pformat(constraints))

        # TODO: If constraints give zero pressure or zero mass, then remove the molecules or report
        # an error.

        all_pressures: bool = all([constraint.field == "pressure" for constraint in constraints])

        if all_pressures and not use_fsolve:
            logger.info(
                "Pressure constraints only so attempting to solve a linear reaction network"
            )
            self._log10_pressures = self._reaction_network.solve(
                constraints=constraints,
                planet=self.planet,
                fo2_constraint=fo2_constraint,
            )
        else:
            if all_pressures and use_fsolve:
                msg = "Pressure constraints only and solving with fsolve"
            else:
                msg: str = "Mixed pressure and mass constraints so attempting to solve a "
                msg += "non-linear system of equations"
            logger.info(msg)
            self._log10_pressures = self._solve_fsolve(
                constraints=constraints,
                planet=self.planet,
                fo2_constraint=fo2_constraint,
            )

        logger.info(pprint.pformat(self.pressures_dict))

        return self.pressures_dict

    def _solve_fsolve(self, **kwargs) -> np.ndarray:
        """Solves the non-linear system of equations.

        Args:
            **kwargs: Keyword argument. See `self.solve`.
        """
        design_matrix, rhs = self._reaction_network.get_design_matrix_and_rhs(**kwargs)

        mass_constraints: list[SystemConstraint] = [
            constraint for constraint in kwargs["constraints"] if constraint.field == "mass"
        ]

        if len(rhs) + len(mass_constraints) != self.number_molecules:
            num: int = self.number_molecules - (len(rhs) + len(mass_constraints))
            raise ValueError(f"Missing {num} constraint(s) to solve the system")

        for constraint in mass_constraints:
            logger.info("Adding constraint from mass balance: %s", constraint.species)

        initial_log10_pressures: np.ndarray = np.ones_like(self.molecules, dtype="float64")
        logger.debug("initial_log10_pressures = %s", initial_log10_pressures)
        ier: int = 0
        # Count the number of attempts to solve the system by randomising the initial condition.
        ic_count: int = 1
        # Maximum number of attempts to solve the system by randomising the initial condition.
        ic_count_max: int = 10
        sol: np.ndarray = np.zeros_like(initial_log10_pressures)
        infodict: dict = {}

        while ier != 1 and ic_count <= ic_count_max:
            sol, infodict, ier, mesg = fsolve(
                self._objective_func,
                initial_log10_pressures,
                args=(design_matrix, rhs, mass_constraints),
                full_output=True,
            )
            logger.info(mesg)
            if ier != 1:
                logger.info(
                    "Retrying with a new randomised initial condition (attempt %d)",
                    ic_count,
                )
                # Increase or decrease the magnitude of all pressures.
                initial_log10_pressures *= np.random.random_sample()
                logger.debug("initial_log10_pressures = %s", initial_log10_pressures)
                ic_count += 1

        if ic_count == ic_count_max:
            logger.error("Maximum number of randomised initial conditions has been exceeded")
            raise RuntimeError("Solution cannot be found (ic_count == ic_count_max)")

        logger.info("Number of function calls = %d", infodict["nfev"])
        logger.info("Final objective function evaluation = %s", infodict["fvec"])  # type: ignore

        return sol

    def _objective_func(
        self,
        log10_pressures: np.ndarray,
        design_matrix: np.ndarray,
        rhs: np.ndarray,
        mass_constraints: list[SystemConstraint],
    ) -> np.ndarray:
        """Objective function for the non-linear system.

        Args:
            log10_pressures: Log10 of the pressures of each molecule.
            design_matrix: The design matrix from the reaction network.
            rhs: The RHS from the reaction network.
            mass_constraints: Mass constraints to apply.

        Returns:
            The solution, which is the log10 of the pressures for each molecule.
        """
        self._log10_pressures = log10_pressures

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = design_matrix.dot(self.log10_pressures) - rhs
        logger.debug("residual_reaction = %s", residual_reaction)

        # Compute residual for the mass balance.
        residual_mass: np.ndarray = np.zeros_like(mass_constraints, dtype="float64")
        for constraint_index, constraint in enumerate(mass_constraints):
            for molecule_index, molecule in enumerate(self.molecules):
                residual_mass[constraint_index] += molecule.mass(
                    planet=self.planet,
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
