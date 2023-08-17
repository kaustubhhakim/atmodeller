"""Core classes and functions."""

import logging
import pprint
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg
from scipy.optimize import fsolve

from atmodeller import GAS_CONSTANT
from atmodeller.thermodynamics import (
    BufferedFugacity,
    IronWustiteBufferHirschmann,
    Molecule,
    NoSolubility,
    Planet,
    Solubility,
    StandardGibbsFreeEnergyOfFormationJANAF,
    StandardGibbsFreeEnergyOfFormationProtocol,
    composition_solubilities,
)

logger: logging.Logger = logging.getLogger(__name__)


class SystemConstraint(Protocol):
    """A constraint to apply to an interior-atmosphere system."""

    @property
    def field(self) -> str:
        ...

    @property
    def species(self) -> str:
        ...

    def get_value(self, **kwargs) -> float:
        ...


@dataclass(kw_only=True)
class _ValueConstraint:
    """A value constraint to apply to an interior-atmosphere system.

    Args:
        species: The species to constrain. Usually a molecule for a pressure constraint or an
            element for a mass constraint.
        value: Imposed value in kg for masses and bar for pressures.
        field: Either 'fugacity' or 'mass'.

    Attributes:
        See Args.
    """

    species: str
    value: float
    field: str

    def get_value(self, **kwargs) -> float:
        del kwargs
        return self.value


@dataclass(kw_only=True)
class FugacityConstraint(_ValueConstraint):
    field: str = "fugacity"


@dataclass(kw_only=True)
class MassConstraint(_ValueConstraint):
    field: str = "mass"


@dataclass(kw_only=True)
class BufferedFugacityConstraint:
    """A buffered fugacity constraint to apply to an interior-atmosphere system.

    Args:
        species: The species (molecule) that is buffered by `buffer`. Defaults to 'O2'.
        fugacity: A BufferedFugacity. Defaults to IronWustiteBufferHirschmann
        log10_shift: Log10 shift relative to the buffer.

    Attributes:
        See Args.
    """

    species: str = "O2"
    fugacity: BufferedFugacity = field(default_factory=IronWustiteBufferHirschmann)
    log10_shift: float = 0
    field: str = field(default="fugacity", init=False)

    def get_value(self, *, temperature: float, **kwargs) -> float:
        del kwargs
        return 10 ** self.fugacity(temperature=temperature, fugacity_log10_shift=self.log10_shift)


@dataclass(kw_only=True)
class ReactionNetwork:
    """Determines the necessary (often formation) reactions to solve a chemical network.

    Args:
        molecules: A list of molecules.
        gibbs_data: Standard Gibbs free energy of formation.

    Attributes:
        molecules: A list of molecules.
        gibbs_data: Standard Gibbs free energy of formation.
        molecule_names: The names of the molecules.
        number_molecules: The number of molecules.
        elements: The elements in the molecule and their counts.
        number_elements: The number of (unique) elements in the molecule.
        number_reactions: The number of reactions.
        molecule_matrix: The stoichiometry matrix of the molecules in terms of elements.
        reaction_matrix: The reaction stoichiometry matrix.
    """

    molecules: list[Molecule]
    gibbs_data: StandardGibbsFreeEnergyOfFormationProtocol

    def __post_init__(self):
        self.molecule_names: list[str] = [molecule.name for molecule in self.molecules]
        logger.info("Molecules = %s", self.molecule_names)
        self.number_molecules: int = len(self.molecules)
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
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the log10 of the reaction equilibrium constant.

        From the Gibbs free energy, we can calculate logKf as:
        logKf = - G/(ln(10)*R*T)

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            log10 of the reaction equilibrium constant.
        """
        gibbs_energy: float = self.get_reaction_gibbs_energy_of_formation(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        equilibrium_constant: float = -gibbs_energy / (np.log(10) * GAS_CONSTANT * temperature)

        return equilibrium_constant

    def get_reaction_gibbs_energy_of_formation(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the Gibb's free energy of formation for a reaction.

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            The Gibb's free energy of the reaction.
        """
        gibbs_energy: float = 0
        for molecule_index, molecule in enumerate(self.molecules):
            gibbs_energy += self.reaction_matrix[
                reaction_index, molecule_index
            ] * self.gibbs_data.get(molecule, temperature=temperature, pressure=pressure)
        return gibbs_energy

    def get_reaction_equilibrium_constant(
        self, *, reaction_index: int, temperature: float, pressure: float
    ) -> float:
        """Gets the equilibrium constant of a reaction Kf

        Args:
            reaction_index: Row index of the reaction as it appears in `self.reaction_matrix`.
            temperature: Temperature.
            pressure: Pressure.

        Returns:
            The equilibrium constant of the reaction.
        """
        equilibrium_constant: float = 10 ** self.get_reaction_log10_equilibrium_constant(
            reaction_index=reaction_index, temperature=temperature, pressure=pressure
        )
        return equilibrium_constant

    def get_design_matrix_and_rhs(
        self,
        *,
        constraints: list[SystemConstraint],
        planet: Planet,
        fugacities_dict: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the design matrix and the right hand side (RHS) vector.

        Args:
            constraints: Constraints for the system of equations.
            planet: Planet properties.
            fugacities_dict: The pressures of the molecules (bar) in a dictionary.

        Returns:
            A dictionary of all the molecules and their partial pressures.
        """

        # TODO: Formally fugacities_dict should actually be the partial pressure dict.  In which
        # case fugacities for the equivalent partial pressures should be computed here.

        pressure: float = sum(fugacities_dict.values())

        pressure_constraints: list[SystemConstraint] = [
            constraint for constraint in constraints if constraint.field == "fugacity"
        ]

        number_pressure_constraints: int = len(pressure_constraints)
        nrows: int = number_pressure_constraints + self.number_reactions

        if nrows == self.number_molecules:
            msg: str = "The necessary number of constraints will be applied to "
            msg += "the reaction network to solve the system"
            logger.info(msg)
        else:
            num: int = self.number_molecules - nrows
            # Logger convention is to avoid f-string. pylint: disable=consider-using-f-string
            msg = "%d additional (not fugacity) constraint(s) are necessary " % num
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
                reaction_index=reaction_index,
                temperature=planet.surface_temperature,
                pressure=pressure,
            )

        for index, constraint in enumerate(pressure_constraints):
            row_index: int = self.number_reactions + index
            molecule_index: int = self.molecule_names.index(constraint.species)
            logger.info("Row %02d: Setting %s partial pressure", row_index, constraint.species)
            coeff[row_index, molecule_index] = 1
            rhs[row_index] = np.log10(constraint.get_value(temperature=planet.surface_temperature))

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
        gibbs_data: Standard Gibbs free energy of formation. Defaults to a linear fit to JANAF.

    Attributes:
        molecules: A list of molecules.
        gibbs_data: Standard Gibbs free energy of formation. Defaults to a linear fit to JANAF.
        planet: A planet. Defaults to a molten Earth.
        molecule_names: A list of the molecule names.
        number_molecules: The number of molecules.
        pressures: The molecule pressures (bar).
        log10_pressures: Log10 of the molecule pressures.
        atmospheric_total_pressure: Total atmospheric pressure.
        atmospheric_mean_molar_mass: Mean molar mass of the atmosphere.
        fugacities_dict: The pressures of the molecules (bar) in a dictionary.
    """

    molecules: list[Molecule]
    gibbs_data: StandardGibbsFreeEnergyOfFormationProtocol = field(
        default_factory=StandardGibbsFreeEnergyOfFormationJANAF
    )
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
        self._conform_solubilities_to_composition()
        self._log10_pressures = np.zeros_like(self.molecules, dtype="float64")
        self._reaction_network = ReactionNetwork(
            molecules=self.molecules, gibbs_data=self.gibbs_data
        )

    def _conform_solubilities_to_composition(self) -> None:
        """Ensure that the solubilities of the species are consistent with the melt composition."""
        if self.planet.melt_composition is not None:
            msg: str = (
                # pylint: disable=consider-using-f-string
                "Setting solubilities to be consistent with the melt composition (%s)"
                % self.planet.melt_composition
            )
            logger.info(msg)
            try:
                solubilities: dict[str, Solubility] = composition_solubilities[
                    self.planet.melt_composition.casefold()
                ]
            except KeyError:
                logger.error("Cannot find solubilities for %s", self.planet.melt_composition)
                raise

            for molecule in self.molecules:
                try:
                    molecule.solubility = solubilities[molecule.name]
                    logger.info(
                        "Found Solubility for %s: %s",
                        molecule.name,
                        molecule.solubility.__class__.__name__,
                    )
                except KeyError:
                    logger.info("No solubility for %s", molecule.name)
                    molecule.solubility = NoSolubility()

    def _molecule_sorter(self, molecule: Molecule) -> tuple[int, str]:
        """Sorter for the molecules.

        Sorts first by molecule complexity and second by molecule name.

        Args:
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
    def fugacities_dict(self) -> dict[str, float]:
        """Fugacities of all species in a dictionary."""
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

    @property
    def output(self) -> dict:
        """Convenient output for analysis."""
        output_dict: dict = {}
        output_dict["total_pressure_in_atmosphere"] = self.atmospheric_total_pressure
        output_dict["mean_molar_mass_in_atmosphere"] = self.atmospheric_mean_molar_mass
        for molecule in self.molecules:
            output_dict[molecule.name] = molecule.output
        # TODO: Dan to add elemental outputs as well.
        return output_dict

    def solve(
        self,
        constraints: list[SystemConstraint],
        *,
        use_fsolve: Optional[bool] = None,
    ) -> dict[str, float]:
        """Solves the system to determine the partial pressures with provided constraints.

        Depending on the user-input, this can operate with only pressure constraints, only
        mass constraints, or a combination of both.

        Args:
            constraints: Constraints for the system of equations.
            use_fsolve: Use fsolve to solve the system of equations. Defaults to None, which means
                to auto select depending if the system is linear or not (which depends on the
                applied constraints).

        Returns:
            The pressures in bar.
        """

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
            )
        else:
            if all_pressures and use_fsolve:
                msg = "Pressure constraints only and solving with fsolve"
            else:
                msg: str = "Mixed pressure and mass constraints so attempting to solve a "
                msg += "non-linear system of equations"
            logger.info(msg)
            self._log10_pressures = self._solve_fsolve(constraints=constraints)

        # Recompute quantities that depend on the solution, since molecule.mass is not called for
        # the linear reaction network.
        for molecule_index, molecule in enumerate(self.molecules):
            molecule.mass(
                planet=self.planet,
                partial_pressure_bar=self.pressures[molecule_index],
                atmosphere_mean_molar_mass=self.atmospheric_mean_molar_mass,
                fugacities_dict=self.fugacities_dict,
            )

        logger.info(pprint.pformat(self.fugacities_dict))

        return self.fugacities_dict

    def _solve_fsolve(self, constraints: list[SystemConstraint]) -> np.ndarray:
        """Solves the non-linear system of equations.

        Args:
            constraints: Constraints for the system of equations.
        """

        # TODO: Now moved into iteration. To eventually remove.
        # design_matrix, rhs = self._reaction_network.get_design_matrix_and_rhs(**kwargs)
        # mass_constraints: list[SystemConstraint] = [
        #    constraint for constraint in kwargs["constraints"] if constraint.field == "mass"
        # ]
        # if len(rhs) + len(mass_constraints) != self.number_molecules:
        #    num: int = self.number_molecules - (len(rhs) + len(mass_constraints))
        #    raise ValueError(f"Missing {num} constraint(s) to solve the system")
        # for constraint in mass_constraints:
        #    logger.info("Adding constraint from mass balance: %s", constraint.species)

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
                args=(constraints),
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
        constraints: list[SystemConstraint],
    ) -> np.ndarray:
        """Objective function for the non-linear system.

        Args:
            log10_pressures: Log10 of the pressures of each molecule.
            constraints: Constraints for the system of equations.

        Returns:
            The solution, which is the log10 of the pressures for each molecule.
        """
        self._log10_pressures = log10_pressures

        design_matrix, rhs = self._reaction_network.get_design_matrix_and_rhs(
            constraints=constraints, planet=self.planet, fugacities_dict=self.fugacities_dict
        )

        # Compute residual for the reaction network.
        residual_reaction: np.ndarray = design_matrix.dot(self.log10_pressures) - rhs
        logger.debug("residual_reaction = %s", residual_reaction)

        mass_constraints: list[SystemConstraint] = [
            constraint for constraint in constraints if constraint.field == "mass"
        ]

        # Compute residual for the mass balance.
        residual_mass: np.ndarray = np.zeros_like(mass_constraints, dtype="float64")
        for constraint_index, constraint in enumerate(mass_constraints):
            for molecule_index, molecule in enumerate(self.molecules):
                residual_mass[constraint_index] += molecule.mass(
                    planet=self.planet,
                    partial_pressure_bar=self.pressures[molecule_index],
                    atmosphere_mean_molar_mass=self.atmospheric_mean_molar_mass,
                    element=constraint.species,
                    fugacities_dict=self.fugacities_dict,
                )
            # Mass values are constant so no need to pass any arguments to get_value().
            residual_mass[constraint_index] -= constraint.get_value()
            # Normalise by target mass to compute a relative residual.
            residual_mass[constraint_index] /= constraint.get_value()
        logger.debug("residual_mass = %s", residual_mass)

        # Combined residual.
        residual: np.ndarray = np.concatenate((residual_reaction, residual_mass))
        logger.debug("residual = %s", residual)

        return residual
