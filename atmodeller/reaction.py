"""Oxygen fugacity buffers and gas phase reactions."""

import logging
import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import linalg, optimize

logger: logging.Logger = logging.getLogger(__name__)

from atmodeller import GAS_CONSTANT, TEMPERATURE_JANAF_HIGH, TEMPERATURE_JANAF_LOW


class _OxygenFugacity(ABC):
    """Oxygen fugacity base class."""

    @abstractmethod
    def _buffer(self, *, temperature: float) -> float:
        """Log10(fo2) of the buffer in terms of temperature.

        Args:
            temperature: Temperature.

        Returns:
            Log10 of the oxygen fugacity.
        """
        raise NotImplementedError

    def __call__(self, *, temperature: float, fo2_shift: float = 0) -> float:
        """log10(fo2) plus an optional shift.

        Args:
            temperature: Temperature.
            fo2_shift: Log10 shift.

        Returns:
            Log10 of the oxygen fugacity including a shift.
        """
        return self._buffer(temperature=temperature) + fo2_shift


class IronWustiteBufferOneill(_OxygenFugacity):
    """Iron-wustite buffer from O'Neill and Eggins (2002). See Table 6.

    https://ui.adsabs.harvard.edu/abs/2002ChGeo.186..151O/abstract
    """

    def _buffer(self, *, temperature: float) -> float:
        """See base class."""
        buffer: float = (
            2
            * (
                -244118
                + 115.559 * temperature
                - 8.474 * temperature * np.log(temperature)
            )
            / (np.log(10) * 8.31441 * temperature)
        )
        return buffer


class IronWustiteBufferFischer(_OxygenFugacity):
    """Iron-wustite buffer from Fischer et al. (2011).

    https://ui.adsabs.harvard.edu/abs/2011E%26PSL.304..496F/abstract
    """

    def _buffer(self, *, temperature: float) -> float:
        """See base class."""
        buffer: float = 6.94059 - 28.1808 * 1e3 / temperature
        return buffer


# TODO: Once the chemical network approach has been tested and verified, the approach below will
# have been superseded.
@dataclass
class _Reaction:
    """A gas phase reaction.

    Args:
        temperature_factor: Factor to multiply 1/temperature for the equilibrium constant.
        constant: Constant factor to add for the equilibrium constant.
        fo2_stoichiometry: The stoichiometry of oxygen in the reaction.
        oxygen_fugacity: The oxygen fugacity model to use for the reaction.
    """

    temperature_factor: float
    constant: float
    fo2_stoichiometry: float
    oxygen_fugacity: _OxygenFugacity = field(default_factory=IronWustiteBufferOneill)

    def equilibrium_constant_log10(self, *, temperature: float) -> float:
        """Log10 of the equilibrium constant.

        Args:
            temperature: Temperature.

        Returns:
            Log10 of the equilibrium constant of the reaction.
        """
        return self.temperature_factor / temperature + self.constant

    def equilibrium_constant(self, *, temperature: float) -> float:
        """Equilibrium constant.

        Args:
            temperature: Temperature.

        Returns:
            The equilibrium constant of the reaction.
        """
        return 10 ** self.equilibrium_constant_log10(temperature=temperature)

    def modified_equilibrium_constant_log10(
        self, *, temperature: float, fo2_shift: float
    ) -> float:
        """Log10 of the 'modified' equilibrium constant, which includes oxygen fugacity.

        Args:
            temperature: Temperature.
            fo2_shift: log10 shift relative to the buffer.

        Returns:
            Log10 of the 'modified' equilibrium constant.
        """
        return self.equilibrium_constant_log10(
            temperature=temperature
        ) - self.fo2_stoichiometry * self.oxygen_fugacity(
            temperature=temperature, fo2_shift=fo2_shift
        )

    def modified_equilibrium_constant(
        self, *, temperature: float, fo2_shift: float
    ) -> float:
        """The 'modified' equilibrium constant, which includes oxygen fugacity.

        Args:
            temperature: Temperature.
            fo2_shift: log10 shift relative to the buffer.

        Returns:
            The 'modified' equilibrium constant.
        """
        return 10.0 ** self.modified_equilibrium_constant_log10(
            temperature=temperature, fo2_shift=fo2_shift
        )


@dataclass
class JanafC(_Reaction):
    """CO2 = CO + 0.5 fo2.

    JANAF log10Keq, 1500 < T < 3000. Fit by P. Sossi.
    """

    temperature_factor: float = -14467.511400133637
    constant: float = 4.348135473316284
    fo2_stoichiometry: float = 0.5


@dataclass
class JanafH(_Reaction):
    """H2O = H2 + 0.5 fo2.

    JANAF log10Keq, 1500 < T < 3000. Fit by P. Sossi.
    """

    temperature_factor: float = -13152.477779978302
    constant: float = 3.038586383273608
    fo2_stoichiometry: float = 0.5


@dataclass
class IvtanthermoC(_Reaction):
    """CO2 = CO + 0.5 fo2.

    IVANTHERMO log10Keq, 298.15 < T < 2000. Fit by Schaefer and Fegley (2017).

    https://ui.adsabs.harvard.edu/abs/2017ApJ...843..120S/abstract
    """

    temperature_factor: float = -14787
    constant: float = 4.5472
    fo2_stoichiometry: float = 0.5


@dataclass
class IvtanthermoCH4(_Reaction):
    """CO2 + 2H2 = CH4 + fo2.

    IVANTHERMO log10Keq, 298.15 < T < 2000. Fit by Schaefer and Fegley (2017).

    https://ui.adsabs.harvard.edu/abs/2017ApJ...843..120S/abstract
    """

    temperature_factor: float = -16276
    constant: float = -5.4738
    fo2_stoichiometry: float = 1


@dataclass
class IvtanthermoH(_Reaction):
    """H2O = H2 + 0.5 fo2.

    IVANTHERMO log10Keq, 298.15 < T < 2000. Fit by Schaefer and Fegley (2017).

    https://ui.adsabs.harvard.edu/abs/2017ApJ...843..120S/abstract
    """

    temperature_factor: float = -12794
    constant: float = 2.7768
    fo2_stoichiometry: float = 0.5


# TODO: Once the chemical network approach has been tested and verified, the approach above will
# have been superseded.


@dataclass
class MolarMasses:
    """Molar masses of atoms and molecules in kg/mol.

    There is a library that could do this, but it would add a dependency and there is always a
    risk it wouldn't be supported in the future:

    https://pypi.org/project/molmass/
    """

    # Define atoms here.
    # pylint: disable=invalid-name
    C: float = 12.0107e-3
    H: float = 1.0079e-3
    N: float = 14.0067e-3
    O: float = 15.9994e-3
    S: float = 32.065e-3

    def __post_init__(self):
        # Define molecules here.
        self.CH4: float = self.C + self.H * 4
        self.CO: float = self.C + self.O
        self.CO2: float = self.C + 2 * self.O
        self.H2: float = self.H * 2
        self.H2O: float = self.H * 2 + self.O
        self.N2: float = self.N * 2
        self.NH3: float = self.N + 3 * self.H
        self.O2: float = self.O * 2
        self.SO2: float = self.S + 2 * self.O


@dataclass(frozen=True)
class FormationEquilibriumConstants:
    """Formation equilibrium constants.

    These parameters result from a linear fit in temperature space to the log Kf column in the
    JANAF data tables for a given molecule. See the jupyter notebook in 'janaf'.

    log10(Kf) = a + b/T

    In the future we could use the Shomate equation to calculate the equilibrium of the gas phase
    reactions.
    """

    # Want to use molecule names therefore pylint: disable=invalid-name
    o2: tuple[float, float] = (0, 0)
    h2: tuple[float, float] = (0, 0)
    n2: tuple[float, float] = (0, 0)
    c: tuple[float, float] = (0, 0)
    ch4: tuple[float, float] = (-5.830066176470588, 4829.067647058815)
    co: tuple[float, float] = (4.319860294117643, 6286.120588235306)
    co2: tuple[float, float] = (-0.028289705882357442, 20753.870588235302)
    h2o: tuple[float, float] = (-3.0385132352941198, 13152.698529411768)
    # TODO: Commented out by Laura so check values.
    # nh3: tuple[float, float] = (-45.9, 192.77)


class ReactionNetwork:
    """Determines the necessary (formation) reactions to solve a chemical network.

    Args:
        molecules: A list of molecules.
    """

    # Elements that can be included in the molecules argument. Ordered by atomic mass.
    possible_elements: tuple[str, ...] = ("H", "He", "C", "N", "O", "Si", "P", "S")

    def __init__(self, molecules: list[str]):
        self.formation_energies: FormationEquilibriumConstants = (
            FormationEquilibriumConstants()
        )
        self.molecules: list[str] = self._check_molecules_input(molecules=molecules)
        logger.info("Molecules = %s", self.molecules)
        self.number_molecules: int = len(molecules)
        self.elements, self.number_elements = self.find_elements()
        self.number_reactions: int = self.number_molecules - self.number_elements
        self.molecule_matrix: np.ndarray = self.find_matrix()
        self.reaction_matrix: np.ndarray = self.partial_gaussian_elimination()
        self.oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill()
        logger.info("Reactions = \n%s", pprint.pformat(self.reactions))
        # TODO: Laura included this but I don't think it is actually used anywhere.
        # self.deltaN = self.get_deltaN()

    def _check_molecules_input(self, *, molecules: list[str]) -> list[str]:
        """Check user molecule input.

        Args:
            molecules: A list of molecules.
        """
        for molecule in molecules:
            if not hasattr(self.formation_energies, molecule.casefold()):
                raise ValueError(f"Formation energy of '{molecule}' is unknown")

        # TODO: Decide whether to sort or not. Algorithmically it is not required.
        # molecules = sorted(molecules, key=self.molecule_complexity)

        return list(molecules)

    # Using formation energies it is actually not necessary to reorder the array.
    def molecule_complexity(self, molecule: str) -> tuple[Any, ...]:
        """A key sorter to sort the molecules in order of complexity, starting with atoms.

        The order ensures that the Gaussian elimination returns formation reactions, i.e. the
        preference of the reaction stoichiometry is given to the first atoms/molecules in
        `self.molecules`.

        Args:
            molecule: The name of the molecule.

        Returns:
            A tuple with the sorting criteria.
        """
        # Rule 1: Single capital letters first.
        if len(molecule) == 1 and molecule.isupper():
            return (0, molecule)

        # Rule 2: Single capital letters followed by a lowercase letter.
        if len(molecule) == 2 and molecule[0].isupper() and molecule[1].islower():
            return (1, molecule)

        # Rule 3: Single capital letters followed by a number.
        if len(molecule) == 2 and molecule[0].isupper() and molecule[1].isdigit():
            return (2, molecule)

        # Rule 4: Single capital letters followed by a lowercase letter and a number.
        if (
            len(molecule) == 3
            and molecule[0].isupper()
            and molecule[1].islower()
            and molecule[2].isdigit()
        ):
            return (3, molecule)

        # Rule 5: All other cases ordered by total length of the string and alphabetical.
        return (4, len(molecule), molecule)

    def find_elements(self) -> tuple[list, int]:
        """Determines the elements that compose the molecules.

        Returns:
            A tuple: (list of elements, number of elements).
        """
        molecules_string: str = "".join(self.molecules)
        elements: list[str] = []
        for element in self.possible_elements:
            found: int = molecules_string.find(element)
            if found != -1:  # Substring exists inside the string.
                elements.append(element)

        return elements, len(elements)

    def find_matrix(self) -> np.ndarray:
        """Creates a matrix where molecules (rows) are split into their element counts (columns).

        Returns:
            For example, self.molecules = ['CO2', 'H2O'] would return:
                [[0, 1, 2],
                 [2, 0, 1]]
            where the columns represent the elements H, C, and O, respectively.
        """
        matrix: np.ndarray = np.zeros((self.number_molecules, self.number_elements))
        for molecule_index, molecule in enumerate(self.molecules):
            for element_index, element in enumerate(self.elements):
                found: int = molecule.find(element)
                if found != -1:  # Substring exists inside the string.
                    try:
                        # Character after the element.
                        character: str = molecule[found + len(element)]
                    except IndexError:
                        # No character (end of the molecule) so the count must be unity.
                        count: int = 1
                    else:
                        try:
                            count = int(character)
                        except ValueError:
                            # Character is not an int so it must be the next element instead.
                            count = 1
                    matrix[molecule_index, element_index] = count
        return matrix

    def partial_gaussian_elimination(self) -> np.ndarray:
        """Performs a partial gaussian elimination to determine the required reactions.

        A copy of `self.molecule_matrix` is first (partially) reduced to row echelon form by
        forward elimination, and then subsequently (partially) reduced to reduced row echelon form
        by backward substitution. Applying the same operations to the identity matrix (as part of
        the augmented matrix) reveals r reactions, where r = number of molecules - number of
        elements. These reactions are given in the last r rows of the reduced matrix. Due to the
        ordering of `self.molecules`, these reactions are given in terms of atoms or simple
        molecules to stay connected to formation reactions.

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

        reduced_matrix1 = augmented_matrix[:, : matrix1.shape[1]]
        reaction_matrix = augmented_matrix[self.number_elements :, matrix1.shape[1] :]
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
                if coeff < 0:
                    reactants += f"{abs(coeff)} {molecule} + "
                elif coeff > 0:
                    products += f"{coeff} {molecule} + "
            reactants = reactants[
                : int(len(reactants) - 3)
            ]  # Removes the extra + at the end.
            products = products[
                : int(len(products) - 3)
            ]  # Removes the extra + at the end.
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction
        # reactions_str: str = pprint.pformat(reactions)

        return reactions

    def get_formation_equilibrium_constant(
        self, *, molecule: str, temperature: float
    ) -> float:
        """Gets the formation equilibrium constant (log Kf in the JANAF tables) of a molecule.

        Args:
            molecule: Name of the molecule.
            temperature: Temperature.

        Returns:
            The formation equilibrium constant.
        """
        constant, temp_factor = getattr(self.formation_energies, molecule.casefold())
        return constant + temp_factor / temperature

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
            ] * self.get_formation_equilibrium_constant(
                molecule=molecule, temperature=temperature
            )
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
        temperature: float,
        input_pressures: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Builds the coefficient matrix and the RHS.

        Args:
            temperature: Temperature.
            input_pressures: A dictionary of {molecule: partial pressure} that should be imposed,
                where each molecule should be in the network (i.e. in `self.molecules`).

        Returns:
            A dictionary of all the molecules and their partial pressures.
        """

        # for molecule, pressure in input_pressures.items():
        #    input_pressures[molecule] = np.log10(pressure)  # Note log10.
        # Add fO2 as an enforced constraint. This can be easily relaxed in the future.
        # input_pressures["O2"] = self.oxygen_fugacity(
        #    temperature=temperature, fo2_shift=fo2_shift
        # )
        # input_pressures["O2"] = oxygen_fugacity(
        #    temperature=temperature, fo2_shift=fo2_shift
        # )

        logger.info("Input pressures (log10) = %s", input_pressures)

        # Build coefficient matrix and RHS vector.
        coeff: np.ndarray = np.zeros((self.number_molecules, self.number_molecules))
        rhs: np.ndarray = np.zeros(self.number_molecules)

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

        # Defined pressures (including fo2 as prescribed by the buffer).
        for index, (molecule, pressure) in enumerate(input_pressures.items()):
            row_index: int = self.number_reactions + index
            molecule_index: int = self.molecules.index(molecule)
            logger.info("Row %02d: Setting %s partial pressure", row_index, molecule)
            coeff[row_index, molecule_index] = 1
            rhs[row_index] = pressure

        logger.debug("Coefficient matrix = \n%s", coeff)
        logger.debug("RHS vector = \n%s", rhs)

        return coeff, rhs

    def set_input_pressures(
        self,
        *,
        temperature: float,
        input_pressures: dict[str, float],
        oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill(),
        fo2_shift: float = 0,
    ) -> dict[str, float]:
        """Sets the input pressures.

        This takes a list of input pressures and adds an extra pressure constraint (fO2) if
        required to close the system of equations. It also converts the pressures to log10.

        Args:
            temperature: Temperature.
            input_pressures: A dictionary of {molecule: partial pressure} that should be imposed,
                where each molecule should be in the network (i.e. in `self.molecules`).
            oxygen_fugacity: Oxygen fugacity model. Defaults to IronWustiteBufferOneill. This is
                only used if the number of `input_pressures` is one less than the required number.
            fo2_shift: log10 fo2 shift from the buffer. Defaults to 0. This is only used if the
                number of `input_parameters` is one less than the required number.

        Returns:
            A dictionary of all the molecules and their partial pressures.
        """
        if (temperature <= TEMPERATURE_JANAF_LOW) or (
            temperature >= TEMPERATURE_JANAF_HIGH
        ):
            msg: str = "Temperature must be in the range {TEMPERATURE_JANAF_LOW} K to "
            msg += f"{TEMPERATURE_JANAF_HIGH} K"
            raise ValueError(msg)

        input_pressures = {
            molecule: pressure
            for molecule, pressure in input_pressures.items()
            if molecule in self.molecules
        }
        input_number: int = len(list(input_pressures.keys()))
        constraints: int = self.number_elements

        for molecule, pressure in input_pressures.items():
            input_pressures[molecule] = np.log10(pressure)

        if input_number == constraints - 1:
            # Missing one constraint, so we impose oxygen fugacity.
            logger.info(
                "Adding fO2 as an additional constraint using %s with fO2_shift = %f",
                oxygen_fugacity.__class__.__name__,
                fo2_shift,
            )
            input_pressures["O2"] = oxygen_fugacity(
                temperature=temperature, fo2_shift=fo2_shift
            )
        elif input_number != constraints:
            raise ValueError(
                f"You must specify pressures for at least {constraints-1} species. Select from {self.molecules}"
            )

        return input_pressures

    def solve(
        self,
        *,
        temperature: float,
        input_pressures: dict[str, float],
        oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill(),
        fo2_shift: float = 0,
    ) -> dict[str, float]:
        """Solves the reaction network to determine the partial pressures of all species.

        Applies the law of mass action. To solve the reaction network requires that:
            `self.number_molecules` = `self.number_reactions` + number of input pressures.
        If the number of input pressures is one less than required, then oxygen fugacity is by
        default imposed as the final constraint.

        We solve for the log10 of the partial pressures of each species. Operating in log10 space
        has two advantages: 1) The dynamic range of the partial pressures is reduced, for example
        fO2 is typically very small compared to other pressures in the system, and 2) In log10
        space the reaction network can be expressed as a linear system which is trivial to solve.

        One could of course use a different log space (natural log), but log10 is chosen because
        the formation reactions are expressed in terms of log10 as well as the oxygen fugacity.

        Args:
            temperature: Temperature.
            input_pressures: A dictionary of {molecule: partial pressure} that should be imposed,
                where each molecule should be in the network (i.e. in `self.molecules`).
            oxygen_fugacity: Oxygen fugacity model. Defaults to IronWustiteBufferOneill. This is
                only used if the number of `input_pressures` is one less than the required number.
            fo2_shift: log10 fo2 shift from the buffer. Defaults to 0. This is only used if the
                number of `input_parameters` is one less than the required number.

        Returns:
            A dictionary of all the molecules and their partial pressures.
        """
        logger.info("Solving the reaction network")

        input_pressures = self.set_input_pressures(
            temperature=temperature,
            input_pressures=input_pressures,
            oxygen_fugacity=oxygen_fugacity,
            fo2_shift=fo2_shift,
        )

        coeff_matrix, rhs = self.get_coefficient_matrix_and_rhs(
            temperature=temperature, input_pressures=input_pressures
        )

        solution: np.ndarray = linalg.solve(coeff_matrix, rhs)
        logger.debug("Solution = \n%s", solution)
        pressures: np.ndarray = 10.0**solution

        output: dict[str, float] = {
            molecule: pressure
            for (molecule, pressure) in zip(self.molecules, pressures)
        }

        logger.info("Solution is:")
        for species, pressure in sorted(output.items()):
            logger.info("    %s pressure (bar) = %f", species, pressure)

        return output

        # TODO: C (a solid) is always given a pressure (fugacity) of unity. To check and document.
        # if "C" in input_pressures:
        #     input_pressures["C"] = 1  # TODO: Sets activity to unit. Valid?
        # logger.info("Input pressures = %s", input_pressures)

    # TODO: Laura included this but I don't think it is actually used anywhere.
    # def get_deltaN(self):
    #     """
    #     To calculate the K_p from K_c, we have to know how the pressure changes.
    #     """
    #     delta_n = np.zeros(self.number_reactions)
    #     for m, mol in enumerate(self.molecules):
    #         if mol != "C":  # BECAUSE C NOT IN GASPHASE
    #             for r in range(self.number_reactions):
    #                 delta_n[r] += self.reaction_matrix[r, m]
    #     return delta_n


class MassBalance(ReactionNetwork):
    """Class to build and solve the non-linear system of equations."""

    def solve(
        self,
        *,
        temperature: float,
        input_pressures: dict[str, float],
        oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill(),
        fo2_shift: float = 0,
    ) -> dict[str, float]:
        """Solve the non-linear equation set.

        TODO: Currently just solves the reaction network numerically using Ax-b=0. Now need to
        include mass balance constraints.
        """
        logger.info("Solving the mass balance network")

        # TODO: In practice we will use mass balance constraints for at least H and C to provide
        # two closure conditions. The third closure condition can still be imposed fO2.
        input_pressures = self.set_input_pressures(
            temperature=temperature,
            input_pressures=input_pressures,
            oxygen_fugacity=oxygen_fugacity,
            fo2_shift=fo2_shift,
        )

        coeff_matrix, rhs = self.get_coefficient_matrix_and_rhs(
            temperature=temperature, input_pressures=input_pressures
        )

        def func(solution: np.ndarray) -> np.ndarray:
            """Express the reaction network as Ax-b=0 for the non-linear solver."""
            lhs: np.ndarray = coeff_matrix.dot(solution) - rhs
            return lhs

        x0: np.ndarray = np.ones(len(self.molecules))
        solution, _, ier, _ = optimize.fsolve(func, x0, full_output=True)
        logger.debug("ier = %s", ier)
        logger.debug("Solution = \n%s", solution)
        pressures: np.ndarray = 10.0**solution

        output: dict[str, float] = {
            molecule: pressure
            for (molecule, pressure) in zip(self.molecules, pressures)
        }

        logger.info("Solution is:")
        for species, pressure in sorted(output.items()):
            logger.info("    %s pressure (bar) = %f", species, pressure)

        return output
