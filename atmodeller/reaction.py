"""Oxygen fugacity buffers and gas phase reactions."""

import logging
import pprint
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger: logging.Logger = logging.getLogger(__name__)


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


class ReactionNetwork:
    """Determines the necessary (formation) reactions to solve a chemical network.

    Args:
        molecules: A list of molecules.
    """

    # Elements that can be included in the molecules argument. Ordered by atomic mass.
    possible_elements: tuple[str, ...] = ("H", "He", "C", "N", "O", "Si", "P", "S")

    def __init__(self, molecules: list[str]):
        self.molecules: list[str] = sorted(molecules, key=self.molecule_complexity)
        logger.info("Molecules = %s", self.molecules)
        self.number_molecules: int = len(molecules)
        self.elements, self.number_elements = self.find_elements()
        self.number_reactions: int = self.number_molecules - self.number_elements
        self.molecule_matrix: np.ndarray = self.find_matrix()
        self.reaction_matrix: np.ndarray = self.partial_gaussian_elimination()
        logger.info("reactions = \n%s", self.reactions)
        # TODO: Laura included this but I don't think it is actually used anywhere.
        # self.deltaN = self.get_deltaN()

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
            A matrix of the reactions.
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
    def reactions(self) -> str:
        """The reactions as a (dictionary) string."""
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
            reactants = reactants[: int(len(reactants) - 3)] # Removes the extra + at the end.
            products = products[: int(len(products) - 3)] # Removes the extra + at the end.
            reaction: str = f"{reactants} = {products}"
            reactions[reaction_index] = reaction
        reactions_str: str = pprint.pformat(reactions)

        return reactions_str

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


# data_file = "dictionary.dat"


# class GibbsEnergys(ReactionNetwork, parameters):

#     """
#     This class calculates the reaction equilibrium constants K_eq,r of the reactions from the class Reaction Network.
#     The class calculates the K_eq,r from the formation equilibrium constants K_eq,f from the molecules in the reaction.
#     K_eq,f is calculated from a datebase in the formate K_eq,f = a + b/T, where a and b are from the dictionar.
#     And

#     """

#     def __init__(self, molecules):
#         # Since we need the stochometric matrix, we inhertit the reaction network
#         ReactionNetwork.__init__(self, molecules)
#         parameters.__init__(self)

#         self.T = self.global_d["temperature"]  # Kelvin

#         # Calculate the Equilibrium Constant for the formation reactions
#         self.df_FormationEnergies = self.get_FormationEnergies_csv(data_file)
#         self.log_K_f = self.get_logK_f()

#         self.log_K_r = self.get_K_r()
#         self.K_r = 10**self.log_K_r

#     def get_FormationEnergies_csv(self, data_file):
#         """
#         Function finds the csv part describing the Formation Energies of the Molecules
#         and subscribes them with and saves them in a dataframe, which will be called G_form
#         """
#         with open(data_file) as f:
#             lines = f.readlines()
#             num_lines = len(lines)
#             for k, line in zip(range(num_lines + 1), lines):
#                 if line == "FORMATION ENERGIES\n":
#                     start_line = k
#                     break

#             for k, line in zip(
#                 range(start_line, num_lines + 1), lines[start_line : num_lines + 1]
#             ):
#                 if line == "END\n":
#                     end_line = k
#                     break

#         range_to_skip = list(range(0, start_line + 1)) + list(
#             range(end_line, num_lines + 1)
#         )
#         df_form = pd.read_csv(
#             data_file, skiprows=range_to_skip, error_bad_lines=False, comment="#"
#         )
#         return df_form

#     def get_logK_f(self):
#         """
#         Here the dictionary would be accesed to get the formation energies for all molecules in J/mol.
#         """
#         T = self.T
#         df = self.df_FormationEnergies
#         log_K_f = {}  # empty dictionary
#         for mol in self.molecules:
#             y = df.loc[df["SPECIES"] == mol]
#             a, b = float(y["a"]), float(y["b"])
#             log_K_f[mol] = a + b / T

#         return log_K_f

#     def get_K_r(self):
#         """
#         Here the Gibb's Energies of the independent equilibrium reaction (descibed in stoch_matrix)
#         in the gas phase of the Earth's atmosphere are calculated from the formation energies (G_f)
#         of the molecules present in the atmosphere (molecules).
#         The values are saved and can be accesed in the vector G_r[num_reaction].
#         """

#         log_K_r = np.zeros(self.num_r)

#         for r in range(self.num_r):
#             for i, mol in enumerate(self.molecules):
#                 key = self.molecules[i]
#                 log_K_r[r] += self.stoch_matrix[r, i] * self.log_K_f[key]
#         return log_K_r


# class OxygenFugacity(parameters):
#     """log10 oxygen fugacity as a function of temperature"""

#     def __init__(self, model="oneill"):
#         parameters.__init__(self)
#         self.T = self.global_d["temperature"]  # Kelvin

#         self.callmodel = getattr(self, model)

#     def __call__(self, fO2_shift=0):
#         """Return log10 fO2"""
#         """ self.callmodel(T) = self.oneill(T)"""

#         return self.callmodel(self.T) + fO2_shift

#     def fischer(self, T):
#         """Fischer et al. (2011) IW"""
#         # in cases where callmodel = fischer or
#         # model = 'fischer'
#         return 6.94059 - 28.1808 * 1e3 / self.T

#     def oneill(self, T):
#         """O'Neill and Eggin (2002) IW"""
#         return (
#             2
#             * (-244118 + 115.559 * self.T - 8.474 * self.T * np.log(self.T))
#             / (np.log(10) * 8.31441 * self.T)
#         )


# class Atmosphere(GibbsEnergys):
#     def __init__(self, molecules, fO2_model="oneill"):
#         self.molecules = molecules
#         # self.mass_d = self.global_d['molar_mass_d']

#         # gettin oxgen fugactiy from model defined above (here also oneill)
#         self.fO2 = OxygenFugacity(fO2_model)

#         # Default Values
#         self.p_0 = 1

#     def Set_p0(self, p_0):
#         self.p_0 = p_0
#         return 0

#     def Fix_fd(self, f_d):
#         self.Fixed_fd = f_d
#         self.fixed_mols = list(f_d.keys())

#         self.unfixed_mols = []
#         for mol in self.molecules:
#             if mol not in self.fixed_mols:
#                 self.unfixed_mols.append(mol)

#         # print('Molecules with fixed fugacities:' ,self.fixed_mols)
#         # print('Molecules with variable fugacities:' ,self.unfixed_mols)

#         for mol, f in f_d.items():
#             if mol == "C":
#                 try:
#                     assert f == 1
#                 except:
#                     # print('Fugacity of',mol,'must be fixed to: 1.')
#                     f_d["C"] = 1
#                     continue
#             # print('Fugacity of',mol,'is fixed to: ',f)
#         return 0

#     def Calculate(self, f_d, **fO2_shift):
#         if fO2_shift:
#             fO2_shift = fO2_shift["fO2_shift"]
#             log_fO2 = self.fO2(fO2_shift=fO2_shift)
#             f_O2 = 10**log_fO2
#             f_d["O2"] = f_O2

#         self.Fix_fd(f_d)
#         self.molecules = self.fixed_mols + self.unfixed_mols
#         GibbsEnergys.__init__(self, self.molecules)

#         for r in range(self.num_r):
#             "Every reaction that produces a non standart molecule"
#             mol = self.molecules[self.number_elements + r]  # molecules that is produced

#             p_ = self.K_r[r]

#             f_d_calc = {}
#             f_d_calc = self.Fixed_fd
#             # p_d['C']=0
#             reaction_array = self.stoch_matrix[r]

#             for i, f in enumerate(self.Fixed_fd.values()):
#                 exponent = reaction_array[i] * (-1)
#                 p_ *= f**exponent

#             f_d_calc[mol] = p_

#         p_d_calc = {}
#         p_d_calc = f_d
#         p_d_calc["C"] = 0
#         p_tot_calc = sum(p_d_calc.values())
#         # Add a Function that creates a csv file?

#         return p_tot_calc, p_d_calc

#     def Calc_Mu_Atm(self, p_d):
#         """Mean molar mass of the atmosphere"""
#         ptot = sum(p_d.values())
#         mu_atm = 0
#         for key, value in p_d.items():
#             mu_atm += self.mass_d[key] * value
#         mu_atm /= ptot

#         return mu_atm

#     def MassMol_Atm(self, mu_atm, key, p):
#         factor = 4.0 * np.pi * self.global_d["planetary_radius"] ** 2
#         factor *= 1 / (mu_atm * self.global_d["little_g"])
#         mu_i = self.mass_d[key]
#         # print(type(mu_i))
#         m_i = mu_i * p * factor
#         return m_i

#     def Calc_Mass(self, p_d):
#         mu_atm = self.Calc_Mu_Atm(p_d)
#         mass_atm_d = {}
#         for key, p in p_d.items():
#             mass_atm_d[key] = self.MassMol_Atm(mu_atm, key, p)

#         mol_matrix = self.find_matrix()

#         for element, line in zip(self.elements, np.transpose(mol_matrix)):
#             mass_atm_d[element] = 0
#             for mol, num in zip(self.molecules, line):
#                 if num != 0:
#                     mass_atm_d[element] += mass_atm_d[mol] / self.mass_d[mol] * num
#             mass_atm_d[element] *= self.mass_d[element]
#         self.m_d = mass_atm_d
#         return mass_atm_d

#     def Calc_md(self, p_scaled_fixed):
#         sol = self.Get_Scaled_Solution(p_scaled_fixed)
#         self.scaled_pd = sol
#         p_d_ = self.Calc_Real_p_d(sol)
#         self.p_d = p_d_

#         m_atm_d = self.Calc_Mass(p_d_)

#         return m_atm_d
