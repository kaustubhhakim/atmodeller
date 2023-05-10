"""Oxygen fugacity buffers and gas phase reactions."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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


# TODO: Below is code written by Laura Murzakhmetov that needs incorporating into the code.


class ReactionNetwork:
    """This class does the Gaussian Elimination to determine the minimal set of reactions to
    describe the atmosphere. The reaction dictionary outputted is unique, since the class will
    always produce reactions where the last elements in the list, are the once beeing produced
    (having a stochometric coef of 1). This allows easy calculation, for a variety of input
    pressure combinations, without making the calculations different or more complicated.
    """

    # The possible elements the class regognizes (others must be included)

    possible_elements = ["C", "H", "O", "N", "He", "S", "P", "Si"]

    def __init__(self, molecules):
        self.molecules = molecules
        self.num_mol = len(molecules)

        self.elements, self.num_e = self.find_elements()

        self.mol_matrix = self.find_matrix()

        self.gauss_elimated_matrix, self.stoch_reaction_matrix = self.gaussElim()

        self.num_r, self.stoch_matrix, self.reaction_d = self.GetEqReactions()

        self.deltaN = self.get_deltaN()

    def find_elements(self):
        """
        Finds the All Elements metioned in the Molecule vector
        returns:
            elements = array of all elements
            num_e = number of elements
        """
        possible_elements = ["C", "H", "O", "N", "He", "S", "P", "Si"]

        molecules = self.molecules
        num_mol = len(molecules)
        mol_str = "".join(molecules)  # .join('')
        elements = []
        num_e = 0
        # FIND ALL ELEMENTS
        for element in possible_elements:
            found = mol_str.find(element)
            if found + 1:
                elements.append(element)
                num_e += 1
        return elements, num_e

    def find_matrix(self):
        """
        Creates a Matrix containing the number of each element in a molecules by counting.
        """
        # Create Matrix by Counting the Number of each element
        matrix = np.zeros((self.num_mol, self.num_e))

        for i in range(self.num_mol):
            mol = self.molecules[i]
            for k in range(self.num_e):
                element = self.elements[k]
                found = mol.find(element)
                if (
                    found + 1
                ):  # if that element is in the molecule, determine the number the molecule occurs in the molecule
                    num = 1
                    try:
                        next_char = mol[found + len(element)]
                    except:
                        matrix[i, k] = num
                    else:
                        if next_char in "0123456789":
                            num = int(next_char)
                    matrix[i, k] = num
        return matrix

    def permutate(self, a, i, k):
        a[[i, k]] = a[[k, i]]
        return a

    def order_diag(self, a, c, num_e):
        # Sort Coloumns: least entries, more important!
        a_sq = a[:num_e]
        a_sq_T = np.transpose(a_sq)
        num_entries = [[i, np.count_nonzero(a_sq_T[i])] for i in range(num_e)]
        num_entries = sorted(num_entries, key=lambda num_entry: num_entry[1])
        order = [num_entries[i][0] for i in range(num_e)]
        num_entries2 = [[i, np.count_nonzero(a_sq[i])] for i in range(num_e)]
        num_entries2 = sorted(num_entries2, key=lambda num_entry: num_entry[1])
        order2 = [num_entries[i][0] for i in range(num_e)]
        # Now Permutate the entries, starting with the most important
        for i in order:
            # print(i, a[i,:])
            for k in order2:
                if a[i, k] != 0:
                    if k == i:
                        break

                    elif k != i:  # if k=i,then row already in perfect position
                        a = self.permutate(a, i, k)
                        c = self.permutate(c, i, k)
                        break
        return a, c

    def gaussElim(self):
        """
        Input:
            a: Matrix of Elemental Composition
            b: Array of Molecules

        Output:
            a: Gaussian Eliminated Matrix
            b: Array of Molecules
            c: Stochiometric Reaction Matrix
        """

        a = self.mol_matrix

        num_mol = len(a)  # number of molecules
        num_e = len(a[0])  # number of elements
        c = np.eye(num_mol)  # stochiometric matrix

        # Elimination phase
        # No Longer Standard Molecules first!
        # Order the First num_e rows, so that we do not have non-zero entries

        a, c = self.order_diag(a, c, num_e)
        for k in range(0, num_e):  # rows: 0,1,2
            assert a[k, k] != 0.0
            for i in range(k + 1, num_mol):
                if a[i, k] != 0.0:
                    # if not null define λ
                    lam = a[i, k] / a[k, k]
                    # we calculate the new row of the matrix
                    a[i, :] = a[i, :] - lam * a[k, :]

                    # we update c
                    c[i, :] = c[i, :] - lam * c[k, :]

        # Backward elimination -> Getting the diagonal matrix.
        for k in range(num_e - 1, -1, -1):  # 2,1,0
            assert a[k, k] != 0.0
            for i in range(k - 1, -1, -1):
                #
                # if not null define λ
                if a[i, k] != 0:
                    lam = a[i, k] / a[k, k]

                    a[i, :] = a[i, :] - lam * a[k, :]

                    # Update vector c.
                    c[i, :] = c[i, :] - lam * c[k, :]

        # Divide to have everywhere one.
        for k in range(0, num_e):
            lam = a[k, k]
            a[k, :] /= lam
            # Update vector c.
            c[k, :] /= lam

        return a, c

    def GetEqReactions(self):
        num_r = self.num_mol - self.num_e
        mol = self.molecules
        c = self.stoch_reaction_matrix
        stoch_matrix = c[self.num_e :]

        # Create empty reaction dictionary
        reaction_d = {}

        # Go threw each reaction line

        for r in range(num_r):
            r_str = ""
            e_str, p_str = "", ""
            # Go threw each molecule
            for m in range(self.num_mol):
                coeff = stoch_matrix[r, m]
                if coeff < 0:  # molecule is educt
                    e_str += str(abs(coeff)) + " " + str(mol[m]) + " + "

                elif coeff > 0:
                    p_str += str(coeff) + " " + str(mol[m]) + " + "

            e_str = e_str[: int(len(e_str) - 3)]
            p_str = p_str[: int(len(p_str) - 3)]
            r_str = e_str + " = " + p_str

            reaction_d[r] = r_str

        return num_r, stoch_matrix, reaction_d

    def get_deltaN(self):
        """
        To calculate the K_p from K_c, we have to know how the pressure changes.
        """
        delta_n = np.zeros(self.num_r)
        for m, mol in enumerate(self.molecules):
            if mol != "C":  # BECAUSE C NOT IN GASPHASE
                for r in range(self.num_r):
                    delta_n[r] += self.stoch_matrix[r, m]
        return delta_n


data_file = "dictionary.dat"


class GibbsEnergys(ReactionNetwork, parameters):

    """
    This class calculates the reaction equilibrium constants K_eq,r of the reactions from the class Reaction Network.
    The class calculates the K_eq,r from the formation equilibrium constants K_eq,f from the molecules in the reaction.
    K_eq,f is calculated from a datebase in the formate K_eq,f = a + b/T, where a and b are from the dictionar.
    And

    """

    def __init__(self, molecules):
        # Since we need the stochometric matrix, we inhertit the reaction network
        ReactionNetwork.__init__(self, molecules)
        parameters.__init__(self)

        self.T = self.global_d["temperature"]  # Kelvin

        # Calculate the Equilibrium Constant for the formation reactions
        self.df_FormationEnergies = self.get_FormationEnergies_csv(data_file)
        self.log_K_f = self.get_logK_f()

        self.log_K_r = self.get_K_r()
        self.K_r = 10**self.log_K_r

    def get_FormationEnergies_csv(self, data_file):
        """
        Function finds the csv part describing the Formation Energies of the Molecules
        and subscribes them with and saves them in a dataframe, which will be called G_form
        """
        with open(data_file) as f:
            lines = f.readlines()
            num_lines = len(lines)
            for k, line in zip(range(num_lines + 1), lines):
                if line == "FORMATION ENERGIES\n":
                    start_line = k
                    break

            for k, line in zip(
                range(start_line, num_lines + 1), lines[start_line : num_lines + 1]
            ):
                if line == "END\n":
                    end_line = k
                    break

        range_to_skip = list(range(0, start_line + 1)) + list(
            range(end_line, num_lines + 1)
        )
        df_form = pd.read_csv(
            data_file, skiprows=range_to_skip, error_bad_lines=False, comment="#"
        )
        return df_form

    def get_logK_f(self):
        """
        Here the dictionary would be accesed to get the formation energies for all molecules in J/mol.
        """
        T = self.T
        df = self.df_FormationEnergies
        log_K_f = {}  # empty dictionary
        for mol in self.molecules:
            y = df.loc[df["SPECIES"] == mol]
            a, b = float(y["a"]), float(y["b"])
            log_K_f[mol] = a + b / T

        return log_K_f

    def get_K_r(self):
        """
        Here the Gibb's Energies of the independent equilibrium reaction (descibed in stoch_matrix)
        in the gas phase of the Earth's atmosphere are calculated from the formation energies (G_f)
        of the molecules present in the atmosphere (molecules).
        The values are saved and can be accesed in the vector G_r[num_reaction].
        """

        log_K_r = np.zeros(self.num_r)

        for r in range(self.num_r):
            for i, mol in enumerate(self.molecules):
                key = self.molecules[i]
                log_K_r[r] += self.stoch_matrix[r, i] * self.log_K_f[key]
        return log_K_r


class OxygenFugacity(parameters):
    """log10 oxygen fugacity as a function of temperature"""

    def __init__(self, model="oneill"):
        parameters.__init__(self)
        self.T = self.global_d["temperature"]  # Kelvin

        self.callmodel = getattr(self, model)

    def __call__(self, fO2_shift=0):
        """Return log10 fO2"""
        """ self.callmodel(T) = self.oneill(T)"""

        return self.callmodel(self.T) + fO2_shift

    def fischer(self, T):
        """Fischer et al. (2011) IW"""
        # in cases where callmodel = fischer or
        # model = 'fischer'
        return 6.94059 - 28.1808 * 1e3 / self.T

    def oneill(self, T):
        """O'Neill and Eggin (2002) IW"""
        return (
            2
            * (-244118 + 115.559 * self.T - 8.474 * self.T * np.log(self.T))
            / (np.log(10) * 8.31441 * self.T)
        )


class Atmosphere(GibbsEnergys):
    def __init__(self, molecules, fO2_model="oneill"):
        self.molecules = molecules
        # self.mass_d = self.global_d['molar_mass_d']

        # gettin oxgen fugactiy from model defined above (here also oneill)
        self.fO2 = OxygenFugacity(fO2_model)

        # Default Values
        self.p_0 = 1

    def Set_p0(self, p_0):
        self.p_0 = p_0
        return 0

    def Fix_fd(self, f_d):
        self.Fixed_fd = f_d
        self.fixed_mols = list(f_d.keys())

        self.unfixed_mols = []
        for mol in self.molecules:
            if mol not in self.fixed_mols:
                self.unfixed_mols.append(mol)

        # print('Molecules with fixed fugacities:' ,self.fixed_mols)
        # print('Molecules with variable fugacities:' ,self.unfixed_mols)

        for mol, f in f_d.items():
            if mol == "C":
                try:
                    assert f == 1
                except:
                    # print('Fugacity of',mol,'must be fixed to: 1.')
                    f_d["C"] = 1
                    continue
            # print('Fugacity of',mol,'is fixed to: ',f)
        return 0

    def Calculate(self, f_d, **fO2_shift):
        if fO2_shift:
            fO2_shift = fO2_shift["fO2_shift"]
            log_fO2 = self.fO2(fO2_shift=fO2_shift)
            f_O2 = 10**log_fO2
            f_d["O2"] = f_O2

        self.Fix_fd(f_d)
        self.molecules = self.fixed_mols + self.unfixed_mols
        GibbsEnergys.__init__(self, self.molecules)

        for r in range(self.num_r):
            "Every reaction that produces a non standart molecule"
            mol = self.molecules[self.num_e + r]  # molecules that is produced

            p_ = self.K_r[r]

            f_d_calc = {}
            f_d_calc = self.Fixed_fd
            # p_d['C']=0
            reaction_array = self.stoch_matrix[r]

            for i, f in enumerate(self.Fixed_fd.values()):
                exponent = reaction_array[i] * (-1)
                p_ *= f**exponent

            f_d_calc[mol] = p_

        p_d_calc = {}
        p_d_calc = f_d
        p_d_calc["C"] = 0
        p_tot_calc = sum(p_d_calc.values())
        # Add a Function that creates a csv file?

        return p_tot_calc, p_d_calc

    def Calc_Mu_Atm(self, p_d):
        """Mean molar mass of the atmosphere"""
        ptot = sum(p_d.values())
        mu_atm = 0
        for key, value in p_d.items():
            mu_atm += self.mass_d[key] * value
        mu_atm /= ptot

        return mu_atm

    def MassMol_Atm(self, mu_atm, key, p):
        factor = 4.0 * np.pi * self.global_d["planetary_radius"] ** 2
        factor *= 1 / (mu_atm * self.global_d["little_g"])
        mu_i = self.mass_d[key]
        # print(type(mu_i))
        m_i = mu_i * p * factor
        return m_i

    def Calc_Mass(self, p_d):
        mu_atm = self.Calc_Mu_Atm(p_d)
        mass_atm_d = {}
        for key, p in p_d.items():
            mass_atm_d[key] = self.MassMol_Atm(mu_atm, key, p)

        mol_matrix = self.find_matrix()

        for element, line in zip(self.elements, np.transpose(mol_matrix)):
            mass_atm_d[element] = 0
            for mol, num in zip(self.molecules, line):
                if num != 0:
                    mass_atm_d[element] += mass_atm_d[mol] / self.mass_d[mol] * num
            mass_atm_d[element] *= self.mass_d[element]
        self.m_d = mass_atm_d
        return mass_atm_d

    def Calc_md(self, p_scaled_fixed):
        sol = self.Get_Scaled_Solution(p_scaled_fixed)
        self.scaled_pd = sol
        p_d_ = self.Calc_Real_p_d(sol)
        self.p_d = p_d_

        m_atm_d = self.Calc_Mass(p_d_)

        return m_atm_d
