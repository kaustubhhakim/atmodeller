"""Oxygen fugacity buffers and gas phase reactions."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

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
    C: tuple[float, float] = (0, 0)
    CH4: tuple[float, float] = (-5.830066176470588, 4829.067647058815)
    CO: tuple[float, float] = (4.319860294117643, 6286.120588235306)
    CO2: tuple[float, float] = (-0.028289705882357442, 20753.870588235302)
    H2: tuple[float, float] = (0, 0)
    H2O: tuple[float, float] = (-3.0385132352941198, 13152.698529411768)
    N2: tuple[float, float] = (0, 0)
    O2: tuple[float, float] = (0, 0)
    # TODO: Commented out by Laura so check values.
    # NH3: tuple[float, float] = (-45.9, 192.77)


# class MassBalance(ReactionNetworkOld):
#     """Class to build and solve the non-linear system of equations."""

#     def solve(
#         self,
#         *,
#         temperature: float,
#         input_pressures: dict[str, float],
#         oxygen_fugacity: _OxygenFugacity = IronWustiteBufferOneill(),
#         fo2_shift: float = 0,
#     ) -> dict[str, float]:
#         """Solve the non-linear equation set.

#         TODO: Currently just solves the reaction network numerically using Ax-b=0. Now need to
#         include mass balance constraints.
#         """
#         logger.info("Solving the mass balance network")

#         # TODO: In practice we will use mass balance constraints for at least H and C to provide
#         # two closure conditions. The third closure condition can still be imposed fO2.
#         input_pressures = self.set_input_pressures(
#             temperature=temperature,
#             input_pressures=input_pressures,
#             oxygen_fugacity=oxygen_fugacity,
#             fo2_shift=fo2_shift,
#         )

#         coeff_matrix, rhs = self.get_coefficient_matrix_and_rhs(
#             temperature=temperature, input_pressures=input_pressures
#         )

#         def func(solution: np.ndarray) -> np.ndarray:
#             """Express the reaction network as Ax-b=0 for the non-linear solver."""
#             lhs: np.ndarray = coeff_matrix.dot(solution) - rhs
#             return lhs

#         x0: np.ndarray = np.ones(len(self.molecules))
#         solution, _, ier, _ = optimize.fsolve(func, x0, full_output=True)
#         logger.debug("ier = %s", ier)
#         logger.debug("Solution = \n%s", solution)
#         pressures: np.ndarray = 10.0**solution

#         output: dict[str, float] = {
#             molecule: pressure
#             for (molecule, pressure) in zip(self.molecules, pressures)
#         }

#         logger.info("Solution is:")
#         for species, pressure in sorted(output.items()):
#             logger.info("    %s pressure (bar) = %f", species, pressure)

#         return output
