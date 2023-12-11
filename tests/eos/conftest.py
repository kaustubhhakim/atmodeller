import pytest
from pytest import approx

from atmodeller.eos.interfaces import RealGasABC

# Tolerances to compare the test results with target output.
RTOL: float = 1.0e-8
ATOL: float = 1.0e-8


class CheckValues:
    """Helper class with methods to check and confirm values."""

    @staticmethod
    def compressibility(
        temperature: float,
        pressure: float,
        fugacity_model: RealGasABC,
        expected: float,
        *,
        rtol=RTOL,
        atol=ATOL
    ) -> None:
        """Checks the compressibility parameter

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure
            fugacity_model: Fugacity model
            expected: The expected value
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        compressibility: float = fugacity_model.compressibility_parameter(temperature, pressure)

        assert compressibility == approx(expected, rtol, atol)

    @staticmethod
    def fugacity_coefficient(
        temperature: float,
        pressure: float,
        fugacity_model: RealGasABC,
        expected: float,
        *,
        rtol=RTOL,
        atol=ATOL
    ) -> None:
        """Checks the fugacity coefficient.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure
            fugacity_model: Fugacity model
            expected: The expected value
            rtol: Relative tolerance
            atol: Absolute tolerance
        """

        fugacity_coeff: float = fugacity_model.fugacity_coefficient(temperature, pressure)

        assert fugacity_coeff == approx(expected, rtol, atol)

    @staticmethod
    def volume(
        temperature: float,
        pressure: float,
        fugacity_model: RealGasABC,
        expected: float,
        *,
        rtol=RTOL,
        atol=ATOL
    ) -> None:
        """Checks the volume.

        Args:
            temperature: Temperature in kelvin
            pressure: Pressure
            fugacity_model: Fugacity model
            expected: The expected value
        """

        volume: float = fugacity_model.volume(temperature, pressure)

        assert volume == approx(expected, rtol, atol)


@pytest.fixture(scope="module")
def check_values():
    return CheckValues()
