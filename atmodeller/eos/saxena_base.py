@dataclass(kw_only=True)
class ShiSaxenaABC(FugacityModelABC):
    """Shi and Saxena fugacity model.

    Shi and Saxena, Thermodynamic modeling of the C-H-O-S fluid system, American Mineralogist,
    Volume 77, pages 1038-1049, 1992. See table 2, critical data of C-H-O-S fluid phases.

    http://www.minsocam.org/ammin/AM77/AM77_1038.pdf

    Attributes:
        Tc: Critical temperature in kelvin.
        Pc: Critical pressure in bar.
        a_coefficients: a coefficients (see paper).
        b_coefficients: b coefficients (see paper).
        c_coefficients: c coefficients (see paper).
        d_coefficients: d coefficients (see paper).
        scaling: Scaling is unity for bar.
        GAS_CONSTANT: Gas constant with the appropriate units.
    """

    Tc: float
    Pc: float
    # TODO: Different P0 to P0 used in Holland and Powell
    # Integration start for pressure.
    P0: float = field(init=False, default=1)  # 1 bar
    a_coefficients: tuple[float, ...]
    b_coefficients: tuple[float, ...]
    c_coefficients: tuple[float, ...]
    d_coefficients: tuple[float, ...]

    @abstractmethod
    def _get_compressibility_coefficient(
        self, temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation.

        Args:
            temperature: Temperature in kelvin.
            coefficients: Tuple of the coefficients a, b, c, or d.

        Returns
            The relevant coefficient.
        """
        ...

    def a(self, temperature: float) -> float:
        """a parameter.

        Args:
            Temperature: temperature in kelvin.

        Returns:
            a parameter.
        """
        return self._get_compressibility_coefficient(temperature, self.a_coefficients)

    def b(self, temperature: float) -> float:
        """b parameter.

        Args:
            Temperature: temperature in kelvin.

        Returns:
            b parameter.
        """
        return self._get_compressibility_coefficient(temperature, self.b_coefficients)

    def c(self, temperature: float) -> float:
        """c parameter.

        Args:
            Temperature: temperature in kelvin.

        Returns:
            c parameter.
        """
        return self._get_compressibility_coefficient(temperature, self.c_coefficients)

    def d(self, temperature: float) -> float:
        """d parameter.

        Args:
            Temperature: temperature in kelvin.

        Returns:
            d parameter.
        """
        return self._get_compressibility_coefficient(temperature, self.d_coefficients)

    def compressibility_parameter(self, temperature: float, pressure: float) -> float:
        """Compressibility parameter at temperature and pressure.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure in kbar.

        Returns:
            The compressibility parameter, Z.
        """

        Pr: float = self.reduced_pressure(pressure)
        Z: float = (
            self.a(temperature)
            + self.b(temperature) * Pr
            + self.c(temperature) * Pr**2
            + self.d(temperature) * Pr**3
        )

        return Z

    def reduced_pressure(self, pressure: float) -> float:
        """Reduced pressure.

        Args:
            pressure: Pressure in kbar.

        Returns:
            The reduced pressure, which is dimensionless.
        """
        return pressure / self.Pc

    @property
    def reduced_pressure0(self) -> float:
        """Reduced pressure.

        Args:
            pressure: Pressure in kbar.

        Returns:
            The reduced pressure, which is dimensionless.
        """
        return self.P0 / self.Pc

    def reduced_temperature(self, temperature: float) -> float:
        """Reduced temperature.

        Args:
            temperature: Temperature in kelvin.

        Returns:
            The reduced temperature, which is dimensionless.
        """
        return temperature / self.Tc

    def volume(self, temperature: float, pressure: float) -> float:
        """Volume.

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume.
        """
        Z: float = self.compressibility_parameter(temperature, pressure)
        volume: float = Z * self.GAS_CONSTANT * temperature / pressure

        return volume

    def volume_integral(self, temperature: float, pressure: float) -> float:
        """Volume integral (V dP).

        Args:
            temperature: Temperature in kelvin.
            pressure: Pressure.

        Returns:
            Volume integral.
        """
        Pr: float = self.reduced_pressure(pressure)
        P0r: float = self.reduced_pressure0
        volume_integral: float = (
            (
                self.a(temperature) * np.log(Pr / P0r)
                + self.b(temperature) * (Pr - P0r)
                + (1.0 / 2) * self.c(temperature) * (Pr**2 - P0r**2)
                + (1.0 / 3) * self.d(temperature) * (Pr**3 - P0r**3)
            )
            * self.GAS_CONSTANT
            * temperature
        )

        return volume_integral


@dataclass(kw_only=True)
class ShiSaxenaLowPressure(ShiSaxenaABC):
    """Low pressure (< 1 kbar)."""

    def _get_compressibility_coefficient(
        self, temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation.

        Args:
            temperature: Temperature in kelvin.
            coefficients: Tuple of the coefficients a, b, c, or d.

        Returns
            The relevant coefficient.
        """
        Tr: float = self.reduced_temperature(temperature)
        coefficient: float = (
            coefficients[0]
            + coefficients[1] / Tr
            + coefficients[2] / Tr ** (3 / 2)
            + coefficients[3] / Tr**3
            + coefficients[4] / Tr**4
        )

        return coefficient


@dataclass(kw_only=True)
class ShiSaxenaHighPressure(ShiSaxenaABC):
    """High pressure (>=1 kbar)."""

    def _get_compressibility_coefficient(
        self, temperature: float, coefficients: tuple[float, ...]
    ) -> float:
        """General form of the coefficients for the compressibility calculation.

        Args:
            temperature: Temperature in kelvin.
            coefficients: Tuple of the coefficients a, b, c, or d.

        Returns
            The relevant coefficient.
        """
        Tr: float = self.reduced_temperature(temperature)
        coefficient: float = (
            coefficients[0]
            + coefficients[1] * Tr
            + coefficients[2] / Tr
            + coefficients[3] * Tr**2
            + coefficients[4] / Tr**2
            + coefficients[5] * Tr**3
            + coefficients[6] / Tr**3
            + coefficients[7] * np.log(Tr)
        )

        return coefficient
