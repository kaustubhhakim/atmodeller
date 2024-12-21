Output
======

The `Output` class processes the solution to provide output, which can be in the form of a dictionary of arrays, Pandas dataframes, or an Excel file. The dictionary keys (or sheet names in the case of Excel output) provide a complete output of quantities.

Gas species
-----------

Species output have a dictionary key associated with the species name and its state of aggregation (e.g., CO2_g, H2_g).

All gas species
~~~~~~~~~~~~~~~

.. list-table:: Outputs for gas species
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - atmosphere_mass
     - kg
     - Mass in the atmosphere
   * - atmosphere_moles
     - moles
     - Number of moles in the atmosphere
   * - atmosphere_number
     - molecule count
     - Number of molecules in the atmosphere
   * - atmosphere_number_density
     - molecules m\ :math:`^{-3}`
     - Number density in the atmosphere
   * - dissolved_mass
     - kg
     - Mass dissolved in the melt
   * - dissolved_moles
     - moles
     - Number of moles in the melt
   * - dissolved_number
     - molecule count
     - Number of molecules in the melt
   * - dissolved_number_density
     - molecules m\ :math:`^{-3}`
     - Number density in the melt
   * - dissolved_ppmw
     - ppm by weight
     - Dissolved mass relative to melt mass
   * - fugacity
     - bar
     - Fugacity
   * - fugacity_coefficient
     - dimensionless
     - Fugacity relative to (partial) pressure
   * - molar_mass
     - kg mole\ :math:`^{-1}`
     - Molar mass
   * - pressure
     - bar
     - Partial pressure
   * - total_mass
     - kg
     - Mass in all reservoirs
   * - total_moles
     - moles
     - Number of moles in all reservoirs
   * - total_number
     - molecule count
     - Number of molecules in all reservoirs
   * - total_number_density
     - molecules m\ :math:`^{-3}`
     - Number density in all reservoirs
   * - volume_mixing_ratio
     - dimensionless
     - Volume mixing ratio

O2_g additional outputs
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Additional outputs for O2_g
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - log10dIW_1_bar
     - dimensionless
     - Log10 shift relative to the IW buffer at 1 bar
   * - log10dIW_P
     - dimensionless
     - Log10 shift relative to the IW buffer at the total pressure