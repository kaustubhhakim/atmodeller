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
     - molecules
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
     - molecules
     - Number of molecules in the melt
   * - dissolved_number_density
     - molecules m\ :math:`^{-3}`
     - Number density in the melt
   * - dissolved_ppmw
     - kg kg\ :math:`^{-1}` (ppm by weight)
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
     - molecules
     - Number of molecules in all reservoirs
   * - total_number_density
     - molecules m\ :math:`^{-3}`
     - Number density in all reservoirs
   * - volume_mixing_ratio
     - dimensionless
     - Volume mixing ratio (atmosphere)

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

Condensed species
-----------------

Species output have a dictionary key associated with the species name and its state of aggregation (e.g., H2O_l, S_cr).

.. list-table:: Outputs for condensed species
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - activity
     - dimensionless
     - Activity
   * - molar_mass
     - kg mole\ :math:`^{-1}`
     - Molar mass
   * - total_mass
     - kg
     - Mass
   * - total_moles
     - moles
     - Number of moles
   * - total_number
     - molecules
     - Number of molecules
   * - total_number_density
     - molecules m\ :math:`^{-3}`
     - Number density

Elements
--------

Element outputs have a dictionary key associated with the element name with an `element_` prefix (e.g., element_H, element_S).

.. list-table:: Outputs for elements
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
     - atoms
     - Number of atoms in the atmosphere
   * - atmosphere_number_density
     - atoms m\ :math:`^{-3}`
     - Number density in the atmosphere
   * - condensed_mass
     - kg
     - Mass in condensed species
   * - condensed_moles
     - moles
     - Number of moles in condensed species
   * - condensed_number
     - atoms
     - Number of atoms in condensed species
   * - condensed_number_density
     - atoms m\ :math:`^{-3}`
     - Number density in condensed species
   * - degree_of_condensation
     - dimensionless
     - Degree of condensation
   * - dissolved_mass
     - kg
     - Mass dissolved in the melt
   * - dissolved_moles
     - moles
     - Number of moles in the melt
   * - dissolved_number
     - atoms
     - Number of atoms in the melt
   * - dissolved_number_density
     - atoms m\ :math:`^{-3}`
     - Number density in the melt
   * - logarithmic_abundance
     - dimensionless
     - Logarithmic abundance
   * - molar_mass
     - kg mole\ :math:`^{-1}`
     - Molar mass
   * - total_mass
     - kg
     - Mass in all reservoirs
   * - total_moles
     - moles
     - Number of moles in all reservoirs
   * - total_number
     - atoms
     - Number of atoms in all reservoirs
   * - total_number_density
     - atoms m\ :math:`^{-3}`
     - Number density in all reservoirs
   * - volume_mixing_ratio
     - dimensionless
     - Volume mixing ratio (atmosphere)

Planet
------

The planet output has a dictionary key of `planet`.

.. list-table:: Outputs for planet
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - core_mass_fraction
     - kg kg\ :math:`^{-1}`
     - Mass fraction of iron core relative to total planet mass
   * - mantle_mass
     - kg
     - Mass of the silicate mantle
   * - mantle_melt_fraction
     - kg kg\ :math:`^{-1}`
     - Fraction of silicate mantle that is molten
   * - mantle_melt_mass
     - kg
     - Mass of molten silicate
   * - mantle_solid_mass
     - kg
     - Mass of solid silicate
   * - planet_mass
     - kg
     - Total mass of the planet
   * - surface_area
     - m\ :math:`^2`
     - Surface area at the surface radius
   * - surface_gravity
     - m s\ :math:`^{-2}`
     - Gravitational acceleration at the surface radius
   * - surface_radius
     - m
     - Radius of the planetary surface
   * - surface_temperature
     - K
     - Temperature at the planetary surface

Atmosphere
----------

The atmosphere output has a dictionary key of `atmosphere`.

.. list-table:: Outputs for atmosphere
   :widths: 25 25 50
   :header-rows: 1

   * - Name
     - Units
     - Description
   * - species_moles
     - moles
     - Number of moles of species
   * - species_number
     - molecules
     - Number of molecules of species
   * - species_number_density
     - molecules m\ :math:`^{-3}`
     - Number density of species
   * - mass
     - kg
     - Mass
   * - molar_mass
     - kg mole\ :math:`^{-1}`
     - Molar mass
   * - pressure
     - bar
     - Total pressure of the atmosphere
   * - volume
     - m\ :math:`^{3}`
     - Volume of the atmosphere
   * - element_moles
     - moles
     - Number of moles of elements
   * - element_number
     - atoms
     - Number of atoms of elements
   * - element_number_density
     - atoms m\ :math:`^{-3}`
     - Number density of elements
   * - temperature
     - K
     - Temperature of the atmosphere
  
Other output
------------

- constraints: Applied elemental mass and/or species fugacity constraints
- raw_solution: Raw solution from the solver, i.e. number densities and stabilities
- residual: Residuals of the reaction network and mass balance
- solver: Solver quantities