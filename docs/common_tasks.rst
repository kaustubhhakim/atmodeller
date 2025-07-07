Common Tasks
============

Add thermodynamic data
----------------------

Thermodynamic data are contained in the file `nasa_glenn_coefficients.txt`, which is within the `thermodata` sub-package in the `data` directory. These data follow the formulation of :cite:t:`MZG02` (https://ntrs.nasa.gov/citations/20020085330), in which functions for heat capacity, enthalpy, and entropy are encoded using coefficients. You can extend the thermodynamic data (and hence available species in *Atmodeller*) by adding species' data from :cite:t:`MZG02`. For consistency *Atmodeller* uses Hill notation for chemical formulae.

In some cases, :cite:t:`MZG02` may not list the species you wish you add, or alternatively coefficients are not provided for the desired temperature range. In these scenarios you must appropriately fit the thermodynamic data for the species and/or temperature range of interest, whilst ensuring consistency with the standard state (1 bar), the reference temperature (273.15 K) and the definition of reference enthalpy :cite:p:`{See Introduction in }MZG02`. Furthermore, the methods and docstrings associated with the classes in the ``thermodata.core`` module provide guidance for relating thermodynamic data to the JANAF tables :cite:p:`Cha98`.

Add a solubility law
--------------------

Solubility laws are encapsulated in the `solubility` sub-package, with the base class ``Solubility`` located in ``solubility.core``. To define a custom solubility model, inherit from the base class and implement the ``concentration`` method. Private modules (indicated by a leading underscore) separate the solubility laws based on speciation, which is determined by the elemental composition. The ``core`` module also includes concrete classes for commonly used solubility laws, such as the power law, which can be used directly without the need to implement a custom class. Once a new solubility law is added, it should be imported into the ``library`` and referenced in the dictionary returned by ``get_solubility_models``. After that, the solubility law can be accessed and used.

Add a real gas EOS
------------------

Guidance coming soon.