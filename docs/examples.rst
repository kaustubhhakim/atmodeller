.. _Examples:

Examples
========

*Atmodeller* contains three sub-packages that provide thermodynamic data, solubility laws, and real gas equations of state (EOS).

Species and thermodynamic data
------------------------------

The species available in *Atmodeller* can be found in the `thermodata` subpackage, where the suffix of the dictionary key describes the *states of aggregation* in accordance with :cite:t:`Cha98`.

.. code-block:: python

    from atmodeller.thermodata import get_thermodata

    # Get all available data
    thermodata = get_thermodata()
    thermodata.keys()

    # For example, get CO2 gas
    CO2_g = thermodata["CO2_g"]
    # Compute the Gibbs energy relative to RT at 2000 K
    CO2_g.get_gibbs_over_RT(2000.0)
    # Compute the composition
    CO2_g.composition
    # Etc., other methods are available to compute other quantities

Solubility
----------

Solubility laws are available in the `solubility` subpackage.

.. code-block:: python

    from atmodeller.solubility import get_solubility_models

    # Get all available solubility models
    sol_models = get_solubility_models()
    sol_models.keys()

    CO2_basalt = sol_models["CO2_basalt_dixon95"]
    # Compute the concentration at fCO2=0.5 bar, 1300 K, and 1 bar
    # Note that fugacity is the first argument and others are keyword only
    CO2_basalt.concentration(0.5, temperature=1300, pressure=1)

Real gas EOS
------------

Real gas equations of state are available in the `eos` subpackage.

.. code-block:: python

   from atmodeller.eos import get_eos_models

    # Get all available EOS models
    eos_models = get_eos_models()
    eos_models.keys()

    # Get a CH4 model
    CH4_eos_model = eos_models['CH4_beattie_holley58']
    # Compute the fugacity at 800 K and 100 bar
    CH4_eos_model.fugacity(800, 100)
    # Compute the compressibility factor at the same conditions
    CH4_eos_model.compressibility_factor(800, 100)
    # Etc., other methods are available to compute other quantities

Model with mass constraints
---------------------------

A common scenario is to calculate how volatiles partition between a magma ocean and an atmosphere when the total elemental abundances are constrained. `Planet()` defaults to a molten Earth, but the planetary parameters can be changed using input arguments.

.. code-block:: python

    from atmodeller import Species, InteriorAtmosphere, Planet, earth_oceans_to_hydrogen_mass
    from atmodeller.solubility import get_solubility_models
    import numpy as np

    solubility_models = get_solubility_models()

    H2_g = Species.create_gas("H2_g")
    H2O_g = Species.create_gas("H2O_g", solubility=solubility_models["H2O_peridotite_sossi23"])
    O2_g = Species.create_gas("O2_g")

    species = (H2_g, H2O_g, O2_g)

    # Planet has input arguments that you can change. See the class documentation.
    planet = Planet()
    interior_atmosphere = InteriorAtmosphere(species)

    oceans = 1
    h_kg = earth_oceans_to_hydrogen_mass(oceans)
    o_kg = 6.25774e20
    mass_constraints = {
        "H": h_kg,
        "O": o_kg,
    }

    # If you do not specify an initial solution guess then a default will be used
    # Initial solution guess number density (molecules/m^3)
    # initial_log_number_density = 50 * np.ones(len(species), dtype=np.float_)

    interior_atmosphere.initialise_solve(
        planet=planet,
        # initial_log_number_density=initial_log_number_density,
        mass_constraints=mass_constraints,
    )
    output = interior_atmosphere.solve()

    # Quick look at the solution
    solution = output.quick_look()

    # Get complete solution as a dictionary
    solution_asdict = output.asdict()
    logger.info(solution_asdict)

    # Get the complete solution as dataframes
    # solution_dataframes = output.to_dataframes()

    # Write the complete solution to Excel
    # output.to_excel("example_single")

Model with mixed constraints
----------------------------

Sometimes it is convenient to define the oxygen fugacity as a constraint on the system.

.. code-block:: python

    pass