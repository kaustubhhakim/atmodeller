.. _TroubleshootingFile:

Troubleshooting
===============

Pipeline integration
--------------------
:ref:`Installation <InstallationFile>` of *Atmodeller* is straightforward, but some package dependencies may conflict with others in a larger Python pipeline. In general, *Atmodeller* has relatively flexible requirements, though the JAX-related dependencies are typically pinned to recent versions to ensure optimal performance and access to the latest features.

If you encounter compatibility issues when integrating *Atmodeller* into your pipeline, please contact the development team for assistance.

The solver fails
----------------

*Atmodeller* assembles and solves systems of non-linear equations, and is therefore subject to all the usual considerations that arise when dealing with such systems. Under the hood, it uses the JAX-based solver library `optimistix <https://github.com/patrick-kidger/optimistix>`_, whose documentation is available `here <https://docs.kidger.site/optimistix>`_. Pay particular attention to the FAQ section. 

Because the solution process is compiled and optimized by JAX, it can be challenging to inspect the solver's internal workings without using `JAX debugging tools <https://jax.readthedocs.io/en/latest/debugging.html>`_. However, the nature of the failure can often provide insight into the underlying issue:

- A solution cannot be found
- The solver detects an NaN (Not a Number) or infinity in a function

A solution cannot be found
~~~~~~~~~~~~~~~~~~~~~~~~~~

- The initial guess may not be sufficient to reach convergence. Try modifying the initial guess---especially if you're iterating systematically over a range of parameters. In such cases, reusing the previous solution as the initial guess for the next step can improve convergence.
  
- Ensure that a solution physically exists for the system you've constructed. While *Atmodeller* allows arbitrary user-defined constraints (e.g., fugacity, mass balance), this does not guarantee a valid solution. If no solution is found, it may be due to over-constraining or conflicting conditions. To diagnose this, simplify the system by reducing the number of species and/or constraints. If a solution can be found for a simpler case, gradually reintroduce complexity to identify when the system becomes unsolvable.

- *Atmodeller* uses a bounded solver, with upper and lower limits specified in the package's `__init__`. These bounds are generally generous, but it is still possible that a solution lies outside them---particularly when modelling large atmospheres in the absence of solubility. Consider adjusting or inspecting the bounds if appropriate.

The solver detects an NaN
~~~~~~~~~~~~~~~~~~~~~~~~~

- During solution, *Atmodeller* evaluates various functions involving thermodynamic data, solubility laws, and real gas equations of state. These functions are not guaranteed to return finite values across all inputs. Although the bounded solver helps mitigate this, NaNs may still arise if any function behaves poorly for the given inputs. In such cases, iteratively simplifying the system---removing or isolating specific solubility laws or EOS---can help identify the problematic function.

- Thermodynamic data in *Atmodeller* are typically valid between 200 K and 6000 K, though in some cases extend to 20,000 K. Ensure that your calculations are performed within these valid temperature ranges to avoid undefined behaviour.

Additional information
~~~~~~~~~~~~~~~~~~~~~~

- Several debugging tools are included in the package's `__init__`, though they may be commented out by default. Uncomment these as needed to assist with tracing specific issues during development or testing.