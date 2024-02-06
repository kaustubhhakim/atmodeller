Troubleshooting
===============

1. JANAF data
-------------

*Atmodeller* downloads and caches JANAF data, which means that the first time you construct and solve a model using JANAF data you must have access to the internet to download the thermodynamic data for each species that you include. Subsequent models will run faster and will not require internet access, unless you include additional species that are not already cached, in which case these will also need to be downloaded.

Also note that the cached data is stored in the `Thermochem <https://thermochem.readthedocs.io/en/latest/>`_ package directory, which means you will need to download JANAF data every time you run *Atmodeller* in a Python environment that doesn't already have data cached.

2. Solving interior-atmosphere systems
--------------------------------------

*Atmodeller* assembles and solves a system of non-linear equations and is therefore subject to the same considerations when solving any system of non-linear equations. *Atmodeller* uses `scipy.optimize.root <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html>`_ to select and drive the behaviour of the solver. Arguments can be passed to this function by specifiying arguments when you call ``solve()`` on an ``InteriorAtmosphereSystem``. The default solver is `hybr <https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html#optimize-root-hybr>`_, but you can experiment with other solvers and other parameters such as the tolerance ``tol``.

The following provides some guidance if you are facing challenges with obtaining a solution to your interior-atmosphere system:

1. Confirm that a solution exists. Although *Atmodeller* allows you to build a system of arbitrary user-imposed constraints (pressure, fugacity, mass, etc.) this does not necessarily mean that there is a physical solution to the system. If *Atmodeller* cannot find a solution it might simply be because a solution does not exist for your imposed constraints. In this regard, it can help to impose a total pressure constraint to first uncover the general behaviour of the solution before imposing pressure or fugacity constraints on individual species.

2. Confirm that the species chosen for your reaction network are appropriate for the pressure and temperature conditions of interest. *Atmodeller* does not perform any internal tests to determine whether or not the species you have chosen are thermodynamically stable at the specified conditions. Hence prior knowledge, intuition, or calculations with a Gibbs minimiser are required to guide the choice of species. Related, if the dynamic range of the species abundances is too large then you may encounter problems.

3. Numerical overflow can occur when the solver steps to a region of parameter space that causes a species' pressure :math:`p_i` to become too large. This is because the reaction network component of the non-linear system is formulated in terms of :math:`\log10(p_i)`, but :math:`p_i` is required for mass balance and hence :math:`10^{\log10(p_i)}` is computed. In this regard, reducing ``factor`` (`see here <https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html#optimize-root-hybr>`_) can prevent the solver from stepping too far.

4. Simplify your system by systematically removing species and/or removing non-linear dependences. For example, oxygen fugacity buffers, real gas equations of state, and solubility relations can depend on the total pressure, which adds an additional pressure coupling between the system of equations compared to simpler ideal-gas only systems. Swapping mass constraints for pressure constraints can also help, as long as the pressure constraints are compatible with a solution (see 1 above). In short, starting with a simple ideal-gas only reaction network and adding incremental complexity is a good approach narrow down reasons why a solution cannot be found.

5. Choose an appropriate initial solution. Providing an initial solution closer to the true solution will improve the chances that the solver can locate and converge to the correct (global) minimum. You can specify the initial solution using the ``initial_solution`` argument of ``solve()`` on an ``InteriorAtmosphereSystem``. If you are systematically iterating over a set of parameters, you can often use the previous solution to the system as the initial solution for the next system with perturbed parameters.

6. The notebook `notebooks/3_monte_carlo.ipynb` shows several examples of training a regressor to improve the initial solution and therefore improve the performance of the solver.