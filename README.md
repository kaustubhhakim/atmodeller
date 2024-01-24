# About
Atmodeller is a Python package that computes the partitioning of volatiles between a planetary atmosphere and its rocky interior. It is released under The GNU General Public License v3.0 or later.

Authors:

- Dan J. Bower (main developer)
- Maggie A. Thompson
- Meng Tian
- Paolo Sossi

# Citation

If you use Atmodeller please cite:

- Bower, D.J, Thompson, M. A., Tian, M., and Sossi P.A. (2024), Unravelling the atmospheres of rocky planets with solubility and real gas equations of state, Astrophysical Journal, submitted.

## Installation

Atmodeller is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

### Quick install

The instructions are given in terms of terminal commands for a Mac, but equivalents exist for other systems.

Navigate to a location on your computer and obtain the Atmodeller source code:

```
git clone git@github.com:ExPlanetology/atmodeller.git
cd atmodeller
```

The basic procedure is to install Atmodeller into a virtual environment. For example, if you are using a variant of Conda to create Python environments ([Anaconda](https://www.anaconda.com/download) is recommended), create a new environment to install Atmodeller. Atmodeller requires Python >= 3.10:

```
conda create -n atmodeller python
conda activate atmodeller
```

Install Atmodeller into the virtual environment, where you can include the `-e` option if you want an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html):

```
pip install .
```

You can load the tutorials by specifying the path to the Jupyter notebook. For example, the first tutorial can be loaded with

```
jupyter notebook notebooks/1_basics.ipynb
```

You may need to *trust* the notebook before it will run.

### Developer install

See [this setup guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) to setup your system to develop *Atmodeller* using [VS Code](https://code.visualstudio.com) and [Poetry](https://python-poetry.org).

## Tutorial

Several Jupyter notebook tutorials are provided in `notebooks/`.

## Tests

You can confirm that all tests pass by running `pytest` in the root directory. Please add more tests if you add new features. Note that `pip install .` in the *Quick install* instructions will not install `pytest` so you will need to install `pytest` separately.

## User tips and troubleshooting

### 1. JANAF data

*Atmodeller* downloads and caches JANAF data, which means that the first time you construct and solve a model (using JANAF data) you must have access to the internet to download the thermodynamic data for each species you include. Subsequent models will run faster and will not require internet access, unless you add additional species that are not already cached, in which case these will also need to be downloaded.

Also note that the cached data is stored in the [Thermochem](https://thermochem.readthedocs.io/en/latest/) package directory, which means you will need to download JANAF data every time you run *Atmodeller* in a Python environment that doesn't already have data cached.


### 2. Solving interior-atmosphere systems

At its core, *Atmodeller* assembles and solves a system of non-linear equations and is therefore subject to the same considerations when solving any system of non-linear equations. *Atmodeller* uses `scipy.optimize.root` to select and drive the behaviour of the solver. Arguments can be passed to this function by specifiying arguments when you call `solve()` on an instance of `InteriorAtmosphereSystem`. The default solver is `hybr`, but you can experiment with other solvers as well as the tolerance parameter `tol`.

The following provides some guidance if you are facing challenges with obtaining a solution to your interior-atmosphere system:

1. Confirm that a solution exists. Although *Atmodeller* allows you to build a system of arbitrary user-imposed constraints (pressure, fugacity, mass, etc.) this does not necessarily mean that there is a physical solution to the system. If *Atmodeller* cannot find a solution it might simply be because a solution does not exist for your imposed constraints. In this regard, it can help to impose a total pressure constraint to first uncover the general behaviour of the solution before imposing pressure or fugacity constraints on individual species.

1. Confirm that the species chosen for your reaction network are appropriate for the pressure and temperature conditions of interest. We recall that *Atmodeller* does not perform any internal tests to determine whether or not the species you have chosen are thermodynamically stable at the specified conditions. Hence prior knowledge, intuition, or calculations with a Gibbs minimiser are required to guide the choice of species. Related, if the dynamic range of the species abundances is too large then you may encounter problems.

1. Numerical overflow can occur when the solver steps to a region of parameter space that causes the pressure to become too large. This is because the reaction network component of the non-linear system is formulated in terms of log10(pressure), but the actual pressure is required for mass balance and hence 10**log10(pressure) is computed. In this regard, reducing the `factor` parameter can prevent the solver from stepping too far [(read more)](https://docs.scipy.org/doc/scipy/reference/optimize.root-hybr.html).

1. Simplify your system by systematically removing species and/or removing non-linear dependences. For example, oxygen fugacity buffers, real gas equations of state, and solubility relations can depend on the total pressure, which adds an additional pressure coupling between the system of equations compared to simpler ideal-gas only systems. Swapping mass constraints for pressure constraints can also help, as long as the pressure constraints are compatible with a solution (see point 1 above). In short, starting with a simple ideal-gas only reaction network and adding incremental complexity is a good approach narrow down reasons why a solution cannot be found.

1. Choose an appropriate initial guess. Providing an initial guess closer to the true solution will improve the chances that the solver can locate and converge to the correct (global) minimum. You can specify the initial guess using the `initial_solution` argument of `solve()` on an instance of `InteriorAtmosphereSystem`. If you are systematically iterating over a set of parameters, you can often use the previous solution to the system as the initial guess for the next system with perturbed parameters.

1. The notebook `notebooks/3_monte_carlo.ipynb` shows several examples of training a regressor to improve the initial solution guess and therefore improve the performance of the solver.
