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