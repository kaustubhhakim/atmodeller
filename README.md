# Advanced preview access (beta testing)

You have been granted advanced preview access of *Atmodeller* and therefore the development team requests the following:

1. Do not distribute or share any parts of the code. Ask the development team directly if you would like other group members or colleagues to have access to the code.
2. If you are planning to use *Atmodeller* to prepare results for a manuscript, you are requested to discuss this with the development team prior to commencing. This is to avoid any potential overlaps with ongoing projects within the development team. The development team may request that one or several members of the development team are included as authors on your manuscript.
3. Involving members of the development team from the outset of your *Atmodeller* projects is always encouraged, since we can help you to get the best performance from the code for your particular applications.
4. Please provide feedback to the development team. We want to improve *Atmodeller* with your assistance!

# About
*Atmodeller* is a Python package that computes the partitioning of volatiles between a planetary atmosphere and its rocky interior. It will be released under The GNU General Public License v3.0 or later once beta testing is complete and the first papers are submitted and/or published.

Authors:

- Dan J. Bower (main developer)
- Maggie A. Thompson
- Meng Tian
- Paolo Sossi

# Citation

If you use *Atmodeller* please cite (prior to manuscript submission, check back to see if this reference has been updated):

- Bower, D.J, Thompson, M. A., Tian, M., and Sossi P.A. (2024), Diversity of rocky planet atmospheres in the C-H-O-N-S-Cl system with interior dissolution, The Astrophysical Journal, submitted.

# Installation

*Atmodeller* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

## Quick install

If you want a GUI way of installing *Atmodeller*, particularly if you are a Windows or Spyder user, see [here](https://gist.github.com/djbower/c82b4a70a3c3c74ad26dc572edefdd34). Otherwise, the instructions below should work to install *Atmodeller* using the terminal on a Mac or Linux system.

Navigate to a location on your computer and obtain the *Atmodeller* source code:

    git clone git@github.com:ExPlanetology/atmodeller.git
    cd atmodeller

The basic procedure is to install *Atmodeller* into an environment. For example, if you are using a Conda distribution to create Python environments (e.g. [Anaconda](https://www.anaconda.com/download)), create a new environment to install *Atmodeller*. *Atmodeller* requires Python >= 3.10:

    conda create -n atmodeller python
    conda activate atmodeller

Install *Atmodeller* into the environment. The preference is to use [Poetry](https://python-poetry.org) because it allows greater flexibility and control over dependency management, and this is actually required if you want to install the dependencies for testing and documentation that are unfortunately not yet supported by `pip`. However, you can install the main *Atmodeller* package using pip as follows, where you can include the `-e` option if you want an [editable install ](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).


    pip install .


You can load the tutorials by specifying the path to the Jupyter notebook. For example, the first tutorial can be loaded with:


    jupyter notebook notebooks/1_basics.ipynb


You may need to *trust* the notebook before it will run.

## Developer install

See this [developer setup guide](https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3) to set up your system to develop *Atmodeller* using [VS Code](https://code.visualstudio.com) and [Poetry](https://python-poetry.org).

## Documentation

Documentation will eventually be available on readthedocs, but for the time being you can compile (and contribute if you wish) to the documentation in the `docs/` directory. To compile the documentation you will need to use Poetry and the option `--with docs` when you run `poetry install`. See [here](https://python-poetry.org/docs/managing-dependencies/) for further information.

## Tutorial

Several Jupyter notebook tutorials are provided in `notebooks/`.

## Tests

You can confirm that all tests pass by running `pytest` in the root directory of *Atmodeller*. Please add more tests if you add new features. Note that `pip install .` in the *Quick install* instructions will not install `pytest` so you will need to install `pytest` into the environment separately.
