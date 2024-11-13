Installation
============

*Atmodeller* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

Quick install for Mac
----------------------

The instructions are given in terms of terminal commands for a Mac, but equivalents exist for other systems (see Linux instructions below).

Navigate to a location on your computer and obtain the *Atmodeller* source code::

    git clone git@github.com:ExPlanetology/atmodeller.git
    cd atmodeller

The basic procedure is to install *Atmodeller* into an environment. For example, if you are using a Conda distribution to create Python environments (e.g. `Anaconda <https://www.anaconda.com/download>`_), create a new environment to install *Atmodeller*. *Atmodeller* requires Python >= 3.10::

    conda create -n atmodeller python
    conda activate atmodeller

Install Atmodeller into the environment, where you can include the ``-e`` option if you want an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_::

    pip install .

You can load the tutorials by specifying the path to the Jupyter notebook. For example, the first tutorial can be loaded with::

    jupyter notebook notebooks/1_basics.ipynb

You may need to *trust* the notebook before it will run.

Quick Install for Linux systems
-------------------------------

Installing *Atmodeller* on a Linux-based system is very similar to the installation for Mac. 

Navigate to a location on your computer and obtain the *Atmodeller* source code via Github::

    git clone git@github.com:ExPlanetology/atmodeller.git
    cd atmodeller

Create an anaconda environment called atmodeller::

    conda create -n atmodeller python
    conda activate atmodeller 

Note that *Atmodeller* requires Python >= 3.10, so check that your pip installer is up-to-date::

    pip install --upgrade pip

Install *Atmodeller* into your environment::

    pip install . 

If you would like to set up your system to develop, then you will also want to install Poetry in your anaconda environment::

    pip install poetry 

Also see below for additional instructions for developer Installation

If you would like to work with *Atmodeller* in VS Code, open the atmodeller directory in VS Code and activate your anaconda environment in the terminal within VS Code

If you run into issues running the Jupyter notebooks, check that you have the necessary packages installed in your environment (e.g., numpy), installing Intel's Math Kernel Library (MKL) for Python should do the trick::
    
    pip install mkl 


Developer install
-----------------

See this `developer setup guide <https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3>`_ to set up your system to develop *Atmodeller* using `VS Code <https://code.visualstudio.com>`_ and `Poetry <https://python-poetry.org>`_.

Tutorial
--------

Several Jupyter notebook tutorials are provided in `notebooks/`.

Tests
-----

You can confirm that all tests pass by running ``pytest`` in the root directory of *Atmodeller*. Please add more tests if you add new features. Note that ``pip install .`` in the *Quick install* instructions will not install ``pytest`` so you will need to install ``pytest`` separately.