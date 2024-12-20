Installation
============

*Atmodeller* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

Quick install
-------------

With a Python environment active::

    pip install atmodeller

.. _developer_install:

Developer install
-----------------

First steps
^^^^^^^^^^^

The instructions below clone the main repository. However, if you prefer to fork the repository first, you should update the repository address accordingly before cloning.

- Navigate to a location on your computer and obtain the source code. To clone using ssh, where you must use a password-protected SSH key::

    git clone git@github.com:ExPlanetology/atmodeller.git
    cd atmodeller

Instructions for connecting to GitHub with SSH are available `here <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>`_.

- If you do not have SSH keys set up, instead you can clone using HTTPS::

    git clone https://github.com/ExPlanetology/atmodeller.git
    cd atmodeller

- Set up a Python environment to install *Atmodeller*. This can be achieved via a GUI or command line tools.

- Install *Atmodeller* into the environment using either (a) `Poetry <https://python-poetry.org>`_ or (b) `pip <https://pip.pypa.io/en/stable/getting-started/>`_.

Option 1: Poetry
^^^^^^^^^^^^^^^^

This requires that you have `Poetry <https://python-poetry.org>`_ installed:

.. code-block:: shell

    poetry install

Option 2: pip
^^^^^^^^^^^^^

Alternatively, use `pip`, where you can include the `-e` option if you want an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_.

.. code-block:: shell

    pip install .

If desired, you will need to manually install the dependencies for testing and documentation because these are automatically installed by Poetry but not by `pip`. See the additional group dependencies to install in `pyproject.toml`.

Additional information
^^^^^^^^^^^^^^^^^^^^^^

- See this `developer setup guide <https://gist.github.com/djbower/c66474000029730ac9f8b73b96071db3>`_ to set up your system to develop *Atmodeller* using `VS Code <https://code.visualstudio.com>`_ and `Poetry <https://python-poetry.org>`_.
- This `Gist <https://gist.github.com/djbower/e9538e7eb5ed3deaf3c4de9dea41ebcd>`_ provides further information about the interoperability of Poetry and pip.