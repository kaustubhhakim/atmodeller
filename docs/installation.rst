Installation
============

*Atmodeller* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

Python environment
------------------
It is recommended to install *Atmodeller* into a virtual environment, see for example: https://docs.python.org/3/library/venv.html.

1. Quick install
----------------

Use pip::

    pip install atmodeller

Downloading the source code is also recommended if you'd like access to the example notebooks in ``notebooks/``.

.. _developer_install:

2. Developer install
--------------------

2a. Fork and clone the repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to contribute to *Atmodeller* or want to work independently, it's recommended to first fork the repository to your own GitHub account. This gives you full control over your changes without affecting the original repository.

Follow these steps:

- Visit the main repository on GitHub: https://github.com/ExPlanetology/atmodeller
- Click the **Fork** button (usually in the top-right corner).
- Choose whether to fork to your personal account or an organization you belong to.

Once you have forked the repository, you can then proceed to clone **your forked copy** instead of the original.

When cloning your fork:

- If using SSH::

    git clone git@github.com:<your-username>/atmodeller.git
    cd atmodeller

- If using HTTPS::

    git clone https://github.com/<your-username>/atmodeller.git
    cd atmodeller

Replace ``<your-username>`` with your actual GitHub username. You can now work on your fork independently, create branches, and later submit a pull request to the main repository if desired.

.. note::

    It is also possible to simply clone the main repository directly, without forking, if you do not intend to make contributions or want to keep things simple.

2b. Install *Atmodeller*
^^^^^^^^^^^^^^^^^^^^^^^^

Use either Option 1: `Poetry <https://python-poetry.org>`_ or Option 2: `pip <https://pip.pypa.io/en/stable/getting-started/>`_.

Option 1: Poetry
^^^^^^^^^^^^^^^^

This requires that you have `Poetry <https://python-poetry.org>`_ installed:

To install the core *Atmodeller* package::

    poetry install

To include developer tools (e.g. testing and linting)::

    poetry install --extras "dev"

To also include documentation build dependencies::

    poetry install --extras "dev docs"

Option 2: pip
^^^^^^^^^^^^^

Alternatively, use pip. You may use the ``-e`` option for an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_::

    pip install -e .

To install additional dependencies:

- For development tools::

      pip install -e .[dev]

- For documentation tools::

      pip install -e .[docs]

- For both::

      pip install -e .[dev,docs]

.. note::

    Zsh treats square brackets (`[ ]`) as globbing characters. You must quote or escape them when using `pip`. Use either of the following::

        pip install -e '.[dev]'
        # or
        pip install -e .\[dev\]