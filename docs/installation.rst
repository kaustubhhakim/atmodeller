.. _InstallationFile:

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

If you plan to contribute to *Atmodeller* or want to work independently, it's recommended to first fork the repository to your own GitHub account. This gives you full control over your changes and the ability to manage your own private branches.

Follow these steps:

- Visit the main repository on GitHub: https://github.com/ExPlanetology/atmodeller
- Click the **Fork** button (typically located in the top-right corner of the page).
- Choose whether to fork the repository to your personal account or to an organization you belong to.

After forking the repository, you should clone **your forked copy** (not the original) to begin development.

Cloning your fork:

- Using SSH::

    git clone git@github.com:<your-username>/atmodeller.git
    cd atmodeller

- Using HTTPS::

    git clone https://github.com/<your-username>/atmodeller.git
    cd atmodeller

Replace ``<your-username>`` with your actual GitHub username. You can now work on your fork independently, create branches, and make changes as needed.

To keep your fork in sync with the original repository---or to submit changes via pull requests---you can follow the instructions in the `GitHub documentation <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork>`_ on configuring a remote upstream. This setup allows you to fetch updates from the main repository and integrate them into your fork.

.. note::

    You can also clone the main repository directly without forking, but this approach provides less flexibility and does not allow you to submit pull requests unless you have write access to the main repository.

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