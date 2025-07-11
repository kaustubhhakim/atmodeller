.. _InstallationFile:

Installation
============

*Atmodeller* is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux).

Python environment
------------------
It is recommended to install *Atmodeller* in a virtual environment, whether you proceed with a quick install or a full developer setup. This helps isolate dependencies and maintain reproducibility.

For more information, see the `Python documentation on venv <https://docs.python.org/3/library/venv.html>`_.

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

If you plan to contribute to *Atmodeller* or want to work independently, it's recommended to first fork the repository to your personal account or to an organisation you belong to. This gives you full control over your changes and the ability to manage your own branches.

Follow these steps:

- Visit the main repository on GitHub: https://github.com/ExPlanetology/atmodeller
- Click the **Fork** button (usually in the top-right corner of the page).
- Choose whether to fork the repository to your personal account or to an organisation account.

After forking the repository, you should **clone your fork** (not the original) to begin development.

Cloning your fork:

- Using SSH::

    git clone git@github.com:<your-account>/atmodeller.git
    cd atmodeller

- Using HTTPS::

    git clone https://github.com/<your-account>/atmodeller.git
    cd atmodeller

Replace ``<your-account>`` with your actual **GitHub username or organisation name**. You can now work on your fork independently, create branches, and make changes as needed.

To keep your fork in sync with the original repository---or to submit changes via pull requests---you can follow the instructions in the `GitHub documentation <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork>`_ on configuring a remote upstream. This setup allows you to fetch updates from the main repository and integrate them into your fork.

.. note::

    You can also clone the main repository directly without forking, but this approach provides less flexibility and does not allow you to submit pull requests unless you have write access to the main repository.

2b. Install *Atmodeller*
^^^^^^^^^^^^^^^^^^^^^^^^

To install the package, you may use a Python project/package manager such as `uv <https://docs.astral.sh/uv>`_, `poetry <https://python-poetry.org>`_, or the standard `pip <https://pip.pypa.io/en/stable/getting-started/>`_ tool. These tools manage dependencies using the project's ``pyproject.toml`` file.

Recommended: use ``uv`` or ``poetry`` or similar for reproducible and locked environments.  
Alternatively, ``pip`` can be used for simpler workflows.

Option 1: uv (recommended)
^^^^^^^^^^^^^^^^^^^^^^^^^^

This requires ``uv`` to be installed.

Create a virtual environment (if you haven't already), typically at the uppermost level of the source code repository::

    uv venv

Activate the virtual environment::

    source .venv/bin/activate

Install the *Atmodeller* package, including development tools::

    uv sync

Optional extras:

- To install documentation dependencies::

      uv sync --extra docs

- To install everything (core + dev + docs)::

      uv sync --extra docs

Option 2: pip
^^^^^^^^^^^^^

Create a virtual environment (if you haven't already), typically at the uppermost level of the source code repository.  
Make sure that the Python version used is compatible with *Atmodeller*'s requirements::

    python -m venv .venv

Activate the virtual environment::

    source .venv/bin/activate

You may use the ``-e`` option for an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_::

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