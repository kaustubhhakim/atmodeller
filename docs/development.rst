Developer's Guide
=================

Introduction
------------

Community development of the code is strongly encouraged so please contact the lead developer if you or your team would like to contribute. *Atmodeller* uses JAX so familiarise yourself with `How to think in JAX <https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html>`_ and `JAX - The Sharp Bits <https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_, as well as other resources offered on the JAX site and the web in general. You are welcome to enquire about the reasoning behind the structure and design of the code with the development team.

Installation
------------

For instructions on developer installation, see :ref:`developer_install`. Note that you may need to install the group dependencies for dev and docs manually if you are using `pip` to install.

Pre-commit
----------

Before issuing a pull request to the main repository, run pre-commit

.. code-block:: shell

    pre-commit run --all-files

Documentation
-------------

Documentation will eventually be available on Read the Docs, but for the time being, you can compile (and contribute, if you wish) to the documentation in the `docs/` directory. If you are using Poetry you need to use the `--with docs` option when you run `poetry install`. See `here <https://python-poetry.org/docs/managing-dependencies/>`_ for further information.

Once the necessary dependencies are installed to compile the documentation, you can navigate into the `docs/` directory and run:

.. code-block:: shell

    make html

And/or to compile the documentation as a PDF where `latexpdf` must be available on your system:

.. code-block:: shell

    make latexpdf

This will build the documentation in the appropriately named subdirectory in `_build`.

Tests
-----

You can confirm that all tests pass by running::
    
    pytest
    
in the root directory of *Atmodeller*. Please add a corresponding unit test for new features that you develop.