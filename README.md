# atmodeller
A Python package for computing the partitioning of volatiles between a planetary atmosphere and its interior.

See the Jupyter notebook tutorials in the package directory `docs/`.

You are encouraged to fork this repository or create your own branch to add new features. You will need to issue a pull request to be able to merge your code into the main branch. Follow the [Google Python style guide](https://google.github.io/styleguide/pyguide.html) and familiarise yourself with [SOLID principles](https://realpython.com/solid-principles-python/).

## Installation

### Development

The instructions below follow <https://denisrosset.github.io/2022-planets/> for using VSCode and Poetry on a Mac. Formally you don't have to use VSCode or Poetry, but using them makes it easier to develop *atmodeller* as a community.

1. Install [VSCode](https://code.visualstudio.com)
1. In VSCode you are recommended to install the following extensions:
	- Python Pylance
	- Region Viewer
	- Todo Tree
	- Jupyter
	- Code Spell Checker
	- IntelliCode
1. Install [Poetry](https://python-poetry.org)
1. Clone this repository (*atmodeller*) to a local directory
1. In VSCode, go to *File* and *Open Folder...* and select the *atmodeller* directory
1. We want to set up a virtual Python environment in the root directory of *atmodeller*. An advantage of using a virtual environment is that it remains completely isolated from any other Python environments on your system (e.g. Conda or otherwise). You must have a Python interpreter available to build the virtual environment according to the dependency in `pyproject.toml`, which could be a native version on your machine or a version from a Conda environment that is currently active. You only need a Python binary so it is not required to install any packages. You can create a virtual environment by using the terminal in VSCode, where you may need to update `python` to reflect the location of the Python binary file. This will create a local Python environment in the `.venv` directory:
	
    ```
    python -m venv .venv
    ```
1. Open a new terminal window in VSCode and VSCode should recognise that you have a virtual environment in .venv, and load this environment automatically. You should see `(.venv)` as the prefix in the terminal prompt.
1. Install the project using poetry to install all the required Python package dependencies:

    ```
    poetry install
    ```

To ensure that all developers are using the same settings there is a `settings.json` file in the `.vscode` directory. These settings will take precedence over your user settings for this project only.


### Tarball

If you prefer, I can send you a package tarball that you can install using the terminal. In the long-term I will set up a package repository, but this serves as an intermediate solution. The following instructions are for a Mac with Conda installed, although the instructions are (presumably) similar for a Windows or Linux system.

1. If you have conda installed you can create a new environment *atmodeller*. You can choose any version of python equal to or greater than 3.10:
	
    ```
    conda create --name atmodeller python=3.10
    ```
2. Activate the environment:

    ```
    conda activate atmodeller
    ```
3. Install the *atmodeller* package into the conda environment, where the filename will be something like *atmodeller-0.1.0.tar.gz*:

    ```
    pip install atmodeller-0.1.0.tar.gz
    ````
4. To locate the example Jupyter notebooks, enter python:

    ```
    python
    ````
5. Once in python type: 

    ```
    import atmodeller
    atmodeller.__file__
    ```
This will report the location of the *atmodeller* package on your system, from which you can determine the path to *atmodeller/docs*. This directory contains the Jupyter notebook tutorials, which you can copy to a different location if you wish.

6. With the *docs* location known, you can now access the Jupyter notebook tutorials. Exit the Python command line first, and then in the terminal type:

    ```
    jupyter notebook docs/1_basics.ipynb
    ```
where the path name is updated according to where *docs* is on your system and the notebook you want to open.

7. In the Jupyter notebook window you may have to *trust* the notebook for all features to work.

## Tests

You can confirm that all tests pass by running `pytest` in the root directory. Please add more tests if you add new features.
