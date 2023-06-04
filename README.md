# atmodeller
A Python package for computing the partitioning of volatiles between a planetary atmosphere and its interior.

See the Jupyter notebook tutorials in the package directory `docs/`.

## Installation with a package tarball

If you prefer, I can send you a package tarball that you can install using the terminal. In the long-term I will set up a package repository, but this serves as an intermediate solution. The following instructions are for a Mac with Conda installed, although the instructions are (presumably) similar for a Windows or Linux system.

1. If you have conda installed you can create a new environment *atmodeller*. You can choose any version of python equal to or greater than 3.10: 
	
	`conda create --name atmodeller --python=3.10`
2. Activate the environment:

	`conda activate atmodeller`
3. Install the *atmodeller* package into the conda environment, where the filename will be something like *atmodeller-0.1.0.tar.gz*:

	`pip install atmodeller-0.1.0.tar.gz`
4. To locate the example Jupyter notebooks, enter python:

	`python`

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
