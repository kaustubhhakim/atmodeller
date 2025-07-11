<p align="center">
<img src="https://github.com/ExPlanetology/atmodeller/blob/main/docs/logo.png" alt="atmodeller logo" width="300"/>
</p>

# Atmodeller

[![Release 0.8.1](https://img.shields.io/badge/Release-0.8.1-blue.svg)](https://github.com/ExPlanetology/atmodeller/releases/tag/v0.8.1)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-yellow.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python package](https://github.com/ExPlanetology/atmodeller/actions/workflows/python-package.yml/badge.svg)](https://github.com/ExPlanetology/atmodeller/actions/workflows/python-package.yml)
[![Test coverage](https://img.shields.io/badge/Coverage-88%25-brightgreen)](https://github.com/ExPlanetology/atmodeller)

## About
Atmodeller is a Python package that uses [JAX](https://jax.readthedocs.io/en/latest/index.html) to compute the partitioning of volatiles between a planetary atmosphere and its rocky interior. It is released under [The GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Documentation

The documentation is available online, with options to download it in EPUB or PDF format:

[https://atmodeller.readthedocs.io/en/latest/](https://atmodeller.readthedocs.io/en/latest/)

## Quick install

Atmodeller is a Python package that can be installed on a variety of platforms (e.g. Mac, Windows, Linux). It is recommended to install Atmodeller in a dedicated Python environment. Before installation, create and activate the environment, then run:

```pip install atmodeller```

Downloading the source code is also recommended if you'd like access to the example notebooks in `notebooks/`.

## Citation

If you use Atmodeller, or data from Atmodeller, please cite:

- Bower, D. J., Thompson, M. A., Hakim, K., Tian, M., and Sossi, P. A. (2025), Diversity of rocky planet atmospheres in the C-H-O-N-S-Cl system with interior dissolution, non-ideality and condensation: Application to TRAPPIST-1e and sub-Neptunes, The Astrophysical Journal, submitted. ArXiv e-print [2507.00499](https://arxiv.org/abs/2507.00499).

The data from the above study are also available for download at https://doi.org/10.17605/OSF.IO/PC5TD.

## Basic usage

Jupyter notebooks in the `notebooks/` directory demonstrate how to perform single and batch calculations, and how to integrate Atmodeller into time-dependent simulations. A simple example of how to use Atmodeller is provided below:

```
from atmodeller import (
    InteriorAtmosphere,
    Planet,
    Species,
    SpeciesCollection,
    earth_oceans_to_hydrogen_mass,
)
from atmodeller.solubility import get_solubility_models

solubility_models = get_solubility_models()
# Get the available solubility models
print("solubility models = ", solubility_models.keys())

H2_g = Species.create_gas("H2")
H2O_g = Species.create_gas("H2O", solubility=solubility_models["H2O_peridotite_sossi23"])
O2_g = Species.create_gas("O2")

species = SpeciesCollection((H2_g, H2O_g, O2_g))
planet = Planet()
interior_atmosphere = InteriorAtmosphere(species)

oceans = 1
h_kg = earth_oceans_to_hydrogen_mass(oceans)
o_kg = 6.25774e20
mass_constraints = {
    "H": h_kg,
    "O": o_kg,
}

# If you do not specify an initial solution guess then a default will be used
# Initial solution guess number density (molecules/m^3)
initial_log_number_density = 50

interior_atmosphere.solve(
    planet=planet,
    initial_log_number_density=initial_log_number_density,
    mass_constraints=mass_constraints,
)
output = interior_atmosphere.output

# Quick look at the solution
solution = output.quick_look()

# Get complete solution as a dictionary
solution_asdict = output.asdict()
print("solution_asdict =", solution_asdict)

# Write the complete solution to Excel
output.to_excel("example_single")
```

## Funding
Atmodeller was created as part of a SERI-funded ERC Starting grant '2ATMO' granted to P. Sossi (Contract no. MB22.00033), with additional funding provided through a Swiss National Science Foundation (SNSF) Eccellenza Professorship (#203668).