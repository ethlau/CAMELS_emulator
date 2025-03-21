# CAMELS emulator

[![DOI](https://zenodo.org/badge/352484591.svg)](https://doi.org/10.5281/zenodo.14714176) [![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

CAMELS_emulator is a repository creating an emulator for the CAMELS profiles, based on https://github.com/ethlau/emu_CAMELS

## File Types and Formats
- **Python Scripts (.py)**: 
  - `profile_emulator.py`: Core implementation for emulating density and temperature profiles
  - `emulator_helper_functions_LH.py`: Helper functions for Latin Hypercube parameter sampling
  - `build_profile_emulator_LH_M200c.py`: Script for building emulators for halo mass-based profiles
- **Data Files**:
  - `emulator_files/*.dill` files: Serialized emulator models and radius information
- **Input Text Files**: 
  - `CosmoAstroSeed_params_LH_*.txt`: Simulation parameter files for Latin Hypercube sampling
- **Notebook**:
  - `Profile_emulator_M200c.ipynb`: Example ipython notebook illustrating how the emulator works

## Required Tools and Dependencies
- **[ostrich](https://github.com/ethlau/ostrich)**: Custom machine learning package for emulation (with `emulate` and `interpolate` modules)
- **[colossus](https://bdiemer.bitbucket.io/colossus/)**: For cosmological calculations and halo mass conversions
- **[kllr](https://github.com/afarahi/kllr)**: Kernel-localized linear regression for data fitting
- **[cgm_toolkit](https://github.com/ethlau/cgm_toolkit)**: Calculations for X-ray properties for halo gas
- **[astropy](https://www.astropy.org/)**: For unit conversions
- Numpy, scipy, and matplotlib

## Script and Code Functions
- **Profile Emulation Functions**:
  - `build_profile_emulator_M200c()`: Creates emulators for X-ray profiles as function of halo mass
  - `build_profile_emulator_Mstar()`: Creates emulators for X-ray profiles as function of stellar mass
  - `emulated_profile_LH()`: Generates profile predictions from emulators
  - `load_profiles_M200c()`: Loads X-ray profiles based on halo mass from simulation data
  - `load_profiles_Mstar()`: Loads X-ray profiles based on stellar mass from simulation data

- **Scaling Relation Emulators**:
  - `build_Lx_Mstar_emulator()`: Creates emulators for X-ray luminosity-stellar mass relations
  - `build_Tx_Mstar_emulator()`: Creates emulators for X-ray temperature-stellar mass relations
  - `build_Mstar_M200c_emulator()`: Creates emulators for stellar mass-halo mass relations
  - `build_Lx_M500c_emulator()`: Creates emulators for X-ray luminosity-halo mass relations

## Relationship to Research
This code implements the emulator framework described in "[X-raying CAMELS: Constraining Baryonic Feedback in the Circum-Galactic Medium](https://ui.adsabs.harvard.edu/abs/2024arXiv241204559L/abstract)" (Lau et al., 2025). Specifically:

1. The emulator uses Principal Component Analysis (PCA) with Radial Basis Function interpolation to predict X-ray surface brightness profiles and X-ray luminosity-stellar mass relations.

2. It processes data from 1000 CAMELS Latin Hypercube simulations per simulation suite (IllustrisTNG, SIMBA, Astrid) across three different redshift snapshots.

3. The code handles 6 key parameters that are varied in the simulations:
   - Two cosmological parameters: Ω_M and σ_8
   - Four SN and AGN feedback parameters implemented in the simulations. 
  
4. The emulator enables direct comparison between simulated X-ray properties and eROSITA All-Sky Survey observations to determine the optimal feedback parameters that reproduce observed X-ray profiles of the circumgalactic medium.
