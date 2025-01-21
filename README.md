# CAMELS_emulator

[![DOI](https://zenodo.org/badge/352484591.svg)](https://doi.org/10.5281/zenodo.14714176)

CAMELS_emulator is a repository creating an emulator for the CAMELS profiles, based on https://github.com/ethlau/emu_CAMELS

A different emulator is created for LH set fpr each simulation suite (SIMBA/IllustrisTNG/Astrid), feedback parameters (ASN1/2, AAGN1/2) and profiles (density and temperature).

The emulator is constructed using the profiles found in emulator_profiles and the interpolation methods of the repository *ostrich*, https://github.com/ethlau/ostrich, (which is forked from https://github.com/dylancromer/ostrich) .

## Dependencies:

numpy, scipy, astropy, colossus, kllr, 

ostrich - should download its own dependencies when cloning the repo
