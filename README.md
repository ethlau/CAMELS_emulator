# CAMELS_emulator

CAMELS_emulator is a repository creating an emulator for the CAMELS profiles, based on https://github.com/ethlau/emu_CAMELS

A different emulator is created for each simulation suite (SIMBA/IllustrisTNG), feedback parameter (ASN1/2, AAGN1/2) and profile (rho mean/median, pth mean/median).

The emulator is constructed using the profiles found in emulator_profiles and the interpolation methods of the repository *ostrich*, https://github.com/ethlau/ostrich, (which is forked from https://github.com/dylancromer/ostrich) .

## Dependencies:

numpy, scipy

ostrich - should download its own dependencies when cloning the repo
