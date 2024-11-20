import matplotlib.pyplot as plt
import numpy             as np
import h5py
import math
import os
import sys
import ostrich.emulate as emulate
import emulator_helper_functions_LH as em

plt.rcParams.update({'font.size': 18})
plt.rc('legend',fontsize=14)

mp = 1.67e-24
Mpc = 3.0856e24
kpc = Mpc/1000.0
erg_to_keV = 6.242e+8
Zsun = 0.0127
Msun = 1.989e33 #g cm^-3
kb = 1.38e-16 # erg/K
erg_to_keV = 6.242e+8
K_to_keV = kb * erg_to_keV
XH = 0.76 #primordial hydrogen fraction
mu = 0.58824; # X=0.76 assumed 
mu_e = mue = 2.0/(1.0+XH); # X=0.76 assumed

num_pca_components = 12

def return_LH_emulator(profile_type, 
                       prof_dir='./simulation_profiles/', 
                       suite='IllustrisTNG', 
                       ssfr_str='all_ssfr', 
                       interpolation_type='linear'):
    '''
    Inputs:
        A: feedback strength (float)
        z: redshift (float)
        logM: halo mass in log10 Msun
    Return:
        r: radial bins in Mpc (numpy array)
        profile: emulator profile values in cgs units (numpy array)
    '''
    samples,radius,y,emulator=em.build_profile_emulator_M200c(prof_dir,
                                                              suite,p
                                                              rofile_type,
                                                              interpolation_type,
                                                              ssfr_str, 
                                                              num_components=num_pca_components)
    return radius, emulator, samples, y

def return_emulated_profile(emulator, 
                            OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2, z, logM):

    #the order here is important, A, then z, then logM!
    params=[[OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2, z, logM]] 
    profile = emulator(params).ravel()
    return profile

prof_dir='./simulation_profiles/' #point to your profiles
suite_list = [sys.argv[1]] #SIMBA or IllustrisTNG
interpolation_type='linear' #this is the Rbf interpolation function

profiles = ['hot_density', 'hot_temperature', 'hot_metallicity', 'xsb']
rerun_building_emulator = True

#ssfr_list = ['all_ssfr', 'upper_ssfr', 'lower_ssfr']
ssfr_list = ['all_ssfr']

emulator_path = './emulator_files/'

for suite in suite_list :
    
    for ssfr_str in ssfr_list :
        
        for p in profiles :
            emulator_file =emulator_path + suite+'_LH_'+p+'_M200c_'+ssfr_str+'_emulator.dill' 
            radius_file =emulator_path + suite+'_LH_'+p+'_M200c_'+ssfr_str+'_radius.dill' 

            if (not os.path.exists(emulator_file)) or rerun_building_emulator :
                
                emu_radius, emulator, samples, y = return_LH_emulator(p, 
                                                                      prof_dir=prof_dir, 
                                                                      suite=suite, 
                                                                      ssfr_str=ssfr_str, 
                                                                      interpolation_type=interpolation_type)
                emulate.save_pca_emulator(emulator_file, emulator)
                emulate.save_pca_emulator(radius_file, emu_radius)
