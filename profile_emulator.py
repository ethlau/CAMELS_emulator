import numpy as np
from ostrich import emulate
from colossus.cosmology import cosmology
from colossus.halo import mass_so

XH = 0.76 #primordial hydrogen fraction
mu = 0.58824; # X=0.76 assumed 
mu_e = mue = 2.0/(1.0+XH); # X=0.76 assumed
OmegaM = 0.3089
sigma8 = 0.8102
cosmo = cosmology.setCosmology('myCosmo', 
                                params = cosmology.cosmologies['planck18'], Om0 = OmegaM, sigma8=sigma8)

mp = 1.67e-24
Mpc = 3.0856e24
kpc = Mpc/1000.0
erg_to_keV = 6.242e+8
Zsun = 0.0127
Msun = 1.989e33 #g cm^-3

def emulated_profile_LH(emulator, *plist):

    profile = emulator([list(plist)]).ravel()
    
    return profile

def load_profile_emulator(radius_file, emulator_file) :
    
    emulator = emulate.load_pca_emulator( emulator_file )
    radius = emulate.load_pca_emulator( radius_file )    

    return radius, emulator

def Mgas_M200c_scaling(suite, z, M200c, *params,
                       OmegaM=OmegaM, sigma8=sigma8,
                       ssfr_str='all_ssfr', emulator_dir='emulator_files/'):

    mgas = [ Mgas_M200c_scalar(suite, z, m, *params, 
                               OmegaM=OmegaM, sigma8=sigma8,
                               ssfr_str=ssfr_str, emulator_dir=emulator_dir) 
             for m in M200c  ]

    return np.array(mgas)

def Mgas_M200c_scalar(suite, z, M200c, *params, 
                      OmegaM=OmegaM, sigma8=sigma8,
                      ssfr_str='all_ssfr', emulator_dir='emulator_files/'):
    
    emulator_file = emulator_dir+'/'+suite+"_LH_hot_density_M200c_"+ssfr_str+"_emulator.dill"
    radius_file   = emulator_dir+'/'+suite+"_LH_hot_density_M200c_"+ssfr_str+"_radius.dill"
        
    radius_Mpc, x_emulator = load_profile_emulator(radius_file, emulator_file)
    radius_kpc = radius_Mpc * 1000.0

    plist = OmegaM, sigma8, *params, z, np.log10(M200c)

    radius = np.logspace(1, 3, 20) # in kpc
    # ngas in cm^-3
    pro = np.interp( np.log10(radius), np.log10(radius_kpc),
                     emulated_profile_LH(x_emulator, *plist) )

    #convert ngas to rho_gas in Msun kpc^-3
    pro *= mu * mp * (1./Msun) * kpc**3
    
    
    differential_volume = radius*0.0
    for ir, r in enumerate(radius) :
        if ir ==0 :
            differential_volume[ir] = (4.0*np.pi/3.0) * radius[ir]**3
        else :    
            differential_volume[ir] = (4.0*np.pi/3.0) * (radius[ir]**3 - radius[ir-1]**3)
    
    mgas_profile = np.cumsum(pro*differential_volume) 

    cosmo = cosmology.setCosmology('myCosmo', 
                                params = cosmology.cosmologies['planck18'], Om0 = OmegaM, sigma8=sigma8)
    R200c = mass_so.M_to_R(M200c, z, '200c') # R200c in kpc
    mgas_R200c = 10**np.interp(np.log10(R200c), np.log10(radius), np.log10(mgas_profile))
    
    return mgas_R200c


def density_profile (suite, z, logM200c, radius, *params, 
                     OmegaM=OmegaM, sigma8=sigma8,
                     ssfr_str='all_ssfr', emulator_dir='emulator_files/'):
    
    emulator_file = emulator_dir+'/'+suite+"_LH_hot_density_M200c_"+ssfr_str+"_emulator.dill"
    radius_file   = emulator_dir+'/'+suite+"_LH_hot_density_M200c_"+ssfr_str+"_radius.dill"
        
    radius_Mpc, x_emulator = load_profile_emulator(radius_file, emulator_file)
    radius_kpc = radius_Mpc * 1000.0

    plist = OmegaM, sigma8, *params, z, logM200c
    pro = np.interp(np.log10(radius), np.log10(radius_kpc), emulated_profile_LH(x_emulator, *plist))
        
    return np.abs(pro)

def temperature_profile (suite, z, logM200c, radius, *params, 
                        OmegaM=OmegaM, sigma8=sigma8,
                         ssfr_str='all_ssfr', emulator_dir='emulator_files/'):
    
    emulator_file = emulator_dir+'/'+suite+"_LH_hot_temperature_M200c_"+ssfr_str+"_emulator.dill"
    radius_file   = emulator_dir+'/'+suite+"_LH_hot_temperature_M200c_"+ssfr_str+"_radius.dill"
        
    radius_Mpc, x_emulator = load_profile_emulator(radius_file, emulator_file)
    radius_kpc = radius_Mpc * 1000.0

    plist = OmegaM, sigma8, *params, z, logM200c
    pro = np.interp(np.log10(radius), np.log10(radius_kpc), emulated_profile_LH(x_emulator, *plist))
        
    return np.abs(pro)
