import numpy as np
import ostrich.emulate
import ostrich.interpolate
from astropy import units as u
import matplotlib.pyplot as plt
import warnings
from kllr import kllr_model 
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

z = np.array([0.05, 0.10, 0.15])
snap=['088','086','084']

mass_str_arr=np.array(['12.0-12.3','12.3-12.7','12.7-13.2', '13.2-14.0'])
mass_low_arr=np.array([12.0,12.3,12.7,13.2])
mass_high_arr=np.array([12.3,12.7,13.2,14.0])
mass = np.array([12.15, 12.5, 12.95, 13.6])

mstar_str_arr=np.array(['9.5-10.0', '10.0-10.5','10.5-11.0', '11.0-12.0'])
ms_low_pow_arr=np.array([9.5, 10.0, 10.5, 11.0])
ms_high_pow_arr=np.array([10.0, 10.5, 11.0, 12.0])
mstar = np.array([9.75, 10.25, 10.75, 11.5])


mean_masses_uw={}
mean_masses_w={}
median_masses={}

mean_mstar_uw={}
mean_mstar_w={}
median_mstar={}

#R (Mpc), hot_dens, hot_temp, hot_metal, xsb (mean, errup, errlow, std, median)
profiles = ['hot_density', 'hot_temperature', 'hot_metallicity', 'xsb']

def Ez(z, OmegaM):

    OmegaL = 1.0 - OmegaM

    return np.sqrt(OmegaL + OmegaM*(1+z)**3)

def set_suite(suite):
    
    txt_file = "CosmoAstroSeed_params_LH_"+suite+".txt"

    data = np.loadtxt(txt_file, dtype={'names': ('sim_name', 'omegam', 'sigma8', 'asn1', 'aagn1', 'asn2', 'aagn2', 'seed'),
                                   'formats': ('S10', float, float, float, float, float, float, int )} )

    Sim_name = data['sim_name']
    OmegaM = data['omegam']
    sigma8 = data['sigma8']
    ASN1 = data['asn1']
    ASN2 = data['asn2']
    AAGN1 = data['aagn1']
    AAGN2 = data['aagn2']

    return Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2

def choose_redshift():
    return z

def cartesian_prod(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T

def LH_cartesian_prod(params, redshift, mass):
    arr_shape = (params.shape[0]*redshift.shape[0]* mass.shape[0],params.shape[1]+2)
    arr = np.zeros(arr_shape)    
    for i in range(params.shape[0]):
        for j, z in enumerate(redshift):
            for k, m in enumerate(mass):
                index = k+mass.shape[0]*(j+redshift.shape[0]*i)
                arr[index,-1] = m
                arr[index,-2] = z
                for l, p in enumerate(params[i]):
                    arr[index,l] = p
    return arr

def LH_cartesian_prod_no_mass(params, redshift):
    arr_shape = (params.shape[0]*redshift.shape[0],params.shape[1]+1)
    arr = np.zeros(arr_shape)    
    for i in range(params.shape[0]):
        for j, z in enumerate(redshift):
            index = j+redshift.shape[0]*i
            arr[index,-1] = z
            for l, p in enumerate(params[i]):
                arr[index,l] = p
    return arr

#----------------------------------------------
#general emulator functions

def load_profiles_M200c(home,suite,sims,snap,mass_str_arr,ssfr_str,profiles=profiles):

    #R (Mpc), hot_dens, hot_temp, hot_metal, xsb (mean, errup, errlow, std, median)
    lx = np.logspace(-2, 0.0, 20)

    mean_profiles = {} 
    median_profiles = {} 
    up_profiles = {} 
    dn_profiles = {} 
    std_profiles = {} 
    
    for p in profiles:
        mean_profiles[p] = [] 
        median_profiles[p] = [] 
        up_profiles[p] = [] 
        dn_profiles[p] = []
        std_profiles[p] = []

    for s in np.arange(len(sims)):
        for n in np.arange(len(snap)):
            for m in np.arange(len(mass)):

                mean_profiles_i = {} 
                median_profiles_i = {} 
                up_profiles_i = {} 
                dn_profiles_i = {} 
                std_profiles_i = {} 
 
                f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_uw_m200c'+mass_str_arr[m]+'_'+ssfr_str+'.txt'
                (x,  
                 mean_profiles_i['hot_density'],   median_profiles_i['hot_density'],  std_profiles_i['hot_density'],
                 mean_profiles_i['hot_temperature'], median_profiles_i['hot_temperature'], std_profiles_i['hot_temperature'],  
                 mean_profiles_i['hot_metallicity'],  median_profiles_i['hot_metallicity'], std_profiles_i['hot_metallicity'], 
                 mean_profiles_i['xsb'], median_profiles_i['xsb'], std_profiles_i['xsb'] ) = np.loadtxt(f,unpack=True)

                for p in profiles:
                    yi = mean_profiles_i[p]
                    stddevi = std_profiles_i[p]
                    ymedi = median_profiles_i[p]

                    temp_y      = InterpolatedUnivariateSpline(x, yi,      ext=0)(lx)
                    temp_stddev = InterpolatedUnivariateSpline(x, stddevi, ext=0)(lx)
                    temp_y_med  = InterpolatedUnivariateSpline(x, ymedi,   ext=0)(lx)

                    mean_profiles[p].append(temp_y)
                    std_profiles[p].append(temp_stddev)
                    median_profiles[p].append(temp_y_med)

    for p in profiles:
        mean_profiles[p] = np.nan_to_num(np.array(mean_profiles[p]))
        std_profiles[p] = np.nan_to_num(np.array(std_profiles[p]))
        median_profiles[p] = np.nan_to_num(np.array(median_profiles[p]))
 
    return lx, mean_profiles, median_profiles,  std_profiles


def build_profile_emulator_M200c(home,suite,prof,func_str,ssfr_str,num_components=10):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod(params,z,mass)

    nsamp=samples.shape[0]

    x, mean, median, std = load_profiles_M200c(home,suite,sims,snap,mass_str_arr,ssfr_str,profiles=profiles)
    ly =  np.log10(median[prof])

    #y = np.nan_to_num( ly, copy=False, 
    #                   nan=np.nanmin(ly), 
    #                   posinf=np.nanmax(ly), 
    #                   neginf=np.nanmin(ly)
    #                   )
    y = median[prof]
    
    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=num_components)
    
    return samples,x,y,emulator

def load_profiles_Mstar(home,suite,sims,snap,mstar_str_arr,ssfr_str,profiles=profiles):

    #R (Mpc), hot_dens, hot_temp, hot_metal, xsb (mean, errup, errlow, std, median)
    lx = np.logspace(-2, 0.0, 20)

    mean_profiles = {} 
    median_profiles = {} 
    up_profiles = {} 
    dn_profiles = {} 
    std_profiles = {} 
    
    for p in profiles:
        mean_profiles[p] = [] 
        median_profiles[p] = [] 
        up_profiles[p] = [] 
        dn_profiles[p] = []
        std_profiles[p] = []

    for s in tqdm(np.arange(len(sims))):
        for n in np.arange(len(snap)):
            for m in np.arange(len(mstar)):

                mean_profiles_i = {} 
                median_profiles_i = {} 
                up_profiles_i = {} 
                dn_profiles_i = {} 
                std_profiles_i = {} 
 
                f=home+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_uw_mstar'+mstar_str_arr[m]+'_'+ssfr_str+'.txt'
                (x,  
                 mean_profiles_i['hot_density'],   median_profiles_i['hot_density'],  std_profiles_i['hot_density'],
                 mean_profiles_i['hot_temperature'], median_profiles_i['hot_temperature'], std_profiles_i['hot_temperature'],  
                 mean_profiles_i['hot_metallicity'],  median_profiles_i['hot_metallicity'], std_profiles_i['hot_metallicity'], 
                 mean_profiles_i['xsb'], median_profiles_i['xsb'], std_profiles_i['xsb'],
                 mean_profiles_i['fgas'], median_profiles_i['fgas'], std_profiles_i['fgas'],
                 mean_profiles_i['hot_fgas'], median_profiles_i['hot_fgas'], std_profiles_i['hot_fgas'] ) =np.loadtxt(f,unpack=True)

                for p in profiles:
                    yi = mean_profiles_i[p]
                    stddevi = std_profiles_i[p]
                    ymedi = median_profiles_i[p]

                    temp_y      = InterpolatedUnivariateSpline(x, yi,      ext=0)(lx)
                    temp_stddev = InterpolatedUnivariateSpline(x, stddevi, ext=0)(lx)
                    temp_y_med  = InterpolatedUnivariateSpline(x, ymedi,   ext=0)(lx)
                    
                    mean_profiles[p].append(temp_y)
                    std_profiles[p].append(temp_stddev)
                    median_profiles[p].append(temp_y_med)

    for p in profiles:
        mean_profiles[p] = np.nan_to_num(np.array(mean_profiles[p]))
        std_profiles[p] = np.nan_to_num(np.array(std_profiles[p]))
        median_profiles[p] = np.nan_to_num(np.array(median_profiles[p]))
 
    return lx, mean_profiles, median_profiles,  std_profiles


def build_profile_emulator_Mstar(home,suite,prof,func_str,ssfr_str,num_components=10):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod(params,z,mstar)

    nsamp=samples.shape[0]

    x, mean, median, std = load_xsb_profiles_Mstar(home,suite,sims,snap,mstar_str_arr,ssfr_str,profiles=profiles)
    
    #ly =  np.log10(mean[prof])
    #print (np.nanmin(ly), np.nanmax(ly))
    #y = np.nan_to_num( ly, copy=False, 
    #                   nan=np.nanmin(ly), 
    #                   posinf=np.nanmax(ly), 
    #                   neginf=np.nanmin(ly)
    #                   )
    
    y = median[prof]
    

    y=np.transpose(y)

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=num_components)
    
    return samples,x,y,emulator


def load_Tx_Mstar(home,suite,sims,snap):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    Tx = []
    Mstar = []

    def linear_scaling(x, *p) :
        return (p[0]+p[1]*x)

    for s in tqdm(np.arange(len(sims))):

        Om = OmegaM[s]

        for n in np.arange(len(snap)):

            #print(sims[s], snap[n])

            filename = home+'/'+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_mstar_Lx.npz'

            data = np.load(filename)
            z = data['z']
            tx_data = data['Tx500c']
            mstar_data = data['mstar']*Ez(z, Om)


            mask = (mstar_data > 0) & (tx_data > 0)
            x_data = np.log10(mstar_data[mask])
            y_data = np.log10(tx_data[mask])
            lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.3)

            lg_Mstar_sample, lg_Tx_sample, intercept, slope, scatter, skew, kurt = lm.fit(
                    x_data, y_data, xrange=[9.0,12],bins=6)  
            lg_Tx_mean = np.mean(lg_Tx_sample, axis=0)

            Mstar = lg_Mstar_sample
            Tx.append(lg_Tx_mean)

    Tx = np.array(Tx)

    return Mstar, Tx

def build_Tx_Mstar_emulator(home,suite,func_str,ssfr_str,num_components=12):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod_no_mass(params,z)

    #print(samples.shape) 
    nsamp=samples.shape[0]

    Mstar, Tx = load_Tx_Mstar(home, suite, sims, snap)

    y = np.transpose(Tx)
    x = Mstar

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=num_components)
    return samples,x,y,emulator



def load_Lx_Mstar(home,suite,sims,snap):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    Lx = []
    Mstar = []
    
    def linear_scaling(x, *p) :
        return (p[0]+p[1]*x)

    for s in tqdm(np.arange(len(sims))):

        Om = OmegaM[s]

        for n in np.arange(len(snap)):

            #print(sims[s], snap[n])

            filename = home+'/'+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_mstar_Lx.npz'

            data = np.load(filename)
            z = data['z']
            lx_data = data['Lx500c']/Ez(z, Om)
            mstar_data = data['mstar']*Ez(z, Om)
            mstar200c_data = data['mstar200c']*Ez(z, Om)
            
            mask = (mstar_data > 0)
            x_data = np.log10(mstar_data[mask])
            y_data = np.log10(lx_data[mask])
            lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.3)

            x_sample, y_sample, intercept, slope, scatter, skew, kurt = lm.fit(
                    x_data, y_data, xrange=[9.0,12],bins=6)  
            y_mean = np.mean(y_sample, axis=0)
            
            Lx.append(y_mean)

            #popt, pcov = curve_fit(linear_scaling, x_data, y_data, p0=[1.0,1.0] )
            #lg_Mstar_sample = np.linspace(9.0, 12.0, 6)
            #lg_Lx_mean =  linear_scaling(lg_Mstar_sample, *popt)
            
            #print(lg_Lx_sample.shape, lg_Mstar_sample.shape)


            #digitized = np.digitize(np.log10(mstar_data), Mstar)
            #Lx_means = [ np.log10(lx_data)[digitized == i].mean() for i in range(0, len(Mstar))]           
            
            #lx_array = np.array(Lx_means)

            #mask = (lx_array == lx_array)

            #for il, lx in enumerate(lx_array):
            #    if lx != lx :
            #        lx_array[il] = interp1d(
            #        (Mstar[mask]), (lx_array[mask]),fill_value='extrapolate')(Mstar[il])

            
            #Lx.append(list(lx_array))


    Lx = np.array(Lx)
    Mstar = x_sample

    return Mstar, Lx

def build_Lx_Mstar_emulator(home,suite,func_str,ssfr_str,num_components=12):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod_no_mass(params,z)

    #print(samples.shape) 
    nsamp=samples.shape[0]

    Mstar, Lx = load_Lx_Mstar(home, suite, sims, snap)

    y = np.transpose(Lx)
    x = Mstar

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=num_components)
    return samples,x,y,emulator

def load_Mstar_M200c(home,suite,sims,snap):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    M200c = np.linspace(12.0, 14.0, 5)
    Mstar = []

    def linear_scaling(x, *p) :
        return (p[0]+p[1]*x)

    for s in tqdm(np.arange(len(sims))):

        Om = OmegaM[s]

        for n in np.arange(len(snap)):

            #print(sims[s], snap[n])

            filename = home+'/'+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_mstar_Lx.npz'

            data = np.load(filename)
            z = data['z']
            M200c_data = data['m200c']*Ez(z, Om)
            mstar_data = data['mstar']*Ez(z, Om)
            mstar200c_data = data['mstar200c']*Ez(z, Om)

            mask = (mstar200c_data > 0) & ( M200c_data > 0)
            y_data = np.log10(mstar200c_data[mask])
            x_data = np.log10(M200c_data[mask])
            
            x_sample, y_sample, intercept, slope, scatter, skew, kurt = lm.fit(
                    x_data, y_data, xrange=[12,14],bins=6)  
            y_mean = np.mean(y_sample, axis=0)
            
            Mstar.append(y_mean)

    Mstar = np.array(Mstar)
    M200c = x_sample

    return M200c, Mstar


def build_Mstar_M200c_emulator(home,suite,func_str,ssfr_str,num_components=12):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod_no_mass(params,z)

    #print(samples.shape) 
    nsamp=samples.shape[0]

    M200c, Mstar = load_Mstar_M200c(home, suite, sims, snap)

    y = np.transpose(Mstar)
    x = M200c

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=num_components)

    return samples,x,y,emulator

def load_Lx_M500c(home,suite,sims,snap):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    Lx = []

    def linear_scaling(x, *p) :
        return (p[0]+p[1]*x)

    for s in tqdm(np.arange(len(sims))):

        Om = OmegaM[s]

        for n in np.arange(len(snap)):

            filename = home+'/'+suite+'/'+suite+'_'+sims[s]+'_'+snap[n]+'_mstar_Lx.npz'

            data = np.load(filename)
            z = data['z']
            lx_data = data['Lx500c']/Ez(z, Om)
            m_data = data['m500c']*Ez(z, Om)


            mask = (m_data > 0) & (lx_data > 0)

            x_data = np.log10(m_data[mask])
            y_data = np.log10(lx_data[mask])

            lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.3)

            M_sample, Lx_sample, intercept, slope, scatter, skew, kurt = lm.fit(
                    x_data, y_data, xrange=[12.0,14.0],bins=8)  
            #popt, pcov = curve_fit(linear_scaling, log10_mstar_data[mask], log10_lx_data[mask], p0=[1.0,1.0] )
            #Lx_temp = 10**(linear_scaling(np.log10(Mstar), *popt))
            #print(Lx_sample, Mstar_sample)

            Lx_mean = np.mean(Lx_sample, axis=0)
            M = M_sample
            Lx.append(Lx_mean)

    Lx = np.nan_to_num(np.array(Lx))

    return M, Lx

def build_Lx_M500c_emulator(home,suite,func_str,ssfr_str,num_components=12):

    Sim_name, OmegaM, sigma8, ASN1, AAGN1, ASN2, AAGN2 = set_suite(suite)

    nums=np.linspace(0,999,1000,dtype='int')
    sims=['LH_'+str(i) for i in nums]

    params = np.vstack([OmegaM,sigma8,ASN1,AAGN1,ASN2,AAGN2]).T
    samples=LH_cartesian_prod_no_mass(params,z)

    #print(samples.shape) 
    nsamp=samples.shape[0]

    M500c, Lx = load_Lx_M500c(home, suite, sims, snap)

    y = np.transpose(Lx)
    x = M500c

    emulator=ostrich.emulate.PcaEmulator.create_from_data(
        samples,
        y,
        ostrich.interpolate.RbfInterpolator,
        interpolator_kwargs={'function':func_str},
        num_components=num_components)
    
    return samples,x,y,emulator
