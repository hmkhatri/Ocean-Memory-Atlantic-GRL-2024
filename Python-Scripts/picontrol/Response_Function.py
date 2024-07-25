"""
This script can be used to create response function in SST and heat content due to single NAO event using cmip piControl runs. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xskillscore as xs
import scipy.stats as sc

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------

def data_bootstrap(data, cf_lev = 0.95, num_sample = 1000):
    
    """Compute bootstrap confidence intervals and standard error for data along axis =0
    Parameters
    ----------
    data : xarray DataArray for data
    stat : statisctic required for bootstrapping function, e.g. np.mean, np.std
    cf_lev : confidence level
    num_sample : Number of bootstrap samples to generate
    Returns
    -------
    bootstrap_ci : object contains float or ndarray of
        bootstrap_ci.confidence_interval : confidence intervals
        bootstrap_ci.standard_error : standard error
    """
    
    data = (data,)
    
    bootstrap_ci = sc.bootstrap(data, statistic=np.mean, confidence_level=cf_lev, vectorized=True, axis=0, n_resamples=num_sample,
                                random_state=1, method='BCa')
    
    return bootstrap_ci

### ------------- Main computations ------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

exp = 'piControl'
source_id = ['MOHC/HadGEM3-GC31-MM/', 'NOAA-GFDL/GFDL-CM4/', 'IPSL/IPSL-CM6A-LR/', 'NCAR/CESM2/']

case = ['NAOp', 'NAOn']

var_list = ['SST', 'Heat_Content_200'] #, 'Heat_Content_1000'] 

cf_lev = 0.95 # confidence level
num_sample = 10000 # bootstrap samples to create

for var_name in var_list:

    for model in source_id:
    
        ds1 = []
    
        # combine nao+ and nao- data
        for cas in case:

            # this is for single NAO events
            #d1 = xr.open_dataset(ppdir + model + exp + "/" + cas + "_Composite_" + var_name + ".nc", use_cftime=True)

            # this is for consecutive NAO events
            d1 = xr.open_dataset(ppdir + model + exp + "/" + cas + "_Composite_" + var_name + "_consecutive_NAO.nc", use_cftime=True)
    
            if(cas == 'NAOp'):
                ds1.append(d1)
            else:
                d2 = d1.copy()
                for var in list(d1.keys()):
                    d1[var] = - d2[var]
                    
                ds1.append(d1.drop('time'))
    
        ds1 = xr.concat(ds1, dim='r')
    
        ds1 = ds1.transpose('r', 'time')

        ds1['NAO'] = (ds1['P_south'] - ds1['P_north'])
    
        print("Data reading complete for model: ", model) 
    
        # compute mean response function and errorbars with bootstrapping
        ds_save = xr.Dataset()
    
        for var1 in list(ds1.keys()):
    
            dim_list = ['time'] 
            
            data_var = ds1[var1].compute()
            bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                    
            ds_save[var1] = data_var.mean('r')
            ds_save[var1 + '_standard_error'] = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
            ds_save[var1 + '_confidence_lower'] = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
            ds_save[var1 + '_confidence_upper'] = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)

        # this is for single NAO events
        #ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence interval: subpolar-subtropical SST and heat content response to a single NAO+ event." + 
        #                               " Confidence interval is " + str(cf_lev*100) + "%. ")
        #save_file_path = (ppdir + model + exp + "/Response_Function_" + var_name + ".nc")

        # this is for consecutive NAO events
        ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence interval: subpolar-subtropical SST and heat content response to consecutive NAO+ events." + 
                                       " Confidence interval is " + str(cf_lev*100) + "%. ")
        save_file_path = (ppdir + model + exp + "/Response_Function_" + var_name + "_consecutive_NAO.nc")
        
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")

        
