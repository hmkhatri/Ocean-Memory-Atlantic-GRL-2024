"""
This script can be used to create composites of heat content (0-200 m and 200-1000m) timeseries based on extreme NAO indices using HadSST, EN4 and HadSLP data.
This script can then be used to create response function in SST and heat content due to single NAO event. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import scipy.stats as sc

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------

def Moving_Avg(ds, time = 1., time_len = 12):
    
    """Compute moving averages
    Parameters
    ----------
    ds : xarray Dataset for data variables
    time : time values for computing weights
    time_len : number of grid points for moving avg
    
    Returns
    -------
    ds_avg : Dataset containting moving avg
    """
    
    if(len(time) == 1):
        
        ds_avg = ds.rolling(time = time_len, center = True).mean('time')
        
    else: 
    
        days = time.dt.daysinmonth
        
        ds_avg = ((ds * days).rolling(time = time_len, center = True).mean('time') /
                  days.rolling(time = time_len, center = True).mean('time'))
    
    return ds_avg

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

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


### ------ Main calculations ------------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"


nao_cut_coef = 1.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (1, 12) # give range of years before / after NAO onset for composites

cf_lev = 0.95 # confidence level
num_sample = 10000 # bootstrap samples to create

# remove 20-year moving mean
num_year = 20
mov_avg = num_year * 12

year1, year2 = (1901, 2020) # range years for composite construction
rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

# read data timeseries
ds_NAO = xr.open_dataset(ppdir + "Observations/NAO_HadSLP.nc")
ds_NAO = ds_NAO.sel(time = ds_NAO['time.year'] >= year1) 
ds_NAO = ds_NAO.sel(time = ds_NAO['time.year'] <= year2)

ds_SST = xr.open_dataset(ppdir + "Observations/SST_Index_HadISST.nc")
ds_SST = ds_SST.sel(time = ds_SST['time.year'] >= year1) 
ds_SST = ds_SST.sel(time = ds_SST['time.year'] <= year2)

ds_EN4 = xr.open_dataset(ppdir + "Observations/Heat_Content_EN4.nc")
ds_EN4 = ds_EN4.sel(time = ds_EN4['time.year'] >= year1)
ds_EN4 = ds_EN4.sel(time = ds_EN4['time.year'] <= year2)
ds_EN4 = ds_EN4.sel(depth=slice(0,200.)).sum('depth') # upper 200 m sum

ds = xr.merge([ds_NAO, ds_SST.drop('time'), ds_EN4.drop('time')])

reg_list = ['North_Atlantic', 'North_Atlantic_subpolar', 'North_Atlantic_subpolar_mid', 'North_Atlantic_subtropical', 'Global']
for reg in reg_list:
    ds['Heat_Content_' + reg] = ds['Heat_Content_' + reg] / (ds['Volume_' + reg] * rho_cp)
    
var_drop = ['expvar', 'eofs', 'Volume_North_Atlantic', 'Volume_North_Atlantic_subpolar', 
            'Volume_North_Atlantic_subpolar_mid', 'Volume_North_Atlantic_subtropical', 'Volume_Global']

ds = ds.drop(var_drop).isel(mode=0)
ds = ds.drop(['lat', 'lon', 'year', 'mode', 'month'])

# Remove linear trends
for var in list(ds.keys()):
    ds[var] = detrend(ds[var], ['time'])

# High-pass filter
# (remove variations longer than 20 years to remove multidecadal fluctuations)
# (this is to ensure that NAO-based composites do not have any biases due to multidecadal variations in temperatures)
if(num_year > 0):
    for var in list(ds.keys()):
        var_smooth = Moving_Avg(ds[var], time = ds['time'], time_len = mov_avg)
        ds[var] = (ds[var] - var_smooth)

print("Data reading complete for observations")

# ----- response function and composites using point-based NAO and eof-based NAO

nao_var = ['NAO_point', 'pcs'] # Uses both station-based NAO indices (NAO_point) and EOF-based NAO indices (pcs)

for nao1 in nao_var:
    
    NAO = ds[nao1].copy()
    NAO = NAO.isel(time=slice(2,len(NAO.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
    NAO_season = NAO.resample(time='QS-DEC').mean('time')
    
    nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
    nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')

    # create composites
    if (nao1 == 'pcs'): # pcs are -ve for NAO+ event
        ind_NAOn = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOp = xr.where(nao_DJF <= -nao_cut, 1, 0)
    else:
        ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)

    ds_res1 = [] # dataset for response function
    
    for cas in case:
        
        ds_ens = []
        
        if (cas == 'NAOp'):
            count_NAO = ind_NAOp
        elif (cas == 'NAOn'):
            count_NAO = ind_NAOn
        else:
            print("Choose a valid case")

        # composite for heat content
        for year in range(year_str + int(num_year/2), len(nao_DJF) - year_end - int(num_year/2)):
            
            if(count_NAO.isel(time=year) == 1):
            
                year_val = nao_DJF['time.year'][year]

                tmp = ds.copy()
                tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)
                
                ds_ens.append(tmp.drop('time'))

        tim = tmp.time
        
        ds_ens = xr.concat(ds_ens, dim='r')
        ds_ens = ds_ens.assign(time = tim)

        print("members = ", len(ds_ens['r']), ", case: ", cas, " -- ", nao1)
        
        ds_ens.to_netcdf(ppdir + "Observations/" + cas + "_Composite_Heat_Content_SST_" + nao1 + ".nc")

        # Get response function and errorbars
        if (cas == 'NAOp'):
            ds_res1.append(ds_ens)
        elif (cas == 'NAOn'):
            d2 = ds_ens.copy()
            for var in list(ds_ens.keys()):
                ds_ens[var] = - d2[var]
                    
            ds_res1.append(ds_ens.drop('time'))

        ds_res = xr.concat(ds_res1, dim='r')

        ds_res = ds_res.transpose('r', 'time')


        # compute mean response function and errorbars with bootstrapping
        ds_save = xr.Dataset()
    
        for var1 in list(ds_res.keys()):
    
            dim_list = ['time'] 
            
            data_var = ds_res[var1].compute()
            bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                    
            ds_save[var1] = data_var.mean('r')
            ds_save[var1 + '_standard_error'] = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
            ds_save[var1 + '_confidence_lower'] = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list)
            ds_save[var1 + '_confidence_upper'] = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)

        ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence interval:" + 
                                        " subpolar-subtropical SST and heat content response to a single NAO+ event." + 
                                        " Confidence interval is " + str(cf_lev*100) + "%. ")
        save_file_path = (ppdir + "Observations/" + "Response_Function_" + "Heat_Content_SST_" + nao1 + ".nc")

        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")

