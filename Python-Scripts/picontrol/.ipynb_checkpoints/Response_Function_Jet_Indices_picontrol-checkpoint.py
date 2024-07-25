"""
This script can be used to create composites of SST, heat content (0-200 m) timeseries based on extreme Jet indices using cmip piControl runs. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import xskillscore as xs
import scipy.stats as sc
import glob

import warnings
warnings.filterwarnings("ignore")

# ----------- Functions -------------
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
    
    bootstrap_ci = sc.bootstrap(data, statistic=np.mean, confidence_level=cf_lev, vectorized=True, 
                                axis=0, n_resamples=num_sample,
                                random_state=1, method='BCa')
    
    return bootstrap_ci

### ------ Main calculations ------------------

source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 
experiment_id = ['piControl'] #, 'historical'] 

dir_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

jet_cut_coef = 1.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['Jetp', 'Jetn']
year_str, year_end = (1, 12) # give range of years before / after NAO onset for composites

# remove 20-year moving mean
num_year = 20
mov_avg = num_year * 12

cf_lev = 0.95 # confidence level
num_sample = 10000 # bootstrap samples to create

for exp in experiment_id:
    
    for model in source_id:

        # Read heat content and NAO data
        d1 = xr.open_mfdataset(dir_path + model + exp + "/Heat_Content/Heat*.nc", use_cftime=True, chunks=None)
        d2 = xr.open_dataset(dir_path + model + exp + "/SST_Index.nc", use_cftime=True)
        d3 = xr.open_dataset(dir_path + model + exp + "/Jet_Indices.nc", use_cftime=True)

        ds = xr.merge([d1, d2, d3])

        # compute heat content in upper 200 m and 200-1000 m
        if(model == 'NCAR/CESM2/'):
            ds = ds.sel(lev=slice(0.,20000.)).sum('lev') # get data in upper 200 m
        else:
            ds = ds.sel(lev=slice(0.,200.)).sum('lev') # get data in upper 200 m

        # Remove linear drift
        for var in list(ds.keys()):
            ds[var] = detrend(ds[var], ['time'])

        # Remove climatology
        ds_clim = ds.groupby('time.month').mean('time')
        ds = ds.groupby('time.month') - ds_clim

        # High-pass filter
        if(num_year > 0):
            for var in list(ds.keys()):
                var_smooth = Moving_Avg(ds[var], time = ds['time'], time_len = mov_avg)
                ds[var] = (ds[var] - var_smooth)
                
        print("Data reading complete for model: ", model)

        ## ---------- composites and response function from jet speed ---------------
        Jet_ind = ds['Jet_speed'].isel(time=slice(2,len(ds.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
        Jet_ind_season = Jet_ind.resample(time='QS-DEC').mean('time')

        Jet_ind_cut = jet_cut_coef * Jet_ind_season.std('time', skipna=True).values
        Jet_ind_DJF = Jet_ind_season.sel(time = Jet_ind_season['time.season'] == 'DJF')

        # create composites
        ind_Jet_indp = xr.where(Jet_ind_DJF >= Jet_ind_cut, 1, 0)
        ind_Jet_indn = xr.where(Jet_ind_DJF <= -Jet_ind_cut, 1, 0)

        ds1 = []

        for cas in case:

            ds_ens = []

            if (cas == 'Jetp'):
                count_jet = ind_Jet_indp
            elif (cas == 'Jetn'):
                count_jet = ind_Jet_indn
            else:
                print("Choose a valid case")

            # composite for seasonal nao indices
            for year in range(year_str + int(num_year/2), len(Jet_ind_DJF) - year_end - int(num_year/2)):
        
                if(count_jet.isel(time=year) == 1):

                    year_val = Jet_ind_DJF['time.year'][year]

                    tmp = ds.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)

                    ds_ens.append(tmp.drop('time'))
                    tim = tmp.time

            ds_ens = xr.concat(ds_ens, dim='r')

            if (cas == 'Jetp'):
                ds1.append(ds_ens.assign(time = tim))
            elif (cas == 'Jetn'):
                d2 = ds_ens.copy()
                for var in list(ds_ens.keys()):
                    ds_ens[var] = - d2[var]
                ds1.append(ds_ens)
                
        ds1 = xr.concat(ds1, dim='r')
        ds1 = ds1.transpose('r', 'time')

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
    
        ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence interval:" + 
                                        "subpolar-subtropical SST and heat content response to a single NAO+ event." + 
                                        " Confidence interval is " + str(cf_lev*100) + "%. ")

        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(dir_path + model + exp + "/Response_Function_Jet_Speed.nc")

        print("Jet speed, SST and heat content data saved successfully for model: ", model)

        ds1.close()
        ds_save.close()

        ## ---------- composites and response function from jet latitude ------------
        Jet_ind = ds['Jet_lat'].isel(time=slice(2,len(ds.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
        Jet_ind_season = Jet_ind.resample(time='QS-DEC').mean('time')

        Jet_ind_cut = jet_cut_coef * Jet_ind_season.std('time', skipna=True).values
        Jet_ind_DJF = Jet_ind_season.sel(time = Jet_ind_season['time.season'] == 'DJF')

        # create composites
        ind_Jet_indp = xr.where(Jet_ind_DJF >= Jet_ind_cut, 1, 0)
        ind_Jet_indn = xr.where(Jet_ind_DJF <= -Jet_ind_cut, 1, 0)

        ds1 = []

        for cas in case:

            ds_ens = []

            if (cas == 'Jetp'):
                count_jet = ind_Jet_indp
            elif (cas == 'Jetn'):
                count_jet = ind_Jet_indn
            else:
                print("Choose a valid case")

            # composite for seasonal nao indices
            for year in range(year_str + int(num_year/2), len(Jet_ind_DJF) - year_end - int(num_year/2)):
        
                if(count_jet.isel(time=year) == 1):

                    year_val = Jet_ind_DJF['time.year'][year]

                    tmp = ds.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)

                    ds_ens.append(tmp.drop('time'))
                    tim = tmp.time

            ds_ens = xr.concat(ds_ens, dim='r')

            if (cas == 'Jetp'):
                ds1.append(ds_ens.assign(time = tim))
            elif (cas == 'Jetn'):
                d2 = ds_ens.copy()
                for var in list(ds_ens.keys()):
                    ds_ens[var] = - d2[var]
                ds1.append(ds_ens)
                
        ds1 = xr.concat(ds1, dim='r')
        ds1 = ds1.transpose('r', 'time')

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
    
        ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence interval:" + 
                                        "subpolar-subtropical SST and heat content response to a single NAO+ event." + 
                                        " Confidence interval is " + str(cf_lev*100) + "%. ")

        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(dir_path + model + exp + "/Response_Function_Jet_Latitude.nc")

        print("Jet lat, SST and heat content data saved successfully for model: ", model)

        ds1.close()
        ds_save.close()
        ds.close()
