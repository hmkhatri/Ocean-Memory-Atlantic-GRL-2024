"""
The script computes composites of mean temperatures in the upper 200 m using EN4 dataset and SST using HadSST data.
Bootstrapping is then used to compute the mean and confidence intervals for annual-mean maps. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
from xeofs.xarray import EOF
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

def annaul_mean_data(ds, var_name, num_days, method = 'mean'):
    
    """Compute annual mean of data for bootstrapping
    Means are computed for year = -1, 0, 1, 2, 3-4, 5-6
    Parameters
    ----------
    ds : xarray Dataset for data variables
    var_name : list of avariables for computing annual means
    num_days : Number of days in months
    method : Options - compute 'mean', 'integrate', 'difference' over time 
    
    Returns
    -------
    ds_annual : Dataset containting annual means
    """
    
    ds_annual = xr.Dataset()
    
    for var1 in var_name:
        
        data_var1 = []
        
        ind_correct = 0
        for i in range(0,7): # mean for years = -1, 0, 1-2, 3-4, 5-6, 7-8, 9-10

            if (i<=1):
                days = num_days.dt.days_in_month.isel(time = slice(12*i + 9, 12*i + 12 + 9)) # Oct-Sep annual mean
                data_var = ds[var1].isel(time = slice(12*i + 9, 12*i + 12 + 9))
            else:
                days = num_days.dt.days_in_month.isel(time = slice(12*(i + ind_correct) + 9, 12*(i + ind_correct + 1) + 9 + 12))
                data_var = ds[var1].isel(time = slice(12*(i + ind_correct) + 9, 12*(i + ind_correct + 1) + 9 + 12))
                ind_correct = ind_correct + 1

            if(method == 'mean'):
                data_var = ((data_var * days).sum('time')/ days.sum('time'))
            elif(method == 'integrate'):
                data_var = ((data_var * days).sum('time') * 3600. * 24.)
            elif(method == 'difference'):
                data_var = (data_var.isel(time=-1) - data_var.isel(time=0))
            else:
                print("Method is not valid")
            
            data_var1.append(data_var)
            
        ds_annual[var1] = xr.concat(data_var1, dim='year')
    
    ds_annual = ds_annual.chunk({'year':-1})
        
    return ds_annual

### ------ Main calculations ------------------
ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

year1, year2 = (1901, 2020) # for removing climatology and composite construction

nao_cut_coef = 1.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (1, 12) # give range of years before / after NAO onset for composites

cf_lev = 0.9 # confidence level
num_sample = 4000 # bootstrap samples to create

# remove 20-year moving mean
num_year = 20
mov_avg = num_year * 12

# read NAO, SST and EN4 data
ds_NAO = xr.open_dataset(ppdir + "Observations/NAO_HadSLP.nc")
ds_NAO = ds_NAO.sel(time = ds_NAO['time.year'] >= year1)
ds_NAO = ds_NAO.sel(time = ds_NAO['time.year'] <= year2)
ds_NAO = ds_NAO.isel(mode=0)

ds_SST = xr.open_dataset(ppdir + "Observations/Data/HadISST_sst.nc", chunks={'time':1})
ds_SST['sst'] = ds_SST['sst'].where(ds_SST['sst'] > -100., np.nan)
ds_SST = ds_SST.rename({'latitude':'lat', 'longitude':'lon'})
ds_SST = ds_SST.sel(time = ds_SST['time.year'] >= year1) 
ds_SST = ds_SST.sel(time = ds_SST['time.year'] <= year2)
ds_SST = ds_SST.get(['sst'])

ds_EN4 = xr.open_mfdataset(ppdir + "Observations/Data/EN4_Met_Office/*/*.nc", chunks={'time':1})
ds_EN4['dz'] = (ds_EN4['depth_bnds'].diff('bnds')).isel(time=0, bnds=0)
ds_EN4 = ds_EN4.get(['temperature', 'dz'])
ds_EN4 = ds_EN4.sel(time = ds_EN4['time.year'] >= year1) 
ds_EN4 = ds_EN4.sel(time = ds_EN4['time.year'] <= year2)
ds_EN4 = ds_EN4.sel(depth = slice(0.,200.))

# get mean temperature in upper 200 m in ds_EN4
dz = ds_EN4['dz'] * ds_EN4['temperature'].isel(time=0) / ds_EN4['temperature'].isel(time=0) # get 3D dz
ds_EN4['temperature_200'] = (ds_EN4['temperature'] * dz).sum('depth') / dz.sum('depth')
ds_EN4 = ds_EN4.get(['temperature_200'])


# Remove climatology
ds_clim = ds_EN4['temperature_200'].groupby('time.month').mean('time') # substract climatology
ds_EN4['temperature_200'] = ds_EN4['temperature_200'].copy().groupby('time.month') - ds_clim

ds_clim = ds_SST['sst'].groupby('time.month').mean('time')
ds_SST['sst'] = ds_SST['sst'].copy().groupby('time.month') - ds_clim

# Remove linear trend
ds_EN4['temperature_200'] = detrend(ds_EN4['temperature_200'], ['time'])
ds_SST['sst'] = detrend(ds_SST['sst'], ['time'])

# High-pass filter
if(num_year > 0):
    var_smooth = Moving_Avg(ds_SST['sst'], time = ds_SST['time'], time_len = mov_avg)
    ds_SST['sst'] = (ds_SST['sst'] - var_smooth)

    var_smooth = Moving_Avg(ds_EN4['temperature_200'], time = ds_EN4['time'], time_len = mov_avg)
    ds_EN4['temperature_200'] = (ds_EN4['temperature_200'] - var_smooth)

print("Data reading complete for observations")

## ----- Get composite means
#nao_var = ['pcs'] #['NAO_point']
nao_var = ['NAO_point', 'pcs']

for nao1 in nao_var:
    
    NAO = ds_NAO[nao1].copy()
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

    ds_res1_SST = [] # dataset for annual function
    ds_res1_EN4 = []
    
    for cas in case:
        
        ds_ens_SST = []
        ds_ens_EN4 = []
        
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

                tmp = ds_SST.copy()
                tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)
                
                ds_ens_SST.append(tmp.drop('time'))

                tmp = ds_EN4.copy()
                tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)
                
                ds_ens_EN4.append(tmp.drop('time'))

        tim = tmp.time
        
        ds_ens_SST = xr.concat(ds_ens_SST, dim='r')
        ds_ens_SST = ds_ens_SST.assign(time = tim)

        ds_ens_EN4 = xr.concat(ds_ens_EN4, dim='r')
        ds_ens_EN4 = ds_ens_EN4.assign(time = tim)

        print("members = ", len(ds_ens_SST['r']), ", case: ", cas, " -- ", nao1)

        # Get annual maps and confidence intervals
        if (cas == 'NAOp'):
            ds_res1_SST.append(ds_ens_SST)
            ds_res1_EN4.append(ds_ens_EN4)
        elif (cas == 'NAOn'):
            d2 = ds_ens_SST.copy()
            ds_ens_SST['sst'] = - d2['sst']
            ds_res1_SST.append(ds_ens_SST.drop('time'))

            d2 = ds_ens_EN4.copy()
            ds_ens_EN4['temperature_200'] = - d2['temperature_200']
            ds_res1_EN4.append(ds_ens_EN4.drop('time'))

    ds_res_SST = xr.concat(ds_res1_SST, dim='r')
    ds_res_SST = ds_res_SST.transpose('r', 'time', 'lat', 'lon')

    ds_res_EN4 = xr.concat(ds_res1_EN4, dim='r')
    ds_res_EN4 = ds_res_EN4.transpose('r', 'time', 'lat', 'lon')

    # compute annual means and bootstrap confidece intervals for SST
    ds_SST_annual = annaul_mean_data(ds_res_SST, ['sst'], ds_res_SST['time'], method = 'mean')
    
    ds_save = xr.Dataset()

    sde_var1 = []; cfd_up_var1 = []; cfd_low_var1 = []

    dim_list = list(ds_SST_annual['sst'].dims[2:])

    var1 = 'sst'
    for yr in range(0, len(ds_SST_annual['year'])):

        data_var = ds_SST_annual[var1].isel(year=yr).compute()
        bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                
        sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
        sde_var1.append(sde)
                
        cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
        cfd_up_var1.append(cfd_up)
                
        cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
        cfd_low_var1.append(cfd_low)
                
    ds_save[var1] = ds_SST_annual[var1].mean('r')
    ds_save[var1 + 'timeseries'] = ds_res_SST[var1].mean('r')
    ds_save[var1 + '_standard_error'] = xr.concat(sde_var1, dim='year') 
    ds_save[var1 + '_confidence_lower'] = xr.concat(cfd_low_var1, dim='year') 
    ds_save[var1 + '_confidence_upper'] = xr.concat(cfd_up_var1, dim='year')

    ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence intervals are at " + str(cf_lev*100) + "%. " + 
                                    "Spatial maps at lag years -1, 0, 1-2, 3-4, 5-6, 7-8, 9-10 (year is Oct-Sep)")

    save_file_path = (ppdir + "Observations/" + "Annual_maps_SST_" +  nao1 + ".nc")

    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
    
    print("SST data saved succefully")

    # compute annual means and bootstrap confidece intervals for EN4
    ds_EN4_annual = annaul_mean_data(ds_res_EN4, ['temperature_200'], ds_res_EN4['time'], method = 'mean')

    ds_save = xr.Dataset()

    sde_var1 = []; cfd_up_var1 = []; cfd_low_var1 = []

    dim_list = list(ds_res_EN4['temperature_200'].dims[2:])

    var1 = 'temperature_200'
    for yr in range(0, len(ds_EN4_annual['year'])):

        data_var = ds_EN4_annual[var1].isel(year=yr).compute()
        bootstrap_ci = data_bootstrap(data_var, cf_lev = cf_lev, num_sample = num_sample)
                
        sde = xr.DataArray(data = bootstrap_ci.standard_error, dims=dim_list)
        sde_var1.append(sde)
                
        cfd_up = xr.DataArray(data = bootstrap_ci.confidence_interval[1], dims=dim_list)
        cfd_up_var1.append(cfd_up)
                
        cfd_low = xr.DataArray(data = bootstrap_ci.confidence_interval[0], dims=dim_list) 
        cfd_low_var1.append(cfd_low)
                
    ds_save[var1] = ds_EN4_annual[var1].mean('r')
    ds_save[var1 + 'timeseries'] = ds_res_EN4[var1].mean('r')
    ds_save[var1 + '_standard_error'] = xr.concat(sde_var1, dim='year') 
    ds_save[var1 + '_confidence_lower'] = xr.concat(cfd_low_var1, dim='year') 
    ds_save[var1 + '_confidence_upper'] = xr.concat(cfd_up_var1, dim='year')

    ds_save.attrs['description'] = ("Bootstrapping standard errors and confidence intervals are at " + str(cf_lev*100) + "%. " + 
                                    "Spatial maps at lag years -1, 0, 1-2, 3-4, 5-6, 7-8, 9-10 (year is Oct-Sep)")

    save_file_path = (ppdir + "Observations/" + "Annual_maps_EN4_" +  nao1 + ".nc")

    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
    
    print("EN4 data saved succefully")

        






