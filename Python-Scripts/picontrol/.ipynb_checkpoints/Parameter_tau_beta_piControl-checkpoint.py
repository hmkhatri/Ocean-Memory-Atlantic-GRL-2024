"""
This script uses an analytical form of response function (SST and heat content change due to single NAO event), which are created using cmip piControl runs, to preditc the actual 
SST and ocean heat content timeseries.

Correlation coefficient between predicted and actual signals can be compuetd for a range of tuning paramters (tau, beta in response functions) for 
chossing an optimal set of parameters.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import os

import warnings
warnings.filterwarnings("ignore")

# ------- Functions for calculations ----------
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

# ------ Main code -----------------

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

exp = 'piControl'
source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 

year_int = 30. # length of response function in years

tau_rng = np.arange(0.5, 20.1, 0.5)
beta_rng = np.arange(0.0, 10.1, 0.5) 

var_list = ['NAO', 'sst_subpolar', 'Heat_Content_North_Atlantic_subpolar']

for model in source_id:

    print("Running model: ", model)

    # read picontrol timeseries

    d1 = xr.open_dataset(ppdir + model + exp + "/NAO_SLP.nc", use_cftime=True)
    d2 = xr.open_mfdataset(ppdir + model + exp + "/Heat_Content/*.nc", use_cftime=True)
    d3 = xr.open_dataset(ppdir + model + exp + "/SST_Index.nc", use_cftime=True)

    if(model == 'NCAR/CESM2/'):
        d2 = d2.sel(lev=slice(0.,20000.)).sum('lev') # get data in upper 200 m
    else:
        d2 = d2.sel(lev=slice(0.,200.)).sum('lev') # get data in upper 200 m

    ds = xr.merge([d1, d2, d3])

    ds['NAO'] = ds['P_south'] - ds['P_north']

    # Remove linear trends
    tmp = ds.get(var_list).compute()
    
    for var in list(tmp.keys()):

        tmp[var] = detrend(tmp[var], ['time'])

    # Remove climatology
    ds_clim = tmp.groupby('time.month').mean('time')
    ds = tmp.groupby('time.month') - ds_clim

    # temporary timeseries for getting the response function right
    tim = ds['time.year'] + ds['time.month'] / 12. - ds['time.year'].values[0] - 10.

    NAO = ds['NAO'].copy()

    #HC_Pred = []
    #SST_Pred = []

    Response_function_full = []
    
    for tau in tau_rng:

        # loop for testing beta for sinusodial damping 
        
        #HC_Pred_beta = []
        #SST_Pred_beta = []

        Response_function_tau = []
        
        for beta in beta_rng:

            # Response function for Sinusoidal Damping term
            Response_function1 = np.exp(-(tim) / (tau)) * ( np.cos(tim * beta / tau))
            Response_function1 = xr.where(tim < 0., 0., Response_function1)
            Response_function1 = xr.where((tim > 1.5 * np.pi * tau/beta), 0., Response_function1)
        
            Response_function1 = Response_function1.isel(time=slice(10*12 - 1, 10*12-1 + int(12*year_int)))
            Response_function1 = Response_function1.isel(time=slice(None, None, -1)).drop('time')

            Response_function_tau.append(Response_function1)

        Response_function_tau = xr.concat(Response_function_tau, dim="beta")

        Response_function_full.append(Response_function_tau)

    Response_function_full = xr.concat(Response_function_full, dim="tau")

    # Predict heat content using response function and NAO timseries
    HC_Pred = xr.zeros_like(NAO)
    HC_Pred = HC_Pred / HC_Pred

    (tmp, HC_Pred) = xr.broadcast(Response_function_full.isel(time=0), HC_Pred)
    HC_Pred = HC_Pred.copy() # otherwise runs into "assignment destination is read-only" error

    for j in range(0 + int(12*year_int),len(NAO)):
                
        tmp1 = NAO.isel(time=slice(j-int(12*year_int), j))
        days = (ds.time.dt.days_in_month).isel(time=slice(j-int(12*year_int), j))
        HC_Pred[:,:,j] = (tmp1 * Response_function_full).sum('time') #/ (Response_function_full * days).sum('time') 

    #std_norm = (ds['Heat_Content_North_Atlantic_subpolar'].isel(time=slice(int(12*year_int), len(NAO)))/1e20).std('time')
            
    HC_Pred =  - HC_Pred.copy() #* std_norm / HC_Pred.std(dim = 'time', skipna = True)

    #std_norm = (ds['sst_subpolar'].isel(time=slice(int(12*year_int), len(NAO)))).std('time')

    SST_Pred = HC_Pred.copy() #* std_norm / HC_Pred.std(dim = 'time', skipna = True)

    # Save data

    ds_save = xr.Dataset()

    ds_save['HC200_Pred'] = HC_Pred
    ds_save['HC200_actual'] = ds['Heat_Content_North_Atlantic_subpolar'].copy() / 1e20 # normalize to avoid large values in computations

    ds_save['SST_Pred'] = SST_Pred
    ds_save['SST_actual'] = ds['sst_subpolar']

    ds_save['HC200_Pred'].attrs['units'] = "normalised"
    ds_save['HC200_actual'].attrs['units'] = "10^20 Joules"
    ds_save['SST_Pred'].attrs['units'] = "normalised"
    ds_save['SST_actual'].attrs['units'] = "Deg C"

    ds_save['HC200_Pred'].attrs['long_name'] = "Subpolar North Atlantic upper 200 m Heat Content (45N-70N, 80W-0W) - Predicted from sinusoidal damping relation"
    ds_save['HC200_actual'].attrs['long_name'] = "Subpolar North Atlantic upper 200 m Heat Content (45N-70N, 80W-0W) - Actual from piControl"

    ds_save['SST_Pred'].attrs['long_name'] = "Subpolar North Atlantic SST (45N-70N, 80W-0W) - Predicted from sinusoidal damping relation"
    ds_save['SST_actual'].attrs['long_name'] = "Subpolar North Atlantic SST (45N-70N, 80W-0W) - Actual from piControl"

    ds_save = ds_save.assign_coords(tau = tau_rng)
    ds_save = ds_save.assign_coords(beta = beta_rng)
    
    save_file_path = (save_path + model + exp + "/Predicting_Heat_Content_SST_piControl.nc")

    # temporary directory
    #directory = "/home/users/hkhatri/tmp/" + model 
    #if not os.path.exists(directory):
    #    # If it doesn't exist, create it
    #    os.makedirs(directory)
    #save_file_path = directory + "/Predicting_Heat_Content_SST_piControl.nc"
    # -------------------
    
    ds_save = ds_save.astype(np.float32).compute()
    ds_save.to_netcdf(save_file_path)
    
    print("Data saved succefully")
    
    ds_save.close()
    ds.close()













            

            