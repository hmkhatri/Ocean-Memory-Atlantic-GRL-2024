"""
This script uses an analytical response function (SST due to single NAO event) with NAO timeseris from HadSLP data  to preditc the actual 
SST timeseries for  arange of tunable parameters (tau, beta)

"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import xskillscore as xs

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

year_int = 30. # length of response function in years

num_year = 1.
mov_avg = int(12. * num_year) # number of months for moving avg

mon_window = 12 * 100. # number of months window for moving correlations

year1, year2 = (1901, 2021) # range years for correlations
rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

tau_rng = np.arange(0.5, 40.1, 0.5)
beta_rng = np.arange(0.0, 10.1, 0.5) 

ds_NAO = xr.open_dataset(ppdir + "Observations/NAO_HadSLP.nc")
ds_SST = xr.open_dataset(ppdir + "Observations/SST_Index_HadISST.nc")


# temporary timeseries for getting the response function right
tim = ds_NAO['time.year'] + ds_NAO['time.month'] / 12. - ds_NAO['time.year'].values[0] - 10.

Response_function_full = []
    
for tau in tau_rng:

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

# use both point-based and pc_based
NAO = ds_NAO['NAO_point'].copy()
NAO_pc = - ds_NAO['pcs'].isel(mode=0).copy()

NAO = detrend(NAO, ['time'])
NAO_pc = detrend(NAO_pc, ['time'])

HC_Pred = xr.zeros_like(NAO)
HC_Pred = HC_Pred / HC_Pred

(tmp, HC_Pred) = xr.broadcast(Response_function_full.isel(time=0), HC_Pred)
HC_Pred = HC_Pred.copy() # otherwise runs into "assignment destination is read-only" error

HC_Pred_pc = HC_Pred.copy()

for j in range(0 + int(12*year_int),len(NAO)):
                
    tmp1 = NAO.isel(time=slice(j-int(12*year_int), j))
    tmp2 = NAO_pc.isel(time=slice(j-int(12*year_int), j))
    
    days = (ds_NAO.time.dt.days_in_month).isel(time=slice(j-int(12*year_int), j))
    
    HC_Pred[:,:,j] = (tmp1 * Response_function_full).sum('time')
    HC_Pred_pc[:,:,j] = (tmp2 * Response_function_full).sum('time')
            
HC_Pred =  - HC_Pred.copy() #/ HC_Pred.std(dim = 'time', skipna = True)
HC_Pred_pc =  - HC_Pred_pc.copy() #/ HC_Pred_pc.std(dim = 'time', skipna = True)

# save dataset
ds_save = xr.Dataset()

ds_save['HC200_Pred'] = HC_Pred
ds_save['HC200_Pred_pc'] = HC_Pred_pc
ds_save['HC200_Pred'].attrs['units'] = "unitless - choose alpha based on SST vs theta"
ds_save['HC200_Pred_pc'].attrs['units'] = "unitless - choose alpha based on SST vs theta"
ds_save['HC200_Pred'].attrs['long_name'] = "Predicted from sinusoidal damping relation - used point SLP differences for NAO"
ds_save['HC200_Pred_pc'].attrs['long_name'] = "Predicted from sinusoidal damping relation - used point EOFs for NAO"

ds_save = ds_save.assign_coords(tau = tau_rng)
ds_save = ds_save.assign_coords(beta = beta_rng)

save_file_path = (ppdir + "Observations/" + "Predict_SST_Observations_1850_2021.nc")
ds_save = ds_save.astype(np.float32).compute()
ds_save.to_netcdf(save_file_path)
    
print("Data saved succefully")
    
ds_save.close()

