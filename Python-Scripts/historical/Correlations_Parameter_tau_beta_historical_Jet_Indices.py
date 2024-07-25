"""
This script uses an analytical form of response function (SST and heat content change due to Jet Indices - speed and latitude), which are created using cmip historical runs, to preditc the actual SST and ocean heat content timeseries.

Correlation coefficient between predicted and actual signals can be compuetd for a range of tuning paramters (tau, beta in response functions) for 
chossing an optimal set of parameters.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import glob
import os
import xskillscore as xs

import warnings
warnings.filterwarnings("ignore")

# ----------- Functions -------------
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

### ------ Main calculations ------------------

source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 
exp = ['piControl', 'historical']

dir_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"
ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

year_int = 30. # length of response function in years

tau_rng = np.arange(0.5, 20.1, 0.5)
beta_rng = np.arange(0.0, 10.1, 0.5) 

#var_list = ['Jet_speed', 'Jet_lat', 'sst_subpolar', 'sst_AMV', 'sst_subtropical', 'sst_global', 'sst_subpolar_mid',
#           'Heat_Content_North_Atlantic_subpolar', 'Heat_Content_North_Atlantic', 'Heat_Content_North_Atlantic_subpolar_mid',
#           'Heat_Content_North_Atlantic_subtropical', 'Heat_Content_Global']

for model in source_id:

    print("Running model: ", model)

    # get all historical run ensembles 
    dir_list = glob.glob(ppdir + model + exp[1] + "/r*")
    
    for dir1 in dir_list:

        dir_name = dir1.split('/')[-1].split(',')[0]

        # read historical timeseries

        d1 = xr.open_dataset(dir_path + model + exp[1] + "/" + dir_name + "/Jet_Indices.nc", use_cftime=True)
        d2 = xr.open_dataset(dir_path + model + exp[1] + "/" + dir_name + "/SST_Index.nc", use_cftime=True)
        d3 = xr.open_dataset(dir_path + model + exp[1] + "/" + dir_name + "/Heat_Content.nc", use_cftime=True)

        if(model == 'NCAR/CESM2/'):
            d3 = d3.sel(lev=slice(0.,20000.)).sum('lev') # get data in upper 200 m
        else:
            d3 = d3.sel(lev=slice(0.,200.)).sum('lev') # get data in upper 200 m

        ds = xr.merge([d1, d2, d3])

        # Remove linear drift
        for var in list(ds.keys()):
            ds[var] = detrend(ds[var], ['time'])
            
        # Remove climatology
        ds_clim = ds.groupby('time.month').mean('time')
        ds = ds.groupby('time.month') - ds_clim
        
        # temporary timeseries for getting the response function right
        tim = ds['time.year'] + ds['time.month'] / 12. - ds['time.year'].values[0] - 10.

        Response_function_full = []
    
        for tau in tau_rng:
    
            # loop for testing beta for sinusodial damping 
    
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

        ds_save = ds.copy()

        # Predict heat content using response function and jet speed timseries
        HC_Pred = xr.zeros_like(ds['Jet_speed'])
        HC_Pred = HC_Pred / HC_Pred
    
        (tmp, HC_Pred) = xr.broadcast(Response_function_full.isel(time=0), HC_Pred)
        HC_Pred = HC_Pred.copy() # otherwise runs into "assignment destination is read-only" error

        for j in range(0 + int(12*year_int),len(ds['time'])):
                    
            tmp1 = ds['Jet_speed'].isel(time=slice(j-int(12*year_int), j))
            days = (ds.time.dt.days_in_month).isel(time=slice(j-int(12*year_int), j))
            HC_Pred[:,:,j] = (tmp1 * Response_function_full).sum('time')
            
        HC_Pred =  - HC_Pred.copy()

        ds_save['HC200_Pred_jet_speed'] = HC_Pred
        ds_save['HC200_Pred_jet_speed'].attrs['units'] = "normalised"
        ds_save['HC200_Pred_jet_speed'].attrs['long_name'] = ("Subpolar North Atlantic upper 200 m Heat Content (45N-70N, 80W-0W) " 
                                                              + "- Predicted from sinusoidal damping relation")

        # Predict heat content using response function and jet lat timseries
        HC_Pred = xr.zeros_like(ds['Jet_lat'])
        HC_Pred = HC_Pred / HC_Pred
    
        (tmp, HC_Pred) = xr.broadcast(Response_function_full.isel(time=0), HC_Pred)
        HC_Pred = HC_Pred.copy() # otherwise runs into "assignment destination is read-only" error

        for j in range(0 + int(12*year_int),len(ds['time'])):
                    
            tmp1 = ds['Jet_lat'].isel(time=slice(j-int(12*year_int), j))
            days = (ds.time.dt.days_in_month).isel(time=slice(j-int(12*year_int), j))
            HC_Pred[:,:,j] = (tmp1 * Response_function_full).sum('time')
            
        HC_Pred =  - HC_Pred.copy()

        ds_save['HC200_Pred_jet_lat'] = HC_Pred
        ds_save['HC200_Pred_jet_lat'].attrs['units'] = "normalised"
        ds_save['HC200_Pred_jet_lat'].attrs['long_name'] = ("Subpolar North Atlantic upper 200 m Heat Content (45N-70N, 80W-0W) " 
                                                              + "- Predicted from sinusoidal damping relation")

        # Correlation coefficients with full timeseries 
        HC200_actual = ds['Heat_Content_North_Atlantic_subpolar'].copy()/1.e20 # normalise to avoid very large numbers in computations 
        
        corr_tmp_HC = xs.pearson_r(ds_save['HC200_Pred_jet_speed'], HC200_actual, dim='time', skipna=True)
        corr_p_tmp_HC = xs.pearson_r_p_value(ds_save['HC200_Pred_jet_speed'], HC200_actual, dim='time', skipna=True)

        ds_save['HC200_Corr_p_jet_speed'] = corr_p_tmp_HC.compute()
        ds_save['HC200_Corr_r_jet_speed'] = corr_tmp_HC.compute()
        ds_save['HC200_Corr_r_jet_speed'].attrs['long_name'] = ("Mean correlation for the full piControl run " + 
                                                                "- upper 200 m heat content in subpolar North Atlantic")
        ds_save['HC200_Corr_p_jet_speed'].attrs['long_name'] = ("p-values in correlation for the full piControl run - " + 
                                                                "upper 200 m heat content in subpolar North Atlantic")

        corr_tmp_HC = xs.pearson_r(ds_save['HC200_Pred_jet_lat'], HC200_actual, dim='time', skipna=True)
        corr_p_tmp_HC = xs.pearson_r_p_value(ds_save['HC200_Pred_jet_lat'], HC200_actual, dim='time', skipna=True)

        ds_save['HC200_Corr_p_jet_lat'] = corr_p_tmp_HC.compute()
        ds_save['HC200_Corr_r_jet_lat'] = corr_tmp_HC.compute()
        ds_save['HC200_Corr_r_jet_lat'].attrs['long_name'] = ("Mean correlation for the full piControl run " + 
                                                                "- upper 200 m heat content in subpolar North Atlantic")
        ds_save['HC200_Corr_p_jet_lat'].attrs['long_name'] = ("p-values in correlation for the full piControl run - " + 
                                                                "upper 200 m heat content in subpolar North Atlantic")

        ds_save = ds_save.assign_coords(tau = tau_rng)
        ds_save = ds_save.assign_coords(beta = beta_rng)

        save_file_path = (ppdir + model + exp[1] + "/" + dir_name + "/Predicting_Heat_Content_SST_historical_Jet_Indices.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
    
        print(dir_name, ": Data saved succefully")
    
        ds_save.close()






