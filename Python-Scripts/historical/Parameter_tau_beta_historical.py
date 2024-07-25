"""
This script uses an analytical form of response function (SST and heat content change due to single NAO event), 
which are created using cmip historical runs, to preditc the actual 
SST and ocean heat content timeseries.

Correlation coefficient between predicted and actual signals can be compuetd for a range of tuning paramters 
(tau, beta in response functions) for choosing an optimal set of parameters.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import glob
import os

import warnings
warnings.filterwarnings("ignore")

# ------- Functions for calculations ----------
def detrend_dim(da, da_ref, dim, deg=1):
    # detrend along a single dimension
    p = da_ref.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, da_ref, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, da_ref, dim, deg=deg)
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

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

exp = ['piControl', 'historical']
source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 

year_int = 30. # length of response function in years

tau_rng = np.arange(0.5, 20.1, 0.5)
beta_rng = np.arange(0.0, 10.1, 0.5) 

var_list = ['NAO', 'sst_subpolar', 'sst_AMV', 'sst_subtropical', 'sst_global', 'sst_subpolar_mid',
           'Heat_Content_North_Atlantic_subpolar', 'Heat_Content_North_Atlantic', 'Heat_Content_North_Atlantic_subpolar_mid',
           'Heat_Content_North_Atlantic_subtropical', 'Heat_Content_Global']

for model in source_id:

    print("Running model: ", model)

    # read picontrol timeseries

    d1 = xr.open_dataset(ppdir + model + exp[0] + "/NAO_SLP.nc", use_cftime=True)
    d2 = xr.open_dataset(ppdir + model + exp[0] + "/SST_Index.nc", use_cftime=True)
    d3 = xr.open_mfdataset(ppdir + model + exp[0] + "/Heat_Content/*.nc", use_cftime=True)

    if(model == 'NCAR/CESM2/'):
        d3 = d3.sel(lev=slice(0.,20000.)).sum('lev') # get data in upper 200 m
    else:
        d3 = d3.sel(lev=slice(0.,200.)).sum('lev') # get data in upper 200 m

    ds_ctr = xr.merge([d1, d2, d3])

    ds_ctr['NAO'] = ds_ctr['P_south'] - ds_ctr['P_north']

    ds_ctr = ds_ctr.get(var_list)

    # get all historical run ensembles 

    dir_list = glob.glob(ppdir + model + exp[1] + "/r*")

    for dir1 in dir_list:

        dir_name = dir1.split('/')[-1].split(',')[0]

        # read historical timeseries

        d1 = xr.open_dataset(ppdir + model + exp[1] + "/" + dir_name + "/NAO_SLP.nc", use_cftime=True)
        d2 = xr.open_dataset(ppdir + model + exp[1] + "/" + dir_name + "/SST_Index.nc", use_cftime=True)
        d3 = xr.open_dataset(ppdir + model + exp[1] + "/" + dir_name + "/Heat_Content.nc", use_cftime=True)

        if(model == 'NCAR/CESM2/'):
            d3 = d3.sel(lev=slice(0.,20000.)).sum('lev') # get data in upper 200 m
        else:
            d3 = d3.sel(lev=slice(0.,200.)).sum('lev') # get data in upper 200 m

        ds_hist = xr.merge([d1, d2, d3])

        ds_hist['NAO'] = ds_hist['P_south'] - ds_hist['P_north']

        ds_hist = ds_hist.get(var_list)

        # read tos timeseries to get the branching time of historical run from picontrol run
        ds_tmp = xr.open_mfdataset(cmip_dir + model + exp[1] + "/" + dir_name + 
                                   "/Omon/tos/gn/latest/tos*.nc", chunks={'time':1})
        # find index of branch out point and chunk the relevant part of picontrol simulation
        hist_branch_day = ds_tmp.attrs['branch_time_in_parent']

        print("Branch_parent = ", ds_tmp.attrs['branch_time_in_parent'], ", Branch child = ", ds_tmp.attrs['branch_time_in_child'])

        tim1 = ds_ctr['time']
        day_count = tim1.dt.days_in_month.cumsum('time')
        day_count = day_count.where(day_count < hist_branch_day)
        start_idx = (day_count / day_count).sum('time') #count nan-points, which are days before branching out

        ds2 = ds_ctr.isel(time=slice(int(start_idx.values), int(start_idx.values) + len(ds_hist['time']))).drop('time')
        ds2 = ds2.assign_coords(time=ds_hist['time'])

        # Remove linear model drift and climatology
        ds_clim = ds_hist.copy().groupby('time.month').mean('time')
        ds_hist = ds_hist.copy().groupby('time.month') - ds_clim

        ds_clim = ds2.copy().groupby('time.month').mean('time')
        ds2 = ds2.copy().groupby('time.month') - ds_clim

        tmp = ds_hist.copy()
    
        for var in list(tmp.keys()):

            ds_hist[var] = detrend(tmp[var], ds2[var], ['time'])

        # temporary timeseries for getting the response function right
        tim = ds_hist['time.year'] + ds_hist['time.month'] / 12. - ds_hist['time.year'].values[0] - 10.
    
        NAO = ds_hist['NAO'].copy()

        # use annual NAO timeseries rathar monthly for reconstruction 
        # (this is just to check if temporalfiltergin has any effect - there is no notable change in final results)
        # (in results shown in Khatri et al 2024, monthly NAO timeseries was used) 
        #NAO = Moving_Avg(NAO, time = NAO['time'], time_len = 12)

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

        # Predict heat content using response function and NAO timseries
        HC_Pred = xr.zeros_like(NAO)
        HC_Pred = HC_Pred / HC_Pred
    
        (tmp, HC_Pred) = xr.broadcast(Response_function_full.isel(time=0), HC_Pred)
        HC_Pred = HC_Pred.copy() # otherwise runs into "assignment destination is read-only" error
    
        for j in range(0 + int(12*year_int),len(NAO)):
                    
            tmp1 = NAO.isel(time=slice(j-int(12*year_int), j))
            days = (ds_hist.time.dt.days_in_month).isel(time=slice(j-int(12*year_int), j))
            HC_Pred[:,:,j] = (tmp1 * Response_function_full).sum('time') #/ (Response_function_full * days).sum('time')

        #std_norm = (ds_hist['Heat_Content_North_Atlantic_subpolar'].isel(time=slice(int(12*year_int), len(NAO)))/1e20).std('time')
            
        HC_Pred =  - HC_Pred.copy() #* std_norm / HC_Pred.std(dim = 'time', skipna = True)
    
        #std_norm = (ds_hist['sst_subpolar'].isel(time=slice(int(12*year_int), len(NAO)))).std('time')
    
        SST_Pred = HC_Pred.copy() #* std_norm / HC_Pred.std(dim = 'time', skipna = True)

        ds_save = ds_hist.copy()

        ds_save['HC200_Pred'] = HC_Pred
        ds_save['HC200_Pred'].attrs['units'] = "normalised"
        ds_save['HC200_Pred'].attrs['long_name'] = "Subpolar North Atlantic upper 200 m Heat Content (45N-70N, 80W-0W) - Predicted from sinusoidal damping relation"

        ds_save['SST_Pred'] = SST_Pred
        ds_save['SST_Pred'].attrs['units'] = "normalised"
        ds_save['SST_Pred'].attrs['long_name'] = "Subpolar North Atlantic SST (45N-70N, 80W-0W) - Predicted from sinusoidal damping relation"

        ds_save = ds_save.assign_coords(tau = tau_rng)
        ds_save = ds_save.assign_coords(beta = beta_rng)

        save_file_path = (save_path + model + exp[1] + "/" + dir_name + "/Predicting_Heat_Content_SST_historical.nc")
        #save_file_path = (save_path + model + exp[1] + "/" + dir_name + "/Predicting_Heat_Content_SST_historical_NAO_annual.nc")
        
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")
    
        ds_save.close()

        
        
        




        

