"""
This script can be used to create composites of SST, heat content timeseries based on extreme NAO indices using cmip historical runs. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
import os

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

### ------ Main calculations ------------------

source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 
exp = ['piControl', 'historical']

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"
ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

nao_cut_coef = 1. # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (1, 12) # give range of years before / after NAO onset for composites

var_list = ['NAO', 'sst_subpolar', 'sst_AMV', 'sst_subtropical', 'sst_global', 'sst_subpolar_mid',
           'Heat_Content_North_Atlantic_subpolar', 'Heat_Content_North_Atlantic', 'Heat_Content_North_Atlantic_subpolar_mid',
           'Heat_Content_North_Atlantic_subtropical', 'Heat_Content_Global']

# remove 20-year moving mean
num_year = 20
mov_avg = num_year * 12

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

        # Remove linear drift and climatology
        ds_clim = ds_hist.copy().groupby('time.month').mean('time')
        ds_hist = ds_hist.copy().groupby('time.month') - ds_clim

        ds_clim = ds2.copy().groupby('time.month').mean('time')
        ds2 = ds2.copy().groupby('time.month') - ds_clim

        tmp = ds_hist.copy()
    
        for var in list(tmp.keys()):

            ds_hist[var] = detrend(tmp[var], ds2[var], ['time'])

        # High-pass filter
        if(num_year > 0):
            for var in list(ds_hist.keys()):
                var_smooth = Moving_Avg(ds_hist[var], time = ds_hist['time'], time_len = mov_avg)
                ds_hist[var] = (ds_hist[var] - var_smooth)

        print("Data reading complete for model: ", model)

        # Compute NAO indices
        NAO = (ds_hist['NAO']).copy()
        NAO = NAO.isel(time=slice(2,len(NAO.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
        NAO_season = NAO.resample(time='QS-DEC').mean('time')

        nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
        nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')

        # create composites
        ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)

        for cas in case:
            
            ds_ens = []

            if (cas == 'NAOp'):
                count_NAO = ind_NAOp
            elif (cas == 'NAOn'):
                count_NAO = ind_NAOn
            else:
                print("Choose a valid case")

            # composite for seasonal nao indices
            for year in range(year_str + int(num_year/2), len(nao_DJF) - year_end - int(num_year/2)):
        
                if(count_NAO.isel(time=year) == 1):
        
                    year_val = nao_DJF['time.year'][year]

                    tmp = ds_hist.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)

                    ds_ens.append(tmp.drop('time'))
                    tim = tmp.time

            ds_ens = xr.concat(ds_ens, dim='r')
            ds_ens = ds_ens.assign(time = tim)

            ds_ens.to_netcdf(ppdir + model + exp[1] + "/" + dir_name + "/" + cas + "_Composite_Heat_Content_SST_NAO.nc")

            print("Composite data saved successfully for model: ", model, " - ensemble: ", dir_name)


    

            


