"""
This script can be used to create composites of heat content (0-200 m and 200-1000m) timeseries based on extreme NAO indices using cmip piControl runs. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend

import warnings
warnings.filterwarnings("ignore")

#from dask_mpi import initialize
#initialize()

#client = Client()
#from dask.distributed import Client

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


### ------ Main calculations ------------------

source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 
experiment_id = ['piControl'] #, 'historical'] 

dir_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

nao_cut_coef = 1.0 # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (1, 12) # give range of years before / after NAO onset for composites

# remove 20-year moving mean
num_year = 20
mov_avg = num_year * 12


for exp in experiment_id:
    
    for model in source_id:

        # Read heat content and NAO data
        d1 = xr.open_mfdataset(dir_path + model + exp + "/Heat_Content/Heat*.nc", use_cftime=True, chunks=None)
        d2 = xr.open_dataset(dir_path + model + exp + "/NAO_SLP.nc", use_cftime=True)

        ds = xr.merge([d1, d2]).load()

        # compute heat content in upper 200 m and 200-1000 m
        if(model == 'NCAR/CESM2/'):
            ds_200 = ds.sel(lev=slice(0.,20000.)).sum('lev') # get data in upper 200 m
            ds_1000 = ds.sel(lev=slice(20000.,100000.)).sum('lev') # get data in 200-1000 m
        else:
            ds_200 = ds.sel(lev=slice(0.,200.)).sum('lev') # get data in upper 200 m
            ds_1000 = ds.sel(lev=slice(200.,1000.)).sum('lev') # get data in 200-1000 m

        # Remove linear drift
        for var in list(ds_200.keys()):

            ds_200[var] = detrend(ds_200[var], ['time'])
            ds_1000[var] = detrend(ds_1000[var], ['time'])

        # Remove climatology
        ds_200_clim = ds_200.groupby('time.month').mean('time')
        ds_200 = ds_200.groupby('time.month') - ds_200_clim
        ds_1000_clim = ds_1000.groupby('time.month').mean('time')
        ds_1000 = ds_1000.groupby('time.month') - ds_1000_clim

        # High-pass filter
        if(num_year > 0):
            for var in list(ds_200.keys()):
                var_smooth = Moving_Avg(ds_200[var], time = ds_200['time'], time_len = mov_avg)
                ds_200[var] = (ds_200[var] - var_smooth)

                var_smooth = Moving_Avg(ds_1000[var], time = ds_1000['time'], time_len = mov_avg)
                ds_1000[var] = (ds_1000[var] - var_smooth)

        print("Data reading complete for model: ", model)

        # Compute NAO indices
        NAO = (ds_200['P_south'] - ds_200['P_north'])
        NAO = NAO.isel(time=slice(2,len(NAO.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
        NAO_season = NAO.resample(time='QS-DEC').mean('time')

        nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
        nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')

        # create composites
        ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)

        for cas in case:
    
            ds_ens_200 = []
            ds_ens_1000 = []

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

                    tmp = ds_200.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)

                    ds_ens_200.append(tmp.drop('time'))

                    tmp = ds_1000.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)

                    ds_ens_1000.append(tmp.drop('time'))
                    
                    tim = tmp.time

            ds_ens_200 = xr.concat(ds_ens_200, dim='r')
            ds_ens_200 = ds_ens_200.assign(time = tim)

            ds_ens_1000 = xr.concat(ds_ens_1000, dim='r')
            ds_ens_1000 = ds_ens_1000.assign(time = tim)

            ds_ens_200.to_netcdf(dir_path + model + exp + "/" + cas + "_Composite_Heat_Content_200.nc")
            ds_ens_1000.to_netcdf(dir_path + model + exp + "/" + cas + "_Composite_Heat_Content_1000.nc")

            print("Heat Content data saved successfully for model: ", model)



