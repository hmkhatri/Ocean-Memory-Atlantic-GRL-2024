"""
This script can be used to create composites of SST timeseries based on extreme NAO indices using cmip piControl runs. 
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
#source_id = ['NCAR/CESM2/', 'MPI-M/MPI-ESM1-2-HR/']
experiment_id = ['piControl'] #, 'historical'] 

dir_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

nao_cut_coef = 1. # 1 for outside 1 x sigma and 2 for outside 2 x sigma
case = ['NAOp', 'NAOn']
year_str, year_end = (1, 12) # give range of years before / after NAO onset for composites

# remove 20-year moving mean
num_year = 20
mov_avg = num_year * 12

for exp in experiment_id:
    
    for model in source_id:

        # Read SST and NAO data
        d1 = xr.open_dataset(dir_path + model + exp + "/SST_Index.nc", use_cftime=True)
        d2 = xr.open_dataset(dir_path + model + exp + "/NAO_SLP.nc", use_cftime=True)
    
        ds = xr.merge([d1, d2])

        # Remove linear drift
        for var in list(ds.keys()):
            
            #tmp = linear_trend(ds[var], 'time')
            #tmp_trend = tmp.slope.values * np.linspace(0., len(ds['time']), num = len(ds['time']))
            #da = xr.DataArray(data=tmp_trend, dims=["time"])
            #ds[var] = ds[var] - da

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

        # Compute NAO indices
        NAO = (ds['P_south'] - ds['P_north'])
        NAO = NAO.isel(time=slice(2,len(NAO.time)-1)) # get rid of first Jan-Feb and last Dec for seasonal avg
        NAO_season = NAO.resample(time='QS-DEC').mean('time')

        nao_cut = nao_cut_coef * NAO_season.std('time', skipna=True).values
        nao_DJF = NAO_season.sel(time = NAO_season['time.season'] == 'DJF')

        # create composites
        ind_NAOp = xr.where(nao_DJF >= nao_cut, 1, 0)
        ind_NAOn = xr.where(nao_DJF <= -nao_cut, 1, 0)
        
        for cas in case:
    
            nao_ens = []
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
        
                    tmp1 = NAO_season
                    tmp1 = tmp1.sel(time = tmp1['time.year'] >= year_val - year_str)
                    tmp1 = tmp1.sel(time = tmp1['time.year'] <= year_val + year_end)
        
                    nao_ens.append(tmp1.drop('time'))
                    tim1 = tmp1.time

                    tmp = ds.copy()
                    tmp = tmp.sel(time = tmp['time.year'] >= year_val - year_str)
                    tmp = tmp.sel(time = tmp['time.year'] <= year_val + year_end)

                    ds_ens.append(tmp.drop('time'))
                    tim = tmp.time
                
            nao_ens = xr.concat(nao_ens, dim='r')
            nao_ens = xr.DataArray.to_dataset(nao_ens, name='NAO_seasonal')
            nao_ens = nao_ens.assign(time = tim1)

            ds_ens = xr.concat(ds_ens, dim='r')
            ds_ens = ds_ens.assign(time = tim)

            nao_ens.to_netcdf(dir_path + model + exp + "/" + cas + "_Composite_NAO.nc")

            ds_ens.to_netcdf(dir_path + model + exp + "/" + cas + "_Composite_SST.nc")

            print("NAO and SST data saved successfully for model: ", model)

            

        




