"""
This script can be used to compute mean sea level pressure over Icealand and Azores regions for computing NAO index using cmip6 picontrol simulations. 
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip

import warnings
warnings.filterwarnings("ignore")

#from dask_mpi import initialize
#initialize()

#client = Client()
#from dask.distributed import Client

### ------ Functions for computations ----------
def xmip_wrapper(ds):
    """ Renaming coordinates and dimensions across cmip models
    """
    ds = ds.copy()
    ds = xmip.rename_cmip6(ds)
    ds = xmip.promote_empty_dims(ds)

    return ds

def compute_grid_areas(ds, x='X', y='Y', RAD_EARTH = 6.387e6):

    """Compute grid-cell areas
    Parameters
    ----------
    ds : xarray Dataset for data variables
    
    Returns
    -------
    Cell_area : grid cell areas
    """

    ds = ds.copy()

    dx = np.mean(ds[x].diff(x)) * np.cos(ds[y] * np.pi / 180.) * (2 * np.pi * RAD_EARTH / 360.)
    dy = np.mean(ds[y].diff(y)) * (2 * np.pi * RAD_EARTH / 360.)

    Cell_area = dx * dy

    Cell_area, tmp = xr.broadcast(Cell_area, ds['psl'].isel(time=0))

    return Cell_area


### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

#source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/'] 
source_id = ['NCAR/CESM2/', 'MPI-M/MPI-ESM1-2-HR/']
experiment_id = ['piControl'] #, 'historical'] 

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

var = 'psl'

RAD_EARTH = 6.387e6

for exp in experiment_id:

    for model in source_id:
    
        ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Amon/" + var + 
                                "/g*/latest/" + var + "*.nc", chunks={'time':1}) 
            
        ds1 = xmip_wrapper(ds1)
    
        print("Data reading complete for model: ", model)
    
        # Compure sea-level pressures
    
        dA = compute_grid_areas(ds1, x='x', y='y', RAD_EARTH = RAD_EARTH) # grid cell areas for area-integration
    
        P_south = ((ds1[var].sel(y = slice(36., 40.), x = slice(332., 340.)) * 
                    dA.sel(y = slice(36., 40.), x = slice(332., 340.))).sum(['x','y']) / 
                   dA.sel(y = slice(36., 40.), x = slice(332., 340.)).sum(['x','y']))
    
        P_north = ((ds1[var].sel(y = slice(63., 70.), x = slice(335., 344.)) * 
                    dA.sel(y = slice(63., 70.), x = slice(335., 344.))).sum(['x','y']) / 
                   dA.sel(y = slice(63., 70.), x = slice(335., 344.)).sum(['x','y']))
    
        # save data
        ds_save = xr.Dataset()
        
        ds_save['P_south'] = P_south
        ds_save['P_south'].attrs['units'] = "Pa"
        ds_save['P_south'].attrs['long_name'] = "Mean Sea Level Pressure over Azores Region"
        
        ds_save['P_north'] = P_north
        ds_save['P_north'].attrs['units'] = "Pa"
        ds_save['P_north'].attrs['long_name'] = "Mean Sea Level Pressure over Iceland Region"
    
        save_file_path = (save_path + model + exp + "/NAO_SLP.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")
    
        ds_save.close()
        ds1.close()
    

