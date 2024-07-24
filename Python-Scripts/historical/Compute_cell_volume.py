"""
This scrip for computing total water volume in subpolar, subtropical (and global) regions considered in the study.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------
def xmip_wrapper(ds):
    """ Renaming coordinates and dimensions across cmip models
    """
    ds = ds.copy()
    ds = xmip.rename_cmip6(ds)
    ds = xmip.promote_empty_dims(ds)
    ds = xmip.correct_lon(ds)

    return ds

def area_sum(ds, dA = 1., x='X', y='Y'):
    """Compute spatial-sums
    Parameters
    ----------
    ds : xarray Dataset for data variables
    dA : xarray Dataset for cell areas
    
    Returns
    -------
    ds_mean : timeseris of spatially-integrated dataset
    """
    
    ds_mean = (ds * dA).sum([x, y])
    
    return ds_mean

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

source_id = ['MOHC/HadGEM3-GC31-MM/', 'NOAA-GFDL/GFDL-CM4/', 'IPSL/IPSL-CM6A-LR/', 'NCAR/CESM2/']
experiment_id = ['piControl']

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

var = 'thetao' # var to get only non-nan cells

for exp in experiment_id:
    
    for model in source_id:

        file_list = glob.glob(cmip_dir + model + exp + "/r1i1p1f1/Omon/" + var + "/gn/latest/" + var + "*nc")
        
        area = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Ofx/areacello/gn/latest/*nc")

        # just get the first file for cell volume calculation

        ds1 = xr.open_dataset(file_list[0], chunks={'time':1, 'lev':1})

        ds1 = xr.merge([ds1, area['areacello']])

        ds1 = xmip_wrapper(ds1)

        if(model == 'NCAR/CESM2/'):
            ds1 = ds1.sel(lev=slice(0.,100000.)) # get data in upper 1000 m
        else:
            ds1 = ds1.sel(lev=slice(0.,1000.)) # get data in upper 1000 m

        print("Data reading complete for model: ", model)

        # cell volume calculations
        
        dz = (ds1['lev_bounds'].diff('bnds')).isel(bnds=0).drop('bnds') # this is in meters even for NCAR/CESM2

        cell_volume = ds1['areacello'] * dz
        
        cell_volume = cell_volume.where((ds1[var].isel(time=0) > -10.) & (ds1[var].isel(time=0) < 30.))

        # Perform area-integration

        # 1. North Atlantic Volume
        dV = cell_volume.where((ds1['lat']>=0.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.))
        Volume_NA = area_sum(dV, dA = 1., x='x', y='y')

        # 2. Subpolar North Atlantic Volume
        dV = cell_volume.where((ds1['lat']>=45.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.))
        Volume_NA_subpolar = area_sum(dV, dA = 1., x='x', y='y')

        # 3. Subtropical North Atlantic Volume
        dV = cell_volume.where((ds1['lat']>=0.) & (ds1['lat']<=45.) & (ds1['lon']>=280.) & (ds1['lon']<=360.))
        Volume_NA_subtropical = area_sum(dV, dA = 1., x='x', y='y')

        # 4. Global Volume
        dV = cell_volume
        Volume_global = area_sum(dV, dA = 1., x='x', y='y')

        # 5. Subpolar 45N-60N, 60W-20W index
        dV = cell_volume.where((ds1['lat']>=45.) & (ds1['lat']<=65.) & (ds1['lon']>=300.) & (ds1['lon']<=340.))
        Volume_NA_subpolar_mid = area_sum(dV, dA = 1., x='x', y='y')

        # ---- save file
        ds_save = xr.Dataset()

        ds_save['Volume_North_Atlantic'] = Volume_NA
        ds_save['Volume_North_Atlantic'].attrs['units'] = "m^3"
        ds_save['Volume_North_Atlantic'].attrs['long_name'] = "North Atlantic Volume (0N-70N, 80W-0W) integrated at each depth"
            
        ds_save['Volume_North_Atlantic_subpolar'] = Volume_NA_subpolar
        ds_save['Volume_North_Atlantic_subpolar'].attrs['units'] = "m^3"
        ds_save['Volume_North_Atlantic_subpolar'].attrs['long_name'] = "Subpolar North Atlantic Volume (45N-70N, 80W-0W) integrated at each depth"

        ds_save['Volume_North_Atlantic_subpolar_mid'] = Volume_NA_subpolar_mid
        ds_save['Volume_North_Atlantic_subpolar_mid'].attrs['units'] = "m^3"
        ds_save['Volume_North_Atlantic_subpolar_mid'].attrs['long_name'] = "Subpolar North Atlantic Volume (45N-65N, 60W-20W) integrated at each depth"
            
        ds_save['Volume_North_Atlantic_subtropical'] = Volume_NA_subtropical
        ds_save['Volume_North_Atlantic_subtropical'].attrs['units'] = "m^3"
        ds_save['Volume_North_Atlantic_subtropical'].attrs['long_name'] = "Subtropical North Atlantic Volume (0N-45N, 80W-0W) integrated at each depth"
            
        ds_save['Volume_Global'] = Volume_global
        ds_save['Volume_Global'].attrs['units'] = "m^3"
        ds_save['Volume_Global'].attrs['long_name'] = "Global Volume integrated at each depth"
    
        save_file_path = (save_path + model + exp + "/Cell_Volume.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
    
        print("Data saved succefully")
    
        ds_save.close()
        ds1.close()


        



        
