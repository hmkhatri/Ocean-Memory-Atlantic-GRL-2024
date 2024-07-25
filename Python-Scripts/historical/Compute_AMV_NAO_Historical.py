"""
This script can be used to compute mean SST and total heat content using cmip6 historical simulations. 
Currently, the script is set up for computing mean SST and heat content over North Atlantic and subpolar North Atlantic, and mean sea level pressures for NAO calculations.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
import os

import warnings
warnings.filterwarnings("ignore")

#from dask_mpi import initialize
#initialize()

#from dask.distributed import Client
#client = Client()
                 
### ------ Functions for computations ----------
def xmip_wrapper(ds):
    """ Renaming coordinates and dimensions across cmip models
    """
    ds = ds.copy()
    ds = xmip.rename_cmip6(ds)
    ds = xmip.promote_empty_dims(ds)
    ds = xmip.correct_lon(ds)

    return ds

def area_avg(ds, dA = 1., x='X', y='Y'):
    """Compute spatial-averages
    Parameters
    ----------
    ds : xarray Dataset for data variables
    dA : xarray Dataset for cell areas
    
    Returns
    -------
    ds_mean : timeseris of spatially-averaged dataset
    """
    
    ds_mean = (ds * dA).sum([x, y]) / (dA).sum([x, y])
    
    return ds_mean

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

source_id = ['MOHC/HadGEM3-GC31-MM/', 'NOAA-GFDL/GFDL-CM4/', 'IPSL/IPSL-CM6A-LR/', 'NCAR/CESM2/'] 

experiment_id = ['historical'] 

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

RAD_EARTH = 6.387e6

## -------- Mean SST calculations ------------

var = 'tos' # for spatial averages
var1 = 'psl'
var2 = 'thetao' # spatial integration 
rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

for exp in experiment_id:
    
    for model in source_id:

        # get all historical run ensembles 

        dir_list = glob.glob(cmip_dir + model + exp + "/r*")

        for dir1 in dir_list:

            dir_name = dir1.split('/')[-1].split(',')[0]

            # Ocean data Read

            directory = cmip_dir + model + exp + "/" + dir_name + "/Omon/" + var + "/gn"
            
            if not os.path.exists(directory): # skip if no data
                continue

            #### ocean heat content ######
            ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/" + dir_name + "/Omon/" + 
                                    var2 + "/gn/latest/" + var2 + "*.nc", chunks={'time':1})
            area = xr.open_mfdataset(cmip_dir + model + "/piControl/r1i1p1f1/Ofx/areacello/gn/latest/*nc")

            ds1 = xr.merge([ds1, area['areacello']])
        
            ds1 = xmip_wrapper(ds1)
    
            print("Data reading complete for model: ", model)

            if(model == 'NCAR/CESM2/'):
                ds1 = ds1.sel(lev=slice(0.,100000.)) # get data in upper 1000 m
            else:
                ds1 = ds1.sel(lev=slice(0.,1000.)) # get data in upper 1000 m

            # Perform area-integration
            dz = (ds1['lev_bounds'].diff('bnds')).isel(bnds=0).drop('bnds')
            heat_content = ds1[var2] # dz * rho_cp is multiplied below

            cell_area = ds1['areacello'].where((heat_content.isel(time=0, lev=0) > -10.) & 
                                               (heat_content.isel(time=0, lev=0) < 30.)).compute()

            # 1. North Atlantic Heat Content
            dA = cell_area.where((ds1['lat']>=0.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)).compute()
            Heat_Content_NA = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA = Heat_Content_NA * dz * rho_cp
    
            # 2. Subpolar North Atlantic Heat Content
            dA = cell_area.where((ds1['lat']>=45.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)).compute()
            Heat_Content_NA_subpolar = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA_subpolar = Heat_Content_NA_subpolar * dz * rho_cp
    
            # 3. Subtropical North Atlantic Heat Content
            dA = cell_area.where((ds1['lat']>=0.) & (ds1['lat']<=45.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)).compute() 
            Heat_Content_NA_subtropical = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA_subtropical = Heat_Content_NA_subtropical * dz * rho_cp
    
            # 4. Global Heat Content
            dA = cell_area
            Heat_Content_global = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_global = Heat_Content_global * dz * rho_cp

            # 5. Subpolar 45N-60N, 60W-20W index
            dA = cell_area.where((ds1['lat']>=45.) & (ds1['lat']<=65.) & (ds1['lon']>=300.) & (ds1['lon']<=340.)).compute()
            Heat_Content_NA_subpolar_mid = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA_subpolar_mid = Heat_Content_NA_subpolar_mid * dz * rho_cp

            # ---- save file
            ds_save = xr.Dataset()
            ds_save['Heat_Content_North_Atlantic'] = Heat_Content_NA
            ds_save['Heat_Content_North_Atlantic'].attrs['units'] = "Joules"
            ds_save['Heat_Content_North_Atlantic'].attrs['long_name'] = "North Atlantic Heat Content (0N-70N, 80W-0W) integrated at each depth"
            
            ds_save['Heat_Content_North_Atlantic_subpolar'] = Heat_Content_NA_subpolar
            ds_save['Heat_Content_North_Atlantic_subpolar'].attrs['units'] = "Joules"
            ds_save['Heat_Content_North_Atlantic_subpolar'].attrs['long_name'] = "Subpolar North Atlantic (45N-70N, 80W-0W) Heat Content integrated at each depth"

            ds_save['Heat_Content_North_Atlantic_subpolar_mid'] = Heat_Content_NA_subpolar_mid
            ds_save['Heat_Content_North_Atlantic_subpolar_mid'].attrs['units'] = "Joules"
            ds_save['Heat_Content_North_Atlantic_subpolar_mid'].attrs['long_name'] = "Subpolar North Atlantic (45N-65N, 60W-20W) Heat Content integrated at each depth"
            
            ds_save['Heat_Content_North_Atlantic_subtropical'] = Heat_Content_NA_subtropical
            ds_save['Heat_Content_North_Atlantic_subtropical'].attrs['units'] = "Joules"
            ds_save['Heat_Content_North_Atlantic_subtropical'].attrs['long_name'] = "Subtropical North Atlantic Heat Content (0N-45N, 80W-0W) integrated at each depth"
            
            ds_save['Heat_Content_Global'] = Heat_Content_global
            ds_save['Heat_Content_Global'].attrs['units'] = "Joules"
            ds_save['Heat_Content_Global'].attrs['long_name'] = "Global Heat Content integrated at each depth"

            # Check if the directory exists
            directory = save_path + model + exp + "/" + dir_name
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)
    
            save_file_path = (save_path + model + exp + "/" + dir_name + "/Heat_Content.nc")
            ds_save = ds_save.astype(np.float32).compute()
            ds_save.to_netcdf(save_file_path)
    
            print("Data saved succefully")
    
            ds1.close()
            ds_save.close()
            

            ####### surface temeperatures ######
            
            ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/" + dir_name + "/Omon/" + 
                                    var + "/gn/latest/" + var + "*.nc", chunks={'time':1})
        
            area = xr.open_mfdataset(cmip_dir + model + "/piControl/r1i1p1f1/Ofx/areacello/gn/latest/*nc") 
        
            ds1 = xr.merge([ds1[var], area['areacello']])
        
            ds1 = xmip_wrapper(ds1)
    
            print("Data reading complete for model: ", model)
    
            # Perform area-mean 
            sst = ds1[var]
            cell_area = ds1['areacello'].where((sst.isel(time=0) > -10.) & (sst.isel(time=0) < 30.)).compute()
            
            # 1. AMV index
            dA = cell_area.where((ds1['lat']>=0.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)) 
            sst_AMV = area_avg(sst, dA = dA, x='x', y='y')
    
            # 2. Subpolar SST index
            dA = cell_area.where((ds1['lat']>=45.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)) 
            sst_subpolar = area_avg(sst, dA = dA, x='x', y='y')
    
            # 3. Subtropical SST index
            dA = cell_area.where((ds1['lat']>=0.) & (ds1['lat']<=45.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)) 
            sst_subtropical = area_avg(sst, dA = dA, x='x', y='y')
    
            # 4. Global mean index
            dA = cell_area 
            sst_global = area_avg(sst, dA = dA, x='x', y='y')
    
            # 5. Subpolar 45N-60N, 60W-20W index
            dA = cell_area.where((ds1['lat']>=45.) & (ds1['lat']<=65.) & (ds1['lon']>=300.) & (ds1['lon']<=340.))
            sst_subpolar_mid = area_avg(sst, dA = dA, x='x', y='y')
    
            # ---- save file
            ds_save = xr.Dataset()
            ds_save['sst_AMV'] = sst_AMV
            ds_save['sst_subpolar'] = sst_subpolar
            ds_save['sst_subtropical'] = sst_subtropical
            ds_save['sst_global'] = sst_global
            ds_save['sst_subpolar_mid'] = sst_subpolar_mid

            # Check if the directory exists
            directory = save_path + model + exp + "/" + dir_name
            if not os.path.exists(directory):
                # If it doesn't exist, create it
                os.makedirs(directory)
    
            save_file_path = (save_path + model + exp + "/" + dir_name + "/SST_Index.nc")
            ds_save = ds_save.astype(np.float32).compute()
            ds_save.to_netcdf(save_file_path)
    
            print("Data saved succefully")
    
            ds_save.close()
            ds1.close()

            ####### Atmospheric Data read ######

            ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/" + dir_name + "/Amon/" + var1 + 
                                    "/g*/latest/" + var1 + "*.nc", chunks={'time':1}) 
            
            ds1 = xmip.rename_cmip6(ds1)
        
            print("Data reading complete for model: ", model)
        
            # Compure sea-level pressures
        
            dA = compute_grid_areas(ds1, x='x', y='y', RAD_EARTH = RAD_EARTH) # grid cell areas for area-integration
        
            P_south = ((ds1[var1].sel(y = slice(36., 40.), x = slice(332., 340.)) * 
                        dA.sel(y = slice(36., 40.), x = slice(332., 340.))).sum(['x','y']) / 
                       dA.sel(y = slice(36., 40.), x = slice(332., 340.)).sum(['x','y']))
        
            P_north = ((ds1[var1].sel(y = slice(63., 70.), x = slice(335., 344.)) * 
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
        
            save_file_path = (save_path + model + exp + "/" + dir_name + "/NAO_SLP.nc")
            ds_save = ds_save.astype(np.float32).compute()
            ds_save.to_netcdf(save_file_path)
        
            print("Data saved succefully")
        
            ds_save.close()
            ds1.close()
            