"""
This script can be used to compute mean SST and total heat content using cmip6 simulations. 
Currently, the script is set up for computing mean SST over North Atlantic and subpolar North Atlantic.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob

import warnings
warnings.filterwarnings("ignore")

from dask_mpi import initialize
initialize()

from dask.distributed import Client
client = Client()
                 
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

### ------ Main calculations ------------------

cmip_dir = "/badc/cmip6/data/CMIP6/CMIP/"

source_id = ['MOHC/HadGEM3-GC31-MM/', 'NOAA-GFDL/GFDL-CM4/', 'IPSL/IPSL-CM6A-LR/', 'NCAR/CESM2/'] 
#source_id = ['MPI-M/MPI-ESM1-2-HR/']
experiment_id = ['piControl'] #, 'historical'] 

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

## -------- Mean SST calculations ------------

"""

var = 'tos' # for spatial averages

for exp in experiment_id:
    
    for model in source_id:

        # Data Read
    
        ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Omon/" + 
                                var + "/gn/latest/" + var + "*.nc", chunks={'time':1})
    
        area = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Ofx/areacello/gn/latest/*nc") 
    
        #basin = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Ofx/basin/gn/latest/*nc") 
    
        ds1 = xr.merge([ds1[var], area['areacello']]) #, basin['basin']])
    
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
        ds_save['sst_subtropical_mid'] = sst_subtropical_mid

        save_file_path = (save_path + model + exp + "/SST_Index_new.nc")
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)

        print("Data saved succefully")

        ds_save.close()
        ds1.close()
        
"""
## -------- Heat content calculations at each level ------------

var = 'thetao' # spatial integration 
rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

for exp in experiment_id:
    
    for model in source_id:

        file_list = glob.glob(cmip_dir + model + exp + "/r1i1p1f1/Omon/" + var + "/gn/latest/" + var + "*nc")

        # first get name of vertical dimension for proper chunking
        #tmp = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Omon/" + var + 
        #                        "/gn/latest/" + var + "*185*.nc", chunks={'time':1}) # chunks={'time':1, tmp[var].dims[1]:1}

        # Data Read
    
        #ds1 = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Omon/" + var + 
        #                        "/gn/latest/" + var + "*.nc", chunks={'time':1, tmp[var].dims[1]:1})

        area = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Ofx/areacello/gn/latest/*nc") 
        
        #basin = xr.open_mfdataset(cmip_dir + model + exp + "/r1i1p1f1/Ofx/basin/gn/latest/*nc") 

        for i in range(0,len(file_list)):

            ds1 = xr.open_dataset(file_list[i], chunks={'time':1})
            #ds1 = xr.decode_cf(ds1)
        
            ds1 = xr.merge([ds1, area['areacello']]) #, basin['basin']])
        
            ds1 = xmip_wrapper(ds1)

            # don't use this rechunking. dak-mpi fails due to out-of-memory issue with rechuncking. don't know why.
            #ds1 = ds1.chunk({"time": 1, "lev":10, "x":-1, "y":-1}) 
    
            print("Data reading complete for model: ", model)
    
            if(model == 'NCAR/CESM2/'):
                ds1 = ds1.sel(lev=slice(0.,100000.)) # get data in upper 1000 m
            else:
                ds1 = ds1.sel(lev=slice(0.,1000.)) # get data in upper 1000 m
    
            # Perform area-integration
            dz = (ds1['lev_bounds'].diff('bnds')).isel(bnds=0).drop('bnds')
            heat_content = ds1[var] # dz * rho_cp is multiplied below

            cell_area = ds1['areacello'].where((heat_content.isel(time=0, lev=0) > -10.) & 
                                               (heat_content.isel(time=0, lev=0) < 30.)).compute()
    
            # 1. North Atlantic Heat Content
            dA = cell_area.where((ds1['lat']>=0.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)).compute()
            #dV  = (dA * dz * rho_cp) #.compute()
            Heat_Content_NA = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA = Heat_Content_NA * dz * rho_cp
    
            # 2. Subpolar North Atlantic Heat Content
            dA = cell_area.where((ds1['lat']>=45.) & (ds1['lat']<=70.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)).compute()
            #dV  = (dA * dz * rho_cp) #.compute()
            Heat_Content_NA_subpolar = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA_subpolar = Heat_Content_NA_subpolar * dz * rho_cp
    
            # 3. Subtropical North Atlantic Heat Content
            dA = cell_area.where((ds1['lat']>=0.) & (ds1['lat']<=45.) & (ds1['lon']>=280.) & (ds1['lon']<=360.)).compute() 
            #dV  = (dA * dz * rho_cp) #.compute()
            Heat_Content_NA_subtropical = area_sum(heat_content, dA = dA, x='x', y='y')
            Heat_Content_NA_subtropical = Heat_Content_NA_subtropical * dz * rho_cp
    
            # 4. Global Heat Content
            dA = cell_area
            #dV  = (dA * dz * rho_cp) #.compute()
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
    
            save_file_path = (save_path + model + exp + "/Heat_Content/Heat_Content_" + str(i+1) + ".nc")
            ds_save = ds_save.astype(np.float32).compute()
            ds_save.to_netcdf(save_file_path)
    
            print("Data saved succefully")
    
            ds_save.close()
            ds1.close()

