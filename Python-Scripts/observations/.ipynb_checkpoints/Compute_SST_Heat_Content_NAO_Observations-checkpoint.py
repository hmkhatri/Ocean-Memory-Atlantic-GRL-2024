"""
This script can be used to compute NAO indices, mean SST and total heat content using HadISST, HadSLP and EN4 data. 
Currently, the script is set up for computing mean SST over North Atlantic and subpolar North Atlantic.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
import glob
from xeofs.xarray import EOF

import warnings
warnings.filterwarnings("ignore")

### ------ Functions for computations ----------
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

def compute_grid_areas(ds, var='psl', x='X', y='Y', RAD_EARTH = 6.387e6):

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

    Cell_area = np.abs(dx * dy)

    Cell_area, tmp = xr.broadcast(Cell_area, ds[var])

    return Cell_area

### ------ Main calculations ------------------
ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

RAD_EARTH = 6.387e6
rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015
year1, year2 = (1901, 2020) # for removing climatology

# read SLP, SST and EN4 data
ds_SST = xr.open_dataset(ppdir + "Observations/Data/HadISST_sst.nc", chunks={'time':1})
ds_SST['sst'] = ds_SST['sst'].where(ds_SST['sst'] > -100., np.nan)
ds_SST = ds_SST.rename({'latitude':'lat', 'longitude':'lon'})

ds_SLP = xr.open_dataset(ppdir + "Observations/Data/HadSLP2r.nc", chunks={'time':1})
ds_SLP = ds_SLP.rename({'latitude':'lat', 'longitude':'lon'})

ds_EN4 = xr.open_mfdataset(ppdir + "Observations/Data/EN4_Met_Office/*/*.nc", chunks={'depth':1})
ds_EN4['dz'] = (ds_EN4['depth_bnds'].diff('bnds')).isel(time=0, bnds=0)
ds_EN4 = ds_EN4.get(['temperature', 'dz'])

# compute grid areas and cell volumes
area = compute_grid_areas(ds_SST, var='sst', x='lon', y='lat', RAD_EARTH = RAD_EARTH) # grid cell areas for area-integration
ds_SST['cell_area'] = area.where(ds_SST['sst'] > -100.)

area = compute_grid_areas(ds_EN4, var='temperature', x='lon', y='lat', RAD_EARTH = RAD_EARTH) 
ds_EN4['cell_volume'] = (area * ds_EN4['dz']).where(ds_EN4['temperature'] > -1000.)

print("Data read complete")

# ------------- NAO compute ––––––––––––- #

tmp = ds_SLP.copy().sel(time = ds_SLP['time.year'] >= year1) 
tmp = tmp.sel(time = tmp['time.year'] <= year2)

ds_clim = tmp.groupby('time.month').mean('time') # substract climatology
ds_SLP = ds_SLP.copy().groupby('time.month') - ds_clim

ds_save = ds_save = xr.Dataset()

ds_save['P_south'] = ds_SLP['unknown'].sel(lat=40., lon=-25.) 
ds_save['P_north'] = ds_SLP['unknown'].sel(lat=65., lon=-20.)

ds_save['NAO_point'] = (ds_save['P_south'] - ds_save['P_north']) # NAO based on station-pressure differences

ds_save['P_south'].attrs['units'] = "hPa"
ds_save['P_north'].attrs['units'] = "hPa"
ds_save['NAO_point'].attrs['units'] = "hPa"

psl_anom = ds_SLP['unknown']

psl_anom = psl_anom.sel(lon = slice(-90., 41.))
psl_anom = psl_anom.isel(lat=slice(2,15)) #where((psl_anom.lat >= 20)) #& (psl_anom.lat <= 80))

model = EOF(psl_anom, n_modes=4, norm=False, weights='coslat', dim='time')
model.solve()
ds_save['expvar'] = model.explained_variance_ratio()
ds_save['eofs'] = model.eofs()
ds_save['pcs'] = model.pcs() # NAO based on EOFs

save_file_path = (ppdir + "Observations/NAO_HadSLP.nc")
save_file_path = "/home/users/hkhatri/tmp/NAO_HadSLP.nc"
ds_save = ds_save.astype(np.float32).compute()
ds_save.to_netcdf(save_file_path)

print("NAO Data saved succefully")

ds_save.close()

# ------------ Mean SST ----------------- #
tmp = ds_SST.copy().sel(time = ds_SST['time.year'] >= year1) 
tmp = tmp.sel(time = tmp['time.year'] <= year2)

ds_clim = tmp.groupby('time.month').mean('time') # substract climatology

sst = ds_SST['sst'].copy().groupby('time.month') - ds_clim['sst']
cell_area = ds_SST['cell_area'].copy()
        
# 1. AMV index (mean SST anomaly over the whole North Atlantic)
dA = cell_area.where((ds_SST['lat']>=0.) & (ds_SST['lat']<=70.) & (ds_SST['lon']>=-80.) & (ds_SST['lon']<=0.)) 
sst_AMV = area_avg(sst, dA = dA, x='lon', y='lat')

# 2. Subpolar SST index
dA = cell_area.where((ds_SST['lat']>=45.) & (ds_SST['lat']<=70.) & (ds_SST['lon']>=-80.) & (ds_SST['lon']<=0.)) 
sst_subpolar = area_avg(sst, dA = dA, x='lon', y='lat')

# 3. Subtropical SST index
dA = cell_area.where((ds_SST['lat']>=0.) & (ds_SST['lat']<=45.) & (ds_SST['lon']>=-80.) & (ds_SST['lon']<=0.)) 
sst_subtropical = area_avg(sst, dA = dA, x='lon', y='lat')

# 4. Global mean index
dA = cell_area 
sst_global = area_avg(sst, dA = dA, x='lon', y='lat')

# 5. Subpolar 45N-60N, 60W-20W index
dA = cell_area.where((ds_SST['lat']>=45.) & (ds_SST['lat']<=65.) & (ds_SST['lon']>=-60.) & (ds_SST['lon']<=-20.))
sst_subpolar_mid = area_avg(sst, dA = dA, x='lon', y='lat')

# ---- save file
ds_save = xr.Dataset()
ds_save['sst_AMV'] = sst_AMV
ds_save['sst_subpolar'] = sst_subpolar
ds_save['sst_subtropical'] = sst_subtropical
ds_save['sst_global'] = sst_global
ds_save['sst_subpolar_mid'] = sst_subpolar_mid

save_file_path = (ppdir + "Observations/SST_Index_HadISST.nc")
ds_save = ds_save.astype(np.float32).compute()
ds_save.to_netcdf(save_file_path)

print("SST Data saved succefully")

ds_save.close()


# ------------ Mean Heat Content --------- #
tmp = ds_EN4.copy().sel(time = ds_EN4['time.year'] >= year1) 
tmp = tmp.sel(time = tmp['time.year'] <= year2)

ds_clim = tmp.groupby('time.month').mean('time') # substract climatology

heat_content = ds_EN4['temperature'].copy().groupby('time.month') - ds_clim['temperature']
cell_volume = ds_EN4['cell_volume'].copy()

# 1. North Atlantic Heat Content
dV = cell_volume.where((ds_EN4['lat']>=0.) & (ds_EN4['lat']<=70.) & (ds_EN4['lon']>=280.) & (ds_EN4['lon']<=360.))
Heat_Content_NA = area_sum(heat_content, dA = dV, x='lon', y='lat')
Heat_Content_NA = Heat_Content_NA * rho_cp

Volume_NA = dV.sum(['lat', 'lon'])
    
# 2. Subpolar North Atlantic Heat Content
dV = cell_volume.where((ds_EN4['lat']>=45.) & (ds_EN4['lat']<=70.) & (ds_EN4['lon']>=280.) & (ds_EN4['lon']<=360.))
Heat_Content_NA_subpolar = area_sum(heat_content, dA = dV, x='lon', y='lat')
Heat_Content_NA_subpolar = Heat_Content_NA_subpolar * rho_cp

Volume_NA_subpolar = dV.sum(['lat', 'lon'])

# 3. Subtropical North Atlantic Heat Content
dV = cell_volume.where((ds_EN4['lat']>=0.) & (ds_EN4['lat']<=45.) & (ds_EN4['lon']>=280.) & (ds_EN4['lon']<=360.))
Heat_Content_NA_subtropical = area_sum(heat_content, dA = dV, x='lon', y='lat')
Heat_Content_NA_subtropical = Heat_Content_NA_subtropical * rho_cp

Volume_NA_subtropical = dV.sum(['lat', 'lon'])
    
# 4. Global Heat Content
dV = cell_volume
Heat_Content_global = area_sum(heat_content, dA = dV, x='lon', y='lat')
Heat_Content_global = Heat_Content_global * rho_cp

Volume_global = dV.sum(['lat', 'lon'])

# 5. Subpolar 45N-60N, 60W-20W index
dV = cell_volume.where((ds_EN4['lat']>=45.) & (ds_EN4['lat']<=65.) & (ds_EN4['lon']>=300.) & (ds_EN4['lon']<=340.))
Heat_Content_NA_subpolar_mid = area_sum(heat_content, dA = dV, x='lon', y='lat')
Heat_Content_NA_subpolar_mid = Heat_Content_NA_subpolar_mid * rho_cp

Volume_NA_subpolar_mid = dV.sum(['lat', 'lon'])

# save file
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

save_file_path = (ppdir + "Observations/Heat_Content_EN4.nc")
ds_save = ds_save.astype(np.float32).compute()
ds_save.to_netcdf(save_file_path)

print("EN4 heat content data saved succefully")

ds_save.close()
