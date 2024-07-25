"""
The script computes correlations between actual heat content (SST) timeseries and predicted timeseries using NAO indices in the North Atlantic.
The correlation coefficients are computed for a range of tau and beta paratemers used in the ocean response function.
"""
# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import xskillscore as xs
import glob

import warnings
warnings.filterwarnings("ignore")

# ------- Functions for calculations ----------
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

ppdir = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

exp = 'historical'
source_id = ['MOHC/HadGEM3-GC31-MM/', 'IPSL/IPSL-CM6A-LR/', 'NOAA-GFDL/GFDL-CM4/', 'NCAR/CESM2/'] 

num_year = 1.
mov_avg = int(12. * num_year) # number of months for moving avg

mon_window = 12 * 100. # number of months window for moving correlations 

for model in source_id:

    print("Running model: ", model)

    dir_list = glob.glob(ppdir + model + exp + "/r*")

    for dir1 in dir_list:

        dir_name = dir1.split('/')[-1].split(',')[0]

        # read historical timeseries

        ds = xr.open_dataset(ppdir + model + exp + "/" + dir_name + "/Predicting_Heat_Content_SST_historical.nc", use_cftime=True)
        #ds = xr.open_dataset(ppdir + model + exp + "/" + dir_name + 
        #                     "/Predicting_Heat_Content_SST_historical_NAO_annual.nc", use_cftime=True)

        ds['HC200_actual'] = ds['Heat_Content_North_Atlantic_subpolar'].copy()/1.e20 # normalise to avoid very large numbers in computations  
        ds['SST_actual'] = ds['sst_subpolar'].copy()

        # linear detrend
        for var in list(ds.keys()):
            ds[var] = detrend(ds[var], ['time'])
        
        # remove signals faster than 1 year  
        ds_mov = ds.copy() #Moving_Avg(ds, time = ds['time'], time_len = mov_avg)

        # Correlation coefficients with full timeseries  
        corr_tmp_HC = xs.pearson_r(ds_mov['HC200_Pred'], ds_mov['HC200_actual'], dim='time', skipna=True)
        corr_tmp_SST = xs.pearson_r(ds_mov['SST_Pred'], ds_mov['SST_actual'], dim='time', skipna=True)
    
        corr_p_tmp_HC = xs.pearson_r_p_value(ds_mov['HC200_Pred'], ds_mov['HC200_actual'], dim='time', skipna=True)
        corr_p_tmp_SST = xs.pearson_r_p_value(ds_mov['SST_Pred'], ds_mov['SST_actual'], dim='time', skipna=True)

        ds_save = xr.Dataset()

        ds_save['HC200_Corr_p'] = corr_p_tmp_HC.compute()
        ds_save['HC200_Corr_r'] = corr_tmp_HC.compute()
    
        ds_save['HC200_Corr_r'].attrs['long_name'] = "Mean correlation for the full piControl run - upper 200 m heat content in subpolar North Atlantic"
        ds_save['HC200_Corr_p'].attrs['long_name'] = "p-values in correlation for the full piControl run - upper 200 m heat content in subpolar North Atlantic"
    
        ds_save['SST_Corr_p'] = corr_p_tmp_SST.compute()
        ds_save['SST_Corr_r'] = corr_tmp_SST.compute()
    
        ds_save['SST_Corr_r'].attrs['long_name'] = "Mean correlation for the full piControl run - SST in subpolar North Atlantic"
        ds_save['SST_Corr_p'].attrs['long_name'] = "p-values in correlation for the full piControl run - SST in subpolar North Atlantic"
    
        # Correlation coefficients with 100-year window
        Cor_HC200 = xr.zeros_like(ds_mov['HC200_Pred'])
        Cor_HC200 = Cor_HC200 / Cor_HC200
        
        Cor_SST = xr.zeros_like(ds_mov['HC200_Pred'])
        Cor_SST = Cor_SST / Cor_SST
    
        Cor_HC200 = Cor_HC200.copy()
        Cor_SST = Cor_SST.copy()
    
        Cor_p_HC200 = Cor_HC200.copy()
        Cor_p_SST = Cor_SST.copy()
        
        for i in range(int(mon_window/2), len(ds_mov['time']) - int(mon_window/2)):
            
            Cor_HC200[:,:,i] = xs.pearson_r(ds_mov['HC200_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                            ds_mov['HC200_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
            Cor_SST[:,:,i] = xs.pearson_r(ds_mov['SST_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                          ds_mov['SST_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
            Cor_p_HC200[:,:,i] = xs.pearson_r_p_value(ds_mov['HC200_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                                      ds_mov['HC200_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                                      dim='time', skipna=True)
    
            Cor_p_SST[:,:,i] = xs.pearson_r_p_value(ds_mov['SST_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                                    ds_mov['SST_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                                    dim='time', skipna=True)
    
        ds_save['SST_Corr_r_moving'] = Cor_SST
        ds_save['HC200_Corr_r_moving'] = Cor_HC200
        
        ds_save['HC200_Corr_r_moving'].attrs['long_name'] = "Mean correlation for 100-year windows - upper 200 m heat content in subpolar North Atlantic"
        ds_save['SST_Corr_r_moving'].attrs['long_name'] = "Mean correlation for 100-year windows - SST in subpolar North Atlantic"
        
        ds_save['SST_Corr_p_moving'] = Cor_p_SST
        ds_save['HC200_Corr_p_moving'] = Cor_p_HC200
        
        ds_save['HC200_Corr_p_moving'].attrs['long_name'] = "p-values for 100-year windows - upper 200 m heat content in subpolar North Atlantic"
        ds_save['SST_Corr_p_moving'].attrs['long_name'] = "p-values for 100-year windows - SST in subpolar North Atlantic"
        
        # Correlation coefficients with low-pass and high-pass filtering
        #timescale = np.arange(2., 100.)
        timescale = np.arange(1., 20.)
        
        Cor_HC200 = xr.zeros_like(ds_mov['HC200_Pred'].isel(time=0))
        Cor_HC200 = Cor_HC200 / Cor_HC200
            
        Cor_SST = xr.zeros_like(ds_mov['HC200_Pred'].isel(time=0))
        Cor_SST = Cor_SST / Cor_SST
        
        Cor_HC200 = Cor_HC200.expand_dims(dim={"HP_LP":2, "timescale": timescale})
        Cor_SST = Cor_SST.expand_dims(dim={"HP_LP":2, "timescale": timescale})
        
        Cor_HC200 = Cor_HC200.copy()
        Cor_SST = Cor_SST.copy()
        
        Cor_p_HC200 = Cor_HC200.copy()
        Cor_p_SST = Cor_SST.copy()
        
        for i in range(0, len(timescale)):
                
            mov_avg_win = int(12. * timescale[i]) 
            ds_low_pass = Moving_Avg(ds, time = ds['time'], time_len = mov_avg_win)
            ds_high_pass = ds_mov - ds_low_pass
        
            Cor_HC200[0,i,:,:] = xs.pearson_r(ds_high_pass['HC200_Pred'], ds_high_pass['HC200_actual'], dim='time', skipna=True)
            Cor_HC200[1,i,:,:] = xs.pearson_r(ds_low_pass['HC200_Pred'], ds_low_pass['HC200_actual'], dim='time', skipna=True)
        
            Cor_SST[0,i,:,:] = xs.pearson_r(ds_high_pass['SST_Pred'], ds_high_pass['SST_actual'], dim='time', skipna=True)
            Cor_SST[1,i,:,:] = xs.pearson_r(ds_low_pass['SST_Pred'], ds_low_pass['SST_actual'], dim='time', skipna=True)
        
            Cor_p_HC200[0,i,:,:] = xs.pearson_r_p_value(ds_high_pass['HC200_Pred'], ds_high_pass['HC200_actual'], dim='time', skipna=True)
            Cor_p_HC200[1,i,:,:] = xs.pearson_r_p_value(ds_low_pass['HC200_Pred'], ds_low_pass['HC200_actual'], dim='time', skipna=True)
        
            Cor_p_SST[0,i,:,:] = xs.pearson_r_p_value(ds_high_pass['SST_Pred'], ds_high_pass['SST_actual'], dim='time', skipna=True)
            Cor_p_SST[1,i,:,:] = xs.pearson_r_p_value(ds_low_pass['SST_Pred'], ds_low_pass['SST_actual'], dim='time', skipna=True)
        
        ds_save['SST_Corr_r_HP_LP'] = Cor_SST
        ds_save['HC200_Corr_r_HP_LP'] = Cor_HC200
        
        ds_save['HC200_Corr_r_HP_LP'].attrs['long_name'] = "Mean correlation for high-pass and low-pass signal - upper 200 m heat content in subpolar North Atlantic"
        ds_save['SST_Corr_r_HP_LP'].attrs['long_name'] = "Mean correlation for high-pass and low-pass signal - SST in subpolar North Atlantic"
        
        ds_save['SST_Corr_p_HP_LP'] = Cor_p_SST
        ds_save['HC200_Corr_p_HP_LP'] = Cor_p_HC200
        
        ds_save['HC200_Corr_p_HP_LP'].attrs['long_name'] = "p-values for high-pass and low-pass signal - upper 200 m heat content in subpolar North Atlantic"
        ds_save['SST_Corr_p_HP_LP'].attrs['long_name'] = "p-values for high-pass and low-pass signal - SST in subpolar North Atlantic"
        
        #save_file_path = (ppdir + model + exp + "/" + dir_name + "/Correlations_Heat_Content_SST_historical.nc")
        save_file_path = (ppdir + model + exp + "/" + dir_name + "/Correlations_Heat_Content_SST_historical_NAO_annual.nc")
        
        ds_save = ds_save.astype(np.float32).compute()
        ds_save.to_netcdf(save_file_path)
            
        print("Data saved succefully")
            
        ds_save.close()
        ds.close()
    



