"""
This script uses an analytical response function (SST and heat content change due to single NAO event) with NAO timeseris from HadSLP data to predict the actual SST and ocean heat content anomaly timeseries.

Correlation coefficient between predicted and actual signals can be compuetd for a range of tuning paramters (tau, beta in response functions) for 
chossing an optimal set of parameters.
"""

# ------- load libraries ------------
import xarray as xr
import numpy as np
import xmip.preprocessing as xmip
from xarrayutils.utils import linear_trend
import xskillscore as xs

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

save_path = "/gws/nopw/j04/snapdragon/hkhatri/CMIP6_Ctr_Hist/"

year_int = 30. # length of response function in years

num_year = 1.
mov_avg = int(12. * num_year) # number of months for moving avg

mon_window = 12 * 100. # number of months window for moving correlations

year1, year2 = (1901, 2021) # range years for correlations
rho_cp = 4.09 * 1.e6 # constant from Williams et al. 2015

tau_rng = np.arange(0.5, 20.1, 0.5)
beta_rng = np.arange(0.0, 10.1, 0.5) 

ds_NAO = xr.open_dataset(ppdir + "Observations/NAO_HadSLP.nc")
ds_SST = xr.open_dataset(ppdir + "Observations/SST_Index_HadISST.nc")
ds_EN4 = xr.open_dataset(ppdir + "Observations/Heat_Content_EN4.nc")
ds_EN4 = ds_EN4.sel(depth=slice(0,200.)).sum('depth') # upper 200 m sum

# temporary timeseries for getting the response function right
tim = ds_NAO['time.year'] + ds_NAO['time.month'] / 12. - ds_NAO['time.year'].values[0] - 10.

Response_function_full = []
    
for tau in tau_rng:

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

# use both point-based and pc_based
NAO = ds_NAO['NAO_point'].copy()
NAO_pc = - ds_NAO['pcs'].isel(mode=0).copy()

NAO = detrend(NAO, ['time'])
NAO_pc = detrend(NAO_pc, ['time'])

HC_Pred = xr.zeros_like(NAO)
HC_Pred = HC_Pred / HC_Pred

(tmp, HC_Pred) = xr.broadcast(Response_function_full.isel(time=0), HC_Pred)
HC_Pred = HC_Pred.copy() # otherwise runs into "assignment destination is read-only" error

HC_Pred_pc = HC_Pred.copy()

for j in range(0 + int(12*year_int),len(NAO)):
                
    tmp1 = NAO.isel(time=slice(j-int(12*year_int), j))
    tmp2 = NAO_pc.isel(time=slice(j-int(12*year_int), j))
    
    days = (ds_NAO.time.dt.days_in_month).isel(time=slice(j-int(12*year_int), j))
    
    HC_Pred[:,:,j] = (tmp1 * Response_function_full).sum('time')
    HC_Pred_pc[:,:,j] = (tmp2 * Response_function_full).sum('time')
            
HC_Pred =  - HC_Pred.copy() #/ HC_Pred.std(dim = 'time', skipna = True)
HC_Pred_pc =  - HC_Pred_pc.copy() #/ HC_Pred_pc.std(dim = 'time', skipna = True)

# save dataset
ds_save = xr.Dataset()

ds_save['HC200_Pred'] = HC_Pred
ds_save['HC200_Pred_pc'] = HC_Pred_pc
ds_save['HC200_Pred'].attrs['units'] = "unitless - choose alpha based on SST vs theta"
ds_save['HC200_Pred_pc'].attrs['units'] = "unitless - choose alpha based on SST vs theta"
ds_save['HC200_Pred'].attrs['long_name'] = "Predicted from sinusoidal damping relation - used point SLP differences for NAO"
ds_save['HC200_Pred_pc'].attrs['long_name'] = "Predicted from sinusoidal damping relation - used point EOFs for NAO"

ds_save = ds_save.assign_coords(tau = tau_rng)
ds_save = ds_save.assign_coords(beta = beta_rng)

# compute correlations (remove signals faster than 1 year)
ds_mov = ds_save.sel(time = ds_save['time.year'] >= year1); 
ds_mov = ds_mov.sel(time = ds_mov['time.year'] <= year2)

tmp_SST = ds_SST['sst_subpolar'].copy()
tmp_SST = tmp_SST.sel(time = tmp_SST['time.year'] >= year1); tmp_SST = tmp_SST.sel(time = tmp_SST['time.year'] <= year2);
ds_mov['SST_actual'] = tmp_SST.drop('time')

tmp_HC = ds_EN4['Heat_Content_North_Atlantic_subpolar'].copy() /1.e20 # normalise to avoid large values in calculations
tmp_HC = tmp_HC.sel(time = tmp_HC['time.year'] >= year1); tmp_HC = tmp_HC.sel(time = tmp_HC['time.year'] <= year2);
ds_mov['HC200_actual'] = tmp_HC.drop('time')

# add rest of the observed vars
ds_save = ds_save.sel(time = ds_save['time.year'] >= year1); 
ds_save = ds_save.sel(time = ds_save['time.year'] <= year2)

reg_list = ['North_Atlantic', 'North_Atlantic_subpolar', 'North_Atlantic_subpolar_mid', 'North_Atlantic_subtropical', 'Global']
for reg in reg_list:
    tmp_var = ds_EN4['Heat_Content_' + reg] / (ds_EN4['Volume_' + reg] * rho_cp)
    tmp_var = tmp_var.sel(time = tmp_var['time.year'] >= year1); tmp_var = tmp_var.sel(time = tmp_var['time.year'] <= year2);
    ds_save['Heat_Content_' + reg] = tmp_var.drop('time')

sst_list = ['sst_subpolar', 'sst_AMV', 'sst_subpolar_mid', 'sst_global']
for var in sst_list:
    tmp_var = ds_SST[var]
    tmp_var = tmp_var.sel(time = tmp_var['time.year'] >= year1); tmp_var = tmp_var.sel(time = tmp_var['time.year'] <= year2);
    ds_save[var] = tmp_var.drop('time')

for var in list(ds_mov.keys()): # linear detrend
    ds_mov[var] = detrend(ds_mov[var], ['time'])
    
 
corr_tmp_HC = xs.pearson_r(ds_mov['HC200_Pred'], ds_mov['HC200_actual'], dim='time', skipna=True)
corr_tmp_SST = xs.pearson_r(ds_mov['HC200_Pred'], ds_mov['SST_actual'], dim='time', skipna=True)

corr_p_tmp_HC = xs.pearson_r_p_value(ds_mov['HC200_Pred'], ds_mov['HC200_actual'], dim='time', skipna=True)
corr_p_tmp_SST = xs.pearson_r_p_value(ds_mov['HC200_Pred'], ds_mov['SST_actual'], dim='time', skipna=True)

ds_save['HC200_Corr_p'] = corr_p_tmp_HC.compute()
ds_save['HC200_Corr_r'] = corr_tmp_HC.compute()

ds_save['HC200_Corr_r'].attrs['long_name'] = "Mean correlation for EN4 observations (1901-2021) - upper 200 m heat content in subpolar North Atlantic"
ds_save['HC200_Corr_p'].attrs['long_name'] = "p-values in correlation for EN4 observations (1901-2021) - upper 200 m heat content in subpolar North Atlantic"

ds_save['SST_Corr_p'] = corr_p_tmp_SST.compute()
ds_save['SST_Corr_r'] = corr_tmp_SST.compute()

ds_save['SST_Corr_r'].attrs['long_name'] = "Mean correlation for HadISST observations (1901-2021) - SST in subpolar North Atlantic"
ds_save['SST_Corr_p'].attrs['long_name'] = "p-values in correlation for HadISST observations (1901-2021) - SST in subpolar North Atlantic"

corr_tmp_HC = xs.pearson_r(ds_mov['HC200_Pred_pc'], ds_mov['HC200_actual'], dim='time', skipna=True)
corr_tmp_SST = xs.pearson_r(ds_mov['HC200_Pred_pc'], ds_mov['SST_actual'], dim='time', skipna=True)

corr_p_tmp_HC = xs.pearson_r_p_value(ds_mov['HC200_Pred_pc'], ds_mov['HC200_actual'], dim='time', skipna=True)
corr_p_tmp_SST = xs.pearson_r_p_value(ds_mov['HC200_Pred_pc'], ds_mov['SST_actual'], dim='time', skipna=True)

ds_save['HC200_pc_Corr_p'] = corr_p_tmp_HC.compute()
ds_save['HC200_pc_Corr_r'] = corr_tmp_HC.compute()

ds_save['HC200_pc_Corr_r'].attrs['long_name'] = ("Mean correlation for EN4 observations (1901-2021) - upper 200 m heat content in subpolar North Atlantic" +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
ds_save['HC200_pc_Corr_p'].attrs['long_name'] = ("p-values in correlation for EN4 observations (1901-2021) - upper 200 m heat content in subpolar North Atlantic" +
                                               "Used first mode pc from EOFs of SLP as NAO index.")

ds_save['SST_pc_Corr_p'] = corr_p_tmp_SST.compute()
ds_save['SST_pc_Corr_r'] = corr_tmp_SST.compute()

ds_save['SST_pc_Corr_r'].attrs['long_name'] = ("Mean correlation for HadISST observations (1901-2021) - SST in subpolar North Atlantic" +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
ds_save['SST_pc_Corr_p'].attrs['long_name'] = ("p-values in correlation for HadISST observations (1901-2021) - SST in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")

# Correlation coefficients with 100-year window
Cor_HC200 = xr.zeros_like(ds_mov['HC200_Pred']); Cor_HC200 = Cor_HC200 / Cor_HC200
Cor_SST = xr.zeros_like(ds_mov['HC200_Pred']); Cor_SST = Cor_SST / Cor_SST
    
Cor_HC200 = Cor_HC200.copy(); Cor_SST = Cor_SST.copy()
Cor_p_HC200 = Cor_HC200.copy(); Cor_p_SST = Cor_SST.copy()

Cor_HC200_pc = Cor_HC200.copy(); Cor_SST_pc = Cor_SST.copy()
Cor_p_HC200_pc = Cor_HC200.copy(); Cor_p_SST_pc = Cor_SST.copy()

for i in range(int(mon_window/2), len(ds_mov['time']) - int(mon_window/2)):

    Cor_HC200[:,:,i] = xs.pearson_r(ds_mov['HC200_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                    ds_mov['HC200_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
    Cor_SST[:,:,i] = xs.pearson_r(ds_mov['HC200_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                  ds_mov['SST_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
    Cor_p_HC200[:,:,i] = xs.pearson_r_p_value(ds_mov['HC200_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                              ds_mov['HC200_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
    Cor_p_SST[:,:,i] = xs.pearson_r_p_value(ds_mov['HC200_Pred'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                            ds_mov['SST_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)

    Cor_HC200_pc[:,:,i] = xs.pearson_r(ds_mov['HC200_Pred_pc'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                    ds_mov['HC200_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
    Cor_SST_pc[:,:,i] = xs.pearson_r(ds_mov['HC200_Pred_pc'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                  ds_mov['SST_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
    Cor_p_HC200_pc[:,:,i] = xs.pearson_r_p_value(ds_mov['HC200_Pred_pc'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                              ds_mov['HC200_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)
    
    Cor_p_SST_pc[:,:,i] = xs.pearson_r_p_value(ds_mov['HC200_Pred_pc'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), 
                                            ds_mov['SST_actual'].isel(time=slice(i-int(mon_window/2), i+int(mon_window/2))), dim='time', skipna=True)

ds_save['SST_Corr_r_moving'] = Cor_SST
ds_save['HC200_Corr_r_moving'] = Cor_HC200
        
ds_save['HC200_Corr_r_moving'].attrs['long_name'] = "Mean correlation for 100-year windows - upper 200 m heat content in subpolar North Atlantic"
ds_save['SST_Corr_r_moving'].attrs['long_name'] = "Mean correlation for 100-year windows - SST in subpolar North Atlantic"
        
ds_save['SST_Corr_p_moving'] = Cor_p_SST
ds_save['HC200_Corr_p_moving'] = Cor_p_HC200
        
ds_save['HC200_Corr_p_moving'].attrs['long_name'] = "p-values for 100-year windows - upper 200 m heat content in subpolar North Atlantic"
ds_save['SST_Corr_p_moving'].attrs['long_name'] = "p-values for 100-year windows - SST in subpolar North Atlantic"

ds_save['SST_Corr_r_moving_pc'] = Cor_SST_pc
ds_save['HC200_Corr_r_moving_pc'] = Cor_HC200_pc
        
ds_save['HC200_Corr_r_moving_pc'].attrs['long_name'] = ("Mean correlation for 100-year windows - upper 200 m heat content in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
ds_save['SST_Corr_r_moving_pc'].attrs['long_name'] = ("Mean correlation for 100-year windows - SST in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
        
ds_save['SST_Corr_p_moving_pc'] = Cor_p_SST_pc
ds_save['HC200_Corr_p_moving_pc'] = Cor_p_HC200_pc
        
ds_save['HC200_Corr_p_moving_pc'].attrs['long_name'] = ("p-values for 100-year windows - upper 200 m heat content in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
ds_save['SST_Corr_p_moving_pc'].attrs['long_name'] = ("p-values for 100-year windows - SST in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")

# Correlation coefficients with low-pass and high-pass filtering
timescale = np.arange(2., 100.)
        
Cor_HC200 = xr.zeros_like(ds_mov['HC200_Pred'].isel(time=0)); Cor_HC200 = Cor_HC200 / Cor_HC200
Cor_SST = xr.zeros_like(ds_mov['HC200_Pred'].isel(time=0)); Cor_SST = Cor_SST / Cor_SST
        
Cor_HC200 = Cor_HC200.expand_dims(dim={"HP_LP":2, "timescale": timescale})
Cor_SST = Cor_SST.expand_dims(dim={"HP_LP":2, "timescale": timescale})
        
Cor_HC200 = Cor_HC200.copy(); Cor_SST = Cor_SST.copy()
Cor_p_HC200 = Cor_HC200.copy(); Cor_p_SST = Cor_SST.copy()

Cor_HC200_pc = Cor_HC200.copy(); Cor_SST_pc = Cor_SST.copy()
Cor_p_HC200_pc = Cor_HC200.copy(); Cor_p_SST_pc = Cor_SST.copy()

for i in range(0, len(timescale)):
                
    mov_avg_win = int(12. * timescale[i]) 
    ds_low_pass = Moving_Avg(ds_mov, time = ds_mov['time'], time_len = mov_avg_win)
    ds_high_pass = ds_mov - ds_low_pass
        
    Cor_HC200[0,i,:,:] = xs.pearson_r(ds_high_pass['HC200_Pred'], ds_high_pass['HC200_actual'], dim='time', skipna=True)
    Cor_HC200[1,i,:,:] = xs.pearson_r(ds_low_pass['HC200_Pred'], ds_low_pass['HC200_actual'], dim='time', skipna=True)
        
    Cor_SST[0,i,:,:] = xs.pearson_r(ds_high_pass['HC200_Pred'], ds_high_pass['SST_actual'], dim='time', skipna=True)
    Cor_SST[1,i,:,:] = xs.pearson_r(ds_low_pass['HC200_Pred'], ds_low_pass['SST_actual'], dim='time', skipna=True)
        
    Cor_p_HC200[0,i,:,:] = xs.pearson_r_p_value(ds_high_pass['HC200_Pred'], ds_high_pass['HC200_actual'], dim='time', skipna=True)
    Cor_p_HC200[1,i,:,:] = xs.pearson_r_p_value(ds_low_pass['HC200_Pred'], ds_low_pass['HC200_actual'], dim='time', skipna=True)
        
    Cor_p_SST[0,i,:,:] = xs.pearson_r_p_value(ds_high_pass['HC200_Pred'], ds_high_pass['SST_actual'], dim='time', skipna=True)
    Cor_p_SST[1,i,:,:] = xs.pearson_r_p_value(ds_low_pass['HC200_Pred'], ds_low_pass['SST_actual'], dim='time', skipna=True)
        
    Cor_HC200_pc[0,i,:,:] = xs.pearson_r(ds_high_pass['HC200_Pred_pc'], ds_high_pass['HC200_actual'], dim='time', skipna=True)
    Cor_HC200_pc[1,i,:,:] = xs.pearson_r(ds_low_pass['HC200_Pred_pc'], ds_low_pass['HC200_actual'], dim='time', skipna=True)
        
    Cor_SST_pc[0,i,:,:] = xs.pearson_r(ds_high_pass['HC200_Pred_pc'], ds_high_pass['SST_actual'], dim='time', skipna=True)
    Cor_SST_pc[1,i,:,:] = xs.pearson_r(ds_low_pass['HC200_Pred_pc'], ds_low_pass['SST_actual'], dim='time', skipna=True)
        
    Cor_p_HC200_pc[0,i,:,:] = xs.pearson_r_p_value(ds_high_pass['HC200_Pred_pc'], ds_high_pass['HC200_actual'], dim='time', skipna=True)
    Cor_p_HC200_pc[1,i,:,:] = xs.pearson_r_p_value(ds_low_pass['HC200_Pred_pc'], ds_low_pass['HC200_actual'], dim='time', skipna=True)
        
    Cor_p_SST_pc[0,i,:,:] = xs.pearson_r_p_value(ds_high_pass['HC200_Pred_pc'], ds_high_pass['SST_actual'], dim='time', skipna=True)
    Cor_p_SST_pc[1,i,:,:] = xs.pearson_r_p_value(ds_low_pass['HC200_Pred_pc'], ds_low_pass['SST_actual'], dim='time', skipna=True)

ds_save['SST_Corr_r_HP_LP'] = Cor_SST
ds_save['HC200_Corr_r_HP_LP'] = Cor_HC200
        
ds_save['HC200_Corr_r_HP_LP'].attrs['long_name'] = "Mean correlation for high-pass and low-pass signal - upper 200 m heat content in subpolar North Atlantic"
ds_save['SST_Corr_r_HP_LP'].attrs['long_name'] = "Mean correlation for high-pass and low-pass signal - SST in subpolar North Atlantic"
        
ds_save['SST_Corr_p_HP_LP'] = Cor_p_SST
ds_save['HC200_Corr_p_HP_LP'] = Cor_p_HC200
        
ds_save['HC200_Corr_p_HP_LP'].attrs['long_name'] = "p-values for high-pass and low-pass signal - upper 200 m heat content in subpolar North Atlantic"
ds_save['SST_Corr_p_HP_LP'].attrs['long_name'] = "p-values for high-pass and low-pass signal - SST in subpolar North Atlantic"

ds_save['SST_Corr_r_HP_LP_pc'] = Cor_SST_pc
ds_save['HC200_Corr_r_HP_LP_pc'] = Cor_HC200_pc
        
ds_save['HC200_Corr_r_HP_LP_pc'].attrs['long_name'] = ("Mean correlation for high-pass and low-pass signal - upper 200 m heat content in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
ds_save['SST_Corr_r_HP_LP_pc'].attrs['long_name'] = ("Mean correlation for high-pass and low-pass signal - SST in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
        
ds_save['SST_Corr_p_HP_LP_pc'] = Cor_p_SST_pc
ds_save['HC200_Corr_p_HP_LP_pc'] = Cor_p_HC200_pc
        
ds_save['HC200_Corr_p_HP_LP_pc'].attrs['long_name'] = ("p-values for high-pass and low-pass signal - upper 200 m heat content in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")
ds_save['SST_Corr_p_HP_LP_pc'].attrs['long_name'] = ("p-values for high-pass and low-pass signal - SST in subpolar North Atlantic." +
                                               "Used first mode pc from EOFs of SLP as NAO index.")

save_file_path = (ppdir + "Observations/" + "Predict_Correlations_Heat_Content_SST_Observations.nc")
ds_save = ds_save.astype(np.float32).compute()
ds_save.to_netcdf(save_file_path)
    
print("Data saved succefully")
    
ds_save.close()




