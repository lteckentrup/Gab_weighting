import numpy as np
import xarray as xr
import pandas as pd
from numpy.linalg import multi_dot
import os, psutil

def readin(file,var,model):
    if model == 'CRUJRA':
        ds = xr.open_dataset('../../LPJ_monthly_corrected/CRUJRA/'+file+
                             '_CRUJRA_1901-2018.nc')
        ds['Time'] = pd.date_range(start='1901-01-01', end='2018-12-31', freq=freq)
    else:
        ds = xr.open_dataset('../../LPJ_monthly_corrected/original/'+model+'/'+file+
                             '_'+model+'_1850-2100.nc')
        ds['Time'] = pd.date_range(start='1850-01-01', end='2100-12-31', freq=freq)

    return(ds[var].sel(Time=slice('1989','2010')))

### List of GCMs
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
          'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6',
          'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
          'NorESM2-MM']

### Calculate weights
def weighting(file, var):
    ### Target
    ds_CRU = readin(file,var,'CRUJRA') 
    
    ### Dataframe for error matrix
    error_df = pd.DataFrame()
    
    ### List of bias correction terms
    BC_term = []

    for m in models:
        ### Read in raw simulated data
        ds_SIM = readin(file,var,m) 
        
        ### Calculate error: target divided by simulation
        bc_term = ds_CRU/ds_SIM 
        
        ### Average over time
        bc_term = bc_term.mean(dim='Time')
        
        ### Set nan/ not finite numbers to zero (?)
        bc_term = bc_term.where(np.isfinite(bc_term),0)
        
        ### Average bias correction term over entire domain
        BC_term.append(bc_term.mean().values)
        
        ### Correct simulation: simulation times error
        ds_SIM_bc =  ds_SIM * bc_term 
        
        ### New error: target divided by corrected simulation
        ds_ERROR =  ds_CRU/ds_SIM_bc 
        
        ### resulting error matrix: each column = 1 model flattened over all 
        ### timesteps, longitudes, and latitudes
        error_df[m] = ds_ERROR.values.flatten() 
    
    ### Calculate covariance matrix
    M_cov = error_df.cov() 
    model_count = len(M_cov.columns)

    unit_col = np.ones((model_count,1))
    M_cov_inv = np.linalg.pinv(M_cov) ### invert covariance matrix

    unit_transpose = unit_col.transpose()

    weights = np.matmul(M_cov_inv, unit_col)/multi_dot([unit_transpose,
                                                        M_cov_inv,
                                                        unit_col])

    df_bc = pd.DataFrame(BC_term).transpose()
    df_bc.columns = models
    df_weights = pd.DataFrame(weights.transpose(),columns = models)

    df_bc.to_csv('BC_'+file+'_'+var+'.csv')
    df_weights.to_csv('weights_'+file+'_'+var+'.csv')

weighting('cpool', 'Total')
