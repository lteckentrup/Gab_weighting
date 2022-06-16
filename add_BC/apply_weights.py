import xarray as xr
import pandas as pd
import numpy as np

models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2-WACCM',
          'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'GFDL-CM4', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6',
          'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',
          'NorESM2-MM']

def apply(model, fname, var):
    ### target
    ds_cru = xr.open_dataset('../LPJ_monthly_corrected/CRUJRA/'+fname+
                             '_CRUJRA_1901-2018.nc')
    ds_cru['Time'] = pd.date_range(start='1901-01-01', end='2018-12-31', freq='A')
    
    ### Read in weights
    weights = pd.read_csv('weights_'+fname+'_'+var+'.csv')
    
    ### Read in raw simulation
    ds = xr.open_dataset('../LPJ_monthly_corrected/original/'+model+'/'+fname+
                         '_'+model+'_1850-2100.nc')

    ds['Time'] = pd.date_range(start='1850-01-01', end='2100-12-31', freq='A')
    
    ### Calculate correction term
    ds_BC = ds.sel(Time=slice('1989','2010')) - \
            ds_cru.sel(Time=slice('1989','2010'))
    
    ### Average over time
    ds_BC = ds_BC.mean(dim='Time')
    ds_var = ds[var]
    
    ### Weight times corrected raw simulation: ds_var-ds_BC[var].values
    ds_weight = weights[model].values*(ds_var-ds_BC[var].values)

    return(ds_weight)

empty = np.zeros([251,68,84])
Time = pd.date_range(start='1850-01-01', end='2100-12-31', freq='A')
Lat = np.arange(-43.75,-9.75,0.5)
Lon = np.arange(112.25,154.25,0.5)

def weighted_array(fname, var):
    da_hybrid = xr.DataArray(empty,
                             coords={'Time': Time,'Lat': Lat, 'Lon': Lon},
                             dims=['Time', 'Lat', 'Lon'])
    
    ### Add all models
    for m in models:
        da_hybrid = da_hybrid + apply(m, fname, var)
    
    da_hybrid = da_hybrid.where((da_hybrid > 0) | da_hybrid.isnull(), 0)

    return(da_hybrid)

da_hybrid = weighted_array('cpool', 'Total')
ds_new = da_hybrid.to_dataset(name='Total')

ds_new.to_netcdf('Total_weighted_1850-2100.nc',
                 encoding={'Time':{'dtype': 'double'},
                           'Lat':{'dtype': 'double'},
                           'Lon':{'dtype': 'double'},
                           'Total':{'dtype': 'float32'}})
