import xarray as xr
import numpy as np
import datetime as dt
from datetime import date
import ipdb
from importlib import reload
import pandas as pd
import xesmf as xe
import multiprocessing
import os

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct
import create_anntraining_LENS as caL


if __name__ == "__main__":

    ### Set the models
    models = ['ACCESS-ESM1-5','CanESM5','cesm2','miroc6','MIROC-ES2L','MPI-ESM1-2-LR'] # 30 enemslbes
    models = ['cesm2'] # 100 members
    models = ['ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1','cesm2', 'EC-Earth3CC',
            'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'UKESM1-0-LL'] # Full models
    models = ['obs1','obs2','obs3','obs4'] # full obs
    models = ['ding22_aice_avg','cesm1_sic_avg','ding23_aice_avg','cesm2_sic_avg']  # For nudging

    ### Set the vars
    psl_vars=[m+'_psl_en' for m in models]; 
    sic_vars=[m+'_sic_en' for m in models]
    sst_vars=[m+'_sst_en' for m in models]
    tas_vars=[m+'_tas_en' for m in models]
    vars = psl_vars + sic_vars + sst_vars + tas_vars # including everything
    vars = tas_vars
    vars = psl_vars + sic_vars + tas_vars
    vars = psl_vars
    vars = psl_vars + tas_vars
    vars = sic_vars

    ### Set the ratio
    psl_ratios = {var:1 for var in psl_vars}; psl_ratios['obs4_psl_en']=100
    # Change (0-100) to (0-1)
    sic_ratios={var:0.01 for var in sic_vars}; sic_ratios['cesm2_sic_en']=1; sic_ratios['cesm2_sic_avg_sic_en']=1;sic_ratios['ding23_aice_avg_sic_en']=1
    sic_ratios['obs1_sic_en']=1; sic_ratios['obs2_sic_en']=1; sic_ratios['obs3_sic_en']=1
    sst_ratios = {var:1 for var in sst_vars}
    tas_ratios = {var:1 for var in tas_vars}
    var_ratios = {**psl_ratios, **sic_ratios, **sst_ratios, **tas_ratios}

    ### Set the ensembles
    en_obs={'obs1':[''],'obs1':[''],'obs2':[''],'obs3':[''],'obs4':['']}
    en_nudging={'ding22_aice_avg':[''],'cesm1_sic_avg':[''],'ding23_aice_avg':[''],'cesm2_sic_avg':['']}
    en_model={m:range(1,2) for m in models} # For fast testing
    en_model={m:range(1,101) for m in models} # For CESM2 only
    en_model={'ACCESS-ESM1-5':range(1,41),'ACCESS-CM2':range(1,11),'CanESM5':range(1,51),'CanESM5-1':range(1,11),'cesm2':range(1,51),'EC-Earth3CC':range(1,11),
            'IPSL-CM6A-LR':range(1,11),'miroc6':range(1,51),'MIROC-ES2L':range(1,31),'MPI-ESM1-2-LR':range(1,31),'MPI-ESM1-2-HR':range(1,11),
            'UKESM1-0-LL':range(1,11)} # For all models
    en_model={m:range(1,11) for m in models} # most models
    ens={**en_model, **en_obs, **en_nudging}
    ensembles = {m+'_%s_en'%var: ens[m] for m in models for var in ['psl','sic','sst','tas']}

    ### Set the periods
    if True: # 20-year (For obs, some periods are limited by data length)
        #hist_st_yr=1980; hist_end_yr=2023 # For observations
        hist_st_yr=1950; hist_end_yr=2055 # For models - the best period
        #hist_st_yr=1850; hist_end_yr=2100 # For models - for sensitivity test
        slicing_period=20; interval=1 # Standard
    else: # 42-year. To create 1980-2022 PSL for obs and models (Never use again)
        hist_st_yr=1980; hist_end_yr=2020; slicing_period=40; interval=1 # For nudging experiment in supp
        hist_st_yr=1980; hist_end_yr=2022; slicing_period=42; interval=1 # For obs

    ### Start the jobs
    argus = [(var,ensembles[var],var_ratios[var],hist_st_yr,hist_end_yr,slicing_period,interval) for i, var in enumerate(vars)]
    reload(caL); reload(tools)
    if False:
        pool_no=int(len(vars))
        pool = multiprocessing.Pool(pool_no)
        pool.starmap(caL.create_training, argus)
    else:  # For testing and debug for the first variables
        for argu in argus:
            caL.create_training(argu[0],argu[1],argu[2],argu[3],argu[4],argu[5],argu[6]) 

def create_training(var, ensembles, var_ratio, hist_st_yr, hist_end_yr, slicing_period, interval):

    def extract_monthly_trend_slp(data_raw, st_yr, end_yr, mons, lats, lons): 
        # 1:Seasonal-mean from monthly data 
        st_yr=str(st_yr).zfill(4); end_yr=str(end_yr).zfill(4)
        st_mon=str(mons[0]); end_mon=str(mons[-1])
        data = data_raw.sel(time=slice('%s-%s-01'%(st_yr,st_mon), '%s-%s-28'%(end_yr,end_mon)))
        mon_mask = data.time.dt.month.isin(mons)
        data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
        # 2: Get trend
        x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
        y=data; ymean=y.mean(dim='time')
        slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 # per deacde
        # 3: Regrid
        ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
        regridder = xe.Regridder(slope.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
        slope = regridder(slope)
        # Weight by cos lat
        #cos_lats = np.cos(np.radians(slope.latitude)) # Do the cos(lat) weight to each grid 
        #slope = (slope*cos_lats).values.reshape(-1)
        return slope

    def extract_monthly_trend_sic(data_raw, st_yr, end_yr, mons, lats, lons): 
        # 1: Seasonal-mean from monthly data 
        # 2: Define sea ice extent
        # 3: Extract the BKS timeseries
        # 4: Get trend
        st_yr=str(st_yr).zfill(4); end_yr=str(end_yr).zfill(4)
        st_mon=str(mons[0]); end_mon=str(mons[-1])
        data = data_raw.sel(time=slice('%s-%s-01'%(st_yr,st_mon), '%s-%s-28'%(end_yr,end_mon)))
        mon_mask = data.time.dt.month.isin(mons)
        data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()

        #lat1=68; lat2=85; lon1=5; lon2=90 # Old region
        lat1=70; lat2=82; lon1=15; lon2=100 ; #For the region in Koenigk et al. 2016 

        # 2:Regrid data (Find a way to avoid double regridding - All are regrided to 1x1 originally)
        #ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
        #regridder = xe.Regridder(data.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
        #data = regridder(data)

        # 4:Extract the BKS region (0-100) or (0-1)
        if False: # For area calculation
            data = xr.where(data<=0.15, np.nan, 1) # Sea ice extent (good for area calculation)
            #data = xr.where(data<=0.15, np.nan, data) # Sea ice area
            data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
            #data=data.mean(dim='latitude').mean(dim='longitude')
            area_grid = np.empty((data.latitude.size, data.longitude.size))
            area_grid[:,:] = np.diff(data.latitude).mean()*111.7 * np.diff(data.longitude).mean()*111.7
            coords = {'latitude':data.latitude, 'longitude':data.longitude}
            area_grid = xr.DataArray(area_grid, dims=[*coords], coords=coords)
            cos_lats = np.cos(area_grid.latitude*np.pi/180)
            area_grid = area_grid * cos_lats
            # Get the summation of sea ice area
            data=(data*area_grid).sum(dim='latitude',skipna=True).sum(dim='longitude',skipna=True)/1e6 #unit as miilion km square
        else: 
            if True: # Sea ice extent (%)
                mask_nan = np.isnan(data); # Get the nan values (all NAN values are the land grids)
                data = xr.where(data<=0.15, 0, 1) # The nan grids will also turn to 1 because these grids are false
                data = xr.where(mask_nan, np.nan, data) # Put the nan back to the data
            else: # Sea ice area - no threshold
                pass
            # Get the SIC cover over BKS
            lons=data.longitude.values; lats=data.latitude.values
            data_ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
            data=xr.DataArray(data_ts,dims=['time'],coords={'time':data.time}) * 100 # from (0-1) to (0-100)

        # Get trend
        x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
        y=data; ymean=y.mean(dim='time')
        slope = ((x-xmean)*(y-ymean)).sum(dim='time',skipna=True) / ((x-xmean)**2).sum(dim='time') * 10 # sea ice change per decade

        return slope

    def extract_monthly_mean_sic(data_raw, st_yr, end_yr, mons=[12,1,2]): # Not used anymore

        # Wintertime mean SIC across the start and end yr
        # 1:Seasonal-mean from monthly data 
        st_mon=str(mons[0]); end_mon=str(mons[-1])
        data = data_raw.sel(time=slice('%s-%s-01'%(st_yr,st_mon), '%s-%s-28'%(end_yr,end_mon)))
        mon_mask = data.time.dt.month.isin(mons) 
        data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
        data = data.mean(dim='time')

        # 2:Remove the too small sic
        #data = xr.where(data<=0.15, np.nan, data)

        # 3:Extract the BKS region (0-100) or (0-1)
        #lat1=68; lat2=85; lon1=5; lon2=90
        lat1=70; lat2=82; lon1=15; lon2=100 ; #For the region in Koenigk et al. 2016 
        data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))

        return data

    ### We start here
    #folder_name='sic_old'
    #folder_name='sic_area'
    folder_name=var[-6:-3]
    for en in ensembles:
        data_raw = tools.read_data(var+str(en),months='all',slicing=False,limit_lat=False) * var_ratio
        if data_raw.latitude[0].item()>data_raw.latitude[-1].item(): # If the latitudes start from 90N, reverse (ERA5, NCEP R2)
            data_raw=data_raw.isel(latitude=slice(None, None, -1)) 
        data_raw = data_raw.sel(latitude=slice(-25,90))
        data_raw = data_raw.sel(time=slice('%s-01-01'%str(hist_st_yr).zfill(4), '%s-12-01'%str(hist_end_yr).zfill(4)))
        data_raw = data_raw.compute()
        years = set(data_raw.time.dt.year.values)
        for st_yr in range(hist_st_yr, hist_end_yr+1-slicing_period, interval):
            end_yr = st_yr+slicing_period
            print(var, en, st_yr, end_yr)
            if (st_yr not in years) | (end_yr not in years):
                print('Not covering the period')
                continue
            if False: # Stop if the file exists. False to replace
                file_path = '/dx13/pyfsiew/training/training_%syr/%s/%s%s_%s-%s.nc'%(slicing_period,folder_name,var,en,st_yr,end_yr)
                if os.path.exists(file_path):
                    print("File exist. Skip")
                    continue
            ### The trends of circulation, sea ice or SST
            # To get their regrid lats and regrid lons
            if ('psl' in var) or ('tas' in var):
                lats = np.arange(-21,90,3) # the last lat is 87 (otherwise CanESM will give error)
                lons = np.arange(-177,180,3).tolist(); lons.remove(0) # no 0 (for models) and no 180 (for MERRA2)
                #lats = np.arange(-21,88,2) # the last lat is 87 (otherwise CanESM will give error)
                #lons = np.arange(-178,180,2).tolist(); lons.remove(0); lons.remove(-2); lons.remove(2)
            elif 'sic' in var:
                lats = np.arange(45.5,90,1) # Then it also covers Pacific sea ice
                lons = np.arange(-179.5,180,1) 
            elif 'sst' in var: 
                lats = np.arange(-21,90,3) 
                lons = np.arange(-177,180,3).tolist(); lons.remove(0) 

            # Each individual month
            #st_yrs = [st_yr+1,st_yr+1] + [st_yr]*10;#end_yrs = [end_yr, end_yr] + [end_yr-1]*10
            #mons_sets = [[2], [1], [12], [11], [10], [9], [8], [7], [6], [5], [4], [3]]
            # DJF, SON, JJA, MAM
            st_yrs = [st_yr, st_yr, st_yr, st_yr]
            end_yrs = [end_yr, end_yr-1, end_yr-1, end_yr-1]
            mons_sets = [(12,1,2), (9,10,11), (6,7,8), (3,4,5)]
            results = []
            for st_yr_psl, end_yr_psl, mons in zip(st_yrs, end_yrs, mons_sets):
                if ('psl' in var) or ('sst' in var) or ('tas' in var):
                    result=extract_monthly_trend_slp(data_raw,st_yr_psl,end_yr_psl,mons,lats,lons)
                elif 'sic' in var:
                    result=extract_monthly_trend_sic(data_raw,st_yr_psl,end_yr_psl,mons,lats,lons)
                results.append(result)
            # Combine them
            slope= xr.concat(results,dim='season')
            slope = xr.DataArray(slope).rename('training')
            #print(slope.shape)
            slope.to_netcdf('/dx13/pyfsiew/training/training_%syr/%s/%s%s_%s-%s.nc'%(slicing_period,folder_name,var,en,st_yr,end_yr))

            if False:### Extract the SIC-mean state in the st_yr
                if 'sic' in var:
                    result=extract_monthly_mean_sic(data_raw, st_yr, end_yr, mons=[12,1,2])
                    sic_mean= xr.DataArray(result).rename('training')
                    sic_mean.to_netcdf('/dx13/pyfsiew/training/training_%syr/%s%s_%s-%s_mean.nc'
                                    %(slicing_period,var,en,st_yr,end_yr))

