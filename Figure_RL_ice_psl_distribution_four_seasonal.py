import ipdb
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from scipy import stats
#import xesmf as xe

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct


if __name__ == "__main__":


    ### Get the internal sea ice distribution from the models


    ### WHen extreme sea ice happen, what are the sea level pressure trends

    models = ['ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'cesm2', 
             'EC-Earth3CC', 'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 
             'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    models = ['cesm2']
    psl_vars=[m+'_psl_en' for m in models]
    sic_vars=[m+'_sic_en' for m in models]
    vars=psl_vars+sic_vars 

    # Set ensembles
    ens={m:range(1,11) for m in models}
    #ens={m:range(1,101) for m in models}
    ensembles = {m+'_%s_en'%var: ens[m] for m in models for var in ['psl','sic','tas']}

    hist_st_yr=1950; hist_end_yr=2055 # This seems to be the best result
    #hist_st_yr=1980; hist_end_yr=2023 # This seems to be the best result

    # Periods and intervals
    slicing_period=20; interval=1 #Default (produce better result)

    # Create the model range
    model_years=range(hist_st_yr, hist_end_yr+1-slicing_period, interval)

    # Set the grids
    aaa=0; bbb=4 # All season (Default)

    # Default (only the Euro-Atlantic SLP)
    lat1=54; lat2=88; lon1=-60; lon2=102.5; grid_no=648 # 2592 grids for all seasons
    lat1=48; lat2=88; lon1=-60; lon2=102.5; grid_no=648 # 2592 grids for all seasons

    # Others
    tt='training' # Get the name

    # Read in the predictor and preditands
    X1, X2, Y = [], [], []
    ens_record = []
    for i, var in enumerate(vars):
        for en in ensembles[var]:
            for st_yr in model_years:
                end_yr = st_yr+slicing_period
                if 'psl' in var: # Geth the PSL
                    ens_record.append((var,en,st_yr,end_yr))
                    data_raw=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/psl/%s%s_%s-%s.nc'%(slicing_period,var,en,st_yr,end_yr))[tt]
                    data=data_raw.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    #data=data.values.reshape(-1)
                    #data=data[aaa*grid_no:bbb*grid_no]
                    X1.append(data)
                elif 'sic' in var: # Get the SIC
                    data_sic = xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic/%s%s_%s-%s.nc'
                                        %(slicing_period,var,en,st_yr,end_yr))[tt]
                    data_sic = data_sic.isel(season=0)
                    Y.append(data_sic.item())

    # Append X1 and X2 as two predictors
    Y=np.array(Y)
    X1=xr.concat(X1,dim='en')


    ### Get the sea ice distribution
    # Remove the ensemble means in the same period for each model. Remove only Y but not X
    groups = [i[0]+'-'+str(i[2])+'-'+str(i[3]) for i in ens_record] # has no information of the ensemble number
    for group in set(groups):
        idx = (np.array(groups)==group).nonzero()[0]
        Y_mean = Y[idx].mean()
        Y[idx] = Y[idx]-Y_mean # Internal variability


    ### Get the PSL distribution
    Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,20,90 # urals
    Ilat1, Ilat2, Ilon1, Ilon2 = 55,70,-45,-1 # Iceland
    # Get the sea ice pressure difference between Urals and Icealand region
    diffs = {}
    large_index={}
    for season in [0,1,2,3]: # Four season
        urals=X1.isel(season=season).sel(latitude=slice(Ulat1,Ulat2)).sel(longitude=slice(Ulon1,Ulon2)).mean(dim='latitude').mean(dim='longitude')
        iceland=X1.isel(season=season).sel(latitude=slice(Ilat1,Ilat2)).sel(longitude=slice(Ilon1,Ilon2)).mean(dim='latitude').mean(dim='longitude')
        diff=urals-iceland
        diffs[season]=diff
        threshold=np.percentile(diff,90)
        large_index[season]=(diff>threshold).values
    ###
    # Get the distribution of SLP trend
    threshold=np.percentile(Y,10)
    small_index=Y<threshold
    plt.close()
    fig, axs = plt.subplots(1,4,figsize=(10,2))
    bins=20
    for i, season in enumerate([0,1,2,3]): # Four season
        axs[i].hist(diffs[season], bins=bins, edgecolor='lightgray', fc="lightgray", lw=1, linewidth=1)
        #axs[i].hist(diffs[season][small_index], bins=bins, edgecolor='royalblue', fc="royalblue", lw=1, linewidth=1)
        axs[i].hist(diffs[season][large_index[season]], bins=bins, edgecolor='royalblue', fc="royalblue", lw=1, linewidth=1)
        axs[i].vlines(x=diffs[season][small_index].mean(), color='black', linestyle='-', linewidth=1, ymin=0, ymax=100)
        axs[i].set_xlim(-600,600)
    fig_name = 'ciruclation_index_distrbution'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0)# hspace is the vertical=3 (close)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name),
                bbox_inches='tight', dpi=400, pad_inches=0.01)
    # Plot the distribution of the internal sea ice
    final_index = (large_index[0]) & (large_index[1]) & (large_index[2]) & (large_index[3])
    final_index = (large_index[0]) & (large_index[1])
    ###
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(4,2))
    ax1.hist(Y, bins=bins, edgecolor='lightgray', fc="lightgray", lw=1, linewidth=1)
    ax1.hist(Y[small_index], bins=bins, edgecolor='royalblue', fc="royalblue", lw=1, linewidth=1)
    ax1.vlines(Y[final_index].mean(), color='black', linestyle='-', linewidth=1, ymin=0, ymax=100)
    fig_name = 'seaice_distribution'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0)# hspace is the vertical=3 (close)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name),
                bbox_inches='tight', dpi=400, pad_inches=0.01)
