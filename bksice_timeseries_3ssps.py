import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import ipdb
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import pandas as pd
import cartopy.crs as ccrs

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct

if __name__ == "__main__":

    reload(tools)

    lat1=70; lat2=82; lon1=15; lon2=100 # Koenigk's box
    st_mon='12'; end_mon='02'
    mons=[6,7,8,9] # Summer
    mons=[12,1,2] # Winter
    vars=['CanESM5_sic_en','MIROC6_sic_en','MIROC6_sicssp245_en']
    vars=['MIROC6_sic_en','MIROC6_sicssp245_en','MIROC6_sicssp370_en','MIROC6_sicssp585_en',
            'CanESM5_sic_en','CanESM5_sicssp245_en','CanESM5_sicssp370_en','CanESM5_sicssp585_en',
            'ACCESS-ESM1-5_sic_en','ACCESS-ESM1-5_sicssp245_en','ACCESS-ESM1-5_sicssp370_en','ACCESS-ESM1-5_sicssp585_en']
    vars=['MIROC6_sic_en','MIROC6_sicssp245_en','MIROC6_sicssp370_en','MIROC6_sicssp585_en',
            'CanESM5_sic_en','CanESM5_sicssp245_en','CanESM5_sicssp370_en','CanESM5_sicssp585_en']
    vars=['MIROC6_sicssp245_en','MIROC6_sicssp370_en','MIROC6_sicssp585_en',
          'CanESM5_sicssp245_en','CanESM5_sicssp370_en','CanESM5_sicssp585_en']
    ensembles=range(1,11)
    ratio=0.01 # Models are not CESM2 and Obs. They are (0-100%)

    bks_tss = {var: {en:None for en in ensembles} for var in vars}
    for var in vars:
        print(var)
        for en in ensembles:
            data_raw = tools.read_data(var+str(en), months='all', slicing=False, reverse_lat=False, limit_lat=False) 
            data_raw = data_raw.compute() * ratio
            data = data_raw
            if "ssp" in var:
                st_yr='2015'; end_yr='2100'
            else:
                st_yr='1980'; end_yr='2016'
            # (1) Do the seasonal mean 
            data = data.sel(time=slice('%s-%s-01'%(st_yr,st_mon), '%s-%s-28'%(end_yr,end_mon)))
            mon_mask = data.time.dt.month.isin(mons)
            data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
            # (2) Define extent. If False ==> Do area
            mask_nan = np.isnan(data) # Get the nan values (all NAN values are the land grids)
            data = xr.where(data<=0.15, 0, 1) # The nan grids will also turn to 1 because these grids are false
            data = xr.where(mask_nan, np.nan, data) # Put the nan back to the data
            # (3) Extract the SIC cover over BKS
            adj=0
            data=data.sel(latitude=slice(lat1+adj,lat2-adj)).sel(longitude=slice(lon1+adj,lon2-adj))
            # (4) Weight the index using the old method
            lons=data.longitude.values; lats=data.latitude.values
            bks_ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
            bks_ts=xr.DataArray(bks_ts,dims=['time'],coords={'time':data.time}) 
            bks_tss[var][en]=bks_ts*100 # turn back to 100%

    ### Get the mean and standard deviation
    bks_tss_mean={var:None for var in vars}
    bks_tss_std={var:None for var in vars}
    for var in vars:
        bks_tss_mean[var] = xr.concat([bks_tss[var][en] for en in ensembles],dim='en').mean(dim='en')
        bks_tss_std[var] = xr.concat([bks_tss[var][en] for en in ensembles],dim='en').std(dim='en')

    # Plot the two timeseries
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(8,2))
    colors={ 'MIROC6_sic_en':'k','MIROC6_sicssp245_en':'yellow','MIROC6_sicssp370_en':'orange','MIROC6_sicssp585_en':'brown',
            'CanESM5_sic_en':'k','CanESM5_sicssp245_en':'cyan','CanESM5_sicssp370_en':'dodgerblue','CanESM5_sicssp585_en':'blue',
            'ACCESS-ESM1-5_sic_en':'k','ACCESS-ESM1-5_sicssp245_en':'greenyellow','ACCESS-ESM1-5_sicssp370_en':'lime','ACCESS-ESM1-5_sicssp585_en':'green'}
    labels={ 'MIROC6_sic_en':'MIROC6 hist','MIROC6_sicssp245_en':'MIROC6 SSP245','MIROC6_sicssp370_en':'MIROC6 SSP370',
            'MIROC6_sicssp585_en':'MIROC6 SSP585',
            'CanESM5_sic_en':'CanESM5 hist','CanESM5_sicssp245_en':'CanESM5 SSP245','CanESM5_sicssp370_en':'CanESM5 SSP370',
            'CanESM5_sicssp585_en':'CanESM5 SSP585',
            'ACCESS-ESM1-5_sic_en':'ACCESS-ESM1-5- hist','ACCESS-ESM1-5_sicssp245_en':'ACCESS-ESM1-5 SSP245','ACCESS-ESM1-5_sicssp370_en':'ACCESS-ESM1-5- SSP370','ACCESS-ESM1-5_sicssp585_en':'ACCESS-ESM1-5- SSSP585'}
    for var in vars:
        years=bks_tss_mean[var].time.dt.year.values
        ax1.plot(years,bks_tss_mean[var],color=colors[var],label=labels[var])
        ax1.fill_between(years, bks_tss_mean[var]-bks_tss_std[var], bks_tss_mean[var]+bks_tss_std[var], fc=colors[var], zorder=100,alpha=0.1)
    ax1.set_ylabel("BKS SIC (%)")
    for i in ['right', 'top']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    # Set xticks
    #ax1.setxticks(years[::10])
    #xticklabels=[str(yr-1)+'/'+str(yr) for yr in years]
    #ax1.set_xticklabels(xticklabels)
    ax1.set_xlim(2016,2100)
    ax1.legend(bbox_to_anchor=(0.5,0.5), ncol=2, loc='lower left', frameon=False, columnspacing=1,handletextpad=0.4, labelspacing=0.3)
    # Save
    fig_name = 'BKS_ts_ssp_models'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)

