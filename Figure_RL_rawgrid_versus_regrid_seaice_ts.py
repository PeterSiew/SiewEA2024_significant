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

    lat1=68; lat2=85; lon1=5; lon2=90 # Old box
    lat1=70; lat2=82; lon1=15; lon2=100 # Koenigk's box
    st_mon='12'; end_mon='02'
    mons=[12,1,2]
    obs_or_cesm2='obs'; st_yr='1980'; end_yr='2023'; nn=1
    obs_or_cesm2='cesm2'; st_yr='1950'; end_yr='2055'; nn=10

    ### Raw sea ice data
    # Read data
    if obs_or_cesm2=='obs': # Reading obs
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/monthly_raw_data/seaice_conc_monthly_nh_197811_202309_v04r00.nc'
        var_name='cdr_seaice_conc_monthly'
        data_raw = xr.open_dataset(path)
        data = data_raw[var_name].compute()
        new_time = pd.date_range(start='1978-11-01', end='2023-09-01', freq='MS') 
        data = data.rename({'tdim':'time'})
        data = data.assign_coords({'time':new_time})
        data = xr.where(data.isin([2.51,2.52,2.53,2.54,2.55]), np.nan, data)
    elif obs_or_cesm2=='cesm2': # Reading CESM2 1st member
        var='cesm2_sicraw_en'; en='1'
        data_raw = tools.read_data(var+str(en), months='all', slicing=False, reverse_lat=False, limit_lat=False) 
        data_raw = data_raw.compute()
        data = data_raw
    # (1) Do the seasonal mean first
    data = data.sel(time=slice('%s-%s-01'%(st_yr,st_mon), '%s-%s-28'%(end_yr,end_mon)))
    mon_mask = data.time.dt.month.isin(mons)
    data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
    if True: # (2) Define extent
        mask_nan = np.isnan(data) # Get the nan values (all NAN values are the land grids)
        data = xr.where(data<=0.15, 0, 1) # The nan grids will also turn to 1 because these grids are false
        data = xr.where(mask_nan, np.nan, data) # Put the nan back to the data
    # (3) Extract BKS
    if obs_or_cesm2=='obs':
        lats=data_raw.latitude
        lons=data_raw.longitude
    elif obs_or_cesm2=='cesm2':
        lats=data_raw.TLAT
        lons=data_raw.TLON
    lat_mask=(lats>=lat1) & (lats<=lat2)
    lon_mask=(lons>=lon1) & (lons<=lon2)
    lat_lon_mask = lat_mask & lon_mask
    data_bks= xr.where(lat_lon_mask,data,np.nan)
    if True: # (4) Weight the boxes before averging
        bks_ts_rawgrid=[]
        for t in range(data_bks.time.size):
            points=data_bks.isel(time=t).values.flatten()
            mask=~np.isnan(points)
            points_valid=points[mask]
            lats_valid = lats.values.flatten()[mask]
            lons_valid = lons.values.flatten()[mask]
            weight_lats = np.cos(lats_valid*np.pi/180)
            points_weight = np.sum(points_valid*weight_lats)/np.sum(weight_lats)
            bks_ts_rawgrid.append(points_weight)
        bks_ts_rawgrid=np.array(bks_ts_rawgrid)
        bks_ts_rawgrid=xr.DataArray(bks_ts_rawgrid,dims=['time'],coords={'time':data_bks.time})
    else: # Don't do weighting
        bks_ts_rawgrid=data_bks.mean(dim='x').mean(dim='y')
    if True: # (5) Plot the lat-lon map with bks values
        points=data_bks.isel(time=0).values.flatten()
        mask_data=np.isnan(points)
        #mask=np.isnan(points)
        #mask_data=(points==1)
        lats_bks = lats.values.flatten()[mask]
        lons_bks = lons.values.flatten()[mask]
        # Plot on map
        plt.close()    
        fig = plt.figure(figsize=(2,2)) 
        projection=ccrs.NorthPolarStereo(); xsize=1.5; ysize=1.5
        ax1 = fig.add_subplot(1,1,1, projection=projection)
        ax1.scatter(lons_bks, lats_bks, color='blue', linestyle='--', transform=ccrs.PlateCarree(), s=0.0000000001)
        ax1.coastlines(color='darkgray', linewidth=1)
        g1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='lightgray', linestyle='-',
                 xlocs=[-180, -120, -60, 0, 60, 120], ylocs=[-60,0,30,60])
        map_extents = {'left':-180, 'right':180, 'bottom':40, 'top':90}
        ax1.set_extent([map_extents['left'], map_extents['right'], map_extents['bottom'], map_extents['top']], ccrs.PlateCarree())
        fig_name = 'testing_mask_pos'
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=300)


    ### Read regrid data
    if obs_or_cesm2=='obs':
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_monthly_nh_197811to202309_regrid_0.25x0.25.nc'
        path = '/dx13/pyfsiew/noaa_nsidc_seaice_conc_data_v4/cdr_seaice_conc_monthly_nh_197811to202309_regrid_1x1.nc'
        data_raw = xr.open_dataset(path)
        var_name='cdr_seaice_conc_monthly'
        data_raw = xr.open_dataset(path)
        data = data_raw[var_name].compute()
        new_time = pd.date_range(start='1978-11-01', end='2023-09-01', freq='MS') 
        data = data.assign_coords({'time':new_time})
    elif obs_or_cesm2=='cesm2': # Reading CESM2 1st member
        var='cesm2_sic_en'; en='1'
        data_raw = tools.read_data(var+str(en), months='all', slicing=False, reverse_lat=False, limit_lat=False) 
        data_raw = data_raw.compute()
        data = data_raw
    # (1) Do the seasonal mean 
    data = data.sel(time=slice('%s-%s-01'%(st_yr,st_mon), '%s-%s-28'%(end_yr,end_mon)))
    mon_mask = data.time.dt.month.isin(mons)
    data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
    if True: # (2) Define extent. If False ==> Do area
        mask_nan = np.isnan(data) # Get the nan values (all NAN values are the land grids)
        data = xr.where(data<=0.15, 0, 1) # The nan grids will also turn to 1 because these grids are false
        data = xr.where(mask_nan, np.nan, data) # Put the nan back to the data
    # (3) Extract the SIC cover over BKS
    adj=0
    data_bks=data.sel(latitude=slice(lat1+adj,lat2-adj)).sel(longitude=slice(lon1+adj,lon2-adj))
    # (4) Create the weighted box ts
    if True:  # New weighting
        lats,lons=np.meshgrid(data_bks.latitude.values, data_bks.longitude.values)
        lats=lats.T; lons=lons.T
        bks_ts_regrid=[]
        for t in range(data_bks.time.size):
            points=data_bks.isel(time=t).values.flatten()
            mask=~np.isnan(points)
            points_valid=points[mask]
            lats_valid = lats.flatten()[mask]
            lons_valid = lons.flatten()[mask]
            weight_lats = np.cos(lats_valid*np.pi/180)
            points_weight = np.sum(points_valid*weight_lats)/np.sum(weight_lats)
            bks_ts_regrid.append(points_weight)
        bks_ts_regrid=np.array(bks_ts_regrid)
        bks_ts_regrid=xr.DataArray(bks_ts_regrid,dims=['time'],coords={'time':data_bks.time})
    elif False: # Weight the index using the old method
        lons=data_bks.longitude.values; lats=data_bks.latitude.values
        bks_ts=ct.weighted_area_average(data_bks.values,lat1,lat2,lon1,lon2,lons,lats)
        bks_ts_regrid=xr.DataArray(bks_ts,dims=['time'],coords={'time':data.time}) 
    else: # No weighting
        bks_ts_regrid=data_bks.mean(dim='latitude').mean(dim='longitude')
    if False: # (5) Plot the lat-lon map with bks values
        points=data_bks.isel(time=0).values.flatten()
        mask=np.isnan(points)
        #mask=(points==0)
        #mask=(points==1)
        lats,lons=np.meshgrid(data_bks.latitude.values, data_bks.longitude.values)
        lats=lats.T; lons=lons.T
        lats_bks = lats.flatten()[mask]
        lons_bks = lons.flatten()[mask]
        # Plot on map
        plt.close()    
        fig = plt.figure(figsize=(2,2)) 
        projection=ccrs.NorthPolarStereo(); xsize=1.5; ysize=1.5
        ax1 = fig.add_subplot(1,1,1, projection=projection)
        ax1.scatter(lons_bks, lats_bks, color='blue', linestyle='--', transform=ccrs.PlateCarree(), s=0.0001)
        ax1.coastlines(color='darkgray', linewidth=1)
        g1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.5, color='lightgray', linestyle='-',
                 xlocs=[-180, -120, -60, 0, 60, 120], ylocs=[-60,0,30,60])
        map_extents = {'left':-180, 'right':180, 'bottom':40, 'top':90}
        ax1.set_extent([map_extents['left'], map_extents['right'], map_extents['bottom'], map_extents['top']], ccrs.PlateCarree())
        fig_name = 'testing_mask_pos_regrid'
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=300)
    
    if True: # Plot the two timeseries
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,1))
        years=bks_ts_rawgrid.time.dt.year.values
        ax1.plot(years,bks_ts_regrid*100,label='Atmospheric 1x1 grid')
        ax1.plot(years,bks_ts_rawgrid*100,label='Raw ocean grid')
        ax1.set_xticks(years[::nn])
        xticklabels=[str(yr-1)+'/'+str(yr)[2:] for yr in years][::nn]
        ax1.legend(bbox_to_anchor=(-0.05, 1.1), ncol=2, loc='lower left', frameon=False, columnspacing=1, 
                    handletextpad=0.4, labelspacing=0.3)
        ax1.set_xticklabels(xticklabels,size=7,rotation=90)
        ax1.set_ylabel("BKS SIC (%)")
        ax1.set_xlabel("Years")
        ax1.set_xlim(years[0],years[-1])
        fig_name = 'BKS_ts_rawgrid_versus_native_grid'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)






