import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
import xesmf as xe

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct

if __name__ == "__main__":

    if True: # A figure in response letter
        #vars = ['noaa_20CR_slp', 'obs1_psl_en', 'obs2_psl_en']
        vars_slp = ['obs2_psl_en']; vars_tas = ['obs2_tas_en']
        vars_slp = ['obs3_psl_en']; vars_tas = ['obs3_tas_en']
        vars_slp = ['obs1_psl_en']; vars_tas = ['obs1_tas_en'] #ERA5
        ensembles = [['']] + [['']] + [['']]
        #st_years=range(1836,2081)
        slicing_period = 20
        st_years=range(1980,2024-slicing_period)

    ##################################################################
    ### Ciruclation trends
    mons = [12,1,2]; st_mon='12'; end_mon='02'; end_yr_adjust=0
    slp_trends = {var:{st_yr:[] for st_yr in st_years} for var in vars_slp}
    Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,25,90 # urals
    Ilat1, Ilat2, Ilon1, Ilon2 = 52,70,-37,0 # Iceland
    for i, var in enumerate(vars_slp):
        for en in ensembles[i]:
            print(var,en)
            data_raw= tools.read_data(var+str(en), months='all', slicing=False, limit_lat=True)
            data_raw = data_raw.sel(latitude=slice(45,89)).compute()
            # Regrid the data (to be consistent between Figure 4A)
            lats = np.arange(45,90,3) # Create the same grids as the training data
            lons = np.arange(-54,108,3).tolist(); lons.remove(0) 
            ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
            regridder = xe.Regridder(data_raw.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
            data_raw = regridder(data_raw)
            lats=data_raw.latitude.values; lons=data_raw.longitude.values
            for st_yr in st_years:
                end_yr = st_yr+slicing_period
                data = data_raw.sel(time=slice('%s-%s-01'%(str(st_yr).zfill(4),st_mon),'%s-%s-28'
                                %(str(end_yr+end_yr_adjust).zfill(4),end_mon)))
                if data.time.size<3:
                    slp_trends[var][st_yr].append(None)
                    continue
                mon_mask = data.time.dt.month.isin(mons)
                data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
                if data.time.size!=slicing_period:
                    slp_trends[var][st_yr].append(None)
                    continue
                if True: # Compute the area-weighted mean
                    urals_ts=ct.weighted_area_average(data.values,Ulat1,Ulat2,Ulon1,Ulon2,lons,lats,
                                                lon_reverse=False,return_extract3d=False) 
                    iceland_ts=ct.weighted_area_average(data.values,Ilat1,Ilat2,Ilon1,Ilon2,lons,lats,
                                                lon_reverse=False, return_extract3d=False) 
                    circulation_ts = urals_ts - iceland_ts
                # Put the timeseries into xarray
                data = xr.DataArray(circulation_ts, dims=['time'], coords={'time':range(urals_ts.size)})
                # Get the trend of the timeseries
                x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
                y=data; ymean=y.mean(dim='time')
                slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 
                slp_trends[var][st_yr].append(slope.item())

    ###########################################################
    ### TAS trends
    Tlat1=70; Tlat2=82; Tlon1=15; Tlon2=100 ; #For the region in Koenigk et al. 2016 
    tas_trends = {var:{st_yr:[] for st_yr in st_years} for var in vars_tas}
    for i, var in enumerate(vars_tas):
        for en in ensembles[i]:
            print(var,en)
            data_raw= tools.read_data(var+str(en), months='all', slicing=False, limit_lat=True)
            data_raw = data_raw.sel(latitude=slice(45,89)).compute()
            # Regrid the data (to be consistent between Figure 4A)
            lats = np.arange(45,90,3) # Create the same grids as the training data
            lons = np.arange(-54,108,3).tolist(); lons.remove(0) 
            ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
            regridder = xe.Regridder(data_raw.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
            data_raw = regridder(data_raw)
            lats=data_raw.latitude.values; lons=data_raw.longitude.values
            for st_yr in st_years:
                end_yr = st_yr+slicing_period
                data = data_raw.sel(time=slice('%s-%s-01'%(str(st_yr).zfill(4),st_mon),'%s-%s-28'
                                %(str(end_yr+end_yr_adjust).zfill(4),end_mon)))
                if data.time.size<3:
                    tas_trends[var][st_yr].append(None)
                    continue
                mon_mask = data.time.dt.month.isin(mons)
                data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
                if data.time.size!=slicing_period:
                    tas_trends[var][st_yr].append(None)
                    continue
                if True: # Compute the area-weighted mean
                    tas_ts =ct.weighted_area_average(data.values,Tlat1,Tlat2,Tlon1,Tlon2,lons,lats,
                                                lon_reverse=False,return_extract3d=False) 
                # Put the timeseries into xarray
                data = xr.DataArray(tas_ts, dims=['time'], coords={'time':range(urals_ts.size)})
                # Get the trend of the timeseries
                x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
                y=data; ymean=y.mean(dim='time')
                slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 
                tas_trends[var][st_yr].append(slope.item())

    ### Plot the timeseries with time
    var_names = {'noaa_20CR_slp':'20CRV3','obs1_psl_en':'MERRA2', 'obs2_psl_en':'ERA5', 'obs3_psl_en':'JRA55',
            'ACCESS-ESM1-5_psl_en':'ACCESS-ESM1-5',\
         'ACCESS-CM2_psl_en':'ACCESS-CM2','CanESM5_psl_en':'CanESM5', 'CanESM5-1_psl_en':'CanESM-1',
         'cesm2_psl_en':'CESM2', 'EC-Earth3CC_psl_en':'EC-Earth3-CC',
         'IPSL-CM6A-LR_psl_en':'IPSL-CM6A-LR','miroc6_psl_en':'MIROC6',
         'MIROC-ES2L_psl_en':'MIROC-ES2L', 'MPI-ESM1-2-LR_psl_en':'MPI-ESM1-2-LR',
         'MPI-ESM1-2-HR_psl_en':'MPI-ESM1-2-HR','UKESM1-0-LL_psl_en':'UKESM1-0-LL'}
    vars_labels = [var_names[var] for var in vars_slp]
    colors=['k','dimgray','royalblue','r','g','orange','slategrey','lime','cyan','gold','pink','violet',
                    'brown','peru','orchid','crimson']
    zorders = [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(7,2))
    ### For SLP
    for i, var in enumerate(vars_slp):
        years = list(slp_trends[var].keys())
        y = [slp_trends[var][yr][0] for yr in years] # Only capture the first ensemble
        ax1.plot(years,y,color='blue',label=vars_labels[i], zorder=zorders[i])
        #ax1.plot(years,y,color=colors[i],label=vars_labels[i], zorder=zorders[i])
    ### For TAS
    ax2=ax1.twinx()
    for i, var in enumerate(vars_tas):
        years = list(tas_trends[var].keys())
        y = [tas_trends[var][yr][0] for yr in years] # Only capture the first ensemble
        #ax2.plot(years,y,color=colors[i],label=vars_labels[i], zorder=zorders[i], linestyle='-.')
        ax2.plot(years,y,color='gray',label=vars_labels[i], zorder=zorders[i], linestyle='-')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=1, xmin=0.01, xmax=0.99)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=1, xmin=0.01, xmax=0.99)
    #ax1.axvline(x=1997, color='black', linestyle='--', linewidth=1, zorder=1)
    ax1.axvline(x=1996, color='black', linestyle='--', linewidth=1, zorder=1)
    ax1.set_title('Trends of circulation dipole', loc='left')
    ax1.set_ylabel('Pa/decade')
    ylim=700
    ax1.set_ylim(-ylim,ylim)
    ax1.set_yticks([-500,-250,0,250,500])
    ax1.set_xlim(1836,2080)
    nn=10
    xticks=range(1835,2081)[::nn]
    ax1.set_xticks(xticks)
    #ax1.grid()
    xticklabels=[str(yr)+'-'+str(yr+slicing_period) for yr in xticks]
    ax1.set_xticklabels(xticklabels, rotation=90)
    for i in ['right', 'top']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    ax1.legend(bbox_to_anchor=(-0.1,1.1),ncol=4,loc='lower left',frameon=False,columnspacing=1.5,handletextpad=0.6)
    fig_name = 'circulation_change_with_times'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)

    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(3,3))
    ### For SLP
    for i, var in enumerate(vars_slp):
        years = list(slp_trends[var].keys())
        x = [slp_trends[var][yr][0] for yr in years] # Only capture the first ensemble
    ### For TAS
    for i, var in enumerate(vars_tas):
        years = list(tas_trends[var].keys())
        y = [tas_trends[var][yr][0] for yr in years] # Only capture the first ensemble
    ax1.scatter(x,y)
    for i, yr in enumerate(years):
        ax1.annotate('%s'%yr+'-'+'%s'%(yr+20), xy=(x[i], y[i]), xycoords='data',fontsize=5)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=-10, xmin=0.01, xmax=0.99)
    ax1.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=-10, ymin=0.01, ymax=0.99)
    ax1.set_xlabel("SLP dipole trend (Pa/decade)")
    ax1.set_ylabel("TAS trend over BKS (K/decade)")
    corr=round(tools.correlation_nan(x,y),2)
    #ax1.annotate(corr, xy=(0.02, 0.98), xycoords='axes fraction',fontsize=10)
    ax1.annotate(r"$\rho$=%s"%corr, xy=(0.02, 0.98), xycoords='axes fraction',fontsize=10)
    # Remove the box
    for i in ['right', 'top']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    fig_name = 'SLP_TAS_scatter_relationships'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)
