import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib
from importlib import reload
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import ipdb
import xesmf as xe

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct

if __name__ == "__main__":


    vars = ['noaa_20CR_slp', 'obs1_psl_en', 'ACCESS-ESM1-5_psl_en','ACCESS-CM2_psl_en','CanESM5_psl_en',
            'CanESM5-1_psl_en','cesm2_psl_en', 'EC-Earth3CC_psl_en', 'IPSL-CM6A-LR_psl_en', 'miroc6_psl_en', 
            'MIROC-ES2L_psl_en','MPI-ESM1-2-LR_psl_en', 'MPI-ESM1-2-HR_psl_en', 'UKESM1-0-LL_psl_en']
    ensembles = [['']] + [['']] + [range(1,2)]*(len(vars)-1)
    st_years=range(1836,2081)
    slicing_period = 20

    if True: # A figure in response letter
        vars = ['noaa_20CR_slp', 'obs1_psl_en', 'obs2_psl_en']
        ensembles = [['']] + [['']] + [['']]
        st_years=range(1980,2023-20)
        slicing_period = 20

    #################################
    mons = [9,10,11]; st_mon='09'; end_mon='11'; end_yr_adjust=-1
    mons = [12,1,2]; st_mon='12'; end_mon='02'; end_yr_adjust=0
    trends = {var:{st_yr:[] for st_yr in st_years} for var in vars}
    if False: # Old region in the original revised manuscirpt
        Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,20,90 # urals
        Ilat1, Ilat2, Ilon1, Ilon2 = 55,70,-45,-1 # Iceland
    else: # New region according to 1996-2016
        Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,25,90 # urals
        Ilat1, Ilat2, Ilon1, Ilon2 = 52,70,-37,0 # Iceland
    for i, var in enumerate(vars):
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
                    trends[var][st_yr].append(None)
                    continue
                mon_mask = data.time.dt.month.isin(mons)
                data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
                if data.time.size!=slicing_period:
                    trends[var][st_yr].append(None)
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
                trends[var][st_yr].append(slope.item())

    # Plot the timeseries with time
    var_names = {'noaa_20CR_slp':'20CRV3','obs1_psl_en':'MERRA2', 'obs2_psl_en':'ERA5', 'ACCESS-ESM1-5_psl_en':'ACCESS-ESM1-5',\
         'ACCESS-CM2_psl_en':'ACCESS-CM2','CanESM5_psl_en':'CanESM5', 'CanESM5-1_psl_en':'CanESM-1',
         'cesm2_psl_en':'CESM2', 'EC-Earth3CC_psl_en':'EC-Earth3-CC',
         'IPSL-CM6A-LR_psl_en':'IPSL-CM6A-LR','miroc6_psl_en':'MIROC6',
         'MIROC-ES2L_psl_en':'MIROC-ES2L', 'MPI-ESM1-2-LR_psl_en':'MPI-ESM1-2-LR',
         'MPI-ESM1-2-HR_psl_en':'MPI-ESM1-2-HR','UKESM1-0-LL_psl_en':'UKESM1-0-LL'}
    vars_labels = [var_names[var] for var in vars]
    colors=['k','dimgray','royalblue','r','g','orange','slategrey','lime','cyan','gold','pink','violet',
                    'brown','peru','orchid','crimson']
    zorders = [1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(7,2))
    for i, var in enumerate(vars):
        years = list(trends[var].keys())
        y = [trends[var][yr][0] for yr in years] # Only capture the first ensemble
        ax1.plot(years,y,color=colors[i],label=vars_labels[i], zorder=zorders[i])
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=1, xmin=0.01, xmax=0.99)
    ax1.axvline(x=1997, color='black', linestyle='--', linewidth=1, zorder=1)
    ax1.axvline(x=1996, color='black', linestyle='--', linewidth=1, zorder=1)
    ax1.set_title('Trends of circulation dipole', loc='left')
    ax1.set_ylabel('Pa/decade')
    ylim=700
    ax1.set_ylim(-ylim,ylim)
    ax1.set_yticks([-500,-250,0,250,500])
    if False:
        ax1.set_xlim(1836,2080)
        nn=10
        xticks=range(1835,2081)[::nn]
    else:
        ax1.set_xlim(1980,2002)
        nn=1
        xticks=range(1980,2003)
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


