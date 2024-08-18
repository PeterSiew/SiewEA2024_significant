import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import multiprocessing
import matplotlib
from importlib import reload
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
#from eofs.xarray import Eof
import ipdb
import xesmf as xe

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct
import fig4_amip_cmip_slptrends_boxplots as fig4

#import os; os.environ['MKL_NUM_THREADS']='1'
#import os; os.environ['OMP_NUM_THREADS']='1'

if __name__ == "__main__":

    vars = ['obs1_psl_en', 'obs2_psl_en', 'obs3_psl_en', 'noaa_20CR_slp',
            'amip_CAM5_slp_control','amip_CAM5_slp_en',
            'amip_CAM6_slp_control', 'amip_CAM6_slp_en',
            'amip_ECHAM5_slp_en', 'amip_GFSv2_slp_en',
            'CESM2_PI_psl', 'CESM2_psl_en',
            'MIROC6_PI_psl', 'MIROC6_psl_en',
            'CanESM5_PI_psl', 'CanESM5_psl_en']
    vars = ['obs1_psl_en', 'obs2_psl_en', 'obs3_psl_en', 'noaa_20CR_slp', # Obs
            'amip_CAM5_slp_en', 'amip_CAM6_slp_en', 'amip_ECHAM5_slp_en', 'amip_GFSv2_slp_en', # Atmosphere-only forced
            'amip_CAM5_slp_control', 'amip_CAM6_slp_control', # Atmosphere-only unforced
            'ACCESS-ESM1-5_psl_en','ACCESS-CM2_psl_en','CanESM5_psl_en','CanESM5-1_psl_en', 'CESM2_psl_en', # Cupled forced
            'EC-Earth3CC_psl_en', 'IPSL-CM6A-LR_psl_en', 'MIROC6_psl_en', 'MIROC-ES2L_psl_en', 
            'MPI-ESM1-2-LR_psl_en', 'MPI-ESM1-2-HR_psl_en', 'UKESM1-0-LL_psl_en',
            'CESM2_PI_psl','CanESM5_PI_psl', 'MIROC6_PI_psl'] # Coupled unforced

    vars_types ={'obs1_psl_en':'MERRA2', 'obs2_psl_en':'ERA5', 'obs3_psl_en':'JRA55', 'noaa_20CR_slp':'20CR', 
             'amip_CAM5_slp_control':'AMIP_control', 'amip_CAM5_slp_en':'AMIP_global',
             'amip_CAM6_slp_control':'AMIP_control', 'amip_CAM6_slp_en':'AMIP_global', 
             'amip_ECHAM5_slp_en':'AMIP_global', 'amip_GFSv2_slp_en':'AMIP_global',
             'CESM2_PI_psl':'CMIP_control', 'CESM2_psl_en':'CMIP_global',
             'MIROC6_PI_psl':'CMIP_control','MIROC6_psl_en':'CMIP_global',
             'CanESM5_PI_psl':'CMIP_control', 'CanESM5_psl_en':'CMIP_global',
             'ACCESS-ESM1-5_psl_en':'CMIP_global','ACCESS-CM2_psl_en':'CMIP_global','CanESM5-1_psl_en':'CMIP_global',
             'EC-Earth3CC_psl_en':'CMIP_global', 'IPSL-CM6A-LR_psl_en':'CMIP_global', 'MIROC-ES2L_psl_en':'CMIP_global',
             'MPI-ESM1-2-LR_psl_en':'CMIP_global', 'MPI-ESM1-2-HR_psl_en':'CMIP_global', 'UKESM1-0-LL_psl_en':'CMIP_global'}

    if False: # 1997-2017 (supplementary figure)
        year_test='1997-2017'
        st_years={'obs1_psl_en':range(1997,1998), 'obs2_psl_en':range(1997,1998), 'obs3_psl_en':range(1997,1998),
                 'noaa_20CR_slp':range(1836,2016), # last year of 20CR is 2015
                 'amip_CAM5_slp_control':range(1,980), 'amip_CAM5_slp_en':range(1997,1998), # 979+20=999
                 'amip_CAM6_slp_control':range(1,980), 'amip_CAM6_slp_en':range(1997,1998),
                 'amip_CAM6_toga_slp_en':range(1997,1998),
                 'amip_ECHAM5_slp_en':range(1997,1998), 'amip_ECHAM5_climsic_slp_en':range(1997,1998),
                 'amip_GFSv2_slp_en':range(1997,1998),
                 'CESM2_PI_psl':range(1,781), 'CESM2_psl_en':range(1997,1998),'CESM2_pacemaker_slp_en':range(1997,1998), # 780+20=90
                 'MIROC6_PI_psl':range(1,781),'MIROC6_psl_en':range(1997,1998),
                 'CanESM5_PI_psl':range(1,781),'CanESM5_psl_en':range(1997,1998),
                 'ACCESS-ESM1-5_psl_en':range(1997,1998),'ACCESS-CM2_psl_en':range(1997,1998),'CanESM5-1_psl_en':range(1997,1998),
                 'EC-Earth3CC_psl_en':range(1997,1998), 'IPSL-CM6A-LR_psl_en':range(1997,1998), 'MIROC-ES2L_psl_en':range(1997,1998),
                 'MPI-ESM1-2-LR_psl_en':range(1997,1998), 'MPI-ESM1-2-HR_psl_en':range(1997,1998), 'UKESM1-0-LL_psl_en':range(1997,1998)}
    else: # 1996-2016
        year_test='1996-2016'
        st_years={'obs1_psl_en':range(1996,1997), 'obs2_psl_en':range(1996,1997),'obs3_psl_en':range(1996,1997),
                 'noaa_20CR_slp':range(1836,2016), # last year of 20CR is 2015
                 'amip_CAM5_slp_control':range(1,980), 'amip_CAM5_slp_en':range(1996,1997), # 979+20=999
                 'amip_CAM6_slp_control':range(1,980), 'amip_CAM6_slp_en':range(1996,1997),
                 'amip_CAM6_toga_slp_en':range(1996,1997),
                 'amip_ECHAM5_slp_en':range(1996,1997), 'amip_ECHAM5_climsic_slp_en':range(1996,1997),
                 'amip_GFSv2_slp_en':range(1996,1997),
                 'CESM2_PI_psl':range(1,781), 'CESM2_psl_en':range(1996,1997),'CESM2_pacemaker_slp_en':range(1996,1997), # 780+20=90
                 'MIROC6_PI_psl':range(1,781),'MIROC6_psl_en':range(1996,1997),
                 'CanESM5_PI_psl':range(1,781),'CanESM5_psl_en':range(1996,1997),
                 'ACCESS-ESM1-5_psl_en':range(1996,1997),'ACCESS-CM2_psl_en':range(1996,1997),'CanESM5-1_psl_en':range(1996,1997),
                 'EC-Earth3CC_psl_en':range(1996,1997), 'IPSL-CM6A-LR_psl_en':range(1996,1997), 'MIROC-ES2L_psl_en':range(1996,1997),
                 'MPI-ESM1-2-LR_psl_en':range(1996,1997), 'MPI-ESM1-2-HR_psl_en':range(1996,1997), 'UKESM1-0-LL_psl_en':range(1996,1997)}


    ensembles = {'obs1_psl_en':[''], 'obs2_psl_en':[''], 'obs3_psl_en':[''], 'noaa_20CR_slp':[''], 
             'amip_CAM5_slp_control':[''], 'amip_CAM5_slp_en':range(1,41),
             'amip_CAM6_slp_control':[''], 'amip_CAM6_slp_en':range(1,11), 'amip_CAM6_toga_slp_en':range(1,11),
             'amip_ECHAM5_slp_en':range(1,51), 'amip_ECHAM5_climsic_slp_en':range(1,51),
             'amip_GFSv2_slp_en':range(1,51),
             'CESM2_PI_psl':[''], 'CESM2_psl_en':range(1,11),'CESM2_pacemaker_slp_en':range(1,11),
             'MIROC6_PI_psl':[''],'MIROC6_psl_en':range(1,11),
             'CanESM5_PI_psl':[''], 'CanESM5_psl_en':range(1,11),
             'ACCESS-ESM1-5_psl_en':range(1,11),'ACCESS-CM2_psl_en':range(1,11),'CanESM5-1_psl_en':range(1,11),
             'EC-Earth3CC_psl_en':range(1,11), 'IPSL-CM6A-LR_psl_en':range(1,11), 'MIROC-ES2L_psl_en':range(1,11),
             'MPI-ESM1-2-LR_psl_en':range(1,11), 'MPI-ESM1-2-HR_psl_en':range(1,11), 'UKESM1-0-LL_psl_en':range(1,11)}

    if False: # For testing
        ensembles = {'obs1_psl_en':[''], 'obs2_psl_en':[''], 'obs3_psl_en':[''], 'noaa_20CR_slp':[''], 
                 'amip_CAM5_slp_control':[''], 'amip_CAM5_slp_en':range(1,3),
                 'amip_CAM6_slp_control':[''], 'amip_CAM6_slp_en':range(1,3), 'amip_CAM6_toga_slp_en':range(1,3),
                 'amip_ECHAM5_slp_en':range(1,3), 'amip_ECHAM5_climsic_slp_en':range(1,3),
                 'amip_GFSv2_slp_en':range(1,3),
                 'CESM2_PI_psl':[''], 'CESM2_psl_en':range(1,3),'CESM2_pacemaker_slp_en':range(1,3),
                 'MIROC6_PI_psl':[''],'MIROC6_psl_en':range(1,3),
                 'CanESM5_PI_psl':[''], 'CanESM5_psl_en':range(1,3),
                 'ACCESS-ESM1-5_psl_en':range(1,3),'ACCESS-CM2_psl_en':range(1,3),'CanESM5-1_psl_en':range(1,3),
                 'EC-Earth3CC_psl_en':range(1,3), 'IPSL-CM6A-LR_psl_en':range(1,3), 'MIROC-ES2L_psl_en':range(1,3),
                 'MPI-ESM1-2-LR_psl_en':range(1,3), 'MPI-ESM1-2-HR_psl_en':range(1,3), 'UKESM1-0-LL_psl_en':range(1,3)}

    ####
    reload(fig4)
    argus=[]
    records=[]
    for var in vars:
        for en in ensembles[var]:
            argu=(var,en,st_years[var])
            argus.append(argu)
            records.append(vars_types[var])
    if True:
        pool = multiprocessing.Pool(20)
        results = pool.map(fig4.extract_indices, argus)
    else: # Testing and debug
        results=[]
        for argu in argus:
            result=fig4.extract_indices(argu)
            results.append(result)
    trends=[]
    vars_types_records=[]
    for i, result in enumerate(results):
        for j in result: # Result could be a list of trends because the control and multiple years
            trends.append(j)
            vars_types_records.append(records[i]) # With all years included in the control
    trends=np.array(trends)
    vars_types_records=np.array(vars_types_records)

    ### Start plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(3.5,2))
    ### Plot the MERRA2 
    idx = vars_types_records=='MERRA2'
    xs=0; adj=0.1
    scatter1=ax1.scatter(xs-adj, trends[idx], s=9, color='k', marker='D', zorder=5, label='MERRA2 (%s)'%year_test)
    obs_trends=trends[idx]
    # Legend for MERRA2
    lsize=7.5; xstart=-0.22
    legend_scatter=ax1.legend(bbox_to_anchor=(xstart, 1.02), ncol=1, loc='lower left',
            frameon=False, columnspacing=1, handletextpad=-0.5, prop={'size':lsize})
    ax1.add_artist(legend_scatter)
    ### Plot the ERA5
    idx = vars_types_records=='ERA5'
    scatter2=ax1.scatter(xs, trends[idx], s=9, color='k', marker='s', zorder=5, label='ERA5 (%s)'%year_test)
    # Legend for ERA5
    legend_scatter=ax1.legend(bbox_to_anchor=(xstart, 1.06), ncol=1, loc='lower left',
            frameon=False, columnspacing=1, handletextpad=-0.5, prop={'size':lsize})
    #ax1.add_artist(legend_scatter)
    ### Plot the JRA55
    idx = vars_types_records=='JRA55'
    scatter3=ax1.scatter(xs+adj, trends[idx], s=9, color='k', marker='x', zorder=5, label='JRA55 (%s)'%year_test)
    # Legend for JRA55
    legend_scatter=ax1.legend(bbox_to_anchor=(xstart, 1.1), ncol=1, loc='lower left',
            frameon=False, columnspacing=1, handletextpad=-0.5, prop={'size':lsize})
    ax1.add_artist(legend_scatter)
    # Plot the boxplots
    widths=0.2
    xx=0.14; yy=0.28
    xs = [0, 1-xx, 1+xx, 2-xx, 2+xx]; xticks=[0, 1, 2] # xtick sits at the middle
    labels=['20th century\nreanalysis', 'Coupled\nmodel', 'Atmosphere\nmodel']
    plotting_types = ['20CR', 'CMIP_control', 'CMIP_global', 'AMIP_control', 'AMIP_global']
    bpcolor = ['black', 'gray', 'orange', 'gray', 'orange']
    for i, pt in enumerate(plotting_types):
        idx = (vars_types_records==pt)
        bp = ax1.boxplot(trends[idx], positions=[xs[i]], showfliers=True, widths=widths, whis=[5,95], patch_artist=True)
        percentile=sum(trends[idx]>obs_trends)/float(len(trends[idx]))*100
        #percentile=sum(trends[idx]<obs_trends)/float(len(trends[idx]))*100 # For SON (southern Greendland)
        print(percentile)
        ax1.annotate(str(round((percentile),1))+'%', color='black', xy=(xs[i]-0.1,700), xycoords='data', size=8)
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(bp[element], color=bpcolor[i], lw=2.5) # lw=3 in Fig.1
        for box in bp['boxes']:
            box.set(facecolor=bpcolor[i])
        plt.setp(bp['medians'], color='white', lw=2)
        plt.setp(bp['fliers'], marker='o', markersize=0.2, markerfacecolor=bpcolor[i], markeredgecolor=bpcolor[i])
    # Setup the X, Y labels
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=-1, xmin=0.01, xmax=0.99)
    ax1.set_ylabel('Pa/decade')
    ax1.set_ylim(-700,700)
    ax1.set_yticks((-500,-250,0,250,500))
    ax1.set_xlim(xs[0]-0.2, xs[-1]+0.2)
    ax1.tick_params(axis='x',direction="in",length=3,colors='black')
    ax1.tick_params(axis='y',direction="in",length=3,colors='black')
    # Set xticks and xticklabels
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(labels, rotation=0)
    if True: # Setup the legend
        lsize=9
        # For the AMIP
        control_legend=[matplotlib.patches.Patch(facecolor='gray',edgecolor='gray',label='Unforced')]
        goga_legend=[matplotlib.patches.Patch(facecolor='orange',edgecolor='orange',label='Forced\n(%s)'%year_test)]
        legends=control_legend+goga_legend
        legend_amip=ax1.legend(handles=legends, bbox_to_anchor=(0.3, 1.05), ncol=2, loc='lower left',
                frameon=False, columnspacing=1, handletextpad=0.3, prop={'size':lsize})
    # Set the box boundary
    for i in ['right', 'top', 'bottom','left']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=0); ax1.tick_params(axis='y', which='both',length=2)
    # Save the figure 
    fig_name = 'fig4_amip_cmip'
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.001)


def extract_indices(argus):
    var = argus[0]
    en = argus[1]
    st_years = argus[2]
    slicing_period = 20 #1997-Dec to 2017-Feb
    #DJF_circulation=False # SON
    DJF_circulation=True # DJF
    if DJF_circulation: # DJF
        mons = [12,1,2]; st_mon='12'; end_mon='02'; end_yr_adjust=0
        #Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,20,90 # urals (old region)
        #Ilat1, Ilat2, Ilon1, Ilon2 = 55,70,-45,-1 # Iceland (old region)
        # New Regions 
        Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,25,90 # urals
        Ilat1, Ilat2, Ilon1, Ilon2 = 52,70,-37,0 # Iceland
    ###############################3
    data_raw= tools.read_data(var+str(en), months='all', slicing=False, limit_lat=True)
    data_raw = data_raw.sel(latitude=slice(45,89)).compute()
    # Regrid the data  (might not necessary for SLP-diff)
    #lats = np.arange(45,90,3) # Create the same grids as the training data
    #lons = np.arange(-54,108,3).tolist(); lons.remove(0) 
    lats = np.arange(45,90,1) 
    lons = np.arange(-54,108,1).tolist(); lons.remove(0) 
    ds_out = xr.Dataset({"latitude": (["latitude"], lats), "longitude": (["longitude"],lons),})
    regridder = xe.Regridder(data_raw.to_dataset(name='hihi'), ds_out, "bilinear", unmapped_to_nan=True) 
    data_raw = regridder(data_raw)
    lons=data_raw.longitude.values; lats=data_raw.latitude.values
    slopes=[]
    for st_yr in st_years:
        end_yr = st_yr+slicing_period
        print(var,en,st_yr,end_yr)
        data=data_raw.sel(time=slice('%s-%s-01'%(str(st_yr).zfill(4),st_mon),'%s-%s-28'%(str(end_yr+
                    end_yr_adjust).zfill(4),end_mon)))
        if data.time.size<len(mons):
            # Don't do the calculation if the time has less than 3 units (may only have Dec data in 2015 for 20CR)
            # otherwise it will have error in the coarsen - mean
            print('Skip - Special')
            continue
        mon_mask = data.time.dt.month.isin(mons)
        # 1 Extract DJF - do the mean
        data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
        if data.time.size!=slicing_period: # Don't do the calculation if the time doesn't fit 20 or 25
            print('Skip')
            continue
        # 2. Extract the two boxes, get the differences
        if True:  # Compute the area-weighted mean; Select regions and circulation index
            urals_ts=ct.weighted_area_average(data.values,Ulat1,Ulat2,Ulon1,Ulon2,lons,lats)
            iceland_ts=ct.weighted_area_average(data.values, Ilat1, Ilat2, Ilon1, Ilon2, lons,lats)
            circulation_ts = urals_ts-iceland_ts
        else:  # Only Urals without Iceland
            urals_ts=ct.weighted_area_average(data.values,Ulat1,Ulat2,Ulon1,Ulon2,lons,lats)
            circulation_ts = urals_ts
        data = xr.DataArray(circulation_ts, dims=['time'], coords={'time':range(circulation_ts.size)})
        # 3. Get the trend of the time-series
        x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
        y=data; ymean=y.mean(dim='time')
        slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 
        slopes.append(slope.item())
    return slopes

