import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
sys.path.insert(0, '/home/pyfsiew/codes/trend_study/')
import tools
import create_timeseries as ct

if __name__ == "__main__":

    vars = ['obs1_sic_en', 'ACCESS-ESM1-5_sic_en','ACCESS-CM2_sic_en','CanESM5_sic_en','CanESM5-1_sic_en',
             'cesm2_sic_en', 'EC-Earth3CC_sic_en', 'IPSL-CM6A-LR_sic_en', 'miroc6_sic_en', 'MIROC-ES2L_sic_en', 
             'MPI-ESM1-2-LR_sic_en', 'MPI-ESM1-2-HR_sic_en', 'UKESM1-0-LL_sic_en']
    var_name = {'obs1_sic_en':'NSIDC','ACCESS-ESM1-5_sic_en':'ACCESS-ESM1-5 (SSP370)',\
         'ACCESS-CM2_sic_en':'ACCESS-CM2 (SSP585)', 
         'CanESM5_sic_en':'CanESM5 (SSP370)', 'CanESM5-1_sic_en':'CanESM-1 (SSP370)',
         'cesm2_sic_en':'CESM2 (SSP370)', 'EC-Earth3CC_sic_en':'EC-Earth3-CC (SSP245)',
         'IPSL-CM6A-LR_sic_en':'IPSL-CM6A-LR (SSP370)','miroc6_sic_en':'MIROC6 (SSP370)',
         'MIROC-ES2L_sic_en':'MIROC-ES2L (SSP245)', 'MPI-ESM1-2-LR_sic_en':'MPI-ESM1-2-LR (SSP370)',
         'MPI-ESM1-2-HR_sic_en':'MPI-ESM1-2-HR (SSP370)','UKESM1-0-LL_sic_en':'UKESM1-0-LL (SSP370)'}
    ensembles = [['']] + [range(1,11,1)]*(len(vars)-1)
    sic_ratios={var:0.01 for var in vars}; sic_ratios['cesm2_sic_en']=1; sic_ratios['obs1_sic_en']=1 # to (0-1)

    st_yr=1980; end_yr=2100
    st_yr=2015; end_yr=2100
    lat1=68; lat2=85; lon1=5; lon2=90
    ### Compute the timeseries
    mons = [12,1,2]
    data_ts = {var:[] for var in vars}
    for i, var in enumerate(vars):
        for en in ensembles[i]:
            print(var,en)
            data = tools.read_data(var+str(en), months='all', slicing=False) * sic_ratios[var]
            data = data.sel(time=slice('%s-12-01'%st_yr, '%s-02-28'%end_yr))
            #data = data.sel(time=slice('1979-03-01', '2021-03-31'))
            mon_mask = data.time.dt.month.isin(mons)
            data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
            data = data.compute()
            # Only select the BKS region
            # Keep the nan values (all NAN values are the land grids)
            mask_nan = np.isnan(data)
            data = xr.where(data<=0.15, 0, 1) 
            data = xr.where(mask_nan, np.nan, data) 
            lons=data.longitude.values; lats=data.latitude.values
            ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
            ts=xr.DataArray(ts,dims=['time'],coords={'time':data.time}) * 100 # Turn them to concentration
            data_ts[var].append(ts.compute())

    ### ax2 (right is the timeseries of observed sea ice)
    plt.close()
    fig, ax2 = plt.subplots(1,1,figsize=(6,2))
    obs_ts = data_ts[vars[0]][0]
    #x = np.arange(obs_ts.time.size)
    x = obs_ts.time
    ax2.plot(x, obs_ts, linestyle='-', color='k', lw=2, label='NSIDC', zorder=2)
    model_colors=['royalblue','r','g','orange','slategrey','lime','cyan','gold','pink','violet',
                    'brown','gray','peru','orchid','crimson']
    model_ts_means=[] # The mean of each model
    # Plot the timesries of indivial models in different color
    for i, var in enumerate(vars[1:]):
        model_ts = data_ts[var]
        model_ts_mean = xr.concat(model_ts,dim='en').mean(dim='en')
        model_ts_means.append(model_ts_mean)
        #model_ts_max = xr.concat(model_ts,dim='en').max(dim='en')
        model_ts_max = xr.concat(model_ts,dim='en').quantile(q=0.75,dim='en')
        #model_ts_min = xr.concat(model_ts,dim='en').min(dim='en')
        model_ts_min = xr.concat(model_ts,dim='en').quantile(q=0.25,dim='en')
        x = model_ts_mean.time
        #ax2.plot(x, model_ts_mean, linestyle='-', color=model_colors[i], lw=2, label=var.split('_sic')[0], zorder=0)
        ax2.plot(x, model_ts_mean, linestyle='-', color=model_colors[i], lw=2, label=var_name[var], zorder=0)
        ax2.fill_between(x, model_ts_min, model_ts_max, alpha=0.1, fc=model_colors[i])
        if False: # Plot individual members
            for ts in model_ts:
                ax2.plot(x, ts, linestyle='-', color='royalblue', lw=0.5, alpha=0.1)
    if False: ### Plot the multi-model mean
        ax2.plot(x, xr.concat(model_ts_means,dim='models').mean(dim='models'), 'royalblue', label='CMIP6 large ensembles')
    # For legend
    ax2.legend(bbox_to_anchor=(-0.05, 1.1), ncol=3, loc='lower left', frameon=False, columnspacing=1, 
                handletextpad=0.4, labelspacing=0.3)
    ### Set the x-axis
    x = model_ts_mean.time
    ax2.set_xlim(x[0], x[-1])
    nn=5; ax2.set_xticks(x[::nn])
    xticklabels = [str(i-1) + '/' + str(i)[2:] for i in model_ts_mean.time.dt.year.values]
    ax2.set_xticklabels(xticklabels[::nn], rotation=50)
    #ax2.tick_params(axis='x', direction="in", length=3, colors='black')
    # Set the yaxis
    ax2.set_yticks([0,30,60,90])
    ax2.yaxis.tick_right()
    title = 'DJF Barents-Kara\nsea ice extent (%)'
    ax2.set_ylabel(title)
    ### Save the file
    fig_name = 'ice_future_projection'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.1, hspace=0) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)

