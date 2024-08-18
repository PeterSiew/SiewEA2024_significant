import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from scipy import stats

import sys
sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct

if __name__ == "__main__":

    # Make sure the obs1 is the ice with -180 to 180 version
    models = ['obs1','ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'CESM2', 
             'EC-Earth3CC', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 
             'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    vars=[m+'_sic_en' for m in models]
    ensembles ={var:range(1,11) for var in vars}; ensembles['obs1_sic_en']=['']
    sic_ratios={var:0.01 for var in vars}; sic_ratios['CESM2_sic_en']=1; sic_ratios['obs1_sic_en']=1 # to (0-1)

    ### Compute the timeseries
    mons = [12,1,2]
    data_ts = {var:[] for var in vars}
    #lat1=68; lat2=85; lon1=5; lon2=90
    lat1=70; lat2=82; lon1=15; lon2=100 ; #For the region in Koenigk et al. 2016 
    st_yr=1980; end_yr=2022
    for i, var in enumerate(vars):
        for en in ensembles[var]:
            print(var,en)
            data = tools.read_data(var+str(en), months='all', slicing=False) * sic_ratios[var]
            data = data.sel(time=slice('%s-12-01'%st_yr, '%s-02-28'%end_yr))
            #data = data.sel(time=slice('1979-03-01', '2021-03-31'))
            mon_mask = data.time.dt.month.isin(mons)
            data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
            data = data.compute()
            if False: # Sea ice extent in area
                # Only select the BKS region
                data = data.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                data = xr.where(data<=0.15, np.nan, 1) # Sea ice extent
                #data = xr.where(data<=0.15, np.nan, data) # Sea ice area
                # Get the sea ice timeseries
                area_grid = np.empty((data.latitude.size, data.longitude.size))
                area_grid[:,:] = np.diff(data.latitude).mean()*111.7 * np.diff(data.longitude).mean()*111.7
                coords = {'latitude':data.latitude, 'longitude':data.longitude}
                area_grid = xr.DataArray(area_grid, dims=[*coords], coords=coords)
                cos_lats = np.cos(area_grid.latitude*np.pi/180)
                area_grid = area_grid * cos_lats
                ts = (data*area_grid).sum(dim='latitude',skipna=True).sum(dim='longitude',skipna=True)/1e6 
                data_ts[var].append(ts.compute())
            else: # Sea ice extent in %
                # Keep the nan values (all NAN values are the land grids)
                mask_nan = np.isnan(data)
                data = xr.where(data<0.15, 0, 1) # The nan grids will also turn to 1 because these grids are false
                data = xr.where(mask_nan, np.nan, data) # Put the nan back to the data
                lons=data.longitude.values; lats=data.latitude.values
                ts=ct.weighted_area_average(data.values,lat1,lat2,lon1,lon2,lons,lats)
                ts=xr.DataArray(ts,dims=['time'],coords={'time':data.time}) * 100
                data_ts[var].append(ts.compute())

    if True: ### Compute the linear trend for observations (map)
        st_yr=1980; end_yr=2023
        mons = [12,1,2]
        var = vars[0]; var_ratio=100
        data = tools.read_data(var,months='all',slicing=False).sel(latitude=slice(25,90))*var_ratio
        data = data.sel(time=slice('%s-12-01'%str(st_yr).zfill(4), '%s-02-28'%str(end_yr).zfill(4)))
        mon_mask = data.time.dt.month.isin(mons)
        data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
        # Compute the climatology
        climatology = data.mean(dim='time')
        # Compute the linear trend
        x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
        y=data; ymean=y.mean(dim='time')
        trend = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 # per deacde
        # Save the trend
        trends_obs = trend.compute()

    ### Starting the plotting
    ### Plot the ax1 (the sea ice linear trend first)
    plt.close()
    projection=ccrs.NorthPolarStereo(); xsize=2; ysize=2
    fig = plt.figure(figsize=(8,2))
    #ax1 = fig.add_subplot(1,4,(1,2), projection=projection)
    ax1 = fig.add_subplot(1,2,1, projection=projection)
    # ax1 (left) is the sea ice lossing map
    shading_grids = [trends_obs+0.00001]
    contour_map_grids = [climatology]
    contour_clevels = [[15]]
    row=1; col=1; grid=row*col
    #mapcolors = ['#fddbc7','#FFFFFF','#FFFFFF','#d1e5f0','#4292c6','#2171b5','#08519c'][::-1]
    mapcolors = ['#d1e5f0','#6baed6','#2171b5','#08519c'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [[-25,-20,-15,-10,-5,0,5,10]]
    shading_level_grid = [[-25,-20,-15,-10,-5]] # SIC
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    region_boxes = None
    region_boxes = [tools.create_region_box(lat1, lat2, lon1, lon2)]
    leftcorner_text=None
    freetext=[r'$\bf{(A)}$']
    freetext_pos=[(-0.01,1.03)]
    xlim=[-180,180]; ylim=(60,90)
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                    left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False,
                    region_boxes=region_boxes, shading_extend='neither', freetext=freetext, freetext_pos=freetext_pos,
                    leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
                    pval_map=None, pval_hatches=None, fill_continent=True, coastlines=False,
                    contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, set_xylim=None,
                    pltf=fig, ax_all=[ax1], set_extent=True, transpose=True,contour_lw=0.6)
    #ax1.annotate('%', xy=(0.97,-0.23), xycoords='axes fraction', size=9)
    ### ax2 (right is the timeseries of observed sea ice)
    #ax2 = fig.add_subplot(1,4,(3,4))
    ax2 = fig.add_subplot(1,2,2)
    obs_ts = data_ts[vars[0]][0]
    x = np.arange(obs_ts.time.size)
    ax2.plot(x, obs_ts, linestyle='-', color='k', lw=2, label='Observations', zorder=5)
    model_colors = ['royalblue', 'red', 'green']
    model_ts_means=[] # The mean of each model
    model_ts_all=[]
    # Plot the timesries of indivial models in different color
    for i, var in enumerate(vars[1:]):
        model_ts = data_ts[var]
        model_ts_mean = xr.concat(model_ts,dim='en').mean(dim='en')
        model_ts_means.append(model_ts_mean)
        #model_ts_max = xr.concat(model_ts,dim='en').max(dim='en')
        #model_ts_min = xr.concat(model_ts,dim='en').min(dim='en')
        #ax2.plot(x, model_ts_mean, linestyle='-', color=model_colors[i], lw=2, label=var)
        #ax2.fill_between(x, model_ts_min, model_ts_max, alpha=0.1, fc=model_colors[i])
        # Plot individual members
        for ts in model_ts:
            ax2.plot(x, ts, linestyle='-', color='royalblue', lw=0.5, alpha=0.1)
            #print(var,ts)
            #model_ts_all.append(ts)
    if False:# Plot the spread of the ensembles (all models)
        ax2.fill_between(x, np.percentile(model_ts_all,5,axis=0),np.percentile(model_ts_all,95,axis=0),alpha=0.1,fc='royalblue')
    # Plot the multi-model mean
    mmm = xr.concat(model_ts_means,dim='models').mean(dim='models')
    mmm_m, mmm_c, _, _, _ = stats.linregress(range(mmm.size),mmm.values)
    obs_m, obs_c, _, _, _ = stats.linregress(range(obs_ts.size),obs_ts.values)
    obs_ts_sel = obs_ts.sel(time=slice('1998-01-01','2017-01-01'))
    obs_sel_m, obs_sel_c, _, _, _ = stats.linregress(range(obs_ts_sel.size),obs_ts_sel.values)
    print(obs_ts) # print observations timeseries
    print(mmm_m*10) # the trend of mmm over the whole period
    print(obs_m*10) # the trend of obs over the whole period
    print(obs_sel_m*10) # the trend of obs over 1997/98 to 2016/17
    ax2.plot(x, mmm, 'royalblue', label='CMIP6')
    ax2.legend(bbox_to_anchor=(0, 0.01), ncol=1, loc='lower left', frameon=False, columnspacing=1, 
                handletextpad=0.4, labelspacing=0.3)
    ### Set the x-axis
    ax2.set_xlim(x[0], x[-1])
    nn=5; ax2.set_xticks(x[::nn])
    xticks = [str(i-1)[-2:] + '/' + str(i)[-2:] for i in obs_ts.time.dt.year.values]
    #xticks = [str(i-1) for i in obs_ts.time.dt.year.values]
    ax2.set_xticklabels(xticks[::nn], rotation=0)
    ax2.tick_params(axis='x', direction="in", length=3, colors='black')
    # Srt the (B) for label
    freetext=r'$\bf{(B)}$'
    ax2.annotate(freetext, xy=freetext_pos[0], xycoords='axes fraction', size=10)
    # Set %/decade for ax1
    ax2.annotate('%/decade', xy=(-0.05,-0.22), xycoords='axes fraction', size=9)
    # Set the '%' unit for ax2
    ax2.annotate('%', xy=(1.015,0.98), xycoords='axes fraction', size=10)
    # Set the yaxis
    ax2.set_yticks([30,60,90])
    ax2.yaxis.tick_right()
    ax2.tick_params(axis='y', direction="in", length=3, colors='black')
    title = 'DJF BKS ice area\n(million km^2)'
    # Save the file
    fig_name = 'Fig1_DJF_sea_ice_timeseries'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.27,hspace=0) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)

