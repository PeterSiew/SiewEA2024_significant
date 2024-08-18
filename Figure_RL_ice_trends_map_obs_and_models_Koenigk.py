
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

    models = ['obs1','ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'cesm2', 
             'EC-Earth3CC', 'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 
             'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    vars = [m+'_sic_en' for m in models]

    ensembles ={var:range(1,11) for var in vars}; ensembles['obs1_sic_en']=['']
    sic_ratios={var:1 for var in vars}; sic_ratios['cesm2_sic_en']=100; sic_ratios['obs1_sic_en']=100
    st_yr=1980; end_yr=2022
    mons = [12,1,2]

    trends={}
    climatologies={}
    for var in vars:
        print(var)
        trend_ensembles=[]
        climatology_ensembles=[]
        for en in ensembles[var]:
            data = tools.read_data(var+str(en), months='all', slicing=False) * sic_ratios[var]
            data = data.sel(time=slice('%s-12-01'%st_yr, '%s-02-28'%end_yr))
            mon_mask = data.time.dt.month.isin(mons)
            data = data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
            data = data.compute()

            # Keep the nan values (all NAN values are the land grids)
            #mask_nan = np.isnan(data)
            #data = xr.where(data<0.15, 0, 1) # The nan grids will also turn to 1 because these grids are false
            #data = xr.where(mask_nan, np.nan, data) # Put the nan back to the data

            # Compute the climatology
            climatology = data.mean(dim='time')
            climatology_ensembles.append(climatology)
            # Compute the linear trend
            x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
            y=data; ymean=y.mean(dim='time')
            trend = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 # per deacde
            trend_ensembles.append(trend)
        climatology_ensembles_mean=xr.concat(climatology_ensembles,dim='en').mean(dim='en')
        climatologies[var] = climatology_ensembles_mean
        trend_ensembles_mean =xr.concat(trend_ensembles,dim='en').mean(dim='en') 
        trends[var] = trend_ensembles_mean 
        

    model_name_dict = {'obs1':'NSIDC','ACCESS-ESM1-5':'ACCESS-ESM1-5',\
         'ACCESS-CM2':'ACCESS-CM2', 'CanESM5':'CanESM5', 'CanESM5-1':'CanESM5-1',
         'cesm2':'CESM2', 'EC-Earth3CC':'EC-Earth3-CC',
         'IPSL-CM6A-LR':'IPSL-CM6A-LR','miroc6':'MIROC6',
         'MIROC-ES2L':'MIROC-ES2L', 'MPI-ESM1-2-LR':'MPI-ESM1-2-LR',
         'MPI-ESM1-2-HR':'MPI-ESM1-2-HR','UKESM1-0-LL':'UKESM1-0-LL'}
    ### Starting the plotting
    plt.close()
    projection=ccrs.NorthPolarStereo(); xsize=2; ysize=2
    shading_grids = [trends[var] for var in vars]
    row, col = 3,5
    grid=row*col
    contour_map_grids = [climatologies[var] for var in vars]
    contour_clevels = [[15]]*grid
    #mapcolors = ['#fddbc7','#FFFFFF','#FFFFFF','#d1e5f0','#4292c6','#2171b5','#08519c'][::-1]
    mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff',
                '#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']
    mapcolors = ['#d1e5f0','#6baed6','#2171b5','#08519c'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [[-25,-20,-15,-10,-5,0,5,10]]
    shading_level_grid = [[-25,-20,-15,-10,-5]]*grid
    #shading_level_grid = [range(-25,25)]*grid
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    if True: # Add region grids
        lat1, lat2, lon1, lon2 = 68,85,5,90
        #region_boxes = None
        region_boxes = [tools.create_region_box(lat1, lat2, lon1, lon2)]*grid
        lat1, lat2, lon1, lon2 = 70,82,15,100
        region_boxes_extra = [tools.create_region_box(lat1, lat2, lon1, lon2)]*grid
    leftcorner_text=None
    freetext=[model_name_dict[m] for m in models]
    freetext_pos=[(0.02,0.9)]*grid
    xlim=[-180,180]; ylim=(60,90)
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                    left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False,
                    region_boxes=region_boxes, shading_extend='min', freetext=freetext, freetext_pos=freetext_pos,
                    leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
                    pval_map=None, pval_hatches=None, fill_continent=True, coastlines=False,
                    contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, set_xylim=None,
                    set_extent=True, transpose=False,contour_lw=1, region_boxes_extra=region_boxes_extra)
    fig_name = 'sea_ice_trends_obs_models'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.27,hspace=0) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)
