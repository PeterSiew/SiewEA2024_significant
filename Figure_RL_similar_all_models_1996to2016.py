import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
import multiprocessing
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
from scipy import stats


if __name__ == "__main__":
    vars = ['obs1_psl_en','CanESM5_psl_en','cesm2_psl_en','miroc6_psl_en','amip_CAM5_slp_en','amip_CAM6_slp_en','amip_ECHAM5_slp_en', 'amip_GFSv2_slp_en']
    ensembles=[[''],range(1,11),range(1,11),range(1,11),range(1,11),range(1,11),range(1,11),range(1,11)]
    ensembles=[[''],range(1,51),range(1,51),range(1,51),range(1,41),range(1,11),range(1,51),range(1,51)]
    vars_name= {'obs1_psl_en':'MERRA2','CanESM5_psl_en':'Coupled\nCanESM5','cesm2_psl_en':'Coupled\nCESM2','miroc6_psl_en':'Coupled\nMIROC6',
            'amip_CAM5_slp_en':'Atmosphere-only\nCAM5','amip_CAM6_slp_en':'Atmosphere-only\nCAM6' ,'amip_ECHAM5_slp_en':'Atmosphere-only\nECHAM5', 'amip_GFSv2_slp_en':'Atmosphere-only\nGFSv2'}
    var_ratio = 1
    en='' # Only for observations
    st_yr=1996; end_yr=2016
    interval=20

    slp_trends={var:[] for var in vars}
    mons=(12,1,2)
    # Read data
    for i, var in enumerate(vars):
        for en in ensembles[i]:
            print(var,en)
            data_raw = tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False) * var_ratio
            data_raw=data_raw.sel(time=slice('%s-%s-01'%(str(st_yr),mons[0]),'%s-%s-28'%(end_yr,mons[-1])))
            # Extract DJF
            mon_mask=data_raw.time.dt.month.isin(mons)
            data=data_raw.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
            if data.time.size!=interval:
                continue
            # Calculate the trend in each grid
            x = xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time})
            xmean = x.mean(dim='time')
            y=data; ymean = y.mean(dim='time')
            results=tools.linregress_xarray(y, x, null_hypo=0)
            slope=results['slope'] * 10
            slope=slope.compute()
            slp_trends[var].append(slope)
            pval = results['pvalues'] 
    ipdb.set_trace()

    # Average across the ensembles
    shading_grids=[xr.concat(slp_trends[var],dim='en').mean(dim='en') for var in vars]
    ### iStart the plotting maps
    contour_grids= None
    contour_pval_grids = None
    row=2; col=4
    grid=row*col
    #mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF','#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff','#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [[-500,-400,-300,-200,-100,100,200,300,400,500]]*grid # For 1996-2016
    shading_level_grid = [np.linspace(-700,700,11)]*grid
    shading_level_grid = [np.linspace(-500,500,13)]*grid
    shading_level_grid = [np.linspace(-450,450,13)]*grid
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    projection=ccrs.NorthPolarStereo(); xsize=2; ysize=2
    xlim = [-180,180]; ylim=(50,90)
    xylims=None; set_extent=True
    pval_map = None
    pval_hatches = None
    fill_continent=False
    region_boxes=None; region_boxes_extra=None
    years=[*slp_trends.keys()]
    leftcorner_text=[vars_name[var] for var in vars]
    tools.map_grid_plotting(shading_grids,row,col,mapcolor_grid,shading_level_grid,clabels_row,
            top_titles=top_title,left_titles=left_title, projection=projection,xsize=xsize,ysize=ysize,
            gridline=False, region_boxes=region_boxes,
            region_boxes_extra=region_boxes_extra,
            leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
            pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=True,
            contour_map_grids=contour_grids, contour_clevels=None, contour_lw=1, set_xylim=xylims,
            set_extent=set_extent, shading_extend='both')
    # Save
    fig_name = 'fig3_all_models'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0)# hspace is the vertical=3 (close)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name),
                bbox_inches='tight', dpi=400, pad_inches=0.01)


