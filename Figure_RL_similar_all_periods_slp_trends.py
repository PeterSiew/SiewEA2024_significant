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
    var = 'obs1_psl_en'
    var_ratio = 1
    en='' # Only for observations
    st_yr=1980
    end_yr=2023
    interval=20

    mons=(12,1,2)
    end_yr_adjusts=0
    slopes={}

    # Read data
    data_raw = tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False) * var_ratio
    data_raw=data_raw.sel(time=slice('%s-%s-01'%(str(st_yr),mons[0]),'%s-%s-28'%(end_yr,mons[-1])))
    slp_trends={}
    # Start the loop
    for st_yr in range(st_yr,end_yr,1):
        end_yr=st_yr+interval
        # Only continue 
        data=data_raw.sel(time=slice('%s-%s-01'%(st_yr,mons[0]),'%s-%s-28'%(end_yr,mons[-1])))
        # Extract DJF
        mon_mask=data.time.dt.month.isin(mons)
        data=data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
        if data.time.size!=interval:
            continue
        # Calculate the trend in each grid
        x = xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time})
        xmean = x.mean(dim='time')
        y=data; ymean = y.mean(dim='time')
        results=tools.linregress_xarray(y, x, null_hypo=0); slope=results['slope'] * 10
        slp_trends[st_yr]=slope
        #pval = results['pvalues'] 


    ipdb.set_trace()
    ### iStart the plotting maps
    shading_grids=[*slp_trends.values()]
    contour_grids= None
    contour_pval_grids = None
    row=5; col=5
    grid=row*col
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF','#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [[-500,-400,-300,-200,-100,100,200,300,400,500]]*grid # For 1996-2016
    shading_level_grid = [[-700,-600,-500,-400,-300,-200,-100,100,200,300,400,500,600,700]]*grid # For 1996-2016
    shading_level_grid = [np.linspace(-700,700,11)]*grid
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    projection=ccrs.NorthPolarStereo(); xsize=1.5; ysize=1.5
    xlim = [-180,180]; ylim=(50,90)
    xylims=None; set_extent=True
    pval_map = None
    pval_hatches = None
    fill_continent=False
    region_boxes=None; region_boxes_extra=None
    years=[*slp_trends.keys()]
    leftcorner_text=[str(yr)+'-'+str(yr+interval) for yr in years]
    if True: # 1996-2016 (New)
        Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,30,90 # urals
        Ilat1, Ilat2, Ilon1, Ilon2 = 50,65,-35,0 # Iceland
        region_boxes = [None]*16 + [tools.create_region_box(Ulat1, Ulat2, Ulon1, Ulon2)] + [None]*30
        region_boxes_extra = [None]*16 + [tools.create_region_box(Ilat1, Ilat2, Ilon1, Ilon2)] + [None]*30
    ###
    tools.map_grid_plotting(shading_grids,row,col,mapcolor_grid,shading_level_grid,clabels_row,
            top_titles=top_title,left_titles=left_title, projection=projection,xsize=xsize,ysize=ysize,
            gridline=False, region_boxes=region_boxes,
            region_boxes_extra=region_boxes_extra,
            leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
            pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=True,
            contour_map_grids=contour_grids, contour_clevels=None, contour_lw=1, set_xylim=xylims,
            set_extent=set_extent, shading_extend='both')
    # Save
    fig_name = 'fig3_all_20-year_periods'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0)# hspace is the vertical=3 (close)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name),
                bbox_inches='tight', dpi=400, pad_inches=0.01)


