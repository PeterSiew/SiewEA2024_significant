import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import pandas as pd
from importlib import reload
import xesmf as xe
import os

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
from scipy import stats

if __name__ == "__main__":

    st_yr=1993; end_yr=2014 # To see the role of PDO
    st_yr=1997; end_yr=2017 # Original manuscirpt
    st_yr=1996; end_yr=2016 # Revused manuscirpt
    ### Read SLP data from long-term reanlaysis
    if False:
        vars=['raw', 'ssts', 'noise']
        filename={'raw':'orig_data','ssts':'yhat','noise':'residual'}
        shading_level_grid=[[-350,-300,-250,-200,-150,-100,-50,50,100,150,200,250,300,350]]*3
        shading_level_grid=[[-400,-300,-200,-100,100,200,300,400]]*3
        shading_level_grid=[np.linspace(-300,300,11)]*3
        leftcorner_text= ['1997-2017 DJF', 'SSTs related', 'Residuals']
    if False: # To further seperate PDO,ENSO and AMO
        vars=['raw', 'ssts', 'noise']
        filename={'raw':'yhat_PDO','ssts':'yhat_ENSO','noise':'yhat_AMO'} 
        shading_level_grid = [np.linspace(-100,100,11)]*3
        leftcorner_text= ['PDO', 'ENSO', 'AMO']
    if True: 
        vars=['raw', 'pdo', 'enso', 'amo', 'noise']
        filename={'raw':'orig_data','pdo':'yhat_PDO','enso':'yhat_ENSO','amo':'yhat_AMO','noise':'residual'} 
        shading_level_grid = [np.linspace(-300,300,11)] + [np.linspace(-100,100,11)]*3 + [np.linspace(-300,300,11)]
        shading_level_grid = [np.linspace(-400,400,11)]*5
        leftcorner_text= ['%s-%s DJF'%(st_yr,end_yr), 'PDO', 'ENSO', 'AMO', 'Residual']
    slopes = []
    mons=[9,10,11]; st_mon='09'; end_mon='11'; end_yr_adjust=-1 # for SON
    mons=[12,1,2]; st_mon='12'; end_mon='02'; end_yr_adjust=0 # for DJF
    for var in vars:
        filepath='/dx15/pyfsiew/olens/%s.nc'%(filename[var])
        slp_raw = xr.open_dataset(filepath)
        varid = [i for i in slp_raw]
        slp_raw = slp_raw[varid[0]]
        slp_raw= slp_raw.rename({'lat':'latitude', 'lon':'longitude'})
        data = slp_raw.sel(time=slice('%s-%s-01'%(str(st_yr),st_mon),'%s-%s-28'%(str(end_yr+end_yr_adjust),end_mon)))
        mon_mask = data.time.dt.month.isin(mons)
        data=data.sel(time=mon_mask).coarsen(time=len(mons), boundary='trim').mean()
        # Calculate the linear trend of the timeseries
        x=xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time}); xmean=x.mean(dim='time')
        y=data; ymean=y.mean(dim='time')
        slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 # sea ice change per decade
        slopes.append(slope)

    shading_grids = slopes
    row=1; col=len(slopes)
    grid = row*col
    mapcolors = ['#2166ac','#4393c3','#92c5de','#d1e5f0','#eef7fa','#ffffff',
                '#ffffff','#fff6e5','#fddbc7','#f4a582','#d6604d','#b2182b']
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', 
                '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    #cmap= 'bwr'
    mapcolor_grid = [cmap] * grid
    contour_map_grids=shading_grids
    contour_map_grids=None
    contour_clevels = shading_level_grid
    clabels_row = [''] * grid
    clabels_row = ['Pa/decade']*grid
    top_title=[''] * col;  left_title=[''] * row
    projection=ccrs.NorthPolarStereo(); xsize=2; ysize=2
    pval_map = None
    matplotlib.rcParams['hatch.linewidth'] = 1; matplotlib.rcParams['hatch.color'] = 'lightgray'
    pval_hatches = [[[0, 0.05, 1000], [None, 'XX']]] * grid # Mask the insignificant regions
    pval_hatches = None
    fill_continent=False
    xlim = [-180,180]; ylim=(45,90)
    region_boxes = None
    region_boxes_extra = None
    #####
    cbarlabel='Pa/decade'
    plt.close()
    fig, ax_all = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    tools.map_grid_plotting(shading_grids, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, 
                    left_titles=left_title, projection=projection, xsize=xsize, ysize=ysize, gridline=False,
                    region_boxes=region_boxes,region_boxes_extra=region_boxes_extra,
                    leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
                    pval_map=None, pval_hatches=None, fill_continent=fill_continent, coastlines=True,
                    contour_map_grids=contour_map_grids, contour_clevels=contour_clevels, set_xylim=None, contour_lw=1,
                    pltf=fig, ax_all=ax_all.flatten(),colorbar=False,
                    indiv_colorbar=[True]*grid)
    ABC = ['A', 'B', 'C', 'D', 'E']
    for i, ax in enumerate(ax_all):
        ax.set_title('('+ABC[i]+')',loc='left')
    if False: ### Setup the colorbar
        cba = fig.add_axes([0.9, 0.2, 0.02, 0.6]) # Figure 4 and 6 and S4
        #cba = fig.add_axes([0.2, 0.09, 0.6, 0.01]) # Figure S6,7
        cNorm  = matplotlib.colors.Normalize(vmin=shading_level_grid[0][0], vmax=shading_level_grid[0][-1])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm,cmap=cmap)
        xticks = xticklabels = [round(i) for i in shading_level_grid[0] if i!=0]
        cb1 = matplotlib.colorbar.ColorbarBase(cba, cmap=cmap, norm=cNorm, orientation='vertical',ticks=xticks, extend='both')
        #cb1.ax.set_xticklabels(xticklabels, fontsize=10)
        cb1.set_label(cbarlabel, fontsize=10, x=1.12, labelpad=10)
    fig_name = 'figS6_decompose_PDO_ENSO_AMO'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=-0.42, hspace=0) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)


