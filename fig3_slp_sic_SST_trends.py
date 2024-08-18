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

    if False:
        vars = ['amip_CAM6_slp_en']; ensembles = [range(1,11)]
        vars = ['amip_CAM5_slp_en']; ensembles = [range(1,41)]
        vars = ['amip_GFSv2_slp_en']; ensembles = [range(40,51)]
        vars = ['amip_ECHAM5_slp_en']; ensembles = [range(1,51)]
        vars = ['cesm2_pacemaker_slp_en']; ensembles = [range(1,11)]
        vars = ['amip_WACCM6sicfixed_slp_en']; ensembles = [range(1,31)]
        vars = ['amip_WACCM6_slp_en']; ensembles = [range(1,31)]
        vars = ['SLP_regrid_1x1_monthly','heat_advection_850hPa_regrid_1x1_monthly']; ensembles=[[''], ['']]
        vars = ['ERA5_MSLP','obs_sic_en']; ensembles=[[''], ['']]

    vars = ['obs2_psl_en','obs2_sic_en','U10M_regrid_1x1_monthly','V10M_regrid_1x1_monthly']
    vars = ['obs3_psl_en','obs3_sic_en','U10M_regrid_1x1_monthly','V10M_regrid_1x1_monthly']
    vars = ['obs1_psl_en','obs1_sic_en','U10M_regrid_1x1_monthly','V10M_regrid_1x1_monthly'] # ALL MERRA2
    ensembles=[['']]*len(vars)
    en='' # Only for observations
    var_ratios = [1,-24*3600]
    var_ratios = [1,100]
    var_ratios = [1,100,1,1]
    st_years=[1996,1997]; interval=20 #1996-2016 and 1997-2017

    # Only DJF
    mons_text=['DJF']; mons=[(12,1,2)]
    st_mons=['12']; end_mons=['02']; end_yr_adjusts=[0] # for DJF

    supp_figure=True
    supp_figure=False
    if supp_figure: # For all four seasons
        mons_text=['DJF','SON','JJA','MAM']; mons=[(12,1,2),(9,10,11),(6,7,8),(3,4,5)]
        st_mons=['12','09','06','03']; end_mons=['02','11','08','05']; end_yr_adjusts=[0,-1,-1,-1] 

    ABCD=['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)']
    leftcorner_text=[str(st_yr)+'-'+str(st_yr+interval)+' '+mt for st_yr in st_years for mt in mons_text]
    slopes = {st_yr: {var:{mon:None for mon in mons} for var in vars} for st_yr in st_years}
    pvals= {st_yr: {var:{mon:None for mon in mons} for var in vars} for st_yr in st_years}
    for st_yr in st_years:
        end_yr=st_yr+interval
        for i, var in enumerate(vars):
            data_raw = tools.read_data(var+str(en), months='all', slicing=False, limit_lat=False) * var_ratios[i]
            data_raw = data_raw.compute()
            for j, mon in enumerate(mons):
                st_mon=st_mons[j]
                end_mon=end_mons[j]
                end_yr_adjust=end_yr_adjusts[j]
                # Read the data
                data=data_raw.sel(time=slice('%s-%s-01'%(str(st_yr),st_mon),'%s-%s-28' %(str(end_yr+end_yr_adjust),end_mon)))
                mon_mask = data.time.dt.month.isin(mon)
                data=data.sel(time=mon_mask).coarsen(time=len(mon), boundary='trim').mean()
                # Calculate the trend in each grid
                x = xr.DataArray(range(data.time.size), dims=['time'], coords={'time':data.time})
                xmean = x.mean(dim='time')
                y=data; ymean = y.mean(dim='time')
                #slope = ((x-xmean)*(y-ymean)).sum(dim='time') / ((x-xmean)**2).sum(dim='time') * 10 # per deacde
                results=tools.linregress_xarray(y, x, null_hypo=0); slope=results['slope'] * 10
                slopes[st_yr][var][mon]=slope
                pval = results['pvalues'] 
                pvals[st_yr][var][mon]=pval

    ### Get the plotting maps
    var=vars[0] # SLP circulation
    contour_grids=[slopes[st_yr][var][mon] for st_yr in st_years for mon in mons]
    contour_pval_grids =[pvals[st_yr][var][mon] for st_yr in st_years for mon in mons]
    sig_contour_grids=[xr.where(pval<=0.1,contour,np.nan) for contour, pval in zip(contour_grids,contour_pval_grids)]
    insig_contour_grids=[xr.where(pval>0.1,contour,np.nan) for contour, pval in zip(contour_grids,contour_pval_grids)]
    var=vars[1] # SIC shading
    shading_grids=[slopes[st_yr][var][mon] for st_yr in st_years for mon in mons]
    var=vars[2] # 10-U wind
    x_vectors=[slopes[st_yr][var][mon] for st_yr in st_years for mon in mons]
    var=vars[3] # 10-V wind
    y_vectors=[slopes[st_yr][var][mon] for st_yr in st_years for mon in mons]
    ### Start the map plotting
    row=len(mons); col=len(st_years)
    if supp_figure:
        row,col=col,row
        for i in [1,2,3]:
            leftcorner_text[i]=leftcorner_text[i].replace('1996-2016','')
        for i in [5,6,7]:
            leftcorner_text[i]=leftcorner_text[i].replace('1997-2017','')
        for i in [0,1,2,3,5,6,7]:
            ABCD[i]=''
        ABCD[0]='(A)'
        ABCD[4]='(B)'
    grid=row*col
    mapcolors = ['#d1e5f0','#6baed6','#2171b5','#08519c'][::-1]
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid = [cmap] * grid
    shading_level_grid = [[-45,-35,-25,-15,-5]]*grid # SIC
    #contour_clevels=[[-250,-200,-150,-100,100,150,200,250]]*grid # Default for 1997-2017
    contour_clevels=[[-500,-400,-300,-200,-100,100,200,300,400,500]]*grid # For 1996-2016
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    #projection=ccrs.Miller(central_longitude=0); xsize=4; ysize=1
    #projection=ccrs.Mercator(central_longitude=0, globe=None, min_latitude=-20,#latitude_true_scale=0); xsize=4; ysize=1
    projection=ccrs.NorthPolarStereo(); xsize=3; ysize=3
    xlim = [-180,180]; ylim=(50,90)
    xylims=None; set_extent=True
    xylims=[((-3500000,4000000), (-4000000,900000))]*grid; set_extent=False
    xylims=[((-3500000,4500000), (-4500000,900000))]*grid; set_extent=False # (x-left, x-right) (y-low, y-up)
    pval_map = None
    matplotlib.rcParams['hatch.linewidth']=1; matplotlib.rcParams['hatch.color'] = 'red'
    pval_hatches = [[[0, 0.05, 1000], [None, 'XX']]] * grid # Mask the insignificant regions
    pval_hatches = None
    fill_continent=False
    fill_continent=True
    region_boxes=None; region_boxes_extra=None
    #Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,20,90 # urals for 1997-2017 (old)
    #Ilat1, Ilat2, Ilon1, Ilon2 = 55,70,-45,-1 # Iceland for 1997-2017 (old)
    # 1996-2016 (New)
    Ulat1, Ulat2, Ulon1, Ulon2 = 50,70,25,90 # urals
    Ilat1, Ilat2, Ilon1, Ilon2 = 52,70,-37,0 # Iceland
    region_boxes = [tools.create_region_box(Ulat1, Ulat2, Ulon1, Ulon2)] * 10
    region_boxes_extra = [tools.create_region_box(Ilat1, Ilat2, Ilon1, Ilon2)] * 10
    if supp_figure:
        region_boxes=[None]*10
        region_boxes_extra=[None]*10
    #
    plt.close()
    fig, axs = plt.subplots(row,col,figsize=(col*xsize, row*ysize), subplot_kw={'projection':projection})
    axs=axs.flatten()
    tools.map_grid_plotting(shading_grids,row,col,mapcolor_grid,shading_level_grid,clabels_row,
            top_titles=top_title,left_titles=left_title, projection=projection,xsize=xsize,ysize=ysize,
            gridline=False, region_boxes=region_boxes,
            region_boxes_extra=region_boxes_extra,
            leftcorner_text=leftcorner_text, ylim=ylim, xlim=xlim, quiver_grids=None,
            pval_map=pval_map, pval_hatches=pval_hatches, fill_continent=fill_continent, coastlines=False,
            contour_map_grids=contour_grids, contour_clevels=contour_clevels, contour_lw=1, set_xylim=xylims,
            pltf=fig, ax_all=axs, set_extent=set_extent, shading_extend='min')
    if True: # Plot the significant SLP
        for i, ax in enumerate(axs):
            map2d=sig_contour_grids[i]
            lons=map2d.longitude.values; lats=map2d.latitude.values
            cs1=ax.contour(lons,lats,map2d,contour_clevels[0],linewidths=2,transform=ccrs.PlateCarree(), colors='k')
    if True: # Plot the wind vectors
        for i, ax in enumerate(axs):
            x_vector=x_vectors[i]; y_vector=y_vectors[i]
            lons=x_vector.longitude.values; lats=x_vector.latitude.values
            transform=ccrs.PlateCarree()
            # Regrid_shape is important. Larger means denser
            Q = ax.quiver(lons, lats, x_vector.values, y_vector.values, headwidth=6, headlength=3,
                    headaxislength=2, units='width', scale_units='inches', pivot='middle', color='green', 
                    width=0.01, scale=5, transform=ccrs.PlateCarree(), regrid_shape=5, zorder=10) 
        axs[-1].quiverkey(Q,1.15,0.23,2,"2 m/s\n/decade", labelpos='S', labelsep=0.05, coordinates='axes', zorder=10)
    ### Plot (A), (B), (C), (D)
    freetext_pos=(-0.01,1.03)
    for i, ax in enumerate(axs):
        freetext=r'$\bf{%s}$'%ABCD[i]
        ax.annotate(freetext, xy=freetext_pos, xycoords='axes fraction', size=10)
    ### For the colorbar label
        axs[-1].annotate('%/decade', xy=(1.04,1.01), xycoords='axes fraction', size=9)
    ### Save it
    fig_name = 'fig3_combined_sic_trend'
    plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=-0.4)# hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name),
                bbox_inches='tight', dpi=500, pad_inches=0.01)


