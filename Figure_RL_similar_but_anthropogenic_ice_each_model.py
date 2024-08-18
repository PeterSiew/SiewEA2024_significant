import ipdb
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
from importlib import reload
from scipy import stats
#import xesmf as xe

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import create_timeseries as ct


if __name__ == "__main__":

    models = ['ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'cesm2', 'EC-Earth3CC', 
            'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    model_name_dict = {'ACCESS-ESM1-5':'ACCESS-ESM1-5 (SSP370)',\
         'ACCESS-CM2':'ACCESS-CM2 (SSP585)', 'CanESM5':'CanESM5 (SSP370)', 'CanESM5-1':'CanESM5-1 (SSP370)',
         'cesm2':'CESM2 (SSP370)', 'EC-Earth3CC':'EC-Earth3-CC (SSP245)',
         'IPSL-CM6A-LR':'IPSL-CM6A-LR (SSP370)','miroc6':'MIROC6 (SSP370)',
         'MIROC-ES2L':'MIROC-ES2L (SSP245)', 'MPI-ESM1-2-LR':'MPI-ESM1-2-LR (SSP370)',
         'MPI-ESM1-2-HR':'MPI-ESM1-2-HR (SSP370)','UKESM1-0-LL':'UKESM1-0-LL (SSP370)'}
    model_marker = {'ACCESS-ESM1-5':'o',\
         'ACCESS-CM2':'x', 'CanESM5':'o', 'CanESM5-1':'o',
         'cesm2':'o', 'EC-Earth3CC':'x',
         'IPSL-CM6A-LR':'o','miroc6':'o',
         'MIROC-ES2L':'x', 'MPI-ESM1-2-LR':'o',
         'MPI-ESM1-2-HR':'o','UKESM1-0-LL':'o'}
    
    model_vars = [m+'_sic_en' for m in models]
    vars = model_vars
    en_model={var:range(1,11) for var in model_vars}
    ensembles={**en_model}

    #hist_st_yr=1980; hist_end_yr=2095
    hist_st_yr=1980; hist_end_yr=2022
    slicing_period=20; interval=1
    st_years=[i for i in range(hist_st_yr, hist_end_yr+1-slicing_period, interval)]
    end_years=[i+slicing_period for i in st_years]
    have_legend=True
    set_ylim=True
    set_title=False; obsname='Observations'

    ###
    plot_mmm=True
    plot_individual_model=True

    sic = []
    ens_record = []
    for i, var in enumerate(vars):
        print(var)
        for en in ensembles[var]:
            for j, st_yr in enumerate(st_years):
                end_yr = end_years[j]
                period = end_yr-st_yr
                data=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic/%s%s_%s-%s.nc'%(period,var,en,st_yr,end_yr))['training']
                data = data.isel(season=0) # Only DJF
                sic.append(data.item())
                ens_record.append((var,en,st_yr,end_yr))

    ### Calculate the ensemble means for different model and periods
    sic=np.array(sic)
    groups = np.array([i[0]+'-'+str(i[2])+'-'+str(i[3]) for i in ens_record] )
    sic_means = {var:[] for var in vars}
    xticks = []
    for i, var in enumerate(vars):
        for j, st_yr in enumerate(st_years):
            end_yr = end_years[j]
            group = var+'-'+str(st_yr)+'-'+str(end_yr)
            idx = (groups==group).nonzero()[0]
            sic_mean = sic[idx].mean() # Average across (30) ensemble in each var
            sic_means[var].append(sic_mean)
            xticks.append(str(st_yr)[2:]+'-'+str(end_yr)[2:]) if i==0 else ''
        # Average across the 20-year periods, and add it as the last element
        average = np.mean(sic_means[var]) 
        sic_means[var].append(average)
        xticks.append('Avg') if i==0 else ''

    ### Start the plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1, figsize=(6,2.5))
    x = np.arange(len(sic_means[vars[0]]))

    if plot_individual_model:
        colors=['royalblue','r','g','orange','gray','lime','cyan','gold','pink','violet','brown','darkviolet','peru','orchid','crimson']
        xoff=np.linspace(-0.5,0.5,len(model_vars))
        for i, var in enumerate(model_vars):
            #ax1.bar(x+xoff[i],sic_means[var],0.05,color='k',label='hi')
            model=var[0:-7]
            size=3
            if model in ['ACCESS-CM2','MIROC-ES2L','EC-Earth3CC']:
                size=13
            ax1.scatter(x,sic_means[var],color=colors[i], s=size, label=model_name_dict[model],marker=model_marker[model])

    if plot_mmm:
        # Get the mean and standard deviation
        lens_vars = model_vars
        lens_slope = np.array([sic_means[var] for var in lens_vars])
        mm_mean= np.mean(lens_slope,axis=0)
        mm_min = np.std(lens_slope,axis=0)*0.5 # Half of the standard deviation
        mm_max = np.std(lens_slope,axis=0)*0.5
        ax1.plot(x, mm_mean, color='red')
        ax1.fill_between(x, mm_mean-mm_min, mm_mean+mm_max, alpha=0.2, fc='tomato') # New std


    ### For other setting
    ax1.set_xlim(x[0]-0.5, x[-1]+0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xticks, rotation=50, size=9)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, zorder=-1)
    if True: # Add a vertical line to the last two columns
        #ax1.axvline(x=x[-1]-1.5, color='black', linestyle='--', linewidth=1)
        ax1.axvline(x=x[-1]-0.5, color='black', linestyle='--', linewidth=1)
    if set_title:
        ax1.set_title(obsname, loc='left', size=12)
    if have_legend:
        ax1.legend(bbox_to_anchor=(0.99, -0.1), ncol=1, loc='lower left', frameon=False, columnspacing=1, 
                    handletextpad=0.4, labelspacing=0.3)
    if set_ylim:
        ax1.set_ylim(-18,3) # Default for %/decade
    else:
        pass
    ax1.set_ylabel('%/decade', rotation=90, labelpad=-3)
    # Remove the box
    for i in ['right', 'top']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    ax1.tick_params(axis='x', direction="in", length=3, colors='black')
    ax1.tick_params(axis='y', direction="in", length=3, colors='black')
    fig_name = 'anthropogenic_sea_ice_indivdual_model'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.5)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)


