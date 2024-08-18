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

    obss = ['obs1','obs2','obs3']
    models = ['ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'cesm2', 'EC-Earth3CC', 
            'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']

    obsi_vars = [m+'_sic_en' for m in obss]
    model_vars = [m+'_sic_en' for m in models]
    vars = obsi_vars + model_vars
    en_model={var:range(1,11) for var in model_vars}
    if False: # For all available ensembles
        en_model_temp={'ACCESS-ESM1-5':range(1,41),'ACCESS-CM2':range(1,11),'CanESM5':range(1,51),'CanESM5-1':range(1,11),
                'cesm2':range(1,51),'EC-Earth3CC':range(1,11),'IPSL-CM6A-LR':range(1,11),'miroc6':range(1,51),
                'MIROC-ES2L':range(1,31),'MPI-ESM1-2-LR':range(1,31),'MPI-ESM1-2-HR':range(1,11),'UKESM1-0-LL':range(1,11)} # For all models
        en_model={m+'_sic_en':en_model_temp[m] for m in en_model_temp.keys()}
    en_obs={var:[''] for var in obsi_vars}
    ensembles={**en_model, **en_obs}

    hist_st_yr=1980; hist_end_yr=2022
    slicing_period=20; interval=1
    st_years=[i for i in range(hist_st_yr, hist_end_yr+1-slicing_period, interval)]
    end_years=[i+slicing_period for i in st_years]
    #st_years = st_years+[1980]; end_years = end_years+[2023]
    set_title=True
    set_title=False; obsname='Observations'
    offset_bool=False # Auto offset
    get_green_line=False# The sum of circu- and anthro- components
    single_model=False # Multi-model
    have_legend=True
    set_ylim=True
    plot_sensitivity=True # Plot the red and blue shading (model sensitivity)
    plot_nudging=False
    plot_mmm=True
    plot_ann_fitting=True
    plot_observations=True
    fixed_legend_pos=True
    fig_name_add=obs_vars if 'obs_vars' in locals() else ""
    obs_legend='NSIDC'

    if 'cesm2_training_only' not in locals(): cesm2_training_only=False
    if cesm2_training_only:
        models = ['cesm2']
        obsi_vars = [m+'_sic_en' for m in obss]
        model_vars = [m+'_sic_en' for m in models]
        vars = obsi_vars + model_vars
        #en_model={var:range(51,101) for var in model_vars}; en_obs={var:[''] for var in obsi_vars}
        en_model={var:range(1,51) for var in model_vars}; en_obs={var:[''] for var in obsi_vars}
        ensembles={**en_model, **en_obs}
        plot_sensitivity=False # Plot the red and blue shading (model sensitivity)
        get_green_line=True 

    # For Figure S2 (all models' first ensembles)
    if 'model_first_en' in locals():
        obsi_vars=obsi_vars_fake
        # Fake observations part
        #obss = ['cesm2_obsfake']
        obss = [obsi_vars_fake+'_obsfake']
        fig_name_add=obss
        obsi_vars = [m+'_sic_en' for m in obss]
        en_obs={var:[obs_en_fake] for var in obsi_vars}
        # Anthropogenic part
        models = [obss[0].replace('_obsfake','')]
        model_vars = [m+'_sic_en' for m in models]
        en_model={var:range(1,11) for var in model_vars}
        ensembles={**en_model, **en_obs}
        vars = obsi_vars + model_vars
        # Other settings
        set_ylim=False
        get_green_line=True# The sum of circu- and anthro- components
        obs_legend= obsi_vars_fake + ' (member %s)'%obs_en_fake
        set_title=True; obsname="(%s) "%ABCDE + obs_legend
        fixed_legend_pos=False

    # For Figure S1 (all reanalysis)
    if 'reanalysis_name' in locals():
        set_title=True; obsname="(%s) "%ABCDE + reanalysis_name

    fig_S3_nudging=True
    fig_S3_nudging=False
    if fig_S3_nudging: # For the nudging (Figure S2)
        include_forced=False
        obss = ['obs1','obs2','obs3']
        models=['ding22_aice_avg','cesm1_sic_avg']; model_name='CESM1'; ABC='A'
        models=['ding23_aice_avg','cesm2_sic_avg']; model_name='CESM2'; ABC='B'
        ###
        obsi_vars = [m+'_sic_en' for m in obss]
        model_vars = [m+'_sic_en' for m in models]
        vars = obsi_vars + model_vars
        ###
        en_obs={var:[''] for var in obsi_vars}
        en_model={var:[''] for var in model_vars}
        ensembles={**en_model, **en_obs}
        hist_st_yr=1980; hist_end_yr=2020
        slicing_period=20; interval=1
        st_years=[i for i in range(hist_st_yr, hist_end_yr+1-slicing_period, interval)]
        end_years=[i+slicing_period for i in st_years]
        plot_observations =True
        plot_ann_fitting=False
        plot_mmm=False
        get_green_line=True
        plot_nudging=True
        fig_name_add='nudging'

    ### Start getting the sea ice trends
    sic = []
    ens_record = []
    for i, var in enumerate(vars):
        var_read=var.replace('_obsfake','') # This is just for reading the files
        print(var)
        for en in ensembles[var]:
            for j, st_yr in enumerate(st_years):
                end_yr = end_years[j]
                period = end_yr-st_yr
                data=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic/%s%s_%s-%s.nc'%(period,var_read,en,st_yr,end_yr))['training']
                #data=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic_area/%s%s_%s-%s.nc'%(period,var_read,en,st_yr,end_yr))['training']
                #data=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic_old/%s%s_%s-%s.nc'%(period,var_read,en,st_yr,end_yr))['training']
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
        # Average across the 20-year periods, and add this as the last element
        average = np.mean(sic_means[var]) 
        sic_means[var].append(average)
        xticks.append('Avg') if i==0 else ''

    ### Start the plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1, figsize=(6,2.5))
    x = np.arange(len(sic_means[vars[0]]))

    if plot_observations: # Plot the observations 
        ice_ts = []
        for var in obsi_vars:
            obs_trends=sic_means[var]
            ice_ts.append(obs_trends)
        ice_ts_min=np.min(ice_ts,axis=0)
        ice_ts_max=np.max(ice_ts,axis=0)
        #ice_ts_mean=np.mean(ice_ts,axis=0) 
        ice_ts_mean=np.array(ice_ts[0]) # Only CDR
        if len(obsi_vars)==1: # plot the line
            ax1.plot(x, ice_ts_min, linestyle='-', color='k', lw=2, label=obs_legend) # This is only for the fake obs
        else: # More than one observations 
            ax1.fill_between(x, ice_ts_min, ice_ts_max, fc='black', label=obs_legend, zorder=100)

    if plot_ann_fitting: # Add circulation components by ANN fitting
        years_keys = [i for i in obs_predicts_central.keys()] 
        if include_forced:
            range_mean = [obs_predicts_central[yr][0] for yr in years_keys] # index 0 is internla variability
            range_max = [obs_predicts_central[yr][0]+combine_std[yr][0]*0.5 for yr in years_keys]
            range_min= [obs_predicts_central[yr][0]-combine_std[yr][0]*0.5 for yr in years_keys]
        else:
            range_mean = [obs_predicts_central[yr] for yr in years_keys]
            range_max = [obs_predicts_central[yr]+combine_std[yr]*0.5 for yr in years_keys]
            range_min = [obs_predicts_central[yr]-combine_std[yr]*0.5 for yr in years_keys]
        range_mean_std = np.std(range_mean[0:-1]) # Don't consider the last element:1980-2021
        # Calculate the average of the 20-years slicing windows
        #range_mean_avg=np.array(np.mean(range_mean))
        #range_max_avg=np.array(np.mean(range_max))
        #range_min_avg=np.array(np.mean(range_min))
        range_mean_avg=np.mean(range_mean)
        range_max_avg=np.mean(range_max)
        range_min_avg=np.mean(range_min)
        # Add them as the last element
        range_mean=range_mean+[range_mean_avg]
        range_max=range_max+[range_max_avg]
        range_min=range_min+[range_min_avg]
        # Setup offset
        xoff=np.zeros(x.size)
        if offset_bool: # no offset for all bars
            xoff=[0 for i in x] # No offset
            #xoff=[0.15 for i in x] # Always offset
        else: # With offset
            offset=0.15 
            # blue bars (mean) is positive, do the offset
            xoff=[0 if rm<0 else offset for rm in range_mean]
        bar_width = 0.45
        ax1.bar(x+xoff, range_mean, bar_width, bottom=0, color='royalblue', label='Internal variability (std:%s)'%round(range_mean_std,2))
        if plot_sensitivity:
            ax1.fill_between(x, range_min, range_max, alpha=0.3, fc='royalblue')

    if include_forced: # Add an additional bar for ML-predicted forced components. This has to be come before plot_mmm
        plot_mmm=False
        mm_mean= [obs_predicts_central[yr][1] for yr in years_keys] # index 0 is internla variability
        mm_max= [obs_predicts_central[yr][1]+combine_std[yr][1]*0.5 for yr in years_keys]
        mm_min= [obs_predicts_central[yr][1]-combine_std[yr][1]*0.5 for yr in years_keys]
        # Calculate the average of the 20-years slicing windows
        mm_mean_avg=np.mean(mm_mean)
        mm_min_avg=np.mean(mm_min)
        mm_max_avg=np.mean(mm_max)
        # Add them as the last element
        mm_mean=mm_mean+[mm_mean_avg]
        mm_min=mm_min+[range_min_avg]
        mm_max=mm_max+[range_max_avg]
        # Calculate the SD
        mm_mean_std = np.std(mm_mean[0:-1]) # Don't count the average element
        if offset_bool: 
            xoff=[0 for i in x]
        else: 
            xoff=[xo*-1 for xo in xoff] # opposite the side the xoff used in blue bars
        ax1.bar(x+xoff,mm_mean,bar_width,bottom=range_mean,color='tomato',label='Anthropogenic (std:%s)'%round(mm_mean_std,2))
        if get_green_line:
            ax1.plot(x+xoff, mm_mean+range_mean, color='green')
        if plot_sensitivity:
            #ax1.fill_between(x, range_mean+mm_mean-mm_min, range_mean+mm_mean+mm_max, alpha=0.3, fc='tomato') # New std
            #ax1.fill_between(x, np.array(range_mean)-mm_min, np.array(range_mean)+mm_max, alpha=0.3, fc='tomato') # New std
            ax1.fill_between(x, np.array(range_mean)+np.array(mm_min), np.array(range_mean)+np.array(mm_max), alpha=0.3, fc='tomato') # New std

    if plot_mmm: # The Multi-model ensemble mean of the anthropogenic sea ice
        lens_vars = model_vars
        lens_slope = np.array([sic_means[var] for var in lens_vars])
        # Do the mean across models' ensemlbe-average 
        mm_mean= np.mean(lens_slope,axis=0)
        mm_min = np.std(lens_slope,axis=0)*0.5 # Half of the standard deviation
        mm_max = np.std(lens_slope,axis=0)*0.5
        mm_mean_std = np.std(mm_mean[0:-1]) # Don't count the average element
        if offset_bool: 
            xoff=[0 for i in x]
            #xoff=[-0.15 for i in x] # always offset
        else: 
            xoff=[xo*-1 for xo in xoff] # opposite the side the xoff used in blue bars
        ax1.bar(x+xoff,mm_mean,bar_width,bottom=range_mean,color='tomato',label='Anthropogenic (std:%s)'%round(mm_mean_std,2))
        if get_green_line:
            ax1.plot(x+xoff, mm_mean+range_mean, color='green')
        if plot_sensitivity:
            #ax1.fill_between(x, range_mean+mm_min, range_mean+mm_max, alpha=0.3, fc='tomato')
            ax1.fill_between(x, range_mean+mm_mean-mm_min, range_mean+mm_mean+mm_max, alpha=0.3, fc='tomato') # New std

        if True: # print the relative contribution of circulation and anthropgoenic processes
            from scipy.signal import detrend
            #ratio=range_mean/(range_mean+mm_mean)*100; print('Relative importance of internal component:',ratio)
            ratio=mm_mean/(range_mean+mm_mean)*100; print('Relative importance of anthropogenic component:',ratio)
            #print(ratio[13:19].mean())
            corr=tools.correlation_nan(range_mean[0:-1],ice_ts_mean[0:-1]); print('Correlation between obs t.s. and circulation component',corr)
            corr=tools.correlation_nan(mm_mean[0:-1],ice_ts_mean[0:-1]); print('Correlation between obs t.s. and anthropogenic component',corr)
            corr=tools.correlation_nan(detrend(range_mean[0:-1]),detrend(ice_ts_mean[0:-1])); print('Detrend correlation between obs t.s. and circulation component',corr)
            corr=tools.correlation_nan(detrend(mm_mean[0:-1]),detrend(ice_ts_mean[0:-1])); print('Detrend correlation between obs t.s. and anthropogenic component',corr)

    if plot_nudging: # The CESM1 and CESM2 Nudging by Ding et al
        xoff=0.2; bar_width=0.5
        sic_nudge=sic_means[vars[3]]
        sic_anthro=sic_means[vars[4]]
        if offset_bool:
            pass
        else:
            offset=0.15
            xoff=[0 if rm<0 else offset for rm in sic_nudge]
        ax1.bar(x+xoff,sic_nudge,bar_width, bottom=0, color='royalblue', label='Internal\nvariability (Nudging by %s)'%model_name)
        ax1.bar(x-xoff,sic_anthro,bar_width,bottom=sic_nudge, color='tomato', label='Anthropogenic (%s only)'%model_name)
        if get_green_line:
            ax1.plot(x, np.array(sic_nudge)+np.array(sic_anthro), color='green')
        ax1.set_title('(%s) %s'%(ABC,model_name), loc='left', size=10)

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
        if fixed_legend_pos:
            ax1.legend(bbox_to_anchor=(0, 0.03), ncol=1, loc='lower left', frameon=False, columnspacing=1, 
                        handletextpad=0.4, labelspacing=0.3)
        else: #Auto-legend position
            ax1.legend(ncol=1, frameon=False, columnspacing=1, 
                        handletextpad=0.4, labelspacing=0.3)
    if set_ylim:
        #ax1.set_ylim(-0.36, 0.05) # Default for million km^2 /decade
        #ax1.set_ylim(-20,10) # Default for %/decade
        #ax1.set_ylim(-12,3) # Default for %/decade
        ax1.set_ylim(-14,3.5) # For revision only
    else:
        #ax1.set_ylim(-16,5) # For CESM2 individual ensemble
        #ax1.set_ylim(-15,15) # For CESM2 individual ensemble
        pass
    ax1.set_ylabel('%/decade', rotation=90, labelpad=-3)
    #ax1.set_yticks([-0.3,-0.2,-0.1,0,0.1])
    # Remove the box
    for i in ['right', 'top']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=2)
        ax1.tick_params(axis='y', which='both',length=2)
    ax1.tick_params(axis='x', direction="in", length=3, colors='black')
    ax1.tick_params(axis='y', direction="in", length=3, colors='black')
    fig_name = 'fig2_%s'%fig_name_add
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0.5)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=500, pad_inches=0.01)


