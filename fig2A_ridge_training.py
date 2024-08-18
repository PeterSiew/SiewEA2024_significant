import xarray as xr
import numpy as np; np.seterr(divide='ignore', invalid='ignore')
from datetime import date
import ipdb
import matplotlib.pyplot as plt; import matplotlib; matplotlib.rcParams['font.sans-serif'] = "URW Gothic"
import multiprocessing
import scipy; from scipy.odr import Model, RealData, ODR
from sklearn.linear_model import Ridge
import random
from timeit import default_timer as timer
from importlib import reload
import torch
device=torch.device("cpu")
torch.set_num_threads(1) # https://github.com/pytorch/pytorch/issues/17199
#export CUDA_VISIBLE_DEVICES="" # Don't use GPU (0 for GPU only)

import os; import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import os; import sys; sys.path.insert(0, '/home/pyfsiew/codes/trend_study')
import tools
import fig2A_ridge_training as annr
import pytorch_ann_model as annm


if __name__ == "__main__":

    reload(annr); reload(annm)

    models = ['ACCESS-ESM1-5','CanESM5','CanESM5-1', 'cesm2',  # Only SSP370 models
              'IPSL-CM6A-LR', 'miroc6',
             'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    models = ['ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'cesm2', 
             'EC-Earth3CC', 'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 
             'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    psl_vars=[m+'_psl_en' for m in models]
    sic_vars=[m+'_sic_en' for m in models]
    tas_vars=[m+'_tas_en' for m in models]
    vars=psl_vars+tas_vars+sic_vars 

    # Set the preidctors
    psl_or_tas='both'
    psl_or_tas='tas'
    psl_or_tas='psl'

    # Set ensembles
    ens={m:range(1,3) for m in models} # for testing
    ens={m:range(1,11) for m in models}
    ensembles = {m+'_%s_en'%var: ens[m] for m in models for var in ['psl','sic','tas']}

    # For the training and testing data
    hist_st_yr=1950; hist_end_yr=2055 # Best result for MERRA2 only
    hist_st_yr=1900; hist_end_yr=1980 # for testing
    hist_st_yr=2010; hist_end_yr=2090 # for testing
    hist_st_yr=1960; hist_end_yr=2040 # Old papers
    if 'hist_yr_sensitivity' in locals(): # This part won't be run if we run this script directly
        hist_st_yr=hist_st_yr_test; hist_end_yr=hist_end_yr_test
    print('Model years for training',hist_st_yr,hist_end_yr)

    # Periods and intervals
    slicing_period=20; interval=1 #Default (produce better result)
    slicing_period=20; interval=5 #Default (produce better result)

    # Create the model range
    model_years=range(hist_st_yr, hist_end_yr+1-slicing_period, interval)
    ### Create the observed range
    obs_st_yr=1980; obs_end_yr=2022
    obs_st_years = list(range(obs_st_yr, obs_end_yr+1-slicing_period,1))
    obs_end_years = [i+slicing_period for i in obs_st_years] 

    # Set the grids
    aaa=1; bbb=2 # SON only
    aaa=0; bbb=3 # DJF, SON and JJA
    aaa=1; bbb=4 # SON,JJA and MAM
    aaa=2; bbb=4 # JJA and MAM
    aaa=0; bbb=1 # DJF only
    aaa=1; bbb=2 # SON only
    aaa=0; bbb=2 # DJF and SON only
    aaa=3; bbb=4 # MAM 
    aaa=2; bbb=3 # JJA
    aaa=0; bbb=4 # All season (Default)

    # Include the pan-Arctic region
    lat1=54; lat2=88; lon1=-180; lon2=180; grid_no=1416 # 5664 grid for all seasons
    # Default (only the Euro-Atlantic SLP)
    lat1=54; lat2=88; lon1=-60; lon2=102.5; grid_no=648 # 2592 grids for all seasons

    if 'obs_vars' not in locals():
        # Set the observations 
        obs_vars=['obs1','obs2','obs3','obs4']; obs_en='' # MERRA2, ERA5 and JRA55
        obs_vars=['obs4']; obs_en='' # NCEP R1
        obs_vars=['obs3']; obs_en='' # Only JRA55
        obs_vars=['obs2']; obs_en='' # Only ERA5
        obs_vars=['obs1']; obs_en='' # MERRA2 
        print('Using Reanalysis %s for prediction'%obs_vars)

    # Others
    ANN=True; debug=False # ANN and use multi-processing
    ANN=False; debug=True # Ridge regresion and use single-core (could track pdb)
    include_forced=True # Include the forced component in the Y predictand
    include_forced=False
    tt='training' # Get the name
    #ridge_alpha=1500 #1500-5500 give similar results (For both SLP and TAS)
    ridge_alpha=200 # For SLP-only

    if 'model_first_en' in locals(): # This part won't be run if we run this script directly
        #obs_vars=['cesm2']; obs_en='1' # This will be supplied by figS1_first_member_models
        obs_vars=[obs_vars_fake]
        obs_en=obs_en_fake
        ensembles[obs_vars[0]+'_psl_en']=[i for i in ensembles[obs_vars[0]+'_psl_en']]
        ensembles[obs_vars[0]+'_psl_en'].remove(int(obs_en))
        ensembles[obs_vars[0]+'_tas_en']=[i for i in ensembles[obs_vars[0]+'_tas_en']]
        ensembles[obs_vars[0]+'_tas_en'].remove(int(obs_en))
        ensembles[obs_vars[0]+'_sic_en']=[i for i in ensembles[obs_vars[0]+'_sic_en']]
        ensembles[obs_vars[0]+'_sic_en'].remove(int(obs_en))

    cesm2_training_only=True
    cesm2_training_only=False
    if cesm2_training_only:
        models = ['cesm2', 'fakecesm2']
        psl_vars=[m+'_psl_en' for m in models]
        sic_vars=[m+'_sic_en' for m in models]
        tas_vars=[m+'_tas_en' for m in models]
        vars=psl_vars+tas_vars+sic_vars 
        ens={'cesm2':range(1,101),'fakecesm2':range(1,101)} 
        ens={'cesm2':range(1,51),'fakecesm2':range(1,51)} # Use only the first 50 memebrs from CESM2
        ensembles = {m+'_%s_en'%var: ens[m] for m in models for var in ['psl','sic','tas']}

    # Read in the predictor and preditands
    X1, X2, Y = [], [], []
    ens_record = []
    for i, var in enumerate(vars):
        for en in ensembles[var]:
            for st_yr in model_years:
                end_yr = st_yr+slicing_period
                if 'psl' in var: # Geth the PSL
                    ens_record.append((var,en,st_yr,end_yr))
                    data_raw=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/psl/%s%s_%s-%s.nc'%(slicing_period,var,en,st_yr,end_yr))[tt]
                    data=data_raw.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    data=data.values.reshape(-1)
                    data=data[aaa*grid_no:bbb*grid_no]
                    X1.append(data)
                if 'tas' in var: # Get the TAS
                    data_raw=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/tas/%s%s_%s-%s.nc'%(slicing_period,var,en,st_yr,end_yr))[tt]
                    data=data_raw.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
                    data=data.values.reshape(-1)
                    data=data[aaa*grid_no:bbb*grid_no]
                    X2.append(data)
                elif 'sic' in var: # Get the SIC
                    #data_sic = xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic_area/%s%s_%s-%s.nc' %(slicing_period,var,en,st_yr,end_yr))[tt]
                    #data_sic = xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic_old/%s%s_%s-%s.nc' %(slicing_period,var,en,st_yr,end_yr))[tt]
                    data_sic = xr.open_dataset('/dx13/pyfsiew/training/training_%syr/sic/%s%s_%s-%s.nc' %(slicing_period,var,en,st_yr,end_yr))[tt]
                    data_sic = data_sic.isel(season=0)
                    Y.append(data_sic.item())

    # Append X1 and X2 as two predictors
    Y=np.array(Y)
    X1=np.array(X1); X2=np.array(X2)
    if psl_or_tas=='both':
        X = np.column_stack((X1,X2))
    elif psl_or_tas=='psl' :
        X=X1
    elif psl_or_tas=='tas' :
        X=X2
    else:
        raise ValueError('Check here')

    ### Remove the ensemble means in the same period for each model. Remove only Y but not X
    Y_new=np.zeros((len(Y),2))
    groups = [i[0]+'-'+str(i[2])+'-'+str(i[3]) for i in ens_record] # has no information of the ensemble number
    for group in set(groups):
        idx = (np.array(groups)==group).nonzero()[0]
        Y_mean = Y[idx].mean()
        Y_new[idx,0] = Y[idx]-Y_mean # Internal variability
        Y_new[idx,1] = Y_mean # enesmeble mean (forced response)
    if include_forced:
        Y=Y_new[:,:] # Include both column
    else: 
        Y=Y_new[:,0] # Only include the internal component
        Y=Y[:,np.newaxis] # 

    ### Prepare the observations for prediction
    # Add 1980 to 2023 to the end
    #obs_st_years_new=obs_st_years+obs_st_years[0:1]; obs_end_years_new=obs_end_years+obs_end_years[-1:] Do the 1980-2023
    obs_st_years_new=obs_st_years; obs_end_years_new=obs_end_years
    X_obss={}
    obs_vars_combine =[(v1,v2) for v1 in obs_vars for v2 in obs_vars]
    for obs_var in obs_vars_combine:
        X1_obss=[]; X2_obss=[]
        for st_yr, end_yr in zip(obs_st_years_new, obs_end_years_new):
            obs_slicing_period = end_yr-st_yr # This might be 20 or 42
            # Read psl
            obs_var_psl=obs_var[0]
            X1_obs=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/psl/%s_psl_en%s_%s-%s.nc'%(obs_slicing_period,obs_var_psl,obs_en,st_yr,end_yr))[tt]
            X1_obs=X1_obs.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
            X1_obs=X1_obs.values.reshape(-1)
            X1_obs=X1_obs[aaa*grid_no:bbb*grid_no]
            X1_obss.append(X1_obs)
            # Read tas
            obs_var_tas=obs_var[1]
            X2_obs=xr.open_dataset('/dx13/pyfsiew/training/training_%syr/tas/%s_tas_en%s_%s-%s.nc'%(obs_slicing_period,obs_var_tas,obs_en,st_yr,end_yr))[tt]
            X2_obs=X2_obs.sel(latitude=slice(lat1,lat2)).sel(longitude=slice(lon1,lon2))
            X2_obs=X2_obs.values.reshape(-1)
            X2_obs=X2_obs[aaa*grid_no:bbb*grid_no]
            X2_obss.append(X2_obs)
        if psl_or_tas=='both':
            X_obss[obs_var]=np.column_stack((X1_obss,X2_obss))
        elif psl_or_tas=='psl':
            X_obss[obs_var]=np.array(X1_obss)
        elif psl_or_tas=='tas':
            X_obss[obs_var]=np.array(X2_obss)
        else:
            raise ValueError('Check here')

    if False: # Put all NAN values into 0 if there are (usually there is no NAN values)
        X[np.isnan(X)] = 0
        X[np.isinf(X)] = 0
        X_obss[np.isnan(X_obss)] = 0
        X_obss[np.isinf(X_obss)] = 0

    ### Start the ML-prediction
    if ANN:
        loops=range(0,10)
    else:
        loops=range(0,1) # for testing or for ridge regression
    argus=[]
    start = timer()
    Y_test_actuals = {var: None for var in psl_vars}
    Y_train_mean = {var: None for var in psl_vars}
    Y_train_std = {var: None for var in psl_vars}
    ens_record_test = {var: None for var in psl_vars}
    X_obss_std={}
    for var in psl_vars:
        train_idx = (~(np.array(ens_record)[:,0]==var)).nonzero()[0]# without that variable to form the training set
        test_idx = (np.array(ens_record)[:,0]==var).nonzero()[0]
        X_train=X[train_idx]
        Y_train=Y[train_idx]
        X_test=X[test_idx]
        Y_test_actuals[var]=Y[test_idx]
        ens_record_test[var]=np.array(ens_record)[test_idx]
        # Standardiz the X
        X_mean=X_train.mean(axis=0)
        X_std=X_train.std(axis=0)
        X_train=(X_train-X_mean)/X_std
        X_test=(X_test-X_mean)/X_std
        for obs_var in X_obss.keys():
            X_temp=X_obss[obs_var]
            X_obss_std[obs_var]=(X_temp-X_mean)/X_std
        # Standardize the Y
        Y_train_mean[var]=Y_train.mean(axis=0)
        Y_train_std[var]=Y_train.std(axis=0)
        Y_train = (Y_train-Y_train_mean[var])/Y_train_std[var]
        for loop in loops:
            argu=(var,loop,X_train,Y_train,X_test,X_obss_std,loop*1) # loop is for the seed (*100 is a good seed)
            argus.append(argu)
    ### Start to run the prediction
    if not debug:
        pool = multiprocessing.Pool(len(psl_vars))
        if ANN:
            results = pool.map(annr.predictions_func, argus)
        else:
            results = pool.map(annr.predictions_func_ridge, argus)
    else: # For debugging
        results=[]
        for argu in argus:
            if ANN:
                result=annr.predictions_func(argu) 
            else:
                result=annr.predictions_func_ridge(argu, ridge_alpha=ridge_alpha) 
            results.append(result)
    end=timer(); 

    ### Put the results into dictionary
    print('Running time:',(end-start)/60, 'mins')
    Y_test_predicts = {var: {l:None for l in loops} for var in psl_vars}
    obs_predicts = {obs_var: {var: {l:None for l in loops} for var in psl_vars} for obs_var in X_obss.keys()}
    for v, var in enumerate(psl_vars):
        for l, loop in enumerate(loops):
            no=v*len(loops)+l
            Y_test_predict, obs_predict = results[no]
            # Add the training_mean and std to the predict values
            Y_test_predict = Y_test_predict*Y_train_std[var]+Y_train_mean[var]
            Y_test_predicts[var][loop]= Y_test_predict
            for obs_var in obs_predict.keys():
                predict_temp=obs_predict[obs_var]*Y_train_std[var]+Y_train_mean[var]
                obs_predicts[obs_var][var][loop]=predict_temp

    ### Average the Y_test across loops
    Y_test_predicts_mean = {var: np.array([Y_test_predicts[var][loop] for loop in loops]).mean(axis=0) for var in psl_vars}
    # Add a year dictionary to the obs_predicts ==> obs_predicts_new
    obs_predicts_new = {obs_var: {var: {l:{} for l in loops} for var in psl_vars} for obs_var in X_obss.keys()}
    for obs_var in X_obss.keys():
        for var in psl_vars:
            for loop in loops:
                for i in range(len(obs_st_years_new)):
                    st_yr=obs_st_years_new[i]; end_yr=obs_end_years_new[i]
                    yr_range=str(st_yr)+'-'+str(end_yr)
                    obs_predicts_new[obs_var][var][loop][yr_range] = obs_predicts[obs_var][var][loop][i]

    # new approch for observations
    # Average across variables and loops - only create a dictionary for obs years range
    obs_year_range=obs_predicts_new[obs_var][var][loop].keys()
    obs_predicts_range = {}
    for yr_range in obs_year_range:
        obs_predicts_range[yr_range]=[]
        for obs_var in X_obss.keys():
            # Average across loop and variables
            predict_temp=np.array([obs_predicts_new[obs_var][var][loop][yr_range] for var in psl_vars for loop in loops]).mean(axis=0)
            #predict_temp=np.array([np.array([obs_predicts_new[obs_var][var][loop][yr_range] for loop in loops]).mean(axis=0) for var in psl_vars]).mean()
            obs_predicts_range[yr_range].append(predict_temp)
        # The items are the 9 reanlaysis dataset
        obs_predicts_range[yr_range] = np.array(obs_predicts_range[yr_range])

    # Obtain the central esimate for ML-prediction, and the combined uncertainty range
    if not include_forced:
        actuals=np.array([i.item() for var in psl_vars for i in Y_test_actuals[var]]) # This makes sure that all model can have different ensemble 
        predicts=np.array([i.item() for var in psl_vars for i in Y_test_predicts_mean[var]])
        rmse_all = np.sqrt(np.mean((actuals-predicts)**2))
    else: # if included_forced=True
        actuals=np.array([Y_test_actuals[var] for var in psl_vars]).reshape(-1,2)
        predicts=np.array([Y_test_predicts_mean[var] for var in psl_vars]).reshape(-1,2)
        rmse_all = (((actuals-predicts)**2).mean(axis=0))**0.5
    corr_all=tools.correlation_nan(predicts,actuals)
    print('RMSE:',rmse_all, ',Corr:',corr_all)
    diffs=actuals-predicts
    diff_std=diffs.std(axis=0)
    print('std of the absolute diffs:',diff_std)
    # Standard deviation of observational dataset
    obs_std={}
    combine_std={}
    obs_predicts_central={}
    for yr_range in obs_year_range:
        obs_predicts_central[yr_range]=np.mean(obs_predicts_range[yr_range],axis=0).item()
        obs_std[yr_range]=np.std(obs_predicts_range[yr_range],axis=0) # The standard deviation is 0 if there is only one item
        # Combine uncertainty
        combine_std[yr_range]=((obs_std[yr_range]**2+diff_std**2)**0.5).item()
        # when inclue_forced is True
        #obs_predicts_central[yr_range]=np.mean(obs_predicts_range[yr_range],axis=0)
        #combine_std[yr_range]=((obs_std[yr_range]**2+diff_std**2)**0.5)

    if True: # Do the plotting for validation
        annr.plotting(Y_test_actuals,Y_test_predicts_mean,psl_vars,ens_record_test,model_years,include_forced)

    #ipdb.set_trace()
    if False: ### Obtain bias correction according in different periods
        def linear_func(p, x):
            m, c = p
            return m*x + c
        linear_model = Model(linear_func)
        slopes,intercepts={},{}
        for model_st_yr in model_years:
            model_end_yr=model_st_yr+slicing_period
            yr_range=str(model_st_yr)+'-'+str(model_end_yr)
            year_mask = (ens_record_test[psl_vars[0]][:,2]==str(model_st_yr)) & (ens_record_test[psl_vars[0]][:,3]==str(model_end_yr))
            actual=np.array([Y_test_actuals[var][year_mask].squeeze() for var in psl_vars]).flatten()
            predict=np.array([Y_test_predicts_mean[var][year_mask].squeeze() for var in psl_vars]).flatten()
            data = RealData(predict,actual)
            odr=ODR(data, linear_model, beta0=[1,1]); out=odr.run()
            slope_odr, intercept_odr = out.beta[0], out.beta[1]
            slopes[yr_range]=slope_odr
            intercepts[yr_range]=intercept_odr
        # Do the plotting here
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(5,1))
        years_range=[*slopes]
        x=range(len(years_range))
        ax1.plot(x,[*slopes.values()])
        ax1.set_xticks(x)
        xticklabels=[i for i in years_range]
        ax1.set_xticklabels(xticklabels,size=7,rotation=45)
        ax1.set_ylabel("Regression\nslope")
        #ax1.set_xlabel("20-year period for training")
        ax1.set_xlim(x[0],x[-1])
        ax1.axhline(y=1, color='lightgray', linestyle='--', linewidth=1)
        fig_name = 'slope_change_training_periods'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)

    ### For the permutation seasonal test
    remove_season_permutation=True
    remove_season_permutation=False
    if remove_season_permutation:
        print('')
        print('You are loading the seasonal permutation test')
        results_permutation=[]
        for argu in argus:
            result=annr.predictions_func_ridge_permutation(argu, ridge_alpha=ridge_alpha) 
            results_permutation.append(result)
        rmses= {var: {} for var in psl_vars}
        for v, var in enumerate(psl_vars):
            Y_test_predict = results_permutation[v]
            seasons=Y_test_predict.keys()
            for season in seasons:
                Y_test_predict_season = Y_test_predict[season]*Y_train_std[var]+Y_train_mean[var]
                # Compare this with Y_test_actual
                Y_test_actual = Y_test_actuals[var]
                rmse=np.sqrt(np.mean((Y_test_predict_season-Y_test_actual)**2))
                rmses[var][season]=rmse
        annr.seasonal_permutation_test_plotting(psl_vars,rmses,seasons)
        print('The permutation test finishes')


def plotting(Y_test_actuals, Y_test_predicts_mean, psl_vars, ens_record_test,model_years,include_forced):

    def linear_func(p, x):
        m, c = p
        return m*x + c

    var=psl_vars[0]
    model_name_dict = {'ACCESS-ESM1-5':'ACCESS-ESM1-5',\
         'ACCESS-CM2':'ACCESS-CM2', 'CanESM5':'CanESM5', 'CanESM5-1':'CanESM5-1',
         'cesm2':'CESM2', 'EC-Earth3CC':'EC-Earth3-CC',
         'IPSL-CM6A-LR':'IPSL-CM6A-LR','miroc6':'MIROC6',
         'MIROC-ES2L':'MIROC-ES2L', 'MPI-ESM1-2-LR':'MPI-ESM1-2-LR',
         'MPI-ESM1-2-HR':'MPI-ESM1-2-HR','UKESM1-0-LL':'UKESM1-0-LL', 'fakecesm2':'fakecesm2'}

    # For the scatter plots, do the median 
    models = [var[0:-7] for var in psl_vars]
    models_name=[model_name_dict[m] for m in models]

    year_mask={} # Different var has different ensemlbe numbers
    for var in psl_vars:
        if False: # testing year is 1980-2000
            st_yr=1855; end_yr=st_yr+20
            st_yr=2070; end_yr=st_yr+20
            st_yr=1955; end_yr=st_yr+20
            st_yr=2035; end_yr=st_yr+20
            st_yr=1950; end_yr=st_yr+20
            st_yr=1995; end_yr=st_yr+20
            st_yr=2000; end_yr=st_yr+20
            year_mask[var] = (ens_record_test[var][:,2]==str(st_yr)) & (ens_record_test[var][:,3]==str(end_yr))
        else: # Including all years
            year_mask[var]=np.array([True]*Y_test_actuals[var].shape[0])

    ### Start plotting
    plt.close()
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(3,10))
    ss = 1; fs=8
    colors=['royalblue','r','g','orange','gray','lime','cyan','gold','pink','violet','brown','darkviolet','peru','orchid','crimson']
    actuals, predicts = [], []
    for i, var in enumerate(psl_vars):
        if include_forced:
            actual=Y_test_actuals[var][year_mask[var]][:,0].squeeze()
            predict=Y_test_predicts_mean[var][year_mask[var]][:,0].squeeze()
        else:
            actual=Y_test_actuals[var][year_mask[var]].squeeze()
            predict=Y_test_predicts_mean[var][year_mask[var]].squeeze()
        # Get the overall actual and predict
        actuals.append(actual)
        predicts.append(predict)
        # Start plotting for indivudal models
        ax1.scatter(predict, actual, s=ss, color=colors[i])
        corr_test = tools.correlation_nan(actual,predict)
        rmse_test = np.sqrt(np.mean((actual-predict)**2))
        # Plot the regression line using ODR
        #slope,intercept,_,_,_=scipy.stats.linregress(predict,actual)
        linear_model = Model(linear_func)
        data = RealData(predict,actual)
        odr = ODR(data, linear_model, beta0=[1,1]); out = odr.run()
        slope, intercept = out.beta[0], out.beta[1]
        ax1.plot(predict,predict*slope+intercept,color=colors[i],linewidth=0.5)
        ax1.annotate(r"%s ($\rho$=%s, $\beta$=%s, RMSE=%s)"%(models_name[i],str(round(corr_test,2)),str(round(slope,2)),str(round(rmse_test,2)))
                ,xy=(1.1,-0.2+0.11*i), xycoords='axes fraction', fontsize=fs, bbox=dict(facecolor=colors[i], edgecolor='white'))
    if True: # Plot the regression line for all (Here actually repeats some of the calculation
        actuals=np.array([j for i in actuals for j in i])
        predicts=np.array([j for i in predicts for j in i])
        corr_test = tools.correlation_nan(actuals,predicts)
        rmse_test = np.sqrt(np.mean((actuals-predicts)**2))
        data = RealData(predicts,actuals)
        odr = ODR(data, linear_model, beta0=[1,1]); out = odr.run()
        slope, intercept = out.beta[0], out.beta[1]
        ax1.plot(predicts,predicts*slope+intercept,color='k',linewidth=1.5)
        ax1.annotate(r"$\rho$=%s, $\beta$=%s, RMSE=%s"%(str(round(corr_test,2)),str(round(slope,2)),str(round(rmse_test,2))),xy=(0.01,0.95), 
                    xycoords='axes fraction', fontsize=fs+1)

    # ax2 and ax3: Plot the forced part
    if include_forced:
        # For ax2
        actuals, predicts = [], []
        for i, var in enumerate(psl_vars):
            actual=Y_test_actuals[var][year_mask[var]][:,1]
            predict=Y_test_predicts_mean[var][year_mask[var]][:,1]
            actuals.append(actual)
            predicts.append(predict)
            ax2.scatter(predict, actual, s=ss, color=colors[i])
            corr_test = tools.correlation_nan(actual,predict)
            rmse_test = np.sqrt(np.mean((actual-predict)**2))
            ax2.annotate(r"%s $(\rho$=%s, RMSE=%s)"%(models_name[i],str(round(corr_test,2)),str(round(rmse_test,3))),xy=(1.1,-0.2+0.11*i), 
                        xycoords='axes fraction', fontsize=fs, bbox=dict(facecolor=colors[i], edgecolor='white'))
            # Plot the individual regression line
            slope,intercept,_,_,_=scipy.stats.linregress(predict,actual)
            ax2.plot(predict,predict*slope+intercept,color=colors[i],linewidth=0.5)
        # Plot the regression line for all for ax2
        actuals=np.array([j for i in actuals for j in i])
        predicts=np.array([j for i in predicts for j in i])
        corr_test = tools.correlation_nan(actuals,predicts)
        rmse_test = np.sqrt(np.mean((actuals-predicts)**2))
        data = RealData(predicts,actuals)
        odr = ODR(data, linear_model, beta0=[1,1]); out = odr.run()
        slope, intercept = out.beta[0], out.beta[1]
        ax2.plot(predicts,predicts*slope+intercept,color='k',linewidth=1.5)
        ax2.annotate(r"$\rho$=%s, $\beta$=%s, RMSE=%s"%(str(round(corr_test,2)),str(round(slope,2)),str(round(rmse_test,2))),xy=(0.01,0.95), 
                    xycoords='axes fraction', fontsize=fs+1)
        # ax3: Get the sum
        actuals, predicts = [], []
        for i, var in enumerate(psl_vars):
            actual=Y_test_actuals[var][year_mask[var]][:,0] + Y_test_actuals[var][year_mask[var]][:,1]
            predict=Y_test_predicts_mean[var][year_mask[var]][:,0] + Y_test_predicts_mean[var][year_mask[var]][:,1]
            actuals.append(actual)
            predicts.append(predict)
            ax3.scatter(predict, actual, s=ss, color=colors[i])
            corr_test = tools.correlation_nan(actual,predict)
            rmse_test = np.sqrt(np.mean((actual-predict)**2))
            ax3.annotate(r"%s $(\rho$=%s, RMSE=%s)"%(models_name[i],str(round(corr_test,2)),str(round(rmse_test,3))),xy=(1.1,-0.2+0.11*i), 
                        xycoords='axes fraction', fontsize=fs, bbox=dict(facecolor=colors[i], edgecolor='white'))
            # Plot the individual regression line
            slope,intercept,_,_,_=scipy.stats.linregress(predict,actual)
            ax3.plot(predict,predict*slope+intercept,color=colors[i],linewidth=0.5)
        # Plot the regression line for all for ax2
        actuals=np.array([j for i in actuals for j in i])
        predicts=np.array([j for i in predicts for j in i])
        corr_test = tools.correlation_nan(actuals,predicts)
        rmse_test = np.sqrt(np.mean((actuals-predicts)**2))
        data = RealData(predicts,actuals)
        odr = ODR(data, linear_model, beta0=[1,1]); out = odr.run()
        slope, intercept = out.beta[0], out.beta[1]
        ax3.plot(predicts,predicts*slope+intercept,color='k',linewidth=1.5)
        ax3.annotate(r"$\rho$=%s, $\beta$=%s, RMSE=%s"%(str(round(corr_test,2)),str(round(slope,2)),str(round(rmse_test,2))),xy=(0.01,0.95), 
                    xycoords='axes fraction', fontsize=fs+1)

    ### Set the axis
    for ax in [ax1,ax2,ax3]:
        ax.axvline(x=0, color='lightgray', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='lightgray', linestyle='--', linewidth=1)
        ax.plot([-100,100], [-100,100], lw=1, c='gray', linestyle='--')
        ax.set_xlabel('Predicted sea ice trend \n(%/decade)')
        ax.set_ylabel('Actual sea ice trend \n(%/decade)')
    xylim = 20
    for ax in [ax1]:
        ax.set_xlim(-xylim,xylim)
        ax.set_ylim(-xylim,xylim)
    xylim = 20
    for ax in [ax2]:
        ax.set_xlim(-xylim,xylim)
        ax.set_ylim(-xylim,xylim)
    xylim = 20
    for ax in [ax3]:
        ax.set_xlim(-xylim,xylim)
        ax.set_ylim(-xylim,xylim)
        #ax.set_xlabel('Predicted observed SIC (%s)\n (miilion km^2/decade)'%period)
        #pass
    fig_name = 'actual_versus_predict_new'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=400)
    
    if False: ### Shoe a distribution figure for response letter
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(3,3))
        bins=20
        ax1.hist(actuals, bins=bins, edgecolor='lightgray', fc="lightgray", lw=1, linewidth=1)
        ipdb.set_trace()
        percentile=((actuals<-6.44).sum())/len(actuals)*100
        ax1.annotate(str(round(percentile,1))+'%', xy=(-15,20), xycoords='data', size=9)
        ax1.axvline(x=-6.44, color='black', linestyle='--', linewidth=1)
        ax1.set_xlim(-20,20)
        ax1.set_xlabel('Actual sea ice trend (%/decade)')
        ax1.set_ylabel('Number of samples')
        fig_name = 'ice_distribution'
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=400)

def predictions_func(args, batch_size=128, epoch_no=500, print_status=True, seed=0):
    var = args[0]
    loop = args[1]
    X_train, Y_train, X_test = args[2], args[3], args[4] # Actually Y_test is not useful here
    X_obss=args[5]
    seed=args[6]; torch.manual_seed(seed); np.random.seed(seed)
    if print_status:
        print(var,loop)
    # Traditional ANN (2 layer 10 nodes)
    model=annm.ANNmodel(X_train.shape[1])
    # Do the training
    model.train() 
    # For 1000 samples, a bath size of 32 is very standard
    epoch, loss = annm.train_model(model, X_train, Y_train, num_epochs=epoch_no, batch_size=batch_size) 
    model.eval() # Do it after training

    # Test for testing data
    X_test = torch.tensor(X_test).float()
    Y_test_predict = model(X_test).detach().numpy().squeeze()

    # Test for observations
    Y_obs_predicts={}
    for obs_var in X_obss.keys():
        X_temp=X_obss[obs_var]
        X_temp= torch.tensor(X_temp).float()
        Y_obs_predicts[obs_var]=model(X_temp).detach().numpy().squeeze()
    return Y_test_predict, Y_obs_predicts

def predictions_func_ridge(args, print_status=True, ridge_alpha=None):
    var = args[0]
    loop = args[1]
    X_train, Y_train, X_test = args[2], args[3], args[4] # Actually Y_test is not useful here
    X_obss = args[5]
    #seed=args[6]; torch.manual_seed(seed); np.random.seed(seed)
    if print_status:
        print(var,loop)

    clf = Ridge(alpha=ridge_alpha).fit(X_train,Y_train) # 3000 Looks okay
    Y_test_predict=clf.predict(X_test)
    #coefs = clf.coef_
    #Y_test_predict=np.dot(coefs,X_test.T) # Produce identical result as above

    Y_obs_predicts={}
    for obs_var in X_obss.keys():
        X_temp=X_obss[obs_var]
        Y_obs_predicts[obs_var]=clf.predict(X_temp)

    return Y_test_predict, Y_obs_predicts

def predictions_func_ridge_permutation(args, print_status=False, ridge_alpha=None, grid_no=648):
    import copy
    var = args[0]
    loop = args[1]
    X_train, Y_train, X_test = args[2], args[3], args[4] 
    if print_status:
        print(var,loop)

    clf = Ridge(alpha=ridge_alpha).fit(X_train,Y_train) 

    #none_index=None
    #djf_index=[i for i in range(grid_no*0,grid_no*1)] + [i for i in range(grid_no*4,grid_no*4+grid_no)]
    #son_index=[i for i in range(grid_no*1,grid_no*2)] + [i for i in range(grid_no*5,grid_no*5+grid_no)]
    #jja_index=[i for i in range(grid_no*2,grid_no*3)] + [i for i in range(grid_no*6,grid_no*6+grid_no)]
    #mam_index=[i for i in range(grid_no*3,grid_no*4)] + [i for i in range(grid_no*7,grid_no*7+grid_no)]
    #all_index=[i for i in range(grid_no*0,grid_no*4)] + [i for i in range(grid_no*4,grid_no*8)]
    none_index=None
    djf_index=[i for i in range(grid_no*0,grid_no*1)] 
    son_index=[i for i in range(grid_no*1,grid_no*2)] 
    jja_index=[i for i in range(grid_no*2,grid_no*3)] 
    mam_index=[i for i in range(grid_no*3,grid_no*4)] 
    all_index=[i for i in range(grid_no*0,grid_no*4)] 
    shuffle_indices=[none_index, djf_index, son_index, jja_index, mam_index, all_index]
    seasons=['None','DJF','SON','JJA','MAM','ALL']

    #ipdb.set_trace()

    Y_test_predict = {}
    for i, season in enumerate(seasons):
        X_test_copy=copy.deepcopy(X_test) # make sure it doesn't change
        if season=='None':
            pass
        else:
            index=shuffle_indices[i]
            temp=X_test_copy[:,index]
            np.random.shuffle(temp)
            X_test_copy[:,index]=temp
            #X_test_copy[:,index]=0
        Y_test_predict[season]=clf.predict(X_test_copy)

    # Don't need to work on observations
    return Y_test_predict

def seasonal_permutation_test_plotting(psl_vars,rmses,seasons):

    #ipdb.set_trace()
    ### Do the plotting
    plt.close()
    fig, ax1 = plt.subplots(1,1,figsize=(7,2))
    bar_width=0.03
    plotting_bars = rmses
    x=np.arange(len(seasons))
    xadj=np.linspace(-0.2,0.2,len(psl_vars))
    colors=['royalblue','r','g','orange','gray','lime','cyan','gold','pink','violet','brown','darkviolet','peru','orchid','crimson']
    for v, var in enumerate(psl_vars):
        plotting_vars=rmses[var]
        ax1.bar(x+xadj[v], [*plotting_vars.values()], bar_width, color=colors[v])
    # Plot the mean
    rmse_mean_season=np.array([np.mean([rmses[var][season] for var in psl_vars]) for season in seasons])
    new_x = x+xadj[-1]+np.diff(xadj)[0]
    ax1.bar(new_x, rmse_mean_season, bar_width, color='k')
    for i, season in enumerate(seasons):
        diff = rmse_mean_season[i]-rmse_mean_season[0]
        diff_text = '+'+str(round(diff,2)) if diff>0 else str(diff) # Add positive sign if it is positive
        ax1.annotate(diff_text, xy=(new_x[i],rmse_mean_season[i]+0.2), xycoords='data', size=9)
    # Setup
    ax1.set_ylabel('RMSE')
    ax1.set_xlabel('Information being removed')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seasons)
    fig_name = 'permutation_test'
    for i in ['right', 'top']:
        ax1.spines[i].set_visible(False)
        ax1.tick_params(axis='x', which='both',length=0); ax1.tick_params(axis='y', which='both',length=2)
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name),bbox_inches='tight', dpi=500, pad_inches=0.01)
