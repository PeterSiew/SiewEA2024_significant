import numpy as np


if False: 
    hist_yr_sensitivity=True
    hist_st_years=[1980,1975,1970,1965,1960,1955,1950,1945,1940,1935,1930,1925,1920,1915]
    hist_end_years=[2025,2030,2035,2040,2045,2050,2055,2060,2065,2075,2080,2085,2090,2095]
    rmse_all_save={}
    for hist_st_yr_test, hist_end_yr_test in zip(hist_st_years,hist_end_years):
        print('')
        exec(open("./fig2A_ridge_training.py").read())
        #exec(open("/home/pyfsiew/codes/trend_study/revised_figures/fig2A_ridge_training.py").read())
        rmse_all_save[str(hist_st_yr_test)+'-'+str(hist_end_yr_test)]=rmse_all

if True:
    rmse_all_save={}
    ridge_alphas=np.arange(0,20000,1000)
    ridge_alphas=np.arange(1000,4000,1000)
    ridge_alphas=[10]+[50]+[100]+[200]+[300]+[400]+[500]+[i for i in range(1000,10000,500)]
    for ridge_alpha in ridge_alphas:
        print(ridge_alpha)
        print('')
        exec(open("./fig2A_ridge_training.py").read())
        rmse_all_save[ridge_alpha]=rmse_all

    
plt.close()
fig, ax1 = plt.subplots(1,1,figsize=(5,1))
years_range=[*rmse_all_save]
x=range(len(years_range))
ax1.plot(x,[*rmse_all_save.values()])
ax1.set_xticks(x)
if False:
    xticklabels=[i.replace("-","\nto\n") for i in years_range]
if True:
    xticklabels=[i for i in years_range]
ax1.set_xticklabels(xticklabels,size=7,rotation=45)
ax1.set_ylabel("RMSE")
ax1.set_xlabel("Alpha used in Ridge regression")
ax1.set_xlim(x[0],x[-1])
fig_name = 'RMSE_over_training_periods'
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4) # hspace is the vertical
plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)
