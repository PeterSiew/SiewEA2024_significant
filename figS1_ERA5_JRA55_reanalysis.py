import numpy as np

obs_vars_loop=(['obs1'],['obs2'],['obs3']) # MERRA2; ERA5; JRA55
names=['MERRA2', 'ERA5', 'JRA55']
titles=['A', 'B', 'C']
obs_en=''
for i, obs_vars in enumerate(obs_vars_loop):
    print('')
    print(obs_vars)
    reanalysis_name = names[i]
    ABCDE=titles[i]
    exec(open("./fig2A_ridge_training.py").read())
    exec(open("./fig2B_bar_charats.py").read())
    
import matplotlib.pyplot as plt
from PIL import Image
import datetime as dt
plt.close()
fig, axs = plt.subplots(3,1, figsize=(13,25))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/home/pyfsiew/graphs/%s_fig2_%s.png"%(today_date,i) for i in obs_vars_loop]
titles=['A', 'B', 'C', 'D']
for i in range(len(obs_vars_loop)):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        axs[i].imshow(image, interpolation='none')
        axs[i].axis('off')
fig_name = 'fig2_%s'%'all_reanalsyis'
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.4)
plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)

