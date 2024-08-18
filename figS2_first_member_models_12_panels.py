import numpy as np


fake_obs = ['ACCESS-ESM1-5','ACCESS-CM2','CanESM5','CanESM5-1', 'cesm2', 
         'EC-Earth3CC', 'IPSL-CM6A-LR', 'miroc6', 'MIROC-ES2L', 
         'MPI-ESM1-2-LR', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']

model_first_en=True
#obs_en_fake='2'
obs_en_fake='1' # This can be passed to another script with %run -i in IPYTHON
titles=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
for i, model in enumerate(fake_obs):
    print('')
    print('Fake obs:',model)
    obs_vars_fake=model
    obsi_vars_fake=model
    ABCDE=titles[i]
    exec(open("./fig2A_ridge_training.py").read())
    exec(open("./fig2B_bar_charats.py").read())
    #print('hi')
    

import matplotlib.pyplot as plt
from PIL import Image
import datetime as dt
plt.close()
fig, axs = plt.subplots(4,3, figsize=(25,22))
axs=axs.flatten()
today_date=dt.date.today()
filenames=["/home/pyfsiew/graphs/%s_fig2_['%s_obsfake'].png"%(today_date,i) for i in fake_obs]
for i in range(12):
    with open(filenames[i],'rb') as f:
        image=Image.open(f)
        #image=plt.imread(f)
        axs[i].imshow(image, interpolation='none')
        #axs[i].imshow(image, interpolation='gaussian')
        axs[i].axis('off')
fig_name = 'fig2_%s'%'all_models'
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=-0.5)
plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=400, pad_inches=0.01)

