
import glob, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import glob
import ast
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
dat_size = 16777216
time_res = 64e-6 # in seconds
T_obs = (dat_size*time_res)/60 # in minutes is equal to 17.895 minutes
freq_axis = np.fft.rfftfreq(dat_size, d=64e-6)
freq_res = 1/(T_obs*60)
fft_size = len(freq_axis)

root_dir = '/hercules/scratch/atya/BinaryML/'
run = 'runBC'
#names = glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/pdot_vol/*.npy')
names = []
labels_df = pd.read_csv(root_dir + f'meta_data/labels_{run}.csv')
filenames = labels_df[labels_df['z_max_rel_from_pvol'].isna()]['file_name'].values
for filename in filenames:
    names.append(f'/hercules/results/atya/BinaryML/sims/{run}/pdot_vol/'+filename.replace('.dat','pvol.npy'))


zarray = np.arange(-100.0, 100.0, 4.0/256)

def process(file):
    try:
        vol = np.load(file)
        plt.imshow(vol)
        file_name = re.findall(rf'obs\d+{run[3:]}pvol',file)[0]
        labels_df_file = labels_df[labels_df['file_name'] == file_name[:-4]+'.dat']
        index = labels_df_file.index[0]
        p_middle = labels_df_file.loc[index]['p_middle']
        freq_index = (1/p_middle)/freq_res
        max_index = np.argmax(vol)
        z_max, r_max = np.unravel_index(max_index, vol.shape)
        z_max_from_pvol = zarray[-z_max]
        freq_max_from_pvol_ind_neg = freq_index + (r_max - 1280)
        p_max_from_pvol_neg = 1/(freq_max_from_pvol_ind_neg*freq_res)
        plt.savefig(root_dir+f'raw_data/{run}/pdot_vol_imgs/' + file_name + '.png')
        return {'index':index, 'p_max_from_pvol':p_max_from_pvol_neg, 'z_max_from_pvol':z_max_from_pvol, 'z_max':z_max, 'r_max':r_max}
    except FileNotFoundError:
        print(file)
val = Parallel(n_jobs=-1)(delayed(process)(file=file_to_process) for file_to_process in tqdm(names))

for dic in val:
    if dic is not None:
        labels_df.loc[dic['index'],'p_max_from_pvol'] = dic['p_max_from_pvol']
        labels_df.loc[dic['index'],'z_max_from_pvol'] = dic['z_max_from_pvol']
        labels_df.loc[dic['index'],'z_max_rel_from_pvol'] = dic['z_max']
        labels_df.loc[dic['index'],'r_max_rel_from_pvol'] = dic['r_max']
    else:
        print('Dic is None, cant perform the labels_df assignment')
        
labels_df.to_csv(root_dir + f'meta_data/labels_{run}.csv',index=False)