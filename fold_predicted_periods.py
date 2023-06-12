import glob
import re
import time
import os
from joblib import Parallel, delayed
import pickle
import numpy as np
import pandas as pd
import glob
import sys

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

run = 'runBB'
job_id = sys.argv[1]
#cur_dir = '/hercules/scratch/atya/BinaryML/'
cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
type = 'pred' # 'pred' or 'true'
label_pd = 'pd_pred' # 'pred_pd' or 'pd'
label_period = 'p_fold_pred' # 'pred_fold_period' or 'fold_period'
start = 0
end = 400
labels_df = pd.read_csv(f'/hercules/scratch/atya/BinaryML/meta_data/labels_{run}.csv')

#cur_dir = '/hercules/scratch/atya/BinaryML/'

labels_p = labels_df[labels_df['status']=='test'][label_period].values[start:end]
labels_pd = labels_df[labels_df['status']=='test'][label_pd].values[start:end]
files_to_fold = labels_df[labels_df['status']=='test']['file_name'].values[start:end]

# labels_p = labels_df[label_period].values[start:end]
# labels_pd = labels_df[label_pd].values[start:end]
# files_to_fold = labels_df['file_name'].values[start:end]

myexecute(f'mkdir -p {cur_dir}sims/{run}/dat_inf_files')
myexecute(f'mkdir -p {cur_dir}sims/{run}/fold_output_{type}')

myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Folding all the predicted candidates:\
           \n\n\n ############################################################################## \n\n\n \"')

def prepfold(p,pd,file_to_fold):
    #sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /u/pdeni/fold-tools-2020-11-18-4cca94447feb.simg '
    sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /hercules/scratch/atya/compare_pulsar_search_algorithms.simg '
    output = cur_dir+f'sims/{run}/fold_output_{type}/'+file_to_fold[:-4]
    file_loc = cur_dir+f'sims/{run}/dat_inf_files/'+file_to_fold
    if (p*1000) < 100:
        myexecute(sing_prefix+f'prepfold -topo -p {p} -pd {pd} -o {output} -noxwin {file_loc}')
    else:
        myexecute(sing_prefix+f'prepfold -topo -slow -p {p} -pd {pd} -o {output} -noxwin {file_loc}')


# dat_files = glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.dat')
# dat_files.extend(glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.inf'))
# #dat_files = ['/hercules/scratch/atya/BinaryML/sims/obs2C.dat','/hercules/scratch/atya/BinaryML/sims/obs2C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.dat']
for file in files_to_fold:
    myexecute(f'rsync -Pav /hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{file} {cur_dir}sims/{run}/dat_inf_files/')
for file in files_to_fold:
    myexecute(f'rsync -Pav /hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{file[:-4]}.inf {cur_dir}sims/{run}/dat_inf_files/')


val = Parallel(n_jobs=-1)(delayed(prepfold)(p = label_p,pd =label_pd,file_to_fold=file_to_fold) for label_p,label_pd,file_to_fold in zip(labels_p,labels_pd,files_to_fold))
myexecute('echo \"\n\n\n\n Job finished \n\n\n ')
myexecute(f'rsync -Pav {cur_dir}sims/{run}/fold_output_{type}/* /hercules/results/atya/BinaryML/sims/{run}/fold_output_{type}/ ')

if cur_dir != '/hercules/scratch/atya/BinaryML/': 
    myexecute(f'rm -rf {cur_dir}')