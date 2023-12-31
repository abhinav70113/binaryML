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

run = 'runBD'
job_id = sys.argv[1]
run_at_node = False
if run_at_node:
    cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
    myexecute(f'mkdir -p {cur_dir}sims/{run}/dat_inf_files')
    myexecute(f'mkdir -p {cur_dir}sims/{run}/fold_output_{type}')
else:
    cur_dir = '/hercules/results/atya/BinaryML/'
type = 'theory' # 'pred' or 'true'
label_pb = 'bper' # 'pred_pd' or 'pd'
label_period = 'period' # 'pred_fold_period' or 'fold_period'
label_x = 'asini'
label_To = 'periastron_time_mjd'
start = 0
end = 10000
labels_df = pd.read_csv(f'/hercules/scratch/atya/BinaryML/meta_data/labels_{run}.csv')

#cur_dir = '/hercules/scratch/atya/BinaryML/'
# labels_p = labels_df[labels_df['status']=='test'][label_period].values[start:end]
# labels_pd = labels_df[labels_df['status']=='test'][label_pd].values[start:end]
#files_to_fold = labels_df[labels_df['status']=='test']['file_name'].values[start:end]

labels_p = labels_df[label_period].values[start:end]
labels_pb = labels_df[label_pb].values[start:end]*3600 # in seconds
labels_x = labels_df[label_x].values[start:end]
labels_To = labels_df[label_To].values[start:end]
files_to_fold = labels_df['file_name'].values[start:end]



myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Folding all the predicted candidates:\
           \n\n\n ############################################################################## \n\n\n \"')

def prepfold(p,pb,x,To,file_to_fold):
    #sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /u/pdeni/fold-tools-2020-11-18-4cca94447feb.simg '
    sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /hercules/scratch/atya/compare_pulsar_search_algorithms.simg '
    output = cur_dir+f'sims/{run}/fold_output_{type}/'+file_to_fold[:-4]
    file_loc = cur_dir+f'sims/{run}/dat_inf_files/'+file_to_fold
    myexecute(sing_prefix+f'prepfold -topo -noxwin -p {p} -bin -pb {pb} -x {x} -To {To} -o {output} {file_loc}')


# dat_files = glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.dat')
# dat_files.extend(glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.inf'))
# #dat_files = ['/hercules/scratch/atya/BinaryML/sims/obs2C.dat','/hercules/scratch/atya/BinaryML/sims/obs2C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.dat']
if run_at_node:
    for file in files_to_fold:
        myexecute(f'rsync -Pav /hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{file} {cur_dir}sims/{run}/dat_inf_files/')
    for file in files_to_fold:
        myexecute(f'rsync -Pav /hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{file[:-4]}.inf {cur_dir}sims/{run}/dat_inf_files/')


val = Parallel(n_jobs=-1)(delayed(prepfold)(p = label_p,
                                            pb =label_pb,
                                            x = label_x,
                                            To = label_To,
                                            file_to_fold=file_to_fold) for label_p,
                                                                            label_pb,
                                                                            label_x,
                                                                            label_To,
                                                                            file_to_fold in zip(
                                                                                                labels_p,
                                                                                                labels_pb,
                                                                                                labels_x,
                                                                                                labels_To,
                                                                                                files_to_fold))
myexecute('echo "\n\n\n\n Job finished \n\n\n" ')

if run_at_node:
    myexecute(f'rsync -Pav {cur_dir}sims/{run}/fold_output_{type}/* /hercules/results/atya/BinaryML/sims/{run}/fold_output_{type}/ ')
    if cur_dir != '/hercules/results/atya/BinaryML/' : 
        myexecute(f'rm -rf {cur_dir}')