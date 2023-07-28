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
import argparse
parser = argparse.ArgumentParser()

def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo '%s'"%cmd)
    os.system(cmd)

def parse_indices(value):
    if value.lower() == 'all':
        return 'all'
    elif value.lower() == 'test':
        return 'test'
    else:
        try:
            return [int(idx) for idx in value.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                                             "Invalid value for indices")

run = 'runBD'
type = 'presto_ACCEL_1200' # 'pred' or 'true'
accelcand = 'presto_nearest_cand_num_accel_1200' # 'pred_pd' or 'pd'
labels_df = pd.read_csv(f'/hercules/scratch/atya/BinaryML/meta_data/labels_{run}.csv')
labels_df = labels_df[~labels_df[accelcand].isna()]

parser.add_argument('-j', '--job_id', type=int, dest='job_id', help='Job ID', default=1234)
parser.add_argument('-fold', '--fold', type=parse_indices, dest='fold', help='Indices of candidates to fold or all candidates in "test" or "all"', default = 'all')
parser.add_argument('-run_at_node', '--run_at_node', type=bool, dest='run_at_node', help='Run at node or not', default = False)

args = parser.parse_args()
run_at_node = args.run_at_node
job_id = args.job_id

# start = 0
# end = 400

if run_at_node:
    cur_dir = f'/tmp/Abhinav_DATA{job_id}/'
    myexecute(f'mkdir -p {cur_dir}sims/{run}/dat_inf_files')
    myexecute(f'mkdir -p {cur_dir}sims/{run}/fold_output_{type}')
else:
    cur_dir = '/hercules/results/atya/BinaryML/'

myexecute(f'mkdir -p {cur_dir}sims/{run}/fold_output_{type}')
files_to_fold = labels_df['file_name'].values
accelcands = labels_df[accelcand].values.astype(int)

myexecute('echo \"\n\n\n\n############################################################################## \n\n\n \
          Comment: Folding all the predicted candidates:\
           \n\n\n ############################################################################## \n\n\n \"')

def prepfold(cand,file_to_fold):
    #sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /u/pdeni/fold-tools-2020-11-18-4cca94447feb.simg '
    sing_prefix = 'singularity exec -H $HOME:/home1 -B /hercules:/hercules/  /hercules/scratch/atya/compare_pulsar_search_algorithms.simg '
    output = 'temp_dir/'+file_to_fold[:-4]
    file_loc = cur_dir+f'sims/{run}/dat_inf_files/'+file_to_fold
    myexecute(sing_prefix+f'prepfold -accelcand {cand} -accelfile {file_loc[:-4]+"_ACCEL_1200.cand"} -o {output} -noxwin {file_loc}')

# dat_files = glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.dat')
# dat_files.extend(glob.glob(f'/hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/*.inf'))
# #dat_files = ['/hercules/scratch/atya/BinaryML/sims/obs2C.dat','/hercules/scratch/atya/BinaryML/sims/obs2C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.inf','/hercules/scratch/atya/BinaryML/sims/obs9C.dat']
if run_at_node:
    for file in files_to_fold:
        myexecute(f'rsync -Pav /hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{file} {cur_dir}sims/{run}/dat_inf_files/')
    for file in files_to_fold:
        myexecute(f'rsync -Pav /hercules/results/atya/BinaryML/sims/{run}/dat_inf_files/{file[:-4]}.inf {cur_dir}sims/{run}/dat_inf_files/')


val = Parallel(n_jobs=-1)(delayed(prepfold)(cand = cand,file_to_fold=file_to_fold) for cand,file_to_fold in zip(accelcands,files_to_fold))

myexecute(f'mv temp_dir/* {cur_dir}sims/{run}/fold_output_{type}')
if run_at_node:
    myexecute('echo \"\n\n\n\n Job finished \n\n\n ')
    myexecute(f'rsync -Pav {cur_dir}sims/{run}/fold_output_{type}/* /hercules/results/atya/BinaryML/sims/{run}/fold_output_{type}/ ')

    if cur_dir != '/hercules/results/atya/BinaryML/': 
        myexecute(f'rm -rf {cur_dir}')