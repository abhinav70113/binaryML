import os
import json
import argparse
import csv
from pathlib import Path
from subprocess import check_output
import subprocess
import time
import random
import itertools
import numpy as np
def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo 'Running: %s'"%cmd)
    os.system(cmd)
'''launch hyperparameter tuning tasks one by one, if the job submission limit is reached, wait for 5 seconds and try again'''

start_trial_index = 0
end_trial_index = 1
type = 'deepLayers'
cur_dir = '/hercules/scratch/atya/IsolatedML/'
log_dir = f'hyperparameter_tuning/{type}/'
cur_log_dir = os.path.join(cur_dir,log_dir)
myexecute(f'mkdir -p {cur_log_dir}')

job_name = "tuning"
output = f'{os.path.join(cur_log_dir,"output.%j")}'
error = f'{os.path.join(cur_log_dir,"error.%j")}'
partition = "short.q"
nodes = 1
cpus_per_task = 1
time_max = "00:10:00"
mem = "100GB"
gres = "gpu:1"
wrap_prefix = " module load anaconda/3/2021.11 && source activate /u/atya/conda-envs/tf-gpu4 && "
sbatch_prefix = f"sbatch --job-name={job_name} --output={output} --error={error} -p {partition} --export=ALL --nodes {nodes} --cpus-per-task={cpus_per_task} --time={time_max} --mem={mem} --gres={gres} "
wrap = wrap_prefix + f"python3 hyperparameter_runing_clean.py %j cnn"
#result = subprocess.check_output([sbatch_prefix, "--wrap", wrap], shell=True).decode("utf-8")
result = myexecute('sbatch --job-name=tuning --output=/hercules/scratch/atya/IsolatedML/hyperparameter_tuning/deepLayers/output.%j --error=/hercules/scratch/atya/IsolatedML/hyperparameter_tuning/deepLayers/error.%j -p short.q --export=ALL --nodes 1 --cpus-per-task=1 --time=00:10:00 --mem=100GB --gres=gpu:1 --wrap="module load anaconda/3/2021.11 && source activate /u/atya/conda-envs/tf-gpu4 && python3 hyperparameter_runing_clean.py \$SLURM_JOB_ID cnn"')
print(result)