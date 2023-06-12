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
rd = 3
def myexecute(cmd):
    if 'echo' not in cmd[:20]:
        os.system("echo 'Running: %s'"%cmd)
    os.system(cmd)
'''launch hyperparameter tuning tasks one by one, if the job submission limit is reached, wait for 5 seconds and try again'''

start_trial_index = 0
end_trial_index = 1
type = 'deepLayers'
cur_dir = '/hercules/scratch/atya/BinaryML/'
log_dir = f'hyperparameter_tuning/{type}/'
cur_log_dir = os.path.join(cur_dir,log_dir)
myexecute(f'mkdir -p {cur_log_dir}')
list_of_dicts_dir = os.path.join(cur_log_dir,'list_of_dicts.json')

#check if the list of dictionaries already exists
if os.path.exists(list_of_dicts_dir):
    list_of_dicts = json.load(open(list_of_dicts_dir))
    myexecute(f'echo "Loaded list of dictionaries from {list_of_dicts_dir}"')

else:
    myexecute(f'echo "List of dictionaries not found, creating all search combinations"')
    default_hyperparameters = {
        'num_deep_layers': 8,
        'final_activation': 'def_relu',
        'initial_learning_rate': 0.001,
        'decay_rate': 0.5,
        'dropout':0.0,
        'batch_size': 16,
        'power' : 1,
        'factor_mse':1,
        'trial':0,
        'deep_layer_size':[64,192,160,176,256,32,144,128],
        'epochs':2,
        'patience':200,
        'input_shape':(400,2),
        'loss_function':'mse',
        'min_label' : 33.0,
        'max_label' : 1000.0
    }

    search_space = {
        'num_deep_layers': range(2, 24),
        'final_activation': ['sigmoid','tanh','def_relu','relu'],
        'initial_learning_rate': [round(i, 5) for i in np.logspace(-4, -2, 50)],
        'decay_rate': [round(i, 2) for i in np.arange(0.5, 1.0, 0.05)],
        'dropout': [round(i, 2) for i in np.arange(0,0.9, 0.05)],
        'batch_size': [16, 32, 64, 128],
        'power' : [0,1,2,2.5,3,3.5,4],
        'factor_mse':[1,10,100,1000,10000,50000],
        'deep_layer_trend': ['increasing', 'decreasing', 'random'],
    }
    
    start_time = time.time()
    # Generate all possible combinations of hyperparameters
    all_combinations = list(itertools.product(*search_space.values()))

    search_space_keys = list(search_space.keys())

    list_of_dicts = [{search_space_keys[i]: combination[i] for i in range(len(search_space_keys))} for combination in all_combinations]
    random.Random(4).shuffle(list_of_dicts)

    for i,ele in enumerate(list_of_dicts[:9999]): # set maimum number of trials that can be tested to 10000
        ele['trial'] = i+1 # index starts from 1 since 0 is reserved for default hyperparameters
        if ele['deep_layer_trend'] == 'increasing':
            start = random.sample(range(8, 257, 8), 1)[0]
            ele['deep_layer_size'] = str([start*i for i in range(1,ele['num_deep_layers']+1)])
        elif ele['deep_layer_trend'] == 'decreasing':
            end = random.sample(range(8, 257, 8), 1)[0]
            ele['deep_layer_size'] = str([end*i for i in range(ele['num_deep_layers']+1,1,-1)]) 
        elif ele['deep_layer_trend'] == 'random':
            ele['deep_layer_size'] = str(random.sample(range(8, 2049, 8), ele['num_deep_layers']))
        else:
            raise ValueError('Invalid deep_layer_trend, use "increasing", "decreasing", or "random"')
        ele['epochs'] = 20000
        ele['patience'] = 200
        ele['input_shape'] = (400,2)
        ele['loss_function'] = 'mse'
        ele['min_label'] = 33.0
        ele['max_label'] = 1000.0

    list_of_dicts.insert(0, default_hyperparameters)

    with open(list_of_dicts_dir, 'w') as f:
        json.dump(list_of_dicts[:10000], f)
    
    myexecute(f'echo "Created all search combinations in {time.time() - start_time} seconds"')


myexecute(f'echo "Starting hyperparmeter search for the index range {start_trial_index} to {end_trial_index} inclusive"')
# SBATCH parameters
job_name = "tuning"
output = f'{os.path.join(cur_log_dir,"output.%j")}'
error = f'{os.path.join(cur_log_dir,"error.%j")}'
partition = "short.q"
nodes = 1
cpus_per_task = 1
time_max = "00:10:00"
mem = "100GB"
gres = "gpu:1"
wrap_prefix = "module load anaconda/3/2021.11 && source activate /u/atya/conda-envs/tf-gpu4 && "
sbatch_prefix = f"sbatch --job-name={job_name} --output={output} --error={error} -p {partition} --export=ALL --nodes {nodes} --cpus-per-task={cpus_per_task} --time={time_max} --mem={mem} --gres={gres} "

def check_job_status(job_id):
    squeue_output = os.popen(f"squeue --job={job_id}").read().strip()
    if job_id not in squeue_output:
        return True
    else:
        return False

def check_incomplete_training(job_id,results_file_loc='logs/results_dict'):
    # Assuming the output dictionary is stored as a json file
    # replace this with your actual method of checking the output
    with open(f'{results_file_loc}.{job_id}', 'r') as f:
        output_log = f.read()
        if 'Stopped training due to time limit' in output_log:
            return True
        else:
            return False

def submit_jobs(trial_indices):
    active_jobs = {}
    for trial_index in trial_indices:
        while True:
            try:
                wrap = wrap_prefix + f"python3 hyperparameter_tuning_clean.py --hyperparameters_index {trial_index} --run runI --log_dir {log_dir} "
                result = subprocess.check_output(f'{sbatch_prefix} --wrap "{wrap}"', shell=True).decode("utf-8")
                job_id = result.split()[-1].strip()
                print(f"Submitted job for trial index {trial_index}: job number {job_id} ")
                active_jobs[job_id] = trial_index
                break
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job for trial index {trial_index}. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"Unhandled exception occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)
    return active_jobs

# def monitor_jobs(active_jobs):
#     incomplete_trials = []
#     while active_jobs:
#         for job_id, trial_index in active_jobs.copy().items():
#             if check_job_status(job_id):
#                 del active_jobs[job_id]
#                 if check_incomplete_training(trial_index):
#                     incomplete_trials.append(trial_index)
#         time.sleep(5)
#     return incomplete_trials

trial_indices = range(start_trial_index, end_trial_index + 1)
# while trial_indices:
#     active_jobs = submit_jobs(trial_indices)
    # trial_indices = monitor_jobs(active_jobs)
active_jobs = submit_jobs(trial_indices)
#save active jobs dictionary
with open(os.path.join(cur_log_dir,'jobid_index_dict.json'), 'w') as f:
    json.dump(active_jobs, f)
#os.system("python3 combine_results.py")