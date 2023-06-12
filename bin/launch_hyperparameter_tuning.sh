#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --partition=long.q
#SBATCH --job-name=hyperparameter_tuning

# Temp file
temp_file=$1

# Get the JSON string for this job
json=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$temp_file")

# Call the Python script with the JSON string
python your_script.py "$json"
