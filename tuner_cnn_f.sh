#!/bin/bash

#SBATCH --array=101-1000
#SBATCH -o hyperparameter_tuning/cnn/tuner_%A_%a.out
#SBATCH -e hyperparameter_tuning/cnn/tuner_%A_%a.err
#SBATCH -D ./
#SBATCH -J tuner
#SBATCH -p gpu.q
#SBATCH --nodes=1             # request a full node
#SBATCH --cpus-per-task=1   # assign one core to that first task
#SBATCH --gres=gpu:1
#SBATCH --time=09:00:00
#SBATCH --mem=49GB
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL --mail-user=s6abtyag@uni-bonn.de1

module load anaconda/3/2021.11
source activate /u/atya/conda-envs/tf-gpu4

combined_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
time srun python Zpredict_cnn_paramtuner.py $combined_id $SLURM_ARRAY_TASK_ID #> hyperparameter_tuning/cnn/tuner_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out

