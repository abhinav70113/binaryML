#!/bin/bash

#SBATCH -o logs/output.fake.%j
#SBATCH -e logs/error.fake.%j
#SBATCH -D ./
#SBATCH -J fake
#SBATCH -p long.q
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task
#SBATCH --cpus-per-task=48  # assign one core to that first task
#SBATCH --time=20:00:00
#SBATCH --mem=300GB
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL --mail-user=s6abtyag@uni-bonn.de

module load anaconda/3/2021.11
source activate /u/atya/conda-envs/tf-gpu4
#combined_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
#time srun python fold_predicted_periods.py $SLURM_JOB_ID
#time srun python make_uniformz.py
time srun python create_fake_binaries_arg.py $SLURM_JOB_ID 208
#time srun python create_fake_binaries_arg.py $combined_id $SLURM_ARRAY_TASK_ID
#SBATCH --array=19-208


