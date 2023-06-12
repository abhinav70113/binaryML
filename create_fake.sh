#!/bin/bash

#SBATCH -o logs/output.fold.%j
#SBATCH -e logs/error.fold.%j
#SBATCH -D ./
#SBATCH -J fold
#SBATCH -p short.q
#SBATCH --nodes=1             # request a full node
#SBATCH --ntasks-per-node=1   # only start 1 task
#SBATCH --cpus-per-task=48  # assign one core to that first task
#SBATCH --time=04:00:00
#SBATCH --mem=100GB
#SBATCH --export=ALL
#SBATCH --mail-type=END,FAIL --mail-user=s6abtyag@uni-bonn.de

module load anaconda/3/2021.11
source activate /u/atya/conda-envs/tf-gpu4
time srun python fold_predicted_periods.py $SLURM_JOB_ID
#time srun python make_uniformz.py
#srun python create_fake_binaries_arg.py $SLURM_JOB_ID 4 6

