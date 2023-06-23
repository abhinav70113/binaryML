#!/bin/bash

#SBATCH -o logs/output.train.%j
#SBATCH -e logs/error.train.%j
#SBATCH -D ./
#SBATCH -J train
#SBATCH -p short.q
#SBATCH --nodes=1             # request a full node
#SBATCH --cpus-per-task=1   # assign one core to that first task
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=50GB
#SBATCH --export=ALL



module load anaconda/3/2021.11
source activate /u/atya/conda-envs/tf-gpu4
#srun python make_uniformz.py
#time srun python single_index_predict.py $SLURM_JOB_ID attention > logs/output.train.$SLURM_JOB_ID
#time srun python chunk_classifier.py $SLURM_JOB_ID cnn > logs/output.train.$SLURM_JOB_ID
time srun python single_index_predict_resume.py $SLURM_JOB_ID cnn 985 13500 1272436 > logs/output.train.$SLURM_JOB_ID
#SBATCH --mail-type=END,FAIL --mail-user=s6abtyag@uni-bonn.de1

