#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --time 1000
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
source activate pytorch-transformer 
python -u scripts/baseline/training_script.py "$@"
