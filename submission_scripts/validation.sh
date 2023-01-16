#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --time 180
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u scripts/full_sentence/validation_script.py "$@"