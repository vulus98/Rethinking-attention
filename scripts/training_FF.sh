#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gpus=5
#SBATCH --gres=gpumem:35g
#SBATCH --mem-per-cpu=16000
#SBATCH --time 800
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u training_FF.py "$@"