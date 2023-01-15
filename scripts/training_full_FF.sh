#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --time 600
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u training_full_FF.py --num_of_curr_trained_layer 5